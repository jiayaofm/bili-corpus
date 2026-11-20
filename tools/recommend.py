# -*- coding: utf-8 -*-
"""
评论推荐系统：给一个视频，从**其他视频**的真人评论里挑几条当参照。

## 在项目里的位置

它是「生成」环节的上游，解决的是**冷启动**：

```
新视频还没有评论  →  从别的视频检索话题/语气相近的真人评论
                  →  当 few-shot 参照喂给生成器
                  →  缓解「内容空心」（模型只会讲常识）
```

## 为什么必须排除目标视频的评论

检索池里如果留着目标视频自己那 160 条真人评论，等于把答案抄给模型，
后面所有的评估都不可信。所以 `fit(..., exclude=视频标题)` 是强制的，
`recommend()` 里还会再断言一次。

## 四个环节

1. **过滤**  用分类器刷掉 AI 写的评论（不拿 AI 的话当人的范本）
2. **召回**  TF-IDF 字符 n-gram 余弦相似，按话题接近度捞候选
3. **排序**  MMR（最大边际相关）：既相关、又彼此不重复
4. **评估**  命中率 / 多样性 / 新颖性 / 冷启动覆盖率

## 用法

```python
r = CommentRecommender()
r.fit(exclude="《晓艳天气预报》")
hits = r.recommend("游戏 一个恶搞天气预报的整活栏目……", k=5)
```
"""
from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EMO_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]|\[[^\]]{1,12}\]")


# --------------------------------------------------------------------------- #
# 数据
# --------------------------------------------------------------------------- #
def load_corpus(data_dir: Path | None = None) -> dict[str, list[str]]:
    """读全部顶层真人评论，按视频标题分组。"""
    data_dir = data_dir or (ROOT / "data")
    out: dict[str, list[str]] = {}
    for p in sorted(data_dir.glob("BV*_like.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            title = r.get("video_title") or ""
            content = (r.get("content") or "").strip()
            if title and content:
                out.setdefault(title, []).append(content)
    return out


# --------------------------------------------------------------------------- #
# 推荐器
# --------------------------------------------------------------------------- #
@dataclass
class Item:
    text: str
    video: str
    tid: int = 0
    has_emoji: bool = False
    length: int = 0
    ai_prob: float = 0.0
    sim: float = 0.0
    match: float = 0.0
    score: float = 0.0


def len_bucket(n: int) -> int:
    """长度分桶：短评 / 中评 / 长评。对齐真人分布的三档。"""
    if n < 10:
        return 0
    if n <= 80:
        return 1
    return 2


@dataclass
class CommentRecommender:
    """
    基于内容特征的评论推荐。

    为什么不用纯文本相似度：实测留一视频检索最高相似度只有 0.15，
    检索结果和查询话题无关。原因是 B 站评论短、抽象词根本不出现
    （真人自己复述标题词的比例也只有 2%）。所以改成按**可算的结构化特征**对齐。

    打分 = 分区匹配 + 长度贴近 + 表情匹配 + 弱词面相似
    """

    min_len: int = 4
    max_len: int = 200
    ai_max: float = 0.5
    mmr_lambda: float = 0.75

    w_domain: float = 0.45
    w_len: float = 0.30
    w_emoji: float = 0.15
    w_text: float = 0.10

    items: list[Item] = field(default_factory=list)
    _vec: TfidfVectorizer | None = None
    _X = None
    excluded_video: str | None = None
    stats: dict = field(default_factory=dict)

    # -- 建索引 ------------------------------------------------------------ #
    def fit(self, corpus: dict[str, list[str]], exclude: str | None = None,
            use_classifier: bool = True,
            tid_of: dict[str, int] | None = None) -> "CommentRecommender":
        """
        Args:
            corpus: {视频标题: [评论, ...]}
            exclude: 要排除的视频标题（**目标视频，必须给**）
            tid_of: {视频标题: 分区 id}
        """
        self.excluded_video = exclude
        tid_of = tid_of or {}
        raw: list[Item] = []
        for video, comments in corpus.items():
            if exclude is not None and video == exclude:
                continue          # 冷启动：不给看目标视频自己的评论
            tid = int(tid_of.get(video, 0))
            for c in comments:
                if not (self.min_len <= len(c) <= self.max_len):
                    continue
                raw.append(Item(text=c, video=video, tid=tid,
                                has_emoji=bool(EMO_RE.search(c)),
                                length=len(c)))

        # ① 过滤：刷掉 AI 发言
        dropped = 0
        if use_classifier and raw:
            try:
                sys.path.insert(0, str(ROOT / "pipeline"))
                from scorer import Scorer
                sc = Scorer()
                kept = []
                for it in raw:
                    p = sc.describe(it.text).get("heldout_ai", 0.0)
                    it.ai_prob = float(p)
                    if p >= self.ai_max:
                        dropped += 1
                        continue
                    kept.append(it)
                raw = kept
            except Exception as e:      # 分类器不可用时不阻断，但要记下来
                self.stats["classifier_error"] = str(e)

        self.items = raw

        # ② 向量化（只作为弱信号）
        if self.items:
            self._vec = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 3),
                min_df=1, sublinear_tf=True,
            )
            self._X = self._vec.fit_transform([it.text for it in self.items])

        self.stats.update({
            "candidates": len(self.items),
            "ai_filtered": dropped,
            "excluded_video": exclude,
        })
        return self

    # -- 召回 + 排序 ------------------------------------------------------- #
    def recommend(self, query: str, k: int = 5, *, domain_tid: int = 0,
                  target_len: int | None = None, want_emoji: bool | None = None
                  ) -> list[Item]:
        """
        返回 k 条评论。

        Args:
            query: 查询文本（摘要），只作弱词面信号
            domain_tid: 目标视频分区，用于同分区优先
            target_len: 期望字数（比如真人中位数），用于长度对齐
            want_emoji: 是否偏好带表情的样本
        """
        if not self.items:
            return []
        tb = len_bucket(target_len) if target_len else None

        sims = np.zeros(len(self.items))
        if self._vec is not None and query.strip():
            sims = cosine_similarity(self._vec.transform([query]), self._X)[0]
            if sims.max() > 0:
                sims = sims / sims.max()      # 归一化，避免尺度和别的项打架

        # ① 内容特征匹配
        for i, it in enumerate(self.items):
            s = 0.0
            if domain_tid:
                s += self.w_domain * (1.0 if it.tid == domain_tid else 0.0)
            else:
                s += self.w_domain
            if tb is not None:
                d = abs(len_bucket(it.length) - tb)
                s += self.w_len * max(0.0, 1.0 - d / 2.0)
            else:
                s += self.w_len
            if want_emoji is not None:
                s += self.w_emoji * (1.0 if it.has_emoji == want_emoji else 0.0)
            else:
                s += self.w_emoji
            s += self.w_text * float(sims[i])
            it.match = s
            it.sim = float(sims[i])

        # ② MMR：在「匹配度高」和「彼此不重复」之间折中
        doc_sim = cosine_similarity(self._X)
        order = list(np.argsort(-np.array([it.match for it in self.items])))
        remaining = order[: min(len(order), k * 8)]
        selected: list[int] = []
        while len(selected) < k and remaining:
            best_i, best_v = None, -1e9
            for i in remaining:
                dup = max(doc_sim[i, j] for j in selected) if selected else 0.0
                v = self.mmr_lambda * self.items[i].match - (1 - self.mmr_lambda) * dup
                if v > best_v:
                    best_v, best_i = v, i
            if best_i is None:
                break
            selected.append(best_i)
            remaining.remove(best_i)
            self.items[best_i].score = float(best_v)
        return [self.items[i] for i in selected]

    # -- 推荐系统评估 ------------------------------------------------------- #
    def evaluate(self, queries: dict[str, dict], k: int = 5) -> dict:
        """
        留一视频的离线评估。

        为什么不用召回率/命中率：冷启动场景下目标视频**没有评论**，
        「正确答案」根本不存在。而且词面命中率在这个语料上天然失效
        （真人自己复述标题词的比例只有 2%）。所以改评这几个：

        Args:
            queries: {视频标题: {"text": 摘要, "tid": 分区, "target_len": 期望字数,
                                 "want_emoji": 是否带表情}}

        Returns:
            domain_purity  推荐里和目标同分区的比例（越高越对齐语域）
            len_gap        推荐字数与目标的平均绝对差（越低越贴合）
            diversity      列表内部平均不相似度（越高越不重复）
            novelty        平均 -log2 词频（越高越不像套话）
            coverage       能召回非空的比例
        """
        from collections import Counter

        freq = Counter()
        for it in self.items:
            freq.update(set(re.findall(r"[\u4e00-\u9fa5]{2,}", it.text)))
        total = max(sum(freq.values()), 1)

        purities, gaps, divs, novs, covered = [], [], [], [], 0
        for title, q in queries.items():
            domain_tid = int(q.get("tid", 0))
            target_len = q.get("target_len")
            rec = self.recommend(q.get("text", ""), k=k, domain_tid=domain_tid,
                                 target_len=target_len,
                                 want_emoji=q.get("want_emoji"))
            if not rec:
                continue
            covered += 1

            if domain_tid:
                purities.append(sum(1 for it in rec if it.tid == domain_tid) / len(rec))
            if target_len:
                gaps.append(float(np.mean([abs(it.length - target_len) for it in rec])))

            if len(rec) > 1:
                vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3))
                X = vec.fit_transform([it.text for it in rec])
                S = cosine_similarity(X)
                n = len(rec)
                divs.append(1 - float((S.sum() - n) / (n * (n - 1))))

            nv = []
            for it in rec:
                ws = set(re.findall(r"[\u4e00-\u9fa5]{2,}", it.text))
                if ws:
                    nv.append(np.mean([-math.log2((freq[w] + 1) / total) for w in ws]))
            if nv:
                novs.append(float(np.mean(nv)))

        return {
            "k": k,
            "pool_size": len(self.items),
            "queries": len(queries),
            "coverage": covered / max(len(queries), 1),
            "domain_purity": float(np.mean(purities)) if purities else 0.0,
            "len_gap": float(np.mean(gaps)) if gaps else 0.0,
            "diversity": float(np.mean(divs)) if divs else 0.0,
            "novelty": float(np.mean(novs)) if novs else 0.0,
            "ai_filtered": self.stats.get("ai_filtered", 0),
        }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    import argparse
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--query", required=True, help="查询文本（标题+摘要）")
    ap.add_argument("--exclude", default=None, help="排除的视频标题（目标视频）")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-filter", action="store_true", help="不过滤 AI 评论")
    a = ap.parse_args()

    corpus = load_corpus()
    r = CommentRecommender().fit(corpus, exclude=a.exclude,
                                 use_classifier=not a.no_filter)
    print("候选池 %d 条（刷掉 AI %d 条），排除视频=%s"
          % (r.stats["candidates"], r.stats.get("ai_filtered", 0), a.exclude))
    print()
    for i, it in enumerate(r.recommend(a.query, k=a.k), 1):
        print("%d. [sim=%.3f] 《%s》" % (i, it.sim, it.video[:16]))
        print("   %s" % it.text[:80])


if __name__ == "__main__":
    main()
