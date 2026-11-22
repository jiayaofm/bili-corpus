# -*- coding: utf-8 -*-
"""
鱼腥味检测器：区分「大肥鱼（鱼尚在）写的评论」和「真人写的评论」。

## 为什么需要它

项目总目标是「把大肥鱼的评论改写成真人口吻」。现有的分类器测的是
**通用 AI 腔**（长度、表情、破折号、复述标题），实测：

    大肥鱼写的    P(AI) 0.176
    真人写的      P(AI) 0.157

**它分不开。** 但人一眼就能看出「一股子鱼尚在味儿」——
说明鱼腥味不在现有的 40 个特征里，在**措辞层面**。

没有这个检测器，改写模型训完了也无法证明「鱼腥味真的去掉了」。

## 三条铁律（第一版全违反了，这些都是补丁）

### 1. 必须在同一视频内对比

拿「21 条鱼」对「8792 条全库真人」，模型可能学到的是
**「哪条视频」**或**「谁更长」**，不是**「谁写的」**。

所以：正负样本必须同 bvid，而且**负样本要按视频平衡采样**。

### 2. 样本不平衡会制造假指标

第一版跑出「准确率 0.990」——因为 99.8% 是负样本，
全判「真人」就有 0.99。这种数字毫无意义。

现在：每个视频按 `pos:neg = 1:k` 采样，并只报**平衡准确率**和 **AUC**。

### 3. 数据不够时必须拒绝出数

第一版在只有 21 条正样本的情况下照样打印 AUC 0.800，
那是 3 条正样本算出来的噪声。

现在：达不到门槛就明确说「数据不足」，需要 `--force` 才强行跑，
并且输出会标红警告。

## 门槛（可按需调）

```
视频数      ≥ 5
每视频正样本 ≥ 5
每视频负样本 ≥ 15
正样本总数   ≥ 60
```

## 用法

    python -X utf8 tools/persona_detector.py                 # 正常跑
    python -X utf8 tools/persona_detector.py --mode logodds  # 只看特征词
    python -X utf8 tools/persona_detector.py --force         # 数据不足也跑（看方向）

数据来源（按优先级自动找）：

    正样本（大肥鱼）  data/whale_comments/*.jsonl 的 comments 字段
                      → 没有就退回 data/rewrite_pairs/style_pairs.json 的 fish
    负样本（真人）    data/whale_comments/*.jsonl 的 real_comments 字段
                      → 没有就从 data/corpus.jsonl 里按 bvid 取
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

EMOJI_RE = re.compile(r"\[[^\[\]]{1,12}\]")
PUNCT_RE = re.compile(r"[，。！？、；：…—\-~·]")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")

MIN_VIDEOS = 5
MIN_POS_PER_VIDEO = 5
MIN_NEG_PER_VIDEO = 15
MIN_POS_TOTAL = 60


# ------------------------------------------------------------------ 数据
def load_pairs() -> tuple[list[dict], str]:
    """返回 [{bvid, pos: [文本], neg: [文本]}]，以及正样本来源说明。"""
    groups: dict[str, dict] = {}
    src_note = ""

    whale_files = sorted(glob.glob(str(ROOT / "data" / "whale_comments" / "*.jsonl")))
    used_whale = False
    for fp in whale_files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("skipped"):
                    continue
                bv = r.get("bvid") or ""
                if not bv:
                    continue
                g = groups.setdefault(bv, {"bvid": bv, "pos": [], "neg": []})
                # 正样本：rewrites 里的改写（鱼腔）
                # 负样本：同一条的 origin（真人原话）
                # 这样话题、内容都对齐，剩下的差异只能是「谁写的」。
                for rw in (r.get("rewrites") or []):
                    o = (rw.get("origin") or "").strip()
                    w = (rw.get("rewrite") or "").strip()
                    if o and w and o != w:
                        g["pos"].append(w)
                        g["neg"].append(o)
                        used_whale = True
                # 兼容旧格式：只有 comments/real_comments 时
                if not g["pos"]:
                    cm = [c.strip() for c in (r.get("comments") or []) if c and c.strip()]
                    rc = [c.strip() for c in (r.get("real_comments") or []) if c and c.strip()]
                    if cm and rc:
                        g["pos"] = cm
                        g["neg"] = rc
                        used_whale = True
    if used_whale:
        src_note = f"大肥鱼真实产出（{len(whale_files)} 个文件）"

    if not groups:
        p = ROOT / "data" / "rewrite_pairs" / "style_pairs.json"
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            for pair in d.get("pairs", []):
                bv = pair.get("bvid") or ""
                if not bv:
                    continue
                groups.setdefault(bv, {"bvid": bv, "pos": [], "neg": []})["pos"] += \
                    [c for c in (pair.get("fish") or []) if c]
            src_note = "⚠️ 我模仿人设写的（不是大肥鱼真实产出）"

    corpus = ROOT / "data" / "corpus.jsonl"
    if corpus.exists():
        by_bv: dict[str, list[str]] = collections.defaultdict(list)
        with corpus.open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                c = (r.get("content") or "").strip()
                bv = r.get("bvid") or ""
                # 去掉 @ 开头的回复：那是对着某个人说话，形态不同，会污染
                if bv and 4 <= len(c) <= 200 and not c.startswith("@"):
                    by_bv[bv].append(c)
        for bv, g in groups.items():
            if not g["neg"]:
                g["neg"] = by_bv.get(bv, [])

    return [g for g in groups.values() if g["pos"] and g["neg"]], src_note


def balance(groups: list[dict], k: int, seed: int = 42) -> list[dict]:
    """每个视频按 pos:neg = 1:k 采样，消灭样本不平衡。"""
    rng = random.Random(seed)
    out = []
    for g in groups:
        if not g["pos"] or not g["neg"]:
            continue
        want = min(len(g["neg"]), max(MIN_NEG_PER_VIDEO, len(g["pos"]) * k))
        neg = rng.sample(g["neg"], want)
        out.append({"bvid": g["bvid"], "pos": g["pos"], "neg": neg})
    return out


def check_sufficient(groups: list[dict]) -> list[str]:
    probs = []
    if len(groups) < MIN_VIDEOS:
        probs.append(f"视频只有 {len(groups)} 个，需要 ≥{MIN_VIDEOS}")
    thin = [g["bvid"] for g in groups if len(g["pos"]) < MIN_POS_PER_VIDEO]
    if thin:
        probs.append(f"{len(thin)} 个视频的正样本 <{MIN_POS_PER_VIDEO} 条")
    total = sum(len(g["pos"]) for g in groups)
    if total < MIN_POS_TOTAL:
        probs.append(f"正样本总共 {total} 条，需要 ≥{MIN_POS_TOTAL}")
    noneg = [g["bvid"] for g in groups if len(g["neg"]) < MIN_NEG_PER_VIDEO]
    if noneg:
        probs.append(f"{len(noneg)} 个视频的真人对照 <{MIN_NEG_PER_VIDEO} 条")
    return probs


def ngrams(text: str, lo: int = 1, hi: int = 3) -> list[str]:
    toks: list[str] = []
    for n in range(lo, hi + 1):
        for i in range(len(text) - n + 1):
            s = text[i:i + n]
            if CJK_RE.search(s):
                toks.append(s)
    return toks


# ------------------------------------------------------------------ 方法一：对数几率比
def logodds_analysis(groups: list[dict], top: int = 20) -> None:
    """信息先验 Dirichlet 的对数几率比（Monroe et al. 2008）。

    只在平衡后的样本上算，避免「21 条 vs 8792 条」把 z 值推到荒谬的量级。
    """
    cnt_pos: collections.Counter = collections.Counter()
    cnt_neg: collections.Counter = collections.Counter()
    for g in groups:
        for t in g["pos"]:
            cnt_pos.update(ngrams(t))
        for t in g["neg"]:
            cnt_neg.update(ngrams(t))

    n_pos = sum(cnt_pos.values())
    n_neg = sum(cnt_neg.values())
    if n_pos == 0 or n_neg == 0:
        print("  [skip] 一侧为空")
        return

    vocab = set(cnt_pos) | set(cnt_neg)
    # 先验强度取常数，不取词汇表大小——取词汇表会把平滑吃掉一切，
    # 把 z 推到几百（实测 827）。500 是经验值，z 落在个位数到十几。
    A0 = 500.0
    n_tot = n_pos + n_neg
    scores = []
    for w in vocab:
        cp, cn = cnt_pos.get(w, 0), cnt_neg.get(w, 0)
        if cp + cn < 5:
            continue
        a_w = A0 * (cp + cn) / n_tot          # 信息先验：背景频率
        p = (cp + a_w) / (n_pos + A0)
        q = (cn + a_w) / (n_neg + A0)
        p = min(max(p, 1e-9), 1 - 1e-9)
        q = min(max(q, 1e-9), 1 - 1e-9)
        var = 1.0 / (cp + a_w) + 1.0 / (cn + a_w)
        z = (math.log(p / (1 - p)) - math.log(q / (1 - q))) / math.sqrt(var)
        if not CJK_RE.search(w) and w.strip() not in '[笑哭]':
            continue
        scores.append((z, w, cp, cn))

    if not scores:
        print("  [skip] 没有出现 ≥3 次的 n-gram，样本太少")
        return

    scores.sort(reverse=True)
    print(f"  平衡后：正样本 {n_pos} 个 n-gram / 负样本 {n_neg} 个 / 词汇表 {len(vocab)}")
    print()

    fish = [x for x in scores if x[0] > 0][:top]
    human = [x for x in reversed(scores) if x[0] < 0][:top]

    print("  ── 鱼腥味候选（z 越大越像鱼尚在）──")
    if not fish:
        print("    （一条都没有：正负两边没有区分度够的 n-gram）")
    for z, w, cp, cn in fish:
        print(f"    z={z:6.2f}  {w!r:12}  鱼 {cp:>3} / 真人 {cn:>3}")
    print()
    print("  ── 真人更常用的（对照）──")
    for z, w, cp, cn in human:
        print(f"    z={z:6.2f}  {w!r:12}  鱼 {cp:>3} / 真人 {cn:>3}")


# ------------------------------------------------------------------ 方法二：分类器 + 按视频留一
def cv_analysis(groups: list[dict], balanced: bool = True) -> dict:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline

    bvids = [g["bvid"] for g in groups]
    texts, ys, gid = [], [], []
    for i, g in enumerate(groups):
        for t in g["pos"]:
            texts.append(t); ys.append(1); gid.append(i)
        for t in g["neg"]:
            texts.append(t); ys.append(0); gid.append(i)
    y = np.array(ys)
    gid = np.array(gid)

    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    print(f"  样本 {len(texts)} 条（鱼 {n_pos} / 真人 {n_neg}），视频 {len(groups)} 个")
    if not balanced:
        print(f"  ⚠️ 未平衡：负样本是正样本的 {n_neg / max(1, n_pos):.1f} 倍，"
              f"「准确率」会被负类主导，没有意义")

    aurocs, baccs, skipped = [], [], 0
    for k in range(len(groups)):
        tr, te = gid != k, gid == k
        if len(set(y[tr])) < 2 or y[te].sum() == 0 or y[te].sum() == te.sum():
            skipped += 1
            continue
        clf = make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=1, sublinear_tf=True),
            LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced"),
        )
        clf.fit([texts[i] for i in np.where(tr)[0]], y[tr])
        prob = clf.predict_proba([texts[i] for i in np.where(te)[0]])[:, 1]
        try:
            aurocs.append(roc_auc_score(y[te], prob))
        except ValueError:
            skipped += 1
            continue
        pred = (prob >= 0.5).astype(int)
        # 平衡准确率：(正类召回 + 负类召回) / 2，不受类别比例影响
        tpr = float((pred[y[te] == 1] == 1).mean())
        tnr = float((pred[y[te] == 0] == 0).mean())
        baccs.append((tpr + tnr) / 2)

    print()
    if not aurocs:
        print("  [skip] 没有可用测试折")
        return {"auc": None, "folds": 0}
    print(f"  ── 按视频留一（{len(aurocs)} 折，跳过 {skipped} 折）──")
    print(f"    AUC        均值 {statistics.mean(aurocs):.3f}   "
          f"中位 {statistics.median(aurocs):.3f}   "
          f"范围 {min(aurocs):.3f}~{max(aurocs):.3f}")
    print(f"    平衡准确率  均值 {statistics.mean(baccs):.3f}")
    if len(aurocs) < MIN_VIDEOS:
        print(f"    ⚠️ 只有 {len(aurocs)} 折，这个数字只能当方向参考，不能当结论")

    # 全量训练一次看特征重要性——诚实检查：是不是只看长度
    clf = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=1, sublinear_tf=True),
        LogisticRegression(C=0.3, max_iter=2000, class_weight="balanced"),
    )
    clf.fit(texts, y)
    names = clf.named_steps["tfidfvectorizer"].get_feature_names_out()
    coefs = clf.named_steps["logisticregression"].coef_[0]
    order = np.argsort(coefs)
    print()
    print("  ── 模型最依赖的特征（正=判成鱼，负=判成真人）──")
    for i in list(order[-8:][::-1]) + list(order[:8]):
        print(f"    {coefs[i]:+7.2f}  {names[i]!r}")
    return {"auc": statistics.mean(aurocs), "folds": len(aurocs)}


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["logodds", "cv", "all"], default="all")
    ap.add_argument("--neg-per-pos", type=int, default=3)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--force", action="store_true", help="数据不足也跑（只看方向）")
    a = ap.parse_args()

    raw, note = load_pairs()
    if not raw:
        print("  [error] 没找到配对数据。")
        print("          需要 data/whale_comments/*.jsonl，或 data/rewrite_pairs/style_pairs.json")
        return 1

    print("=" * 66)
    print("  鱼腥味检测器 —— 大肥鱼 vs 真人")
    print("=" * 66)
    print(f"  正样本来源: {note or '未知'}")
    print(f"  原始：视频 {len(raw)} 个 / 鱼 {sum(len(g['pos']) for g in raw)} 条"
          f" / 真人 {sum(len(g['neg']) for g in raw)} 条")

    probs = check_sufficient(raw)
    if probs and not a.force:
        print()
        print("  ❌ 数据不足，拒绝出数（避免打印虚假指标）")
        for p in probs:
            print(f"     · {p}")
        print()
        print("  → 等 data/whale_comments/*.jsonl 到位后重跑")
        print("  → 或者加 --force 强行跑，但结果只能当方向参考")
        return 2
    if probs:
        print()
        print("  ⚠️ 数据不足，--force 强跑；以下数字只能当方向参考，不能写进结论")
        for p in probs:
            print(f"     · {p}")

    groups = balance(raw, a.neg_per_pos)
    print(f"  平衡后：视频 {len(groups)} 个 / 鱼 {sum(len(g['pos']) for g in groups)} 条"
          f" / 真人 {sum(len(g['neg']) for g in groups)} 条（pos:neg = 1:{a.neg_per_pos}）")
    print()

    if a.mode in ("logodds", "all"):
        print("【一】鱼腥味是什么（对数几率比）")
        logodds_analysis(groups, a.top)
        print()

    if a.mode in ("cv", "all"):
        print("【二】能不能分辨出来（按视频留一）")
        cv_analysis(groups, balanced=True)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
