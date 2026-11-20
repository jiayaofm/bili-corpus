# -*- coding: utf-8 -*-
"""
对照实验：词面检索 vs 语义检索。

用户的问题：检索差是因为数据量不够，还是查询词不好？
我的假设：**根因是表示方法**——TF-IDF 只数字面重合，不懂意思。

做法：同一批 1053 条评论、同一批查询，只换表示方法：

    A  TF-IDF 字符 n-gram        （现在的做法）
    B  Qwen3-1.7B 隐状态均值池化  （语义向量）

如果 B 明显好于 A，说明不是数据量问题，是表示问题。
如果 B 也很差，才需要用「多爬数据」来解释。

用法：
    python tools/test_semantic_retrieval.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.stdout.reconfigure(encoding="utf-8")

DOM = {
    "BV1w6ZEYrEsb": "美食", "BV1ArZnYUE9a": "游戏", "BV1H5Z4YhEsn": "游戏",
    "BV1stZAYvELe": "鬼畜", "BV1Qx4g6JEsr": "整活", "BV1edWM63Ewd": "科技",
    "BV1Sbnz6EEix": "科技",
}

# 目标视频 -> (查询摘要, 真人评论里「和这个话题有关」的关键词，用于粗判检索对不对)
QUERIES = {
    "《晓艳天气预报》": (
        "一个恶搞天气预报的整活栏目，主播用正经的新闻播报腔念离谱的天气内容，"
        "包袱全在一本正经说瞎话这个反差上",
        ["天气预报", "播报", "台词", "稿子", "念", "正经", "整活", "搞笑", "笑", "演"],
    ),
    '为什么中国人对一道菜的最高评价是"下饭"?': (
        "从主食结构和饮食习惯聊为什么中国人评价一道菜的最高标准是下饭，"
        "聊到米饭在日常饮食里的地位、汤汁咸鲜对下饭的作用",
        ["下饭", "米饭", "主食", "菜", "吃", "味道", "咸", "汁", "汤", "胃口"],
    ),
    "Claude Fable 5.2实测：94分刷爆了我们的私有题库，但我真用不起了！": (
        "用私有编程题库测 Claude 得 94 分，比 Opus 高四分，但要六小时八十一美元，"
        "日常用不起，建议还是用 Opus",
        ["claude", "opus", "模型", "贵", "订", "美元", "钱", "跑分", "编程", "代码"],
    ),
    "海龟汤（2）": (
        "海龟汤推理游戏第二期，主持人给出离奇故事开头，其他人提问还原真相，"
        "只能回答是或不是、不重要",
        ["海龟汤", "汤面", "汤底", "推理", "故事", "猜", "答案", "问题", "恐怖"],
    ),
    "666，大肥鱼被弹幕骗了！以为自己赚了几万亿！结果大梦一场！": (
        "观众用弹幕假装给主播刷了几万亿礼物，主播以为真赚到了，最后发现是整活",
        ["肥鱼", "鱼", "弹幕", "骗", "礼物", "钱", "赚", "直播", "刷"],
    ),
    "不可能存在方的鸡！除非……": (
        "用诡辩和科学论证为什么不可能存在方的鸡，除非",
        ["鸡", "方", "圆", "羽毛", "物理", "麦块", "梗", "羊驼"],
    ),
    "有人问我怎么接入B站？其实这个号从头到尾都是我自己在运营": (
        "讲怎么把自己的 AI 接入 B 站评论区，其实这个号一直是 up 自己在运营",
        ["ai", "bot", "接入", "评论", "号", "运营", "模型", "自动", "封"],
    ),
}


def load_corpus():
    items, titles = [], []
    for p in sorted((ROOT / "data").glob("BV*_like.jsonl")):
        bv = "BV" + p.name.split("BV")[-1].split("_")[0]
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            c = (r.get("content") or "").strip()
            if c and 4 <= len(c) <= 200:
                items.append(c)
                titles.append(r.get("video_title") or "")
    return items, titles


def tfidf_retrieve(items, titles, query, exclude, k):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    idx = [i for i, t in enumerate(titles) if t != exclude]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=1, sublinear_tf=True)
    X = vec.fit_transform([items[i] for i in idx])
    s = cosine_similarity(vec.transform([query]), X)[0]
    top = np.argsort(-s)[:k]
    return [items[idx[i]] for i in top], [float(s[i]) for i in top]


def semantic_embed(texts, batch=16, pool="mean", center=True):
    """
    Qwen3 隐状态向量。

    踩过的坑：直接用「最后一个 token」的隐状态，所有文本余弦都挤在 0.42~0.45，
    检索全乱（给「天气预报」推「下饭」）。这是因果语言模型的**各向异性**问题：
    表示都挤在同一个锥里，彼此天然相似。

    修法两条一起上：
      1. 均值池化（按 attention_mask 加权平均），比取最后一位稳
      2. 去中心化（减去整体均值），把公共主轴去掉
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "Qwen/Qwen3-1.7B"
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.bfloat16, device_map="cuda",
    ).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            enc = tok(chunk, return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(model.device)
            h = model(**enc, output_hidden_states=True).hidden_states[-1]
            if pool == "mean":
                m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
                v = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            else:
                last = enc["attention_mask"].sum(dim=1) - 1
                v = h[torch.arange(h.size(0)), last]
            out.append(v.float().cpu().numpy())
    del model
    torch.cuda.empty_cache()
    X = np.vstack(out)

    if center:
        mu = X.mean(axis=0, keepdims=True)
        X = X - mu
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return X


def hit_ratio(texts, kws):
    if not kws:
        return 0.0
    hits = 0
    for t in texts:
        tl = t.lower()
        if any(kw.lower() in tl for kw in kws):
            hits += 1
    return hits / len(texts)


def main():
    items, titles = load_corpus()
    print("池子 %d 条 / %d 个视频" % (len(items), len(set(titles))))
    print()

    # 语义向量：全库算一次，查询再算
    # 注意：去中心化的均值必须来自**全库**，查询要用同一个均值减，否则两边不在一个空间
    print("计算 Qwen3-1.7B 语义向量…")
    E_all_raw = semantic_embed(items, center=False)
    q_texts = [v[0] for v in QUERIES.values()]
    E_q_raw = semantic_embed(q_texts, center=False)
    mu = E_all_raw.mean(axis=0, keepdims=True)

    def _norm(X):
        X = X - mu
        return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    E_all, E_q = _norm(E_all_raw), _norm(E_q_raw)
    print("  完成：评论 %s  查询 %s" % (E_all.shape, E_q.shape))
    print("  向量退化的自检：随机两条不相关评论的余弦应在 0.0±0.2 附近")
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, len(E_all), 500), rng.integers(0, len(E_all), 500)
    base = float(np.mean(np.sum(E_all[a] * E_all[b], axis=1)))
    print("  随机配对平均余弦 = %.3f   %s"
          % (base, "正常" if abs(base) < 0.25 else "仍偏高，向量还是退化的"))
    print()

    K = 5
    print("%-34s %10s %10s" % ("目标视频", "TF-IDF", "Qwen语义"))
    print("-" * 58)
    rows_a, rows_b = [], []
    for qi, (title, (query, kws)) in enumerate(QUERIES.items()):
        # A 词面
        topA, simA = tfidf_retrieve(items, titles, query, title, K)
        hA = hit_ratio(topA, kws)
        # B 语义
        cand = np.array([i for i, t in enumerate(titles) if t != title])
        s = E_all[cand] @ E_q[qi]
        topB = [items[cand[j]] for j in np.argsort(-s)[:K]]
        hB = hit_ratio(topB, kws)
        rows_a.append(hA)
        rows_b.append(hB)
        print("%-34s %9.0f%% %9.0f%%" % (title[:32], 100 * hA, 100 * hB))
    print("-" * 58)
    print("%-34s %9.0f%% %9.0f%%" % ("平均命中（含话题关键词）",
                                     100 * np.mean(rows_a), 100 * np.mean(rows_b)))
    print()

    # 举例
    print("=" * 70)
    t = "《晓艳天气预报》"
    query, kws = QUERIES[t]
    print("举例：《%s》" % t)
    print("查询：%s" % query[:52])
    print()
    topA, simA = tfidf_retrieve(items, titles, query, t, K)
    print("【TF-IDF】")
    for x, s in zip(topA, simA):
        print("  [%.3f] %s" % (s, x[:56]))
    print()
    qi = list(QUERIES).index(t)
    cand = np.array([i for i, tt in enumerate(titles) if tt != t])
    s = E_all[cand] @ E_q[qi]
    print("【Qwen 语义】")
    for j in np.argsort(-s)[:K]:
        print("  [%.3f] %s" % (s[j], items[cand[j]][:56]))


if __name__ == "__main__":
    main()
