# -*- coding: utf-8 -*-
"""
语料分析 —— 为「AI 说话像人」项目出第一组指标

输出：
  1. 长度分布（AI 味最直观的一个维度）
  2. 标点密度（破折号/顿号/引号是 AI 文本的强特征）
  3. B站表情标签使用率（[汤圆] [滑稽] 这类是极强的人类信号）
  4. 高赞 vs 低赞的长度对比
  5. 口语/网络用语命中率
  6. LIKE vs TIME 两种排序的差异
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(__file__).parent / "data"

PUNCT = ["，", "。", "！", "？", "、", "「", "」", "“", "”", "—", "…", "~", "；", "："]
EMOTE_RE = re.compile(r"\[[^\[\]]{1,12}\]")          # B站表情标签
URL_RE = re.compile(r"https?://\S+")
MENTION_RE = re.compile(r"@[^\s@]{1,20}")

# 口语 / 网络用语（粗筛，仅示意口径）
COLLOQUIAL = ["卧槽", "我靠", "特么", "妈的", "牛逼", "离谱", "恶心", "笑死", "草",
              "啊这", "哈哈", "属实", "绷不住", "逆天", "整挺", "绝了", "乐",
              "淦", "服了", "无语", "拉胯", "菜", "卷", "破防", "急了", "典"]
# AI 典型套话（用来反证）
AI_CLICHE = ["总的来说", "值得注意的是", "首先", "其次", "最后", "综上所述",
             "不仅", "而且", "因此", "由此可见", "一方面", "另一方面", "极其重要"]


def load(name):
    p = DATA / name
    if not p.exists():
        return []
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def bucket_report(recs, label):
    if not recs:
        return
    lens = sorted(len(r["content"]) for r in recs)
    n = len(lens)
    print(f"\n【{label}】{n} 条")
    print(f"  平均 {sum(lens)/n:.1f} 字   中位 {lens[n//2]} 字   "
          f"最短 {lens[0]}   最长 {lens[-1]}")
    buckets = [(1, 5), (6, 15), (16, 30), (31, 60), (61, 120), (121, 10**9)]
    print("  长度分布:")
    for lo, hi in buckets:
        c = sum(1 for L in lens if lo <= L <= hi)
        pct = c / n * 100
        lab = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
        print(f"    {lab:>8} 字  {c:>4} ({pct:>5.1f}%) {'█' * int(pct / 2)}")


def punct_report(recs, label):
    if not recs:
        return
    total_chars = sum(len(r["content"]) for r in recs)
    print(f"\n【{label} · 标点密度】（每万字出现次数）")
    for p in PUNCT:
        c = sum(r["content"].count(p) for r in recs)
        if c:
            print(f"    {p}  {c:>6} 次   密度 {c / total_chars * 10000:>7.1f} /万字")


def feature_report(recs, label):
    if not recs:
        return
    n = len(recs)
    emote = sum(1 for r in recs if EMOTE_RE.search(r["content"]))
    url = sum(1 for r in recs if URL_RE.search(r["content"]))
    mention = sum(1 for r in recs if MENTION_RE.search(r["content"]))
    multi_line = sum(1 for r in recs if "\n" in r["content"])
    col = sum(1 for r in recs if any(w in r["content"] for w in COLLOQUIAL))
    cliche = sum(1 for r in recs if any(w in r["content"] for w in AI_CLICHE))

    print(f"\n【{label} · 人类信号】{n} 条")
    print(f"    含 B站表情标签 [xx]  {emote:>4} ({emote/n*100:.1f}%)   ← 人味最重的信号")
    print(f"    含 @提及            {mention:>4} ({mention/n*100:.1f}%)")
    print(f"    含链接              {url:>4} ({url/n*100:.1f}%)")
    print(f"    多行（换行分段）    {multi_line:>4} ({multi_line/n*100:.1f}%)")
    print(f"    含口语脏话/梗词     {col:>4} ({col/n*100:.1f}%)")
    print(f"    含 AI 套话          {cliche:>4} ({cliche/n*100:.1f}%)   ← 越低越好")


def like_split(recs, label):
    if len(recs) < 40:
        return
    srt = sorted(recs, key=lambda x: -x["like"])
    q = len(srt) // 4
    top, bottom = srt[:q], srt[-q:]
    at = sum(len(r["content"]) for r in top) / len(top)
    ab = sum(len(r["content"]) for r in bottom) / len(bottom)
    et = sum(1 for r in top if EMOTE_RE.search(r["content"])) / len(top) * 100
    eb = sum(1 for r in bottom if EMOTE_RE.search(r["content"])) / len(bottom) * 100
    print(f"\n【{label} · 高赞 vs 低赞】")
    print(f"    高赞 25%: 平均 {at:>5.1f} 字   表情率 {et:>4.1f}%")
    print(f"    低赞 25%: 平均 {ab:>5.1f} 字   表情率 {eb:>4.1f}%")
    print(f"    → 高赞更{'长' if at > ab else '短'} {abs(at-ab):.1f} 字")


def top_ngrams(recs, label, k=15):
    """二元组，看真实用词，而不是我们的预设词表"""
    cnt = Counter()
    for r in recs:
        t = re.sub(EMOTE_RE, "", r["content"])
        t = re.sub(r"[\s\W_]+", "", t)
        for i in range(len(t) - 1):
            cnt[t[i:i+2]] += 1
    print(f"\n【{label} · 高频二字组（Top {k}）】")
    print("    " + "  ".join(f"{w}:{c}" for w, c in cnt.most_common(k)))


def main():
    like = load("BV1YxWx69Ens_like.jsonl")
    time_ = load("BV1YxWx69Ens_time.jsonl")

    print("=" * 68)
    print("B站评论语料分析 · BV1YxWx69Ens（BLG vs T1 锐评）")
    print("=" * 68)

    bucket_report(like, "LIKE 排序")
    bucket_report(time_, "TIME 排序")

    punct_report(like, "LIKE 排序")
    feature_report(like, "LIKE 排序")
    feature_report(time_, "TIME 排序")
    like_split(like, "LIKE 排序")
    top_ngrams(like, "LIKE 排序")

    print("\n" + "=" * 68)
    print("解读要点")
    print("=" * 68)
    allr = like + time_
    if allr:
        n = len(allr)
        short = sum(1 for r in allr if len(r["content"]) <= 5)
        emote = sum(1 for r in allr if EMOTE_RE.search(r["content"]))
        print(f"  · ≤5 字短评占比 {short/n*100:.1f}% —— 这是'随口说话'的浓度指标")
        print(f"  · 表情标签使用率 {emote/n*100:.1f}% —— AI 几乎不用 [汤圆] 这类站内表情")
        print(f"  · 对比同话题的 AI 输出，主要看：平均长度、，和。的密度、表情率")


if __name__ == "__main__":
    main()
