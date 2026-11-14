# -*- coding: utf-8 -*-
"""
从人工标注里挖 AI 评论的特征词

思路：不预设规则，让数据自己说。
  正样本 = 打分器判为 high/mid 且未被标 x 的（人工复核后暂认为 AI）
  负样本 = low 档 + 人工标 x 的

方法：
  · 字符 n-gram（2–5 字），中文无需分词，对短文本更稳
  · 用 log-odds ratio + 平滑，避免小样本噪声
  · 同时报告支持度（出现条数），低于 2 条的丢弃

输出：analysis/ai_keywords.txt
"""
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

LAB = Path(__file__).parent / "labeling"
OUT = Path(__file__).parent / "analysis"
OUT.mkdir(exist_ok=True)

EMOTE_RE = re.compile(r"\[[^\[\]]{1,12}\]")


def load_rows():
    p = LAB / "candidates.csv"
    rows = []
    with p.open(encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        for d in r:
            sc = (d["score"] or "").strip()
            marked_human = "x" in sc.lower()
            try:
                val = float(sc.lower().replace("x", "") or 0)
            except ValueError:
                val = 0.0
            rows.append({
                "score": val,
                "conf": d["conf"],
                "marked_human": marked_human,
                "content": d["content"].replace("\\n", "\n"),
                "level": d["level"],
                "length": int(d["length"]),
            })
    return rows


def ngrams(text, ns=(2, 3, 4)):
    t = EMOTE_RE.sub("", text)
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"BV[0-9A-Za-z]{10}", "", t)
    t = re.sub(r"\s+", "", t)
    out = set()
    for n in ns:
        for i in range(len(t) - n + 1):
            g = t[i:i + n]
            if g.strip():
                out.add(g)
    return out


def main():
    rows = load_rows()
    # 正：high/mid 且未标 x；负：其余（low + 标 x）
    pos = [r for r in rows if r["conf"] in ("high", "mid") and not r["marked_human"]]
    neg = [r for r in rows if not (r["conf"] in ("high", "mid") and not r["marked_human"])]

    print(f"正样本(AI) {len(pos)} 条   负样本(真人) {len(neg)} 条")
    print(f"其中人工标 x 的 {sum(1 for r in rows if r['marked_human'])} 条已计入负样本\n")

    pos_cnt, neg_cnt = Counter(), Counter()
    for r in pos:
        pos_cnt.update(ngrams(r["content"]))
    for r in neg:
        neg_cnt.update(ngrams(r["content"]))

    Np, Nn = len(pos), len(neg)
    scored = []
    for g, cp in pos_cnt.items():
        if cp < 2:                      # 至少 2 条里出现
            continue
        cn = neg_cnt.get(g, 0)
        if cn / Nn > 0.08:              # 真人里太常见就排除
            continue
        # log-odds with smoothing
        p = (cp + 0.5) / (Np + 1)
        q = (cn + 0.5) / (Nn + 1)
        lo = math.log(p / (1 - p)) - math.log(q / (1 - q))
        scored.append((lo, cp, cn, g))

    scored.sort(key=lambda x: -x[0])

    # 去掉被更长 gram 完全包含的短 gram（保留信息量大的）
    kept = []
    for lo, cp, cn, g in scored:
        if any(g != k[3] and g in k[3] and k[0] >= lo - 0.5 for k in kept[:200]):
            continue
        kept.append((lo, cp, cn, g))
        if len(kept) >= 150:
            break

    lines = []
    lines.append("=" * 70)
    lines.append(f"AI 评论特征词（正样本 {Np} / 负样本 {Nn}）")
    lines.append("=" * 70)
    lines.append(f"{'log-odds':>9} {'AI中':>5} {'真人中':>6}  片段")
    lines.append("-" * 70)
    for lo, cp, cn, g in kept:
        lines.append(f"{lo:>9.2f} {cp:>5} {cn:>6}  {g}")

    # 负向：真人爱用而 AI 不用的
    lines.append("")
    lines.append("=" * 70)
    lines.append("真人高频片段（AI 几乎不用）")
    lines.append("=" * 70)
    rev = []
    for g, cn in neg_cnt.items():
        if cn < 3:
            continue
        cp = pos_cnt.get(g, 0)
        if cp / max(1, Np) > 0.15:
            continue
        p = (cp + 0.5) / (Np + 1)
        q = (cn + 0.5) / (Nn + 1)
        lo = math.log(q / (1 - q)) - math.log(p / (1 - p))
        rev.append((lo, cn, cp, g))
    rev.sort(key=lambda x: -x[0])
    for lo, cn, cp, g in rev[:60]:
        lines.append(f"{lo:>9.2f} {cn:>5} {cp:>6}  {g}")

    text = "\n".join(lines)
    (OUT / "ai_keywords.txt").write_text(text, encoding="utf-8")
    print(text)
    print(f"\n已写入 {OUT / 'ai_keywords.txt'}")


if __name__ == "__main__":
    main()
