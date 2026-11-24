# -*- coding: utf-8 -*-
"""检查 v4 数据的可用率。

对比 v3 那批：1020 对 → 旧阈值 397 对(39%) → 新阈值 254 对(25%)。
v4 的任务书改成「不许压缩、只换说法」，这里看实际效果。
"""
import json
import re
import statistics as st
import sys
from collections import Counter

ROOT = "data/whale_comments/whale_v4_20261122.jsonl"
sys.stdout.reconfigure(encoding="utf-8")

CJK = re.compile(r"[^\u4e00-\u9fff0-9A-Za-z]")
LINK = re.compile(r"https?://|BV[0-9A-Za-z]{10}|b23\.tv")


def bg(t):
    t = CJK.sub("", t)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def jac(a, b):
    A, B = bg(a), bg(b)
    return len(A & B) / len(A | B) if (A | B) else 1.0


def cjk_ratio(t):
    if not t:
        return 0.0
    return len(re.findall(r"[\u4e00-\u9fff]", t)) / len(t)


rows = [json.loads(l) for l in open(ROOT, encoding="utf-8")]
pairs = []
for r in rows:
    for p in (r.get("pairs") or []):
        o = (p.get("origin") or "").strip()
        w = (p.get("rewrite") or "").strip()
        if o or w:
            pairs.append((o, w))

print(f"  总对数 {len(pairs)}")
print()

why = Counter()
ok = []
for o, w in pairs:
    if not o or not w:
        why["空"] += 1
        continue
    if o == w:
        why["一字未改"] += 1
        continue
    if LINK.search(o) or LINK.search(w):
        why["含链接"] += 1
        continue
    if len(o) < 8:
        why["原文太短"] += 1
        continue
    if cjk_ratio(o) < 0.7:
        why["原文不是中文"] += 1
        continue
    r = len(w) / max(1, len(o))
    if not (0.7 <= r <= 1.35):
        why[f"长度比越界(<0.7或>1.35)"] += 1
        continue
    j = jac(o, w)
    if j >= 0.5:
        why["改得不够(相似度≥0.5)"] += 1
        continue
    ok.append((o, w, r, j))

print(f"  可用 {len(ok)} 对（{len(ok)/len(pairs)*100:.0f}%）")
print()
print("  淘汰原因：")
for k, v in why.most_common():
    print(f"    {k:<28} {v:>4} 条")

if ok:
    print()
    rs = [x[2] for x in ok]
    js = [x[3] for x in ok]
    print("  可用样本的长度比  中位 %.2f  P10 %.2f  P90 %.2f" % (
        st.median(rs), sorted(rs)[len(rs)//10], sorted(rs)[len(rs)*9//10]))
    print("  可用样本的相似度  中位 %.2f  P10 %.2f  P90 %.2f" % (
        st.median(js), sorted(js)[len(js)//10], sorted(js)[len(js)*9//10]))
    print()
    print("  对照 v3：")
    print("    v3 旧阈值 397 对(39%) / 新阈值 254 对(25%)")
    print("    v4 新阈值 %d 对(%.0f%%)" % (len(ok), len(ok) / len(pairs) * 100))
