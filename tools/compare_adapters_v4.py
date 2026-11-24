# -*- coding: utf-8 -*-
"""对比两个 adapter 在 v4_dev 上的改写幅度和内容丢失。

核心问题：训练目标（真人原文）和输入（鱼的改写）之间的字符 bigram 相似度
中位是 0.30。模型输出应该也落在这个区间，才算真的"改写"了。
"""
import json
import re
import statistics as st
import sys

sys.stdout.reconfigure(encoding="utf-8")
CJK = re.compile(r"[^\u4e00-\u9fff0-9A-Za-z]")


def bg(t):
    t = CJK.sub("", t)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def jac(a, b):
    A, B = bg(a), bg(b)
    return len(A & B) / len(A | B) if (A | B) else 1.0


rows = [json.loads(l) for l in open("data/rewrite/v4_dev.jsonl", encoding="utf-8")]
xs = [r["_meta"]["x"] for r in rows]
ys = [r["_meta"]["y"] for r in rows]

print("  参考：输入(鱼的改写) 与 目标(真人原文) 的相似度")
jt = [jac(x, y) for x, y in zip(xs, ys)]
print("    中位 %.2f   P10 %.2f   P90 %.2f" % (st.median(jt),
                                              sorted(jt)[len(jt) // 10],
                                              sorted(jt)[len(jt) * 9 // 10]))
print()

for name, path in (("fish-a2", "state/preds/fish-a2-on-v4dev.jsonl"),
                   ("fish-v4", "state/preds/fish-v4.jsonl")):
    preds = {json.loads(l)["x"]: json.loads(l)["pred"]
             for l in open(path, encoding="utf-8")}
    js, lost = [], 0
    for x, y in zip(xs, ys):
        p = preds.get(x)
        if not p:
            continue
        j = jac(x, p)
        js.append(j)
        if len(p) < len(y) * 0.6:
            lost += 1
    print("  %s" % name)
    print("    输出与输入相似度  中位 %.2f   P10 %.2f   P90 %.2f" % (
        st.median(js), sorted(js)[len(js) // 10], sorted(js)[len(js) * 9 // 10]))
    print("    太像输入(≥0.6)   %d/%d (%.0f%%)" % (
        sum(1 for j in js if j >= 0.6), len(js),
        sum(1 for j in js if j >= 0.6) / len(js) * 100))
    print("    太不像(<0.15)    %d/%d (%.0f%%)" % (
        sum(1 for j in js if j < 0.15), len(js),
        sum(1 for j in js if j < 0.15) / len(js) * 100))
    print("    长度不到真人 60%%  %d/%d (%.0f%%)" % (
        lost, len(js), lost / len(js) * 100))
    print()

print("  fish-v4 的五条对照（相似度按高低排）：")
preds = {json.loads(l)["x"]: json.loads(l)["pred"]
         for l in open("state/preds/fish-v4.jsonl", encoding="utf-8")}
pairs = [(jac(x, preds[x]), x, preds[x], y) for x, y in zip(xs, ys) if x in preds]
pairs.sort()
for j, x, p, y in pairs[:3] + pairs[-2:]:
    print("    [相似度 %.2f]" % j)
    print("      输入 %s" % x[:44])
    print("      输出 %s" % p[:44])
    print("      真人 %s" % y[:44])
