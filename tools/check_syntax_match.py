# -*- coding: utf-8 -*-
"""模型的输出，语法上更接近真人还是更接近原来的鱼腔？

拿 fish-a2 的 dev 预测（78 条）比三方：
  模型输出 / 真人原文(y) / 鱼的输入(x)
"""
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.stdout.reconfigure(encoding="utf-8")

from syntax_stats import human_score, stats  # noqa: E402

rows = [json.loads(l) for l in open(ROOT / "state" / "preds" / "fish-a2.jsonl",
                                    encoding="utf-8")]
groups = {"模型输出": [], "真人原文": [], "鱼的输入": []}
for r in rows:
    for k, t in (("模型输出", r["pred"]), ("真人原文", r["y"]), ("鱼的输入", r["x"])):
        s = stats(t)
        if s:
            groups[k].append(s)

keys = list(groups["模型输出"][0].keys())
print(f"  dev {len(rows)} 条")
print()
print("  %-12s %9s %9s %9s   %s" % ("维度", "模型输出", "真人原文", "鱼的输入", "模型更像谁"))
print("  " + "-" * 62)
for k in keys:
    a = st.mean(x[k] for x in groups["模型输出"])
    b = st.mean(x[k] for x in groups["真人原文"])
    c = st.mean(x[k] for x in groups["鱼的输入"])
    closer = "真人" if abs(a - b) < abs(a - c) else "鱼"
    if abs(abs(a - b) - abs(a - c)) < 0.05:
        closer = "都像"
    print("  %-12s %9.2f %9.2f %9.2f   %s" % (k, a, b, c, closer))

print()
print("  综合「像人话」得分（0~1，越高越好）：")
for name, vals in (("模型的输出", "模型输出"), ("真人原文", "真人原文"), ("鱼的输入", "鱼的输入")):
    idx = 0
    texts = {
        "模型输出": [r["pred"] for r in rows],
        "真人原文": [r["y"] for r in rows],
        "鱼的输入": [r["x"] for r in rows],
    }[vals]
    sc = [human_score(t) for t in texts]
    print("    %-8s %.3f" % (name, st.mean(sc)))
