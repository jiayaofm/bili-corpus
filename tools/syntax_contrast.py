# -*- coding: utf-8 -*-
"""对比「大肥鱼的改写」和「真人原文」的语法习惯。

用 986 对配对数据（同视频、同话题），所以差异只来自写法，不来自内容。

目的：找出可量化的语法维度，供后续
  ①  改写模型的评测指标
  ②  候选重排（best-of-N）的打分函数
"""
import glob
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")

SENT_PUNCT = "。！？!?…~"
CLAUSE_PUNCT = "，,。！？!?…~；;：:、"
PARTICLES = "啊吧呢嘛哦诶呀哈嘿咯啦哇嗯"
DISCOURSE = ["就是", "反正", "其实", "那个", "然后", "所以", "但是", "因为",
             "真的", "感觉", "好像", "居然", "竟然", "直接", "根本"]
FORMAL = ["而", "因此", "从而", "不仅", "更是", "以及", "然而", "进而",
          "本质", "层面", "维度", "方向上"]
RHETORIC = ["不是", "难道", "哪有", "何必", "怎么", "凭什么", "谁能"]


def clauses(t):
    return [c for c in re.split(f"[{CLAUSE_PUNCT}]", t) if c.strip()]


def stats(t):
    t = t.strip()
    n = len(t)
    if not n:
        return None
    cl = clauses(t)
    lens = [len(c) for c in cl] or [n]
    han = len(re.findall(r"[\u4e00-\u9fff]", t))
    return {
        "字数": n,
        "分句数": len(cl),
        "平均分句长": sum(lens) / len(lens),
        "最短分句": min(lens),
        "短分句占比": sum(1 for x in lens if x <= 4) / len(lens),
        "句末有标点": 1.0 if t and t[-1] in SENT_PUNCT else 0.0,
        "逗号密度": t.count("，") / max(1, han) * 100,
        "句号密度": t.count("。") / max(1, han) * 100,
        "语气词密度": sum(t.count(c) for c in PARTICLES) / max(1, han) * 100,
        "口语连接词": sum(t.count(w) for w in DISCOURSE) / max(1, han) * 100,
        "书面连接词": sum(t.count(w) for w in FORMAL) / max(1, han) * 100,
        "反问标记": sum(t.count(w) for w in RHETORIC) / max(1, han) * 100,
        "第一人称我": t.count("我") / max(1, han) * 100,
        "数字英文占比": len(re.findall(r"[0-9A-Za-z]", t)) / n * 100,
        "叠字重复": len(re.findall(r"(.)\1", t)) / max(1, han) * 100,
    }


fish, human = [], []
for fp in glob.glob(str(ROOT / "data" / "whale_comments" / "*.jsonl")):
    for line in open(fp, encoding="utf-8"):
        r = json.loads(line)
        if r.get("skipped"):
            continue
        for rw in (r.get("rewrites") or []):
            o = (rw.get("origin") or "").strip()
            w = (rw.get("rewrite") or "").strip()
            if not o or not w or o == w or len(o) < 8:
                continue
            so, sw = stats(o), stats(w)
            if so and sw:
                human.append(so)
                fish.append(sw)

print(f"  配对样本 {len(human)} 对（同视频同话题）")
print()
keys = list(human[0].keys())
print("  %-12s %10s %10s %10s   %s" % ("维度", "鱼的改写", "真人原文", "差", "谁更高"))
print("  " + "-" * 66)
rows = []
for k in keys:
    a = st.mean(x[k] for x in fish)
    b = st.mean(x[k] for x in human)
    d = a - b
    who = "鱼" if d > 0 else "人"
    if abs(d) < 1e-9:
        who = "—"
    rows.append((k, a, b, d, who))
for k, a, b, d, who in rows:
    print("  %-12s %10.2f %10.2f %+10.2f   %s" % (k, a, b, d, who))

print()
print("  按差距排序（相对差 = 差 / 人的均值）：")
rel = sorted(rows, key=lambda r: -abs(r[3]) / max(1e-6, abs(r[2])))
for k, a, b, d, who in rel[:8]:
    print("    %-12s 鱼 %.2f → 人 %.2f   (%s 高 %+.0f%%)"
          % (k, a, b, who, abs(d) / max(1e-6, abs(b)) * 100))
