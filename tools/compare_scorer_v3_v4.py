# -*- coding: utf-8 -*-
"""v3 和 v4 并排打分，看换成 v4 之后误判有没有减少。

重点看两类：
  ① 真人长评 —— v3 因为「长 = AI」会给高分，v4 应该掉下来
  ② 大肥鱼评论 —— v4 应该仍然给高分（否则就是换错了）
"""
import glob
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.stdout.reconfigure(encoding="utf-8")

from comment_server import ClassifierV3  # noqa: E402

v3 = ClassifierV3(ROOT / "state" / "scorer_v3.npz")
v4 = ClassifierV3(ROOT / "state" / "scorer_v4.npz")
print(f"  v3 就绪={v3.ok}  cv_auc={getattr(v3, 'cv_auc', None)}")
print(f"  v4 就绪={v4.ok}  cv_auc={getattr(v4, 'cv_auc', None)}")
print()

# 真人长评：从语料里挑最长的
corp = [json.loads(l) for l in open(ROOT / "data" / "corpus.jsonl", encoding="utf-8")]
humans = [(r.get("content") or "").strip() for r in corp]
humans = [t for t in humans if 60 <= len(t) <= 300 and not t.startswith("@")]
random.Random(0).shuffle(humans)
humans_long = humans[:12]

# 真人短评
humans_short = [t for t in humans if 10 <= len(t) <= 25][:8]

# 大肥鱼评论
fish = []
for fp in sorted(glob.glob(str(ROOT / "data" / "whale_comments" / "*.jsonl"))):
    for line in open(fp, encoding="utf-8"):
        r = json.loads(line)
        if r.get("skipped"):
            continue
        for c in (r.get("comments") or []):
            t = (c if isinstance(c, str) else "").strip()
            if 15 <= len(t) <= 200:
                fish.append(t)
random.Random(1).shuffle(fish)
fish = fish[:12]

groups = [("真人·长评", humans_long), ("真人·短评", humans_short),
          ("大肥鱼", fish)]

print("  %-10s %5s %7s %7s   %s" % ("类别", "字数", "v3", "v4", "样本"))
print("  " + "-" * 70)
for name, texts in groups:
    if not texts:
        continue
    s3 = v3.score(texts)
    s4 = v4.score(texts)
    import statistics as st
    print("  %-10s %5.0f %6.1f%% %6.1f%%   %s" % (
        name, st.median([len(t) for t in texts]),
        st.median(s3) * 100, st.median(s4) * 100, texts[0][:26]))
print()
print("  判成 AI（>0.5）的比例：")
for name, texts in groups:
    if not texts:
        continue
    a3 = sum(1 for v in v3.score(texts) if v > 0.5) / len(texts)
    a4 = sum(1 for v in v4.score(texts) if v > 0.5) / len(texts)
    print("    %-10s v3 %3.0f%%   v4 %3.0f%%" % (name, a3 * 100, a4 * 100))
print()
print("  逐条对照（真人长评，v3 高分的那几条）：")
pairs = list(zip(humans_long, v3.score(humans_long), v4.score(humans_long)))
for t, a, b in sorted(pairs, key=lambda x: -x[1])[:6]:
    print("    v3 %5.1f%% → v4 %5.1f%%   %s" % (a * 100, b * 100, t[:44]))
