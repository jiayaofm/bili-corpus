# -*- coding: utf-8 -*-
"""查 level 字段为何出现 None，以及它与 AI 风格的关联"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
DATA = Path(__file__).parent / "data"

EMOTE_RE = re.compile(r"\[[^\[\]]{1,12}\]")
AI_MARK = ["「", "」", "——", "**"]


def load():
    recs = []
    for fp in sorted(DATA.glob("*_*.jsonl")):
        for l in fp.open(encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                d["_src"] = fp.name
                recs.append(d)
    return recs


recs = load()
print(f"总 {len(recs)} 条\n")

# 1. level 取值分布
lv = Counter(r.get("level", "KEY_MISSING") for r in recs)
print("level 取值分布：")
for k, v in lv.most_common():
    print(f"  {k}: {v}")

# 2. 看一条 level 为 None 的原始记录结构
none_recs = [r for r in recs if r.get("level") is None]
print(f"\nlevel 为 None 的记录数：{len(none_recs)}")
if none_recs:
    r = none_recs[0]
    print("  样例字段：", {k: (v if k not in ('content', 'sub_replies') else f'<{len(str(v))}字符>')
                          for k, v in r.items()})
    print("  内容：", r["content"][:100].replace("\n", "\\n"))

# 3. 按 level 是否缺失，看 AI 标记率
print("\n按 level 是否缺失，统计 AI 风格标记率：")
for label, group in (("level=None", none_recs),
                     ("level=0", [r for r in recs if r.get("level") == 0]),
                     ("level>=5", [r for r in recs if isinstance(r.get("level"), int) and r["level"] >= 5])):
    if not group:
        continue
    n = len(group)
    has_corner = sum(1 for r in group if "「" in r["content"])
    has_dash = sum(1 for r in group if "——" in r["content"])
    has_md = sum(1 for r in group if "**" in r["content"])
    has_semi = sum(1 for r in group if "；" in r["content"])
    emote = sum(1 for r in group if EMOTE_RE.search(r["content"]))
    avg = sum(len(r["content"]) for r in group) / n
    print(f"  {label:<12} n={n:>4}  平均{avg:>6.1f}字  "
          f"「={has_corner:>3}  ——={has_dash:>3}  **={has_md:>3}  ；={has_semi:>3}  表情率={emote/n*100:>4.1f}%")

# 4. 含 markdown 粗体的
print("\n含 markdown 粗体 ** 的评论：")
md = [r for r in recs if "**" in r["content"]]
print(f"  共 {len(md)} 条")
for r in md[:6]:
    print(f"   [{r.get('level')}] {r['content'][:110]!r}")

# 5. 引用 BV 号的
print("\n评论正文里引用 BV 号的：")
bvc = [r for r in recs if re.search(r"BV[0-9A-Za-z]{10}", r["content"])]
print(f"  共 {len(bvc)} 条")
for r in bvc[:6]:
    print(f"   [{r.get('level')}] {r['content'][:110]!r}")
