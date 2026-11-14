# -*- coding: utf-8 -*-
"""
恢复被覆盖的人工标注

背景：label_ai.py v3 运行时覆盖了 candidates.csv，用户标的 8 个 x 丢失。
这 8 条的内容此前已读出，按前缀匹配回语料，落成持久文件 human_marked.json。
今后 label_ai.py 优先从这个文件读标注，不再依赖 CSV。
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(__file__).parent / "data"
OUT = Path(__file__).parent / "labeling"
OUT.mkdir(exist_ok=True)

# 用户标为「真人」的 8 条，用内容前缀定位
PREFIXES = [
    "已三连，up up",
    "视频开头Case的作品在线预览链接",
    "刚刚看到有人大致意思说ai没法有稳定的记忆",
    "怎么可能,获取直播间弹幕不可能用文字识别",
    "🍚🎣肥鱼来",
    "其实它知道的。现在有些直播送真礼物",
    "我要开一层楼",
    "已点赞投币，内个，这是硅基流动的邀请码",
]


def load():
    recs = []
    for fp in sorted(DATA.glob("*_*.jsonl")):
        for l in fp.open(encoding="utf-8"):
            if l.strip():
                recs.append(json.loads(l))
    seen, out = set(), []
    for r in recs:
        if r["rpid"] in seen:
            continue
        seen.add(r["rpid"])
        out.append(r)
    return out


recs = load()
print(f"语料 {len(recs)} 条\n")

found, missing = [], []
for p in PREFIXES:
    hits = [r for r in recs if r["content"].startswith(p)]
    if hits:
        r = hits[0]
        found.append({"rpid": r["rpid"], "prefix": p, "content": r["content"]})
        print(f"[OK] {r['rpid']}  {r['content'][:60]!r}")
    else:
        # 退一步：包含匹配
        hits = [r for r in recs if p in r["content"]]
        if hits:
            r = hits[0]
            found.append({"rpid": r["rpid"], "prefix": p, "content": r["content"]})
            print(f"≈ {r['rpid']}  {r['content'][:60]!r}  (包含匹配)")
        else:
            missing.append(p)
            print(f"[缺失] 未找到: {p[:40]}")

path = OUT / "human_marked.json"
path.write_text(json.dumps(found, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n恢复 {len(found)}/{len(PREFIXES)} 条 → {path}")
if missing:
    print("未找到的（需要你补标）：")
    for m in missing:
        print("  ", m)
