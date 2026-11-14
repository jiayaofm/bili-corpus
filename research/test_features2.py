# -*- coding: utf-8 -*-
"""
第二轮特征测试：从 23 条 AI 样本里实际观察到的句式 + 账号名规则
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(__file__).parent / "data"
LAB = Path(__file__).parent / "labeling"


def load_corpus():
    recs, seen = [], set()
    for fp in sorted(DATA.glob("*_*.jsonl")):
        for l in fp.open(encoding="utf-8"):
            if not l.strip():
                continue
            d = json.loads(l)
            if d["rpid"] in seen:
                continue
            seen.add(d["rpid"])
            recs.append(d)
    return recs


corpus = load_corpus()
by = {r["rpid"]: r for r in corpus}
humans = {x["rpid"] for x in json.loads((LAB / "human_marked.json").read_text(encoding="utf-8"))}

ai_ids = []
with (LAB / "candidates.csv").open(encoding="utf-8-sig", newline="") as f:
    for d in csv.DictReader(f):
        if d["conf"] in ("high", "mid") and d["label"] != "0":
            ai_ids.append(d["rpid"])
AI = [by[i] for i in ai_ids if i in by]
HU = [by[i] for i in humans if i in by]
REST = [r for r in corpus if r["rpid"] not in set(ai_ids) | humans]

print(f"AI {len(AI)}   真人 {len(HU)}   其余 {len(REST)}\n")

# ============ 账号名规则 ============
FISH = re.compile(r"鱼|鲸")
print("=" * 78)
print("账号名规则")
print("=" * 78)
groups = [("AI", AI), ("真人", HU), ("其余", REST)]
for label, g in groups:
    hit = [r for r in g if FISH.search(r.get("user", ""))]
    print(f"  {label:<4} 含鱼/鲸 {len(hit):>3}/{len(g):<4} ({len(hit)/max(1,len(g))*100:>5.1f}%)")

# 名字长度
for label, g in groups:
    lens = [len(r.get("user", "")) for r in g]
    if lens:
        print(f"  {label:<4} 名字平均长度 {sum(lens)/len(lens):.1f}")

# 组合：名字含鱼/鲸 → 在哪些视频出现
print("\n  含鱼/鲸的账号出现在哪些视频：")
for label, g in groups:
    hit = [r for r in g if FISH.search(r.get("user", ""))]
    from collections import Counter
    c = Counter(r.get("video_title", "")[:20] for r in hit)
    for t, n in c.items():
        print(f"    {label:<4} {t!r}: {n}")

# ============ 第二轮句式 ============
print("\n" + "=" * 78)
print("第二轮句式候选（从实测样本里观察到的）")
print("=" * 78)

CAND = [
    ("这条（指代被回复的评论）", r"这条"),
    ("比…更/都", r"比[^，。\n]{2,13}(?:更|都)"),
    ("看得我…", r"看得我"),
    ("有点+X", r"有点[^\s，。]{1,4}"),
    ("…是真的", r"(?:是|也是)真的[，。]"),
    ("这条我信/值得", r"这条[^，。]{0,4}(?:我信|值得|说)"),
    ("能成立/靠的是", r"能成立|靠的是|之所以"),
    ("不必/不必把", r"不必"),
    ("…才叫真…", r"才叫真|才叫"),
    ("一出…大戏/戏", r"演一出|大戏|剧本"),
    ("本鱼/本鲸/本喵", r"本鱼|本鲸|本喵"),
    ("余额/token/白饭", r"余额|token|白饭|饭钱"),
    ("读字幕/我的型号", r"读字幕|我这个型号|派活"),
    ("比…更值得记", r"更值得(?:记|一提|看)"),
    ("反差/寓言/内核/结构", r"反差|寓言|内核|结构"),
]

print(f"\n{'候选':<26}{'AI':>5}{'AI率':>7}{'真人':>5}{'真人率':>7}{'其余率':>8}{'判定':>10}")
print("-" * 78)
for name, pat in CAND:
    rx = re.compile(pat)
    a = sum(1 for r in AI if rx.search(r["content"]))
    h = sum(1 for r in HU if rx.search(r["content"]))
    o = sum(1 for r in REST if rx.search(r["content"]))
    ar, hr, orr = a/len(AI)*100, h/len(HU)*100, o/max(1,len(REST))*100
    if a >= 3 and hr <= 15 and ar > orr + 15:
        v = "有效"
    elif a >= 3 and ar > orr + 8:
        v = "○弱"
    elif hr >= 30:
        v = "真人常用"
    else:
        v = "—"
    print(f"{name:<26}{a:>5}{ar:>6.0f}%{h:>5}{hr:>6.0f}%{orr:>7.0f}%{v:>10}")

# ============ 组合规则效果 ============
print("\n" + "=" * 78)
print("组合规则：名字含鱼/鲸  OR  句式分>=18")
print("=" * 78)
name_hit = [r for r in corpus if FISH.search(r.get("user", ""))]
print(f"  名字含鱼/鲸的账号共 {len(name_hit)} 条评论")
print(f"    其中落在 AI 候选集: {sum(1 for r in name_hit if r['rpid'] in set(ai_ids))}")
print(f"    其中真人确认为真人: {sum(1 for r in name_hit if r['rpid'] in humans)}")
