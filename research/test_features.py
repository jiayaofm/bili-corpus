# -*- coding: utf-8 -*-
"""
特征候选池测试 —— 扩展句式 + 账号名称

目的：把候选特征逐个丢进数据里测「AI 命中 / 真人误伤」，只留有效的。
不再靠拍脑袋，每条候选都给数字。
"""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(__file__).parent / "data"
LAB = Path(__file__).parent / "labeling"

EMOTE_RE = re.compile(r"\[[^\[\]]{1,12}\]")

# ---------- 载入 ----------
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

def load_human_marks():
    p = LAB / "human_marked.json"
    if not p.exists():
        return set()
    return {x["rpid"] for x in json.loads(p.read_text(encoding="utf-8"))}

def load_ai_set():
    """当前打分器判为 high/mid 且未被人标为真人的，视为 AI 候选集"""
    p = LAB / "candidates.csv"
    ai = []
    if not p.exists():
        return ai
    with p.open(encoding="utf-8-sig", newline="") as f:
        for d in csv.DictReader(f):
            if d["conf"] in ("high", "mid") and d["label"] != "0":
                ai.append(d["rpid"])
    return ai

corpus = load_corpus()
by_rpid = {r["rpid"]: r for r in corpus}
humans = load_human_marks()
ai_ids = [i for i in load_ai_set() if i in by_rpid]
AI = [by_rpid[i] for i in ai_ids]
HU = [by_rpid[i] for i in humans if i in by_rpid]
REST = [r for r in corpus if r["rpid"] not in set(ai_ids) | humans]

print(f"语料 {len(corpus)}    AI候选 {len(AI)}    人工确认真人 {len(HU)}    其余 {len(REST)}\n")

# ============================================================
# 一、账号名称分析
# ============================================================
print("=" * 78)
print("一、账号名称")
print("=" * 78)

def name_feats(name):
    return {
        "含鱼": "鱼" in name,
        "含小": "小" in name,
        "含号": "号" in name,
        "纯数字尾": bool(re.search(r"\d+$", name)),
        "含字母": bool(re.search(r"[A-Za-z]", name)),
        "含下划线": "_" in name,
        "长度": len(name),
    }

print("\n【AI 候选的账号名】")
for r in AI:
    print(f"  {r.get('user','')!r:38}  level={r.get('level')}  len={len(r.get('user',''))}")
print("\n【人工确认真人的账号名】")
for r in HU:
    print(f"  {r.get('user','')!r:38}  level={r.get('level')}  len={len(r.get('user',''))}")

print("\n【各特征的命中率对比】")
KEYS = ["含鱼", "含小", "含号", "纯数字尾", "含字母", "含下划线"]
hdr = f"{'特征':<10}{'AI命中':>10}{'AI占比':>10}{'真人命中':>10}{'真人占比':>10}{'其余占比':>10}"
print(hdr)
for k in KEYS:
    a = sum(1 for r in AI if name_feats(r.get("user", ""))[k])
    h = sum(1 for r in HU if name_feats(r.get("user", ""))[k])
    o = sum(1 for r in REST if name_feats(r.get("user", ""))[k])
    print(f"{k:<10}{a:>10}{a/max(1,len(AI))*100:>9.0f}%{h:>10}{h/max(1,len(HU))*100:>9.0f}%{o/max(1,len(REST))*100:>9.0f}%")

# 同名账号出现次数（同一账号多评论）
print("\n【AI 候选里同一账号的评论条数】")
c = Counter(r.get("user", "") for r in AI)
for name, n in c.most_common(10):
    print(f"  {name!r}: {n} 条")

# ============================================================
# 二、扩展句式候选
# ============================================================
print("\n" + "=" * 78)
print("二、句式候选特征测试")
print("=" * 78)

CANDIDATES = [
    # 用户提出的
    ("这笔账/这账", r"这(?:笔)?账"),
    ("算账/账本", r"算账|账本|查账"),
    ("记下了", r"记下|记一笔|记着"),
    ("XX这块", r"[^，。\n]{1,6}这块"),
    # 判断/总结腔
    ("说白了/说实话", r"说白了|说实话|老实说|讲真"),
    ("其实", r"其实"),
    ("确实", r"确实"),
    ("至于", r"至于"),
    ("换句话说", r"换句话说|也就是说|简单说"),
    ("有一说一", r"有一说一|平心而论|讲道理"),
    ("不如说", r"不如说|与其说"),
    ("这叫", r"这叫|这才叫|叫[^，。]{0,4}"),
    # 认知/元评论
    ("我好奇的是", r"我好奇的是|我最在意|我最关心"),
    ("我觉得/我认为", r"我觉得|我认为|在我看来"),
    ("有没有人", r"有没有人|有没有真的|有人真的"),
    ("反而", r"反而"),
    ("本质上", r"本质上|实质上是|说到底"),
    # 程度/修饰
    ("特别+形容词", r"特别[^\s，。]{1,4}"),
    ("有点+形容词", r"有点[^\s，。]{1,4}"),
    ("瞬间/一下子", r"瞬间|一下子|一瞬间"),
    # 结构
    ("多行分段", r"\n"),
    ("句号数>=2", None),      # 特殊处理
    ("问号结尾", r"[？?]\s*$"),
    ("双引号引文", r"“[^”]{4,}”"),
    ("括号补充", r"（[^）]{2,20}）"),
]

def hit(r, pat):
    c = r["content"]
    if pat is None:
        return c.count("。") >= 2
    return bool(re.search(pat, c))

print(f"\n{'候选特征':<16}{'AI命中':>8}{'AI率':>8}{'真人命中':>8}{'真人率':>8}{'其余率':>8}{'判定':>8}")
print("-" * 78)
for name, pat in CANDIDATES:
    a = sum(1 for r in AI if hit(r, pat))
    h = sum(1 for r in HU if hit(r, pat))
    o = sum(1 for r in REST if hit(r, pat))
    ar, hr, orr = a/max(1,len(AI))*100, h/max(1,len(HU))*100, o/max(1,len(REST))*100
    if a >= 3 and hr <= 15 and ar > orr + 20:
        verdict = "有效"
    elif a >= 3 and ar > orr + 10:
        verdict = "○弱"
    elif hr >= 40:
        verdict = "真人常用"
    else:
        verdict = "—"
    print(f"{name:<16}{a:>8}{ar:>7.0f}%{h:>8}{hr:>7.0f}%{orr:>7.0f}%{verdict:>8}")
