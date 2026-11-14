# -*- coding: utf-8 -*-
"""
生成盲标表格

关键设计（为什么要"盲"）：
  · 不显示规则分数 —— 否则你会顺着我的判断走，等于把我的结论抄一遍
  · 不显示命中了哪些特征 —— 同上
  · 不显示账号名 —— 名字含鱼/鲸 是个强特征，但真人粉丝也这么起名，
                    看到名字会污染判断
  · 不显示点赞数 —— 高赞不等于真人
  · 随机打乱顺序 —— 避免按顺序标时形成先入为主

抽样策略：分层
  · 31 条规则候选全给你（提高信息量）
  · 再随机抽 120 条其余评论（覆盖真实分布）
  · 合计 151 条，约 20 分钟

输出：
  labeling/blind_151.csv       ← 你标这个
  labeling/blind_151_map.json  ← 编号↔rpid 对照，别动
"""
import csv
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent
DATA = ROOT / "data"
LAB = ROOT / "labeling"

N_NEG = 120
SEED = 20261111


def load():
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


def main():
    recs = load()
    by = {r["rpid"]: r for r in recs}

    # 规则候选（high/mid 且未被人工标为真人）
    cand_ids = []
    with (LAB / "candidates.csv").open(encoding="utf-8-sig", newline="") as f:
        for d in csv.DictReader(f):
            if d["conf"] in ("high", "mid") and d["label"] != "0":
                cand_ids.append(d["rpid"])

    # 已人工确认的 8 条 —— 排除，避免重复劳动
    known_human = {x["rpid"] for x in
                   json.loads((LAB / "human_marked.json").read_text(encoding="utf-8"))}

    cand = [i for i in cand_ids if i in by and i not in known_human]
    pool = [i for i in by if i not in set(cand) and i not in known_human]

    rng = random.Random(SEED)
    neg_sample = rng.sample(pool, min(N_NEG, len(pool)))

    picked = cand + neg_sample
    rng.shuffle(picked)

    rows = []
    for n, rpid in enumerate(picked, 1):
        c = by[rpid]["content"]
        rows.append({
            "编号": f"{n:03d}",
            "评论内容": c,
            "判断(1=AI 0=真人 ?=不确定)": "",
            "备注": "",
        })

    out_csv = LAB / "blind_151.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

    amap = {
        "seed": SEED,
        "total": len(rows),
        "candidate_count": len(cand),
        "random_count": len(neg_sample),
        "excluded_known_human": len(known_human),
        "items": [
            {"编号": f"{n:03d}", "rpid": rpid,
             "was_candidate": 1 if rpid in set(cand) else 0}
            for n, rpid in enumerate(picked, 1)
        ],
    }
    (LAB / "blind_151_map.json").write_text(
        json.dumps(amap, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"生成完成：{out_csv}")
    print(f"  总条数        {len(rows)}")
    print(f"  规则候选      {len(cand)}")
    print(f"  随机抽样      {len(neg_sample)}")
    print(f"  已排除(已标)  {len(known_human)}")
    print(f"  平均长度      {sum(len(r['评论内容']) for r in rows)/len(rows):.1f} 字")
    print(f"  映射表        {LAB / 'blind_151_map.json'}")
    print()
    print("请打开 CSV，在「判断」列填 1 / 0 / ?，存盘后告诉我。")
    print("  · 单条 6–8 秒即可，凭第一感觉，别反复推敲")
    print("  · 拿不准就填 ?，不要猜")
    print("  · 不要看「备注」以外的东西——表格里本来也没有其他信息")


if __name__ == "__main__":
    main()
