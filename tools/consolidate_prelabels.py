# -*- coding: utf-8 -*-
"""
把 AI 预标注结果并入真值

要点：
  1. 只并入有明确判定（1/0）的，「不确定」的跳过
  2. was_candidate 按 s03 同一定义补齐：规则判 high/mid 即为候选层
  3. source 字段区分人工标注与 AI 预标注 —— 评估时必须分开看，
     否则"用 AI 标的标签测 AI 模型"是循环论证
"""
import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT))

from label_ai import score as rule_score, conf_of  # noqa: E402

GT = ROOT / "labeling" / "ground_truth.csv"
SOURCES = [
    ("labeling/ai_prelabels.csv", "ai_prelabel_r1"),
    ("labeling/ai_prelabels_r2.csv", "ai_prelabel_r2"),
    ("labeling/ai_prelabels_r3.csv", "ai_prelabel_r3"),
]
FIELDS = ["rpid", "label", "strength", "was_candidate", "content", "source"]


def main():
    # 现有真值（人工）
    existing = {}
    for r in csv.DictReader(GT.open(encoding="utf-8-sig", newline="")):
        r.setdefault("source", "human")
        existing[r["rpid"]] = r
    n_human = len(existing)

    added = 0
    skipped = 0
    by_src = {}
    for rel, src in SOURCES:
        p = ROOT / rel
        if not p.exists():
            continue
        cnt = 0
        for r in csv.DictReader(p.open(encoding="utf-8-sig", newline="")):
            lab = (r["label"] or "").strip()
            if lab not in ("0", "1"):
                skipped += 1
                continue
            if r["rpid"] in existing:
                continue
            s, hits = rule_score({"content": r["content"], "user": ""})
            cand = 1 if conf_of(s, hits) in ("high", "mid") else 0
            existing[r["rpid"]] = {
                "rpid": r["rpid"], "label": lab,
                "strength": "x" if lab == "1" else "",
                "was_candidate": cand, "content": r["content"], "source": src,
            }
            cnt += 1
            added += 1
        by_src[src] = cnt

    rows = list(existing.values())
    with GT.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    print(f"人工标注      {n_human} 条")
    print(f"AI 预标注新增 {added} 条  {by_src}")
    print(f"跳过不确定     {skipped} 条")
    print(f"合计          {len(rows)} 条")
    n1 = sum(1 for r in rows if r["label"] == "1")
    print(f"  正类 {n1} / 负类 {len(rows) - n1}")

    # 分层权重需要
    m = json.loads((ROOT / "labeling" / "blind_map.json").read_text(encoding="utf-8"))
    print(f"\n（加权参数沿用它：随机层权重 {m['pool_rest']/m['random_count']:.2f}）")


if __name__ == "__main__":
    main()
