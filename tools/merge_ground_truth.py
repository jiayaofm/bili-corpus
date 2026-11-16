# -*- coding: utf-8 -*-
"""
合并所有历史标注，抢救被覆盖的数据

背景：s04 每次重跑都会用最新盲标表覆盖 ground_truth.csv。
     第二批（blind_sample 157 条）与第一批（blind_151 151 条）只重叠 45 条，
     另外 106 条旧标注被覆盖了，但备份还在 state/ 里。

本脚本把它们合并成一份完整真值，并标注来源批次。

注意：两批标注标准不一致（第一批判 AI 36.4%，第二批 23.6%），
     合并会引入噪声，source 字段用于后续区分。
"""
import csv
import glob
import json
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "ground_truth_merged.csv"


def load(path, source):
    out = {}
    for r in csv.DictReader(open(path, encoding="utf-8-sig", newline="")):
        out[r["rpid"]] = {
            "rpid": r["rpid"],
            "label": int(r["label"]),
            "strength": r.get("strength", ""),
            "was_candidate": r.get("was_candidate", ""),
            "content": r.get("content", ""),
            "source": source,
        }
    return out


def main():
    merged = {}

    # 旧批次（备份），先放进去
    for f in sorted(glob.glob(str(ROOT / "state" / "ground_truth_backup_*.csv"))):
        for k, v in load(f, "batch1_old").items():
            merged.setdefault(k, v)

    # 当前批次覆盖同名（标准更新）
    cur = load(ROOT / "labeling" / "ground_truth.csv", "batch2_current")
    for k, v in cur.items():
        merged[k] = v

    rows = list(merged.values())
    print(f"合并后 {len(rows)} 条唯一标注")
    print(f"  来源：{dict(Counter(r['source'] for r in rows))}")
    print(f"  标签：{dict(Counter(r['label'] for r in rows))}")
    print(f"  AI 占比：{sum(r['label'] for r in rows)/len(rows)*100:.1f}%")

    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["rpid", "label", "strength",
                                          "was_candidate", "content", "source"])
        w.writeheader()
        w.writerows(rows)

    # 重算未标注池
    corpus = []
    for line in (ROOT / "data" / "corpus.jsonl").open(encoding="utf-8"):
        line = line.strip()
        if line:
            corpus.append(json.loads(line))
    labeled = set(merged)
    un = [c for c in corpus if c["rpid"] not in labeled]
    print(f"\n语料 {len(corpus)} 条")
    print(f"  已标注 {len(labeled)} 条")
    print(f"  未标注 {len(un)} 条（原来 851，减少 {851 - len(un)}）")

    (ROOT / "state" / "unlabeled_sorted.json").write_text(
        json.dumps(un, ensure_ascii=False), encoding="utf-8")
    print(f"\n输出：{OUT}")
    print("未标注池已刷新：state/unlabeled_sorted.json")


if __name__ == "__main__":
    main()
