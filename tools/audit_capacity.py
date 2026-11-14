# -*- coding: utf-8 -*-
"""
自动审核能力边界分析

自动审核的核心指标不是 F1，而是：**在给定误报率下能召回多少**。
因为误判真人（误伤）的代价远高于漏掉。

本脚本在 157 条人工真值上扫描阈值，给出：
  · 每个阈值下的加权精确率 / 召回率
  · 达到 90% / 95% / 99% 精确率时的召回率
  · 可用的工作点（operating point）
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT))

from label_ai import score  # noqa: E402

GT = ROOT / "labeling" / "ground_truth.csv"
CORPUS = ROOT / "data" / "corpus.jsonl"
MAP = ROOT / "labeling" / "blind_map.json"


def load():
    m = json.loads(MAP.read_text(encoding="utf-8"))
    w_rand = m["pool_rest"] / m["random_count"]

    corpus = {}
    for line in CORPUS.open(encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            corpus[r["rpid"]] = r

    rows = list(csv.DictReader(GT.open(encoding="utf-8-sig", newline="")))
    recs = []
    for r in rows:
        c = corpus.get(r["rpid"])
        if not c:
            continue
        s, hits = score({"content": c["content"], "user": c.get("user", "")})
        recs.append({
            "score": s,
            "label": int(r["label"]),
            "weight": 1.0 if int(r["was_candidate"]) == 1 else w_rand,
            "content": c["content"],
        })
    return recs, w_rand


def wmetrics(recs, thr):
    tp = sum(r["weight"] for r in recs if r["score"] >= thr and r["label"] == 1)
    fp = sum(r["weight"] for r in recs if r["score"] >= thr and r["label"] == 0)
    fn = sum(r["weight"] for r in recs if r["score"] < thr and r["label"] == 1)
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return p, rc, tp, fp


def main():
    recs, w_rand = load()
    pos = sum(1 for r in recs if r["label"] == 1)
    print(f"真值 {len(recs)} 条  正类 {pos}  随机层权重 {w_rand:.2f}\n")

    print(f"{'阈值':>6}{'精确率':>9}{'召回率':>9}{'命中AI':>9}{'误伤真人':>10}")
    print("-" * 46)
    for thr in [0, 8, 18, 25, 32, 45, 60, 80, 100]:
        p, rc, tp, fp = wmetrics(recs, thr)
        print(f"{thr:>6}{p*100:>8.1f}%{rc*100:>8.1f}%{tp:>9.1f}{fp:>10.1f}")

    print("\n=== 达到指定精确率的最低阈值 ===")
    targets = [0.80, 0.90, 0.95, 0.99]
    all_scores = sorted({r["score"] for r in recs}, reverse=True)
    for tgt in targets:
        best = None
        for thr in all_scores:
            p, rc, _, _ = wmetrics(recs, thr)
            if p >= tgt:
                if best is None or rc > best[1]:
                    best = (thr, rc, p)
        if best:
            print(f"  精确率 >= {tgt*100:.0f}%  →  阈值 {best[0]:>5.0f}，"
                  f"召回率 {best[1]*100:>5.1f}%，实际精确率 {best[2]*100:.1f}%")
        else:
            print(f"  精确率 >= {tgt*100:.0f}%  →  无法达到")

    # 最优 F1 工作点
    best_f1 = max(
        ((thr, *wmetrics(recs, thr)) for thr in all_scores),
        key=lambda x: (2 * x[1] * x[2] / (x[1] + x[2]) if x[1] + x[2] else 0),
    )
    f1 = 2 * best_f1[1] * best_f1[2] / (best_f1[1] + best_f1[2])
    print(f"\n最优 F1 工作点：阈值 {best_f1[0]:.0f}，"
          f"精确率 {best_f1[1]*100:.1f}%，召回率 {best_f1[2]*100:.1f}%，F1 {f1:.3f}")


if __name__ == "__main__":
    main()
