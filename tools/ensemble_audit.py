# -*- coding: utf-8 -*-
"""
融合打分器：规则 + 模型，寻找自动审核的最佳工作点

自动审核的资源约束：
  · 误伤真人 = 严重（用户投诉）
  · 漏掉 AI   = 可接受（少抓几个）

所以优化目标不是 F1，而是：**给定精确率下限，最大化召回率**。

对比三种打分：
  1. 规则分（人工设计的启发式）
  2. 模型概率（随机森林，交叉验证的折外预测）
  3. 融合分（规则 + 模型）
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

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from pipeline import features as F  # noqa: E402

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
        s, _ = score({"content": c["content"], "user": c.get("user", "")})
        recs.append({"rec": c, "rule": float(s), "label": int(r["label"]),
                     "w": 1.0 if int(r["was_candidate"]) == 1 else w_rand})
    return recs


def pr_at_precision(scores, labels, weights, target):
    """在精确率 >= target 的约束下，找召回率最大的阈值"""
    order = np.argsort(-np.asarray(scores))
    best = None
    for k in range(1, len(order) + 1):
        idx = order[:k]
        tp = weights[idx][labels[idx] == 1].sum()
        fp = weights[idx][labels[idx] == 0].sum()
        if tp + fp == 0:
            continue
        p = tp / (tp + fp)
        if p < target:
            continue
        fn = weights[labels == 1].sum() - tp
        r = tp / (tp + fn) if (tp + fn) else 0
        if best is None or r > best[1]:
            best = (float(scores[order[k - 1]]), r, p)
    return best


def main():
    recs = load()
    labels = np.array([r["label"] for r in recs])
    weights = np.array([r["w"] for r in recs])
    rule = np.array([r["rule"] for r in recs])
    print(f"真值 {len(recs)} 条  正类 {int(labels.sum())}  "
          f"误伤成本权重 {weights.max():.2f}\n")

    # 模型折外概率
    cols = F.cols_of(F.CLASSIFY_GROUPS)
    X = np.array([[F.featurize(r["rec"]["content"], r["rec"].get("user", ""))[c]
                   for c in cols] for r in recs], dtype=float)
    X = StandardScaler().fit_transform(X)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    prob = np.zeros(len(recs))
    for tr, te in skf.split(X, labels):
        m = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                    random_state=42)
        m.fit(X[tr], labels[tr])
        prob[te] = m.predict_proba(X[te])[:, 1]

    # 归一到 0-1 的规则分
    rule_n = (rule - rule.min()) / (rule.max() - rule.min() + 1e-9)

    cands = {
        "规则分": rule,
        "模型概率": prob,
        "融合 0.5/0.5": 0.5 * rule_n + 0.5 * prob,
        "融合 0.6规则/0.4模型": 0.6 * rule_n + 0.4 * prob,
        "取最大": np.maximum(rule_n, prob),
        "加权规则为主": 0.7 * rule_n + 0.3 * prob,
    }

    print("=== 精确率约束下的最大召回率 ===")
    print(f"{'打分器':<22}{'P>=90%':>10}{'P>=95%':>10}{'P>=99%':>10}")
    print("-" * 62)
    rows = {}
    for name, s in cands.items():
        line = f"{name:<22}"
        row = {}
        for tgt in (0.90, 0.95, 0.99):
            r = pr_at_precision(list(s), labels, weights, tgt)
            row[tgt] = r
            line += f"{('%.1f%%' % (r[1]*100)) if r else '—':>10}"
        rows[name] = row
        print(line)

    print("\n=== 各打分器在 P>=95% 时的实际工作点 ===")
    for name, row in rows.items():
        r = row[0.95]
        if r:
            print(f"  {name:<22} 阈值 {r[0]:>7.2f}  召回 {r[1]*100:>5.1f}%  "
                  f"实际精确率 {r[2]*100:.1f}%")

    # 推荐
    best_name = max(rows, key=lambda n: (rows[n][0.95][1] if rows[n][0.95] else -1))
    print(f"\n推荐用于自动审核：**{best_name}**")


if __name__ == "__main__":
    main()
