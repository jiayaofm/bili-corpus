# -*- coding: utf-8 -*-
"""
加权评估：消除分层抽样带来的偏差

问题：盲标样本里 AI 占 23.6%（37/157），但全库真实只有 7.9%。
在这个偏斜样本上算的召回率会虚高。

做法：按抽样比给每条样本加权，重新算精确率/召回率。
  · 候选层：全取，权重 1
  · 随机层：963 条里抽 120，权重 963/120 ≈ 8.03

这是论文里应该报告的最终数字。
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
SetPath = Path(__file__).parent
sys.path.insert(0, str(SetPath))

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

from pipeline import features as F

ROOT = SetPath
GT = ROOT / "labeling" / "ground_truth.csv"
CORPUS = ROOT / "data" / "corpus.jsonl"
MAP = ROOT / "labeling" / "blind_map.json"


def main():
    m = json.loads(MAP.read_text(encoding="utf-8"))
    pool = m["pool_rest"]
    n_rand = m["random_count"]
    w_rand = pool / n_rand

    corpus = {}
    for line in CORPUS.open(encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            corpus[r["rpid"]] = r

    rows = list(csv.DictReader(GT.open(encoding="utf-8-sig", newline="")))
    recs, y, w = [], [], []
    for r in rows:
        c = corpus.get(r["rpid"])
        if not c:
            continue
        recs.append(c)
        y.append(int(r["label"]))
        w.append(1.0 if int(r["was_candidate"]) == 1 else w_rand)

    y = np.array(y)
    w = np.array(w)
    print(f"样本 {len(y)} 条   正类 {y.sum()}")
    print(f"随机层放大系数 {w_rand:.3f}")
    print(f"加权后等效样本量 {w.sum():.0f}（应接近全库 1008）\n")

    cols = F.cols_of(F.CLASSIFY_GROUPS)
    X = np.array([[F.featurize(r["content"], r.get("user", ""))[c] for c in cols]
                  for r in recs], dtype=float)
    X = StandardScaler().fit_transform(X)

    def weighted_metrics(y_true, pred, weights):
        tp = weights[(pred == 1) & (y_true == 1)].sum()
        fp = weights[(pred == 1) & (y_true == 0)].sum()
        fn = weights[(pred == 0) & (y_true == 1)].sum()
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f1

    # ---------- 规则基线 ----------
    rule_pred = np.array([int(r["was_candidate"]) for r in rows
                          if r["rpid"] in corpus])
    rp, rr, rf = weighted_metrics(y, rule_pred, w)

    # ---------- 模型（重复 CV 平均概率 → 阈值 0.5）----------
    models = {
        "逻辑回归": LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"),
        "随机森林": RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                            random_state=42),
    }
    print(f"{'方法':<12}{'精确率':>9}{'召回率':>9}{'F1':>9}")
    print("-" * 40)
    print(f"{'规则基线':<12}{rp:>9.3f}{rr:>9.3f}{rf:>9.3f}")

    for name, model in models.items():
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
        ps, rs, fs = [], [], []
        for tr, te in rskf.split(X, y):
            mdl = model.__class__(**model.get_params())
            mdl.fit(X[tr], y[tr])
            pred = mdl.predict(X[te])
            p, r, f = weighted_metrics(y[te], pred, w[te])
            ps.append(p); rs.append(r); fs.append(f)
        print(f"{name:<12}{np.mean(ps):>9.3f}{np.mean(rs):>9.3f}{np.mean(fs):>9.3f}"
              f"   (F1 ±{np.std(fs):.3f})")

    print("\n注：加权后精确率不受抽样影响（条件于预测），召回率才被修正。")
    print("    规则精确率高但召回低，模型召回高但精确率下降——这是真实权衡。")


if __name__ == "__main__":
    main()
