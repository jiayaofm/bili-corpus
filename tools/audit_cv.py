# -*- coding: utf-8 -*-
"""
审核工具的真实性能（严格交叉验证）

关键：模型在训练折上训练、阈值也只在训练折上校准，
测试折完全没见过——这才是不带乐观偏差的数字。

如果直接用全量训练再在同批数据上评估，会得到 100% 精确率的假象。
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

from label_ai import score as rule_score  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from pipeline import features as F  # noqa: E402

RULE_W, MODEL_W = 0.7, 0.3
TARGET = 0.95


def main():
    m = json.loads((ROOT / "labeling" / "blind_map.json").read_text(encoding="utf-8"))
    w_rand = m["pool_rest"] / m["random_count"]

    corpus = {}
    for line in (ROOT / "data" / "corpus.jsonl").open(encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            corpus[r["rpid"]] = r

    rows = list(csv.DictReader(
        (ROOT / "labeling" / "ground_truth.csv").open(encoding="utf-8-sig", newline="")))

    texts, y, w, rule = [], [], [], []
    for r in rows:
        c = corpus.get(r["rpid"])
        if not c:
            continue
        texts.append(c["content"])
        y.append(int(r["label"]))
        w.append(1.0 if int(r["was_candidate"]) == 1 else w_rand)
        rule.append(rule_score({"content": c["content"],
                                "user": c.get("user", "")})[0])

    y = np.array(y)
    w = np.array(w)
    rule = np.array(rule, float)
    cols = F.cols_of(F.CLASSIFY_GROUPS)
    X = np.array([[F.featurize(t, "")[k] for k in cols] for t in texts], float)

    rmin, rmax = rule.min(), rule.max()
    rule_n = (rule - rmin) / (rmax - rmin + 1e-9)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    verdict = np.array(["放行"] * len(y), dtype=object)

    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])

        # 内层 OOF 校准阈值
        inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
        oof = np.zeros(len(tr))
        for itr, ite in inner.split(Xtr, y[tr]):
            s2 = StandardScaler().fit(Xtr[itr])
            mm = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                        random_state=42)
            mm.fit(s2.transform(Xtr[itr]), y[tr][itr])
            oof[ite] = mm.predict_proba(s2.transform(Xtr[ite]))[:, 1]
        fused_tr = RULE_W * rule_n[tr] + MODEL_W * oof
        order = np.argsort(-fused_tr)
        thr = None
        for k in range(1, len(order) + 1):
            idx = order[:k]
            tp = int((y[tr][idx] == 1).sum())
            fp = int((y[tr][idx] == 0).sum())
            if tp + fp and tp / (tp + fp) >= TARGET:
                thr = float(fused_tr[order[k - 1]])
        if thr is None:
            thr = float(fused_tr.max())

        # 外层预测
        rf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                    random_state=42)
        rf.fit(Xtr, y[tr])
        p = rf.predict_proba(Xte)[:, 1]
        fused = RULE_W * rule_n[te] + MODEL_W * p
        verdict[te] = np.where(fused >= thr, "AI",
                               np.where(fused >= thr * 0.45, "待复核", "放行"))

    total_ai = w[y == 1].sum()
    print(f"真值 {len(y)} 条，加权 AI 总量 {total_ai:.0f}\n")
    print(f"{'档位':<12}{'加权命中':>9}{'误伤':>8}{'精确率':>9}{'召回率':>9}")
    print("-" * 48)
    for name, mask in [("AI 档", verdict == "AI"),
                       ("AI+待复核", np.isin(verdict, ["AI", "待复核"]))]:
        tp = w[mask & (y == 1)].sum()
        fp = w[mask & (y == 0)].sum()
        p = tp / (tp + fp) if tp + fp else 0
        print(f"{name:<12}{tp:>9.1f}{fp:>8.2f}{p*100:>8.1f}%{tp/total_ai*100:>8.1f}%")

    n = {k: int((verdict == k).sum()) for k in ("AI", "待复核", "放行")}
    print(f"\n样本内分布：{n}")
    print("\n注意：这是 5 折交叉验证的折外结果，模型没见过测试折。")
    print("      目标精确率 0.95，实际可能因样本量小而波动。")


if __name__ == "__main__":
    main()
