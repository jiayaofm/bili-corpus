# -*- coding: utf-8 -*-
"""
配对比较：加入 AI 预标注后，模型在「纯人工测试集」上是否变好

设计要点（这是让结论可信的关键）：
  · 测试集只用 263 条人工标注，AI 预标注永不进入测试
  · 同一个 5 折划分，两个模型用同一批测试折
  · 唯一的差别是训练集：一个只有人工 263，一个加上 AI 预标 476
  · 指标用加权口径（沿用人盲标的分层权重），与项目其它报告一致

如果 expanded 更好，说明 AI 预标注提供了真实信息；
如果没有提升或变差，说明它只是在重复模型已有的知识。
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

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from pipeline import features as F  # noqa: E402

GT = ROOT / "labeling" / "ground_truth.csv"
MAP = ROOT / "labeling" / "blind_map.json"


def load():
    amap = json.loads(MAP.read_text(encoding="utf-8"))
    w_rand = amap["pool_rest"] / amap["random_count"]

    rows = []
    for r in csv.DictReader(GT.open(encoding="utf-8-sig", newline="")):
        rows.append({
            "rpid": r["rpid"],
            "label": int(r["label"]),
            "w": 1.0 if int(r["was_candidate"]) == 1 else w_rand,
            "human": not r.get("source", "").startswith("ai_prelabel"),
            "content": r["content"],
        })
    return rows


def featurize(rows):
    cols = F.cols_of(F.CLASSIFY_GROUPS)
    X = np.array([[F.featurize(r["content"], "")[c] for c in cols] for r in rows], float)
    y = np.array([r["label"] for r in rows])
    w = np.array([r["w"] for r in rows])
    return X, y, w


def metrics(y, pred, w):
    tp = w[(pred == 1) & (y == 1)].sum()
    fp = w[(pred == 1) & (y == 0)].sum()
    fn = w[(pred == 0) & (y == 1)].sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def main():
    rows = load()
    human = [r for r in rows if r["human"]]
    extra = [r for r in rows if not r["human"]]
    print(f"人工标注 {len(human)} 条（测试集）")
    print(f"AI 预标注 {len(extra)} 条（只进训练集）\n")

    Xh, yh, wh = featurize(human)
    Xe, ye, _we = featurize(extra)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred_base = np.zeros(len(human), int)
    pred_exp = np.zeros(len(human), int)

    for tr, te in skf.split(Xh, yh):
        m1 = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                    random_state=42)
        m1.fit(Xh[tr], yh[tr])
        pred_base[te] = m1.predict(Xh[te])

        m2 = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                    random_state=42)
        m2.fit(np.vstack([Xh[tr], Xe]), np.concatenate([yh[tr], ye]))
        pred_exp[te] = m2.predict(Xh[te])

    print(f"{'方案':<28}{'精确率':>9}{'召回率':>9}{'F1':>9}")
    print("-" * 56)
    for name, pred in [("仅人工 263 条", pred_base),
                       (f"人工 263 + 预标 {len(extra)}", pred_exp)]:
        p, r, f1 = metrics(yh, pred, wh)
        print(f"{name:<28}{p*100:>8.1f}%{r*100:>8.1f}%{f1:>9.3f}")

    # 配对显著性：McNemar 近似
    a = pred_base == yh
    b = pred_exp == yh
    only_a = int(((a) & (~b)).sum())
    only_b = int(((~a) & (b)).sum())
    print(f"\n预测对错变化：仅基线对 {only_a} 条，仅扩展对 {only_b} 条")
    if only_a + only_b:
        from math import comb
        n = only_a + only_b
        k = min(only_a, only_b)
        pv = sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2
        print(f"McNemar 精确检验 p = {min(pv,1):.4f}"
              f"　{'显著' if pv < 0.05 else '不显著'}")

    # 分组看：AI 预标注里正类比例
    print(f"\nAI 预标注正类 {int(ye.sum())} / {len(ye)} = {ye.mean()*100:.1f}%")
    print(f"人工标注正类 {int(yh.sum())} / {len(yh)} = {yh.mean()*100:.1f}%")


if __name__ == "__main__":
    main()
