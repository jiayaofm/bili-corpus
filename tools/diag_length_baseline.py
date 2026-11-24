# -*- coding: utf-8 -*-
"""长度基线：模型到底学到了「AI 腔」，还是只学到「长的就是 AI」？

人工标注里 AI 样本中位 43 字、真人中位 20 字，差了一倍多。
如果光用长度就能做出很高的 AUC，那现在的分数有一大半是长度撑起来的，
对「写得长的真人」会系统性误判——这正好解释了为什么扩展给一些
正常的真人长评打了 90%+。
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")


def lgvo_pred(X, y, g, C=0.03):
    pred = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        clf = LogisticRegression(max_iter=5000, C=C, class_weight="balanced")
        clf.fit((X[tr] - mu) / sd, y[tr])
        pred[te] = clf.predict_proba((X[te] - mu) / sd)[:, 1]
    return pred, np.ones(len(y), bool)


def run(path, label):
    z = np.load(path, allow_pickle=True)
    Xh, E, y, g = z["Xh"], z["E"], z["y"], z["g"]
    cols = [str(c) for c in z["cols"]]

    hits = [i for i, c in enumerate(cols)
            if any(k in c.lower() for k in ("len", "char", "length", "count", "字数"))]
    print(f"  【{label}】样本 {len(y)}（正 {int(y.sum())}）")
    print(f"    长度相关特征：{[cols[i] for i in hits]}")

    if hits:
        L = Xh[:, [hits[0]]]
        p, _ = lgvo_pred(L, y, g)
        print(f"    只用长度：  AUC={roc_auc_score(y, p):.3f}  "
              f"AP={average_precision_score(y, p):.3f}")

    X = np.hstack([Xh, E])
    p2, _ = lgvo_pred(X, y, g)
    print(f"    全特征：    AUC={roc_auc_score(y, p2):.3f}  "
          f"AP={average_precision_score(y, p2):.3f}")

    # 去掉长度以外的差异：把样本按长度分层，只在同长度区间里比较
    if hits:
        ln = Xh[:, hits[0]]
        band = np.digitize(ln, np.percentile(ln, [25, 50, 75]))
        print("    分层看（同一长度区间内的 AUC）：")
        for b in range(4):
            m = band == b
            if m.sum() < 10 or len(set(y[m])) < 2:
                print(f"      第 {b + 1} 段  n={m.sum():<4} 样本太少或单一类别")
                continue
            print(f"      第 {b + 1} 段  n={m.sum():<4} "
                  f"长度 {ln[m].min():.0f}~{ln[m].max():.0f}  "
                  f"AUC={roc_auc_score(y[m], p2[m]):.3f}")
    print()


def main():
    print("=" * 64)
    print("  长度基线：分数有多少是「长 = AI」撑起来的")
    print("=" * 64)
    run(ROOT / "state" / "scorer_feats.npz", "人工标注 263 条")
    run(ROOT / "state" / "scorer_whale.npz", "鲸鱼混合 1751 条")


if __name__ == "__main__":
    main()
