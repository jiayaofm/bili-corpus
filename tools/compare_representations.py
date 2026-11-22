# -*- coding: utf-8 -*-
"""用交叉验证诚实比较三种表示。

背景：data/ground_truth_merged.csv 那 263 条是分类器的**训练集**，
直接在它上面 predict 会得到 AP=1.000 的假成绩。必须交叉验证。
"""
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import pipeline.features as F

rows = [r for r in csv.DictReader(
    open(ROOT / "data" / "ground_truth_merged.csv", encoding="utf-8-sig", newline=""))
    if (r.get("content") or "").strip()]
texts = [r["content"].strip()[:400] for r in rows]
y = np.array([1 if r["label"] == "1" else 0 for r in rows])
print(f"  样本 {len(y)} 条（AI {y.sum()} / 真人 {(y == 0).sum()}）")

cv = StratifiedKFold(5, shuffle=True, random_state=42)


def report(tag, p):
    ap = average_precision_score(y, p)
    auc = roc_auc_score(y, p)
    # 最佳阈值下的精确率/召回率/误伤
    best = max(((t, (p >= t).sum()) for t in np.unique(p)), key=lambda kv: kv[1] and 0 or 0)
    print(f"  {tag:<34} AP={ap:.3f}  AUC={auc:.3f}")
    return ap, auc


# ① 42 手工特征 + 随机森林（重训，交叉验证）
matches = list(F.featurize(texts[0], "").keys())
X = np.array([[F.featurize(t, "")[c] for c in matches] for t in texts], float)
rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
p_rf = cross_val_predict(rf, X, y, cv=cv, method="predict_proba")[:, 1]
report("42 手工特征 + 随机森林(5折CV)", p_rf)

# ② 只用句号和逗号，看标点能走多远
punc = np.array([[t.count("。"), t.count("，"), t.count("！"), len(t)] for t in texts], float)
p_punc = cross_val_predict(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
    punc, y, cv=cv, method="predict_proba")[:, 1]
report("只给 句号/逗号/感叹号/长度", p_punc)

# ③ Qwen3 句向量 + 逻辑回归
import torch
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
mdl = AutoModel.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float16,
                                device_map="auto").eval()
embs = []
with torch.inference_mode():
    for i in range(0, len(texts), 16):
        enc = tok(texts[i:i + 16], return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to(mdl.device)
        out = mdl(**enc).last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        embs.append(((out * mask).sum(1) / mask.sum(1)).float().cpu().numpy())
E = np.vstack(embs)
p_emb = cross_val_predict(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
    E, y, cv=cv, method="predict_proba")[:, 1]
report(f"Qwen3-1.7B 句向量({E.shape[1]}维) + LR", p_emb)

# ④ 两者拼接
X2 = np.hstack([X, E])
p_both = cross_val_predict(
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced")),
    X2, y, cv=cv, method="predict_proba")[:, 1]
report("手工特征 + 句向量 拼接", p_both)

print()
print("  阈值权衡（用最好的那个表示）：")
best_p = p_emb if roc_auc_score(y, p_emb) > roc_auc_score(y, p_rf) else p_rf
print("    阈值   召回    误伤   精确率")
for t in (0.5, 0.6, 0.7, 0.8, 0.9):
    tp = ((best_p >= t) & (y == 1)).sum()
    fp = ((best_p >= t) & (y == 0)).sum()
    rec = tp / max(1, (y == 1).sum())
    fpr = fp / max(1, (y == 0).sum())
    prec = tp / max(1, tp + fp)
    print(f"    {t:.1f}  {rec*100:5.1f}%  {fpr*100:5.1f}%  {prec*100:5.1f}%")
