# -*- coding: utf-8 -*-
"""用 Qwen3 句向量 + 手工特征，重训 AI 腔分类器。

## 为什么换

现有 42 手工特征（随机森林）5 折 CV：AP=0.838 AUC=0.921。
Qwen3-1.7B 句向量 + 逻辑回归：AP=0.888 AUC=0.956。
两者拼接：AP=0.905 AUC=0.960。

手工特征看的是标点统计（「」、分号、句号数），所以对「长句子、讲道理、
带链接」的真人评论会误判——这些是结构特征，AI 和人都会用。
句向量看的是语义本身，正好补这块。

## 产出

`state/scorer_v3.npz`：逻辑回归的系数 + 标准化参数
服务端加载它，推理时用同一套特征。

## 训练数据

`data/ground_truth_merged.csv` 263 条（AI 63 / 真人 200）。
**注意：这是现有唯一带标签的数据，所以 CV 报出的数字有方差，别当精确值。**
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import pipeline.features as F  # noqa: E402

OUT = ROOT / "state" / "scorer_v3.npz"


def load_labeled():
    rows = [r for r in csv.DictReader(
        open(ROOT / "data" / "ground_truth_merged.csv", encoding="utf-8-sig", newline=""))
        if (r.get("content") or "").strip()]
    texts = [r["content"].strip()[:400] for r in rows]
    y = np.array([1 if r["label"] == "1" else 0 for r in rows])
    return texts, y


def hand_features(texts):
    cols = list(F.featurize(texts[0], "").keys())
    return np.array([[F.featurize(t, "")[c] for c in cols] for t in texts], float), cols


def embed(texts, tok, mdl, bs=16):
    import torch
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(mdl.device)
            h = mdl(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1)
            out.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
    return np.vstack(out)


def main():
    print("=" * 62)
    print("  重训 AI 腔分类器：手工特征 + Qwen3 句向量")
    print("=" * 62)

    texts, y = load_labeled()
    print(f"  标注样本 {len(y)} 条（AI {y.sum()} / 真人 {(y == 0).sum()}）")

    print("  计算 42 个手工特征 …", flush=True)
    Xh, cols = hand_features(texts)

    print("  载入 Qwen3-1.7B 计算句向量 …", flush=True)
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    mdl = AutoModel.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float16,
                                    device_map="auto").eval()
    E = embed(texts, tok, mdl)
    print(f"  句向量 {E.shape[1]} 维")

    X = np.hstack([Xh, E])
    mean = X.mean(0)
    std = X.std(0) + 1e-8
    Xs = (X - mean) / std

    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    best = None
    print()
    print("  调参（5 折 CV）：")
    for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        p = cross_val_predict(LogisticRegression(max_iter=5000, C=C,
                                                 class_weight="balanced"),
                              Xs, y, cv=cv, method="predict_proba")[:, 1]
        ap, auc = average_precision_score(y, p), roc_auc_score(y, p)
        star = ""
        if best is None or auc > best[1]:
            best = (C, auc, ap)
            star = "  ←"
        print(f"    C={C:<6} AP={ap:.3f}  AUC={auc:.3f}{star}")

    C, auc, ap = best
    print()
    print(f"  选定 C={C}   CV: AP={ap:.3f}  AUC={auc:.3f}")

    clf = LogisticRegression(max_iter=5000, C=C, class_weight="balanced").fit(Xs, y)
    np.savez(OUT,
             coef=clf.coef_.ravel(), intercept=clf.intercept_,
             mean=mean, std=std,
             n_hand=len(cols), n_emb=E.shape[1],
             cv_auc=auc, cv_ap=ap, C=C,
             cols=np.array(cols))
    print(f"  已保存 → {OUT}")

    # 对若干典型文本的判定
    probe = [
        ("大肥鱼腔", "老唐看菜先看价。沙县 11 块的炒饭对得起 11 块他就说不错；11 块的菜卖 111，他当场开喷。"),
        ("大肥鱼腔", "点歪不点歪都是说法。现在是材料学卡住了，材料一突破，科技立刻爆发。"),
        ("典型AI腔", "这期视频不仅呈现了市井烟火，更是折射出当代消费分层。"),
        ("真人说理", "唐大厨评价菜品是按价格来的，比如沙县一份炒饭11块，能抄出11块的价格他就会评价不错。如果是111块的价格炒11的菜那他就要开喷了"),
        ("真人技术", "已经把缺的把battery-control skill上传到 https://github.com/gychen-NJU/battergi-control 。而且也更新了插件本体，增加了这个依赖检测。"),
        ("真人聊天", "好的好的，谢谢up🥰"),
    ]
    pt = [t for _, t in probe]
    Xp = np.hstack([hand_features(pt)[0], embed(pt, tok, mdl)])
    pp = clf.predict_proba((Xp - mean) / std)[:, 1]
    print()
    print("  典型文本打分：")
    for (tag, t), v in zip(probe, pp):
        print(f"    {tag:<8} {v*100:5.1f}%   {t[:34]}")


if __name__ == "__main__":
    main()
