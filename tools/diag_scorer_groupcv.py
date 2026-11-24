# -*- coding: utf-8 -*-
"""诚实评估：打分器在「没见过的新视频」上到底能到多少。

## 为什么要做这个

现有报告说 5 折 CV 的 AP=0.901 / AUC=0.960（train_scorer_v3.py）。
但那 63 条 AI 正样本只来自 **3 个视频**，263 条标注总共只来自 **6 个视频**。
StratifiedKFold(shuffle=True) 会把同一个视频的评论同时放进训练集和测试集，
模型可以靠「这条在聊老唐 / 聊材料学」这种话题特征认出视频，而不是认出
「AI 腔」。报出来的分数就是虚的。

真实用法是：给一个**没见过的新视频**打分。所以必须按视频分组做
留一视频交叉验证（LGVO）。

顺带修掉另外两个小泄漏：
  - 标准化用全量数据算的均值方差 → 改成每折只用训练集算
  - C 在同一套 CV 上调、又用它报分 → 这里固定 C，只看泛化

## 结论怎么看

LGVO 分数才是能对外报的数字。随机 CV 与 LGVO 的差距，就是「话题泄漏」
贡献的虚高部分。
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import pipeline.features as F  # noqa: E402


def load():
    corp = {}
    for line in open(ROOT / "data" / "corpus.jsonl", encoding="utf-8"):
        r = json.loads(line)
        corp[str(r.get("rpid") or "")] = r.get("bvid", "")
    rows = [r for r in csv.DictReader(
        open(ROOT / "data" / "ground_truth_merged.csv", encoding="utf-8-sig", newline=""))
        if (r.get("content") or "").strip()]
    texts = [r["content"].strip()[:400] for r in rows]
    y = np.array([1 if r["label"] == "1" else 0 for r in rows])
    g = np.array([corp.get(str(r["rpid"]), "?") for r in rows])
    return texts, y, g


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


def eval_cv(X, y, groups, mode, C=0.03):
    """mode = 'random' 或 'group'。返回 (AP, AUC, 每折明细)。"""
    n = len(y)
    pred = np.zeros(n)
    if mode == "random":
        splits = list(StratifiedKFold(5, shuffle=True, random_state=42).split(X, y))
    else:
        splits = list(LeaveOneGroupOut().split(X, y, groups))
    per_fold = []
    for tr, te in splits:
        # 标准化只用训练集，避免泄漏
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        clf = LogisticRegression(max_iter=5000, C=C, class_weight="balanced")
        clf.fit((X[tr] - mu) / sd, y[tr])
        p = clf.predict_proba((X[te] - mu) / sd)[:, 1]
        pred[te] = p
        if len(set(y[te])) == 2:
            per_fold.append((len(te), int(y[te].sum()),
                             roc_auc_score(y[te], p)))
        else:
            per_fold.append((len(te), int(y[te].sum()), float("nan")))
    return (average_precision_score(y, pred), roc_auc_score(y, pred),
            per_fold, pred)


def main():
    print("=" * 66)
    print("  打分器诚实评估：随机 CV vs 留一视频 CV")
    print("=" * 66)

    texts, y, g = load()
    print(f"  样本 {len(y)} 条（AI {y.sum()} / 真人 {(y == 0).sum()}）")
    print(f"  视频 {len(set(g))} 个，其中含 AI 正样本的 "
          f"{len(set(g[y == 1]))} 个")
    print()

    Xh, cols = hand_features(texts)
    print(f"  手工特征 {len(cols)} 个", flush=True)

    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    mdl = AutoModel.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float16,
                                    device_map="auto").eval()
    E = embed(texts, tok, mdl)
    print(f"  句向量 {E.shape[1]} 维", flush=True)

    for name, X in (("手工特征", Xh), ("手工+句向量", np.hstack([Xh, E]))):
        print()
        print(f"  【{name}】")
        ap_r, auc_r, _, _ = eval_cv(X, y, g, "random")
        print(f"    随机 5 折      AP={ap_r:.3f}  AUC={auc_r:.3f}"
              f"   ← 现有报告用的就是这个")
        ap_g, auc_g, folds, pred = eval_cv(X, y, g, "group")
        print(f"    留一视频        AP={ap_g:.3f}  AUC={auc_g:.3f}"
              f"   ← 真实泛化")
        print(f"    虚高            AP {ap_r - ap_g:+.3f}  AUC {auc_r - auc_g:+.3f}")
        print()
        print("    每折明细（留出的视频 / 样本数 / 正样本 / AUC）：")
        vids = sorted(set(g), key=lambda v: -sum(1 for x in g if x == v))
        for v, (nn, npos, auc) in zip(vids, folds):
            mark = "  ← 正样本全在这一折里" if npos == int(y.sum()) else ""
            print(f"      {v}  {nn:>3} 条  正 {npos:>2}  "
                  f"{'AUC=%.3f' % auc if auc == auc else 'AUC=—（单一类别）'}{mark}")

    print()
    print("  说明：含 AI 正样本的视频只有 3 个，留一视频等于「用 2 个视频的正样本")
    print("        去认第 3 个视频」，这是最接近真实使用场景的考法。")


if __name__ == "__main__":
    main()
