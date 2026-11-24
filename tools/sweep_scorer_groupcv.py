# -*- coding: utf-8 -*-
"""在「留一视频」这个诚实标准下，把能试的模型都试一遍。

现有数据的天花板：63 条 AI 正样本只来自 3 个视频，全部 263 条标注只来自
6 个视频。随机 5 折的 AP 0.903 是话题泄漏撑起来的，留一视频只有 0.627。

这个脚本回答一个问题：**在数据不变的条件下，换模型能提升多少？**
如果各种模型都在 0.63 附近，说明瓶颈是数据，不是模型，再调也没用。

特征算一次存盘，后面反复用，不重复跑 Qwen3。
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import pipeline.features as F  # noqa: E402

CACHE = ROOT / "state" / "scorer_feats.npz"


def load_labels():
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


def build_cache():
    texts, y, g = load_labels()
    cols = list(F.featurize(texts[0], "").keys())
    Xh = np.array([[F.featurize(t, "")[c] for c in cols] for t in texts], float)

    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    mdl = AutoModel.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float16,
                                    device_map="auto").eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), 16):
            enc = tok(texts[i:i + 16], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(mdl.device)
            h = mdl(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1)
            out.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
    E = np.vstack(out)
    np.savez(CACHE, Xh=Xh, E=E, y=y, g=g, cols=np.array(cols))
    print(f"  特征已缓存 → {CACHE}")


def grouped_eval(X, y, g, make_clf):
    """留一视频交叉验证。返回逐样本预测。"""
    pred = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        clf = make_clf()
        clf.fit((X[tr] - mu) / sd, y[tr])
        pred[te] = clf.predict_proba((X[te] - mu) / sd)[:, 1]
    return pred


def main():
    if not CACHE.exists():
        print("  首次运行，先算特征 …", flush=True)
        build_cache()
    z = np.load(CACHE, allow_pickle=True)
    Xh, E, y, g = z["Xh"], z["E"], z["y"], z["g"]
    print(f"  样本 {len(y)}（AI {y.sum()}）  视频 {len(set(g))}")
    print("  评测方式：留一视频交叉验证（每折标准化只用训练集）")
    print()

    variants = {
        "手工特征 + LogReg C=0.03": (Xh, lambda: LogisticRegression(
            max_iter=5000, C=0.03, class_weight="balanced")),
        "手工特征 + LogReg C=0.3": (Xh, lambda: LogisticRegression(
            max_iter=5000, C=0.3, class_weight="balanced")),
        "句向量 + LogReg C=0.003": (E, lambda: LogisticRegression(
            max_iter=5000, C=0.003, class_weight="balanced")),
        "句向量 + LogReg C=0.03": (E, lambda: LogisticRegression(
            max_iter=5000, C=0.03, class_weight="balanced")),
        "拼接 + LogReg C=0.03": (np.hstack([Xh, E]), lambda: LogisticRegression(
            max_iter=5000, C=0.03, class_weight="balanced")),
        "拼接 + LogReg C=0.003": (np.hstack([Xh, E]), lambda: LogisticRegression(
            max_iter=5000, C=0.003, class_weight="balanced")),
        "拼接 + LogReg L1 C=0.01": (np.hstack([Xh, E]), lambda: LogisticRegression(
            max_iter=5000, C=0.01, penalty="l1", solver="liblinear",
            class_weight="balanced")),
        "拼接 + 随机森林": (np.hstack([Xh, E]), lambda: RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2, class_weight="balanced",
            random_state=0, n_jobs=-1)),
        "拼接 + 梯度提升": (np.hstack([Xh, E]), lambda: GradientBoostingClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.06, random_state=0)),
    }

    print("  %-30s %8s %8s %8s" % ("模型", "AP", "AUC", "对比基线"))
    print("  " + "-" * 60)
    base = None
    results = {}
    for name, (X, mk) in variants.items():
        try:
            p = grouped_eval(X, y, g, mk)
        except Exception as e:                                 # noqa: BLE001
            print("  %-30s  失败: %s" % (name, str(e)[:28]))
            continue
        ap, auc = average_precision_score(y, p), roc_auc_score(y, p)
        if base is None:
            base = ap
        results[name] = (ap, auc, p)
        print("  %-30s %8.3f %8.3f %+8.3f" % (name, ap, auc, ap - base))

    print()
    best = max(results.items(), key=lambda kv: kv[1][0])
    print(f"  最好：{best[0]}  AP={best[1][0]:.3f} AUC={best[1][1]:.3f}")
    print()
    print("  参考：随机 5 折报的 AP=0.903（虚高），留一视频最优如上。")
    print("        如果各模型都挤在 0.6 附近，说明瓶颈是「正样本只来自 3 个视频」，")
    print("        不是模型选得不好——再怎么调也没用，得补数据。")


if __name__ == "__main__":
    main()
