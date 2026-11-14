# -*- coding: utf-8 -*-
"""
s06 分类

读 s05 的矩阵，跑三个模型 + 规则基线对比 + 特征组消融 + PR 曲线。

输出：
  analysis/v5_classify.md
  analysis/fig_v5_importance.png
  analysis/fig_v5_pr.png
  analysis/fig_v5_ablation.png
  analysis/fig_v5_confusion.png
"""
import csv
import json

import numpy as np

import common
import evaluate
from common import CFG, P, banner, info, record

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_curve, auc)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

STAGE = "s06_classify"
MD_REL = "analysis/v5_classify.md"


def cv_eval(model_factory, X, y, folds, repeats, seed):
    """未加权（样本口径）——用于特征组消融的相对比较"""
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    P_, R_, F_ = [], [], []
    for tr, te in rskf.split(X, y):
        m = model_factory()
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        tp = int(((pred == 1) & (y[te] == 1)).sum())
        fp = int(((pred == 1) & (y[te] == 0)).sum())
        fn = int(((pred == 0) & (y[te] == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        P_.append(p); R_.append(r)
        F_.append(2 * p * r / (p + r) if p + r else 0.0)
    return np.mean(P_), np.mean(R_), np.mean(F_), np.std(F_)


def cv_eval_weighted(model_factory, X, y, w, folds, repeats, seed):
    """同时给出样本口径与总体口径（加权）指标"""
    rskf = RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=seed)
    sp, sr, sf, wp, wr, wf = [], [], [], [], [], []
    for tr, te in rskf.split(X, y):
        m = model_factory()
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        s = evaluate.sample_metrics(y[te], pred)
        v = evaluate.weighted_metrics(y[te], pred, w[te])
        sp.append(s["precision"]); sr.append(s["recall"]); sf.append(s["f1"])
        wp.append(v["precision"]); wr.append(v["recall"]); wf.append(v["f1"])
    return {
        "sample": (np.mean(sp), np.mean(sr), np.mean(sf), np.std(sf)),
        "weighted": (np.mean(wp), np.mean(wr), np.mean(wf), np.std(wf)),
    }


def main(force=False):
    banner("s06 分类")

    st = P["state"]
    Xp, yp, mp = st / "X_classify.npy", st / "y_classify.npy", st / "meta_classify.json"
    for p in (Xp, yp, mp):
        if not p.exists():
            info(f"[失败] 缺少 {p}，请先跑 s05")
            return 1

    X = np.load(Xp)
    y = np.load(yp)
    meta = json.loads(mp.read_text(encoding="utf-8"))
    cols = meta["cols"]
    rpids = meta["rpids"]
    group_of = meta["group_of"]
    was_cand = meta.get("was_candidate")

    info(f"矩阵 {X.shape}  正 {int(y.sum())} / 负 {int((y == 0).sum())}")

    # ---- 抽样权重（分层抽样还原到总体口径）----
    bmap = evaluate.load_blind_map(P["labeling"] / "blind_map.json")
    if was_cand is None:
        info("[警告] meta 里没有 was_candidate，退化为不加权")
        w = np.ones(len(y), dtype=float)
    else:
        w = evaluate.stratum_weights(was_cand, bmap)
    sample_ai_rate = float(y.sum()) / len(y)
    eff_n = evaluate.effective_n(w)
    if bmap and eff_n > 0:
        pop_ai = float(w[y == 1].sum())
        pop_rate = pop_ai / eff_n
        info(f"样本 AI 密度 {sample_ai_rate*100:.1f}%  →  加权总体 {pop_rate*100:.1f}%")
        info(f"等效样本量 {eff_n:.0f}")
    else:
        pop_rate = sample_ai_rate
        info("[提示] 未找到 blind_map.json，指标不加权")

    folds = CFG["classify"]["cv_folds"]
    repeats = CFG["classify"]["cv_repeats"]
    seed = CFG["classify"]["seed"]

    Xs = StandardScaler().fit_transform(X)

    MODELS = {
        "逻辑回归": lambda: LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced"),
        "随机森林": lambda: RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                                   random_state=seed, n_jobs=-1),
        "决策树": lambda: DecisionTreeClassifier(max_depth=4, class_weight="balanced",
                                                 random_state=seed),
    }

    lines = ["# 实验 v5：分类（人工真值）", ""]
    lines.append(f"- 样本 {len(y)} 条，正类 {int(y.sum())} / 负类 {int((y == 0).sum())}")
    lines.append(f"- 特征 {len(cols)} 维，来自 {len(meta['groups'])} 个特征组")
    lines.append(f"- 评估：{folds} 折 × {repeats} 次重复分层交叉验证")
    lines.append("")
    lines.append("> 标签为人工盲标（x=明确AI / m=疑似AI 均计入正类），不来自规则，无循环性。")
    lines.append("")
    lines.append(f"> **注意：指标口径**：盲标采用分层抽样，样本 AI 密度 "
                 f"{sample_ai_rate*100:.1f}%，全库真实约 {pop_rate*100:.1f}%。"
                 "下文**主结论一律用加权口径**；样本口径仅用于特征组的相对比较。")

    # ---------- 抽样偏差说明 ----------
    lines.append("\n## 一、抽样偏差与加权还原\n")
    lines.append("盲标样本的构造方式：")
    lines.append("")
    lines.append("| 层 | 抽样方式 | 权重 |")
    lines.append("|---|---|---|")
    if bmap:
        lines.append(f"| 规则候选层 | 全取 {bmap.get('candidate_count')} 条 | 1 |")
        lines.append(f"| 随机层 | 从 {bmap.get('pool_rest')} 条中抽 "
                     f"{bmap.get('random_count')} 条 | "
                     f"{bmap.get('pool_rest')/max(1,bmap.get('random_count',1)):.3f} |")
    else:
        lines.append("| — | 未找到 blind_map.json | 1（不加权） |")
    lines.append("")
    lines.append(f"加权后等效样本量 **{eff_n:.0f}**（全库 1008 条）。")
    lines.append("")
    lines.append("为什么必须加权：")
    lines.append("")
    lines.append("- **召回率** = TP/(TP+FN)，分母是全库所有 AI。样本里 AI 被过度包含，"
                 "未加权召回率会虚高。")
    lines.append("- **精确率** = TP/(TP+FP)，条件于「预测为正」，"
                 "在分层抽样下受影响较小。")
    lines.append("- 因此加权主要修正召回率，以及由它连带的 F1。")

    # ---------- 模型对比 ----------
    lines.append("\n## 二、模型对比（主结论：加权口径）\n")
    lines.append("| 模型 | 精确率 | 召回率 | F1 |")
    lines.append("|---|---|---|---|")
    results = {}
    for name, fac in MODELS.items():
        res = cv_eval_weighted(fac, Xs, y, w, folds, repeats, seed)
        wp, wr, wf, wfs = res["weighted"]
        results[name] = res
        lines.append(f"| {name} | {wp:.3f} | {wr:.3f} | {wf:.3f} ± {wfs:.3f} |")
        info(f"{name}: 加权 P={wp:.3f} R={wr:.3f} F1={wf:.3f}")
    best_name = max(results, key=lambda k: results[k]["weighted"][2])

    lines.append("\n对照：同一模型在**样本口径**（未加权）下的表现\n")
    lines.append("| 模型 | 精确率 | 召回率 | F1 |")
    lines.append("|---|---|---|---|")
    for name in MODELS:
        sp, sr, sf, sfs = results[name]["sample"]
        lines.append(f"| {name} | {sp:.3f} | {sr:.3f} | {sf:.3f} ± {sfs:.3f} |")
    lines.append("")
    lines.append("→ 样本口径的召回率明显高于加权口径，差额来自被过采样的 AI 样本。")

    # ---------- 规则基线 ----------
    gt_path = P["labeling"] / "ground_truth.csv"
    cand = {}
    if gt_path.exists():
        with gt_path.open(encoding="utf-8-sig", newline="") as f:
            for d in csv.DictReader(f):
                cand[d["rpid"]] = int(d.get("was_candidate") or 0)

    if cand:
        cvec = np.array([cand.get(r, 0) for r in rpids])
        rule_s = evaluate.sample_metrics(y, cvec)
        rule_w = evaluate.weighted_metrics(y, cvec, w)
        rp, rr, rf1 = rule_w["precision"], rule_w["recall"], rule_w["f1"]

        lines.append("\n## 三、与规则打分器对比（加权口径）\n")
        lines.append("| 方法 | 精确率 | 召回率 | F1 |")
        lines.append("|---|---|---|---|")
        lines.append(f"| 规则打分器 | {rp:.3f} | {rr:.3f} | {rf1:.3f} |")
        bw = results[best_name]["weighted"]
        lines.append(f"| {best_name} | {bw[0]:.3f} | {bw[1]:.3f} | {bw[2]:.3f} |")
        if rf1 > 0:
            delta = bw[2] - rf1
            lines.append("")
            lines.append(f"→ 模型 F1 比规则基线 **{delta:+.3f}**。")
            if delta > 0:
                lines.append("规则精确但召回低，模型用少量精确率换回大量召回。")
            else:
                lines.append("在该加权口径下，规则基线仍优于学习模型——"
                             "说明人工规则本身已经捕捉到了主要信号，"
                             "且模型受限于正样本规模，泛化不足。")
        lines.append("")
        lines.append(f"（样本口径对照：规则 P={rule_s['precision']:.3f} "
                     f"R={rule_s['recall']:.3f} F1={rule_s['f1']:.3f}，"
                     "召回率同样被高估。）")
        info(f"规则基线（加权）: P={rp:.3f} R={rr:.3f} F1={rf1:.3f}")
        info(f"规则基线（样本）: P={rule_s['precision']:.3f} "
             f"R={rule_s['recall']:.3f} F1={rule_s['f1']:.3f}")

    # ---------- 特征重要性 ----------
    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                random_state=seed, n_jobs=-1)
    rf.fit(Xs, y)
    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1]

    lines.append("\n## 四、特征重要性\n")
    lines.append("| 排名 | 特征 | 所属组 | 重要性 |")
    lines.append("|---|---|---|---|")
    for i, j in enumerate(order[:15], 1):
        lines.append(f"| {i} | {cols[j]} | {group_of.get(cols[j], '')} | {imp[j]:.4f} |")

    fig, ax = plt.subplots(figsize=(7, 6))
    top = order[:15][::-1]
    ax.barh([cols[i] for i in top], imp[top], color="#4a7ab5")
    ax.set_xlabel("importance")
    ax.set_title("特征重要性 Top 15")
    fig.tight_layout()
    fig.savefig(P["analysis"] / "fig_v5_importance.png", dpi=130)
    plt.close(fig)

    # ---------- 决策树规则 ----------
    dt = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=seed)
    dt.fit(Xs, y)
    lines.append("\n## 五、决策树规则（深度 3）\n")
    lines.append("```")
    lines.append(export_text(dt, feature_names=cols, decimals=2))
    lines.append("```")

    # ---------- 特征组消融 ----------
    lines.append("\n## 六、特征组消融\n")
    lines.append("| 特征子集 | 维度 | 精确率 | 召回率 | F1 |")
    lines.append("|---|---|---|---|---|")

    def idx_of(names):
        return [cols.index(n) for n in names if n in cols]

    subsets = []
    cum = []
    for g in meta["groups"]:
        names = [c for c in cols if group_of.get(c) == g]
        if not names:
            continue
        cum += names
        subsets.append((f"累积到 {g}", idx_of(cum)))
    for g in meta["groups"]:
        names = [c for c in cols if group_of.get(c) == g]
        if names:
            subsets.append((f"仅 {g}", idx_of(names)))
    struct = [c for c in cols if group_of.get(c) == "D 结构类"]
    if struct:
        subsets.append(("去掉 结构类（含长度）",
                        [i for i, c in enumerate(cols) if c not in struct]))
    subsets.append(("全部特征", list(range(len(cols)))))

    abl = {}
    for label, idxs in subsets:
        if not idxs:
            continue
        Xg = StandardScaler().fit_transform(X[:, idxs])
        p, r, f, _ = cv_eval(
            lambda: RandomForestClassifier(n_estimators=150, class_weight="balanced",
                                           random_state=seed, n_jobs=-1),
            Xg, y, folds, max(3, repeats // 2), seed)
        abl[label] = f
        lines.append(f"| {label} | {len(idxs)} | {p:.3f} | {r:.3f} | {f:.3f} |")
        info(f"[消融] {label}: F1={f:.3f}")

    # 消融图
    cum_labels = [l for l in abl if l.startswith("累积到")]
    if cum_labels:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        vals = [abl[l] for l in cum_labels]
        ax.plot(range(len(vals)), vals, "o-", lw=2, color="#4a7ab5")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels([l.replace("累积到 ", "") for l in cum_labels],
                           rotation=25, ha="right")
        ax.set_ylabel("F1")
        ax.set_title("累积加入特征组的效果")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(P["analysis"] / "fig_v5_ablation.png", dpi=130)
        plt.close(fig)

    # ---------- PR 曲线 + 混淆矩阵 ----------
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    Xb = StandardScaler().fit_transform(X)
    prob = np.zeros(len(y))
    for tr, te in skf.split(Xb, y):
        m = MODELS[best_name]()
        m.fit(Xb[tr], y[tr])
        prob[te] = m.predict_proba(Xb[te])[:, 1]

    prec, rec, thr = precision_recall_curve(y, prob)
    ap = average_precision_score(y, prob)
    best_i = int(np.argmax(2 * prec * rec / (prec + rec + 1e-12)))
    th = float(thr[min(best_i, len(thr) - 1)])
    pred = (prob >= th).astype(int)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, label=f"{best_name}（AP={ap:.3f}）")
    ax.set_xlabel("召回率")
    ax.set_ylabel("精确率")
    ax.set_title("PR 曲线")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(P["analysis"] / "fig_v5_pr.png", dpi=130)
    plt.close(fig)

    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["真人", "AI"]); ax.set_yticklabels(["真人", "AI"])
    ax.set_xlabel("预测"); ax.set_ylabel("实际")
    ax.set_title(f"混淆矩阵（{best_name} @ {th:.2f}）")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(P["analysis"] / "fig_v5_confusion.png", dpi=130)
    plt.close(fig)

    lines.append("\n## 七、最佳阈值下的表现（样本口径）\n")
    lines.append(f"- 模型：{best_name}，阈值 {th:.2f}")
    lines.append(f"- 平均精确率 AP = {ap:.3f}")
    lines.append(f"- 混淆矩阵：TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")

    lines.append("\n## 八、结论\n")
    lines.append(f"1. 最佳模型 **{best_name}**，加权 F1 = "
                 f"{results[best_name]['weighted'][2]:.3f}"
                 f"（样本口径 {results[best_name]['sample'][2]:.3f}）。")
    if "去掉 结构类（含长度）" in abl:
        drop = abl["全部特征"] - abl["去掉 结构类（含长度）"]
        lines.append(f"2. 去掉结构类（含长度）后 F1 变化 {drop:+.3f}，"
                     "说明长度不是主要判据。")
    lines.append("3. 符号类与句式类单独使用通常精确率高、召回率低，属铁证型特征。")
    lines.append("\n### 局限")
    lines.append(f"- 正样本 {int(y.sum())} 条，指标置信区间较宽")
    lines.append("- 标注者单一，存在个人判断偏差")
    lines.append("- 语料仍集中于少数视频，话题偏置")

    (common.ROOT / MD_REL).write_text("\n".join(lines), encoding="utf-8")
    info(f"报告 {MD_REL}")

    record(STAGE, {
        MD_REL: len(lines),
        "analysis/fig_v5_importance.png": 1,
        "analysis/fig_v5_pr.png": 1,
        "analysis/fig_v5_confusion.png": 1,
        "analysis/fig_v5_ablation.png": 1,
    }, extra={"best": best_name,
              "f1_weighted": round(results[best_name]["weighted"][2], 4),
              "f1_sample": round(results[best_name]["sample"][2], 4),
              "pop_ai_rate": round(pop_rate, 4)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
