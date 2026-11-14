# -*- coding: utf-8 -*-
"""
实验 v2：把分类做扎实（不依赖新数据）

设计要点：
  1. 特征按语义分 7 组，做**累积消融**（A → A+B → A+B+C …），
     看每组加进来是提升还是拖累
  2. **重复分层交叉验证**（5 折 × 10 次），报告均值±标准差，
     避免单次划分的偶然性
  3. PR 曲线 + ROC 曲线 + 阈值扫描——因为正样本只有 31 条，
     单一阈值下的 F1 不可靠，看整条曲线才稳
  4. 与**规则打分器**做基线对比：规则分数直接当预测分数扫阈值
  5. Bootstrap 置信区间

输出：analysis/v2_report.md + 4 张图
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_curve)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = Path(__file__).parent
DATA = ROOT / "data"
LAB = ROOT / "labeling"
OUT = ROOT / "analysis"
OUT.mkdir(exist_ok=True)

EMOTE_RE = re.compile(r"\[[^\[\]]{1,12}\]")
URL_RE = re.compile(r"https?://\S+")
BV_RE = re.compile(r"BV[0-9A-Za-z]{10}")
MENTION_RE = re.compile(r"@[^\s@]{1,20}")


# ============================================================
# 1. 载入
# ============================================================
def load():
    recs, seen = [], set()
    for fp in sorted(DATA.glob("*_*.jsonl")):
        for l in fp.open(encoding="utf-8"):
            if not l.strip():
                continue
            d = json.loads(l)
            if d["rpid"] in seen:
                continue
            seen.add(d["rpid"])
            recs.append(d)

    ai_ids, score_map = set(), {}
    with (LAB / "candidates.csv").open(encoding="utf-8-sig", newline="") as f:
        for d in csv.DictReader(f):
            score_map[d["rpid"]] = float(d["score"])
            if d["conf"] in ("high", "mid") and d["label"] != "0":
                ai_ids.add(d["rpid"])
    for r in recs:
        r["is_ai"] = 1 if r["rpid"] in ai_ids else 0
        r["rule_score"] = score_map.get(r["rpid"], 0.0)
    return recs


# ============================================================
# 2. 特征（按语义分组）
# ============================================================
def featurize(r):
    c = r["content"]
    n = max(1, len(c))
    uname = r.get("user") or ""
    f = {}
    # A 符号类
    f["有「」"] = int("「" in c or "」" in c)
    f["有破折号"] = int("——" in c)
    f["有分号"] = int("；" in c)
    f["有冒号"] = int("：" in c or ":" in c)
    f["有Markdown"] = int("**" in c)
    # B 句式类
    f["不是A是B"] = int(bool(re.search(r"不是[^，。；\n]{1,18}[，,]?\s*(?:而)?是[^，。；\n]{1,18}", c)))
    f["…是真的"] = int(bool(re.search(r"(?:是|也是)真的[，。]", c)))
    f["比…更/都"] = int(bool(re.search(r"比[^，。\n]{2,13}(?:更|都)", c)))
    f["指代这条"] = int("这条" in c)
    f["另一回事"] = int(bool(re.search(r"另一(?:回事|栏|种|层)", c)))
    f["本质内核腔"] = int(bool(re.search(r"(?:本质|内核|核心|结构)[^，。]{0,6}(?:是|在于|其实是)", c)))
    # C 人设类
    f["自称本鱼"] = int(bool(re.search(r"本鱼|本鲸|本喵", c)))
    f["助手视角"] = int(bool(re.search(r"我这个型号|派活|停机|让我干|我干活|跑一轮题库|读字幕", c)))
    # D 结构类
    f["字数"] = np.log1p(len(c))
    f["行数"] = c.count("\n") + 1
    f["用字丰富度"] = len(set(c)) / n
    # E 账号类
    f["名字含鱼"] = int(bool(re.search(r"鱼|鲸", uname)))
    f["名字长度"] = len(uname)
    # F 标点类
    f["逗号数"] = c.count("，")
    f["句号数"] = c.count("。")
    f["感叹号数"] = c.count("！")
    f["问号数"] = c.count("？")
    f["顿号数"] = c.count("、")
    # G 其他
    f["有表情"] = int(bool(EMOTE_RE.search(c)))
    f["有@"] = int(bool(MENTION_RE.search(c)))
    f["有BV号"] = int(bool(BV_RE.search(c)))
    f["有链接"] = int(bool(URL_RE.search(c)))
    return f


GROUPS = {
    "A 符号类": ["有「」", "有破折号", "有分号", "有冒号", "有Markdown"],
    "B 句式类": ["不是A是B", "…是真的", "比…更/都", "指代这条", "另一回事", "本质内核腔"],
    "C 人设类": ["自称本鱼", "助手视角"],
    "D 结构类": ["字数", "行数", "用字丰富度"],
    "E 账号类": ["名字含鱼", "名字长度"],
    "F 标点类": ["逗号数", "句号数", "感叹号数", "问号数", "顿号数"],
    "G 其他": ["有表情", "有@", "有BV号", "有链接"],
}


# ============================================================
# 3. 评估工具
# ============================================================
def repeated_eval(model, X, y, seed=42, repeats=10):
    """重复分层 CV，返回各指标的均值±标准差"""
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    metrics = {"precision": [], "recall": [], "f1": [], "auc": []}
    for tr, te in rskf.split(X, y):
        m = model.__class__(**model.get_params())
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        prob = m.predict_proba(X[te])[:, 1]
        tp = int(((pred == 1) & (y[te] == 1)).sum())
        fp = int(((pred == 1) & (y[te] == 0)).sum())
        fn = int(((pred == 0) & (y[te] == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        metrics["precision"].append(p)
        metrics["recall"].append(r)
        metrics["f1"].append(f1)
        if len(np.unique(y[te])) > 1:
            metrics["auc"].append(auc(*roc_curve(y[te], prob)[:2]))
    return {k: (np.mean(v), np.std(v)) for k, v in metrics.items() if v}


def bootstrap_f1(y, pred, n=2000, seed=42):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, size=len(y), replace=True)
        ys, ps = y[s], pred[s]
        tp = int(((ps == 1) & (ys == 1)).sum())
        fp = int(((ps == 1) & (ys == 0)).sum())
        fn = int(((ps == 0) & (ys == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        vals.append(2 * p * r / (p + r) if p + r else 0.0)
    return np.percentile(vals, [2.5, 97.5])


# ============================================================
# 4. 主流程
# ============================================================
def main():
    recs = load()
    y = np.array([r["is_ai"] for r in recs])
    print(f"语料 {len(recs)}  正样本 {y.sum()}  负样本 {(y == 0).sum()}\n")

    feats = [featurize(r) for r in recs]
    all_names = list(feats[0].keys())
    X_all = np.array([[f[k] for k in all_names] for f in feats], dtype=float)
    rule = np.array([r["rule_score"] for r in recs], dtype=float)

    lines = ["# 实验 v2：分类（不依赖新数据）", ""]
    lines.append(f"- 语料 {len(recs)} 条，正样本 {y.sum()} 条（silver label），负样本 {(y == 0).sum()} 条")
    lines.append("- 评估：5 折 × 10 次重复分层交叉验证")
    lines.append("")
    lines.append("> 注意：正样本仅 31 条，所有指标都有较宽的置信区间；标签来自规则打分器。")

    # ---------- 4.1 累积消融 ----------
    print("=" * 62)
    print("累积消融（特征组逐步加入）")
    print("=" * 62)
    lines.append("\n## 一、累积消融\n")
    lines.append("| 特征组 | 维度 | 精确率 | 召回率 | F1 | AUC |")
    lines.append("|---|---|---|---|---|---|")

    order = ["A 符号类", "B 句式类", "C 人设类", "D 结构类", "E 账号类", "F 标点类", "G 其他"]
    cum, prev_f1, rows = [], None, []
    lr = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
    for g in order:
        cum += GROUPS[g]
        cols = [all_names.index(k) for k in cum]
        Xg = StandardScaler().fit_transform(X_all[:, cols])
        m = repeated_eval(lr, Xg, y)
        f1, f1s = m["f1"]
        delta = "" if prev_f1 is None else f" ({f1-prev_f1:+.3f})"
        rows.append((g, len(cum), m))
        lines.append(f"| +{g} | {len(cum)} | {m['precision'][0]:.3f} | "
                     f"{m['recall'][0]:.3f} | {f1:.3f}±{f1s:.3f}{delta} | "
                     f"{m['auc'][0]:.3f} |")
        print(f"  +{g:<10} dim={len(cum):>2}  F1={f1:.3f}±{f1s:.3f}{delta}")
        prev_f1 = f1

    # ---------- 4.2 关键子集对比 ----------
    print("\n" + "=" * 62)
    print("关键子集 × 三个模型")
    print("=" * 62)
    SUBSETS = {
        "仅符号 A": GROUPS["A 符号类"],
        "符号 A + 句式 B": GROUPS["A 符号类"] + GROUPS["B 句式类"],
        "符号 A + 人设 C": GROUPS["A 符号类"] + GROUPS["C 人设类"],
        "A + B + C": GROUPS["A 符号类"] + GROUPS["B 句式类"] + GROUPS["C 人设类"],
        "全部 7 组": all_names,
        "仅结构 D（含长度）": GROUPS["D 结构类"],
        "仅账号 E": GROUPS["E 账号类"],
    }
    MODELS = {
        "逻辑回归": LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"),
        "随机森林": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                            random_state=42),
        "决策树": DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42),
    }

    lines.append("\n## 二、关键子集 × 模型\n")
    lines.append("| 子集 | 模型 | 精确率 | 召回率 | F1 | AUC |")
    lines.append("|---|---|---|---|---|---|")
    best = (None, None, -1)
    for sname, cols_names in SUBSETS.items():
        cols = [all_names.index(k) for k in cols_names]
        Xg = StandardScaler().fit_transform(X_all[:, cols])
        for mname, model in MODELS.items():
            m = repeated_eval(model, Xg, y)
            f1 = m["f1"][0]
            lines.append(f"| {sname} | {mname} | {m['precision'][0]:.3f} | "
                         f"{m['recall'][0]:.3f} | {f1:.3f}±{m['f1'][1]:.3f} | "
                         f"{m['auc'][0]:.3f} |")
            print(f"  {sname:<18} {mname:<6} F1={f1:.3f}")
            if f1 > best[2]:
                best = (sname, mname, f1, cols)

    # ---------- 4.3 与规则基线对比 ----------
    print("\n" + "=" * 62)
    print("规则打分器基线")
    print("=" * 62)
    lines.append("\n## 三、与规则打分器对比\n")
    prec_r, rec_r, thr_r = precision_recall_curve(y, rule)
    f1_r = 2 * prec_r * rec_r / (prec_r + rec_r + 1e-12)
    bi = int(np.argmax(f1_r[:-1]))
    rule_f1 = float(f1_r[bi])

    sname, mname, best_f1, best_cols = best
    Xb = StandardScaler().fit_transform(X_all[:, best_cols])
    # 重复 CV 下 cross_val_predict 不可用（样本会多次进测试集），手动累加概率
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    prob_sum, cnt = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in rskf.split(Xb, y):
        m = MODELS[mname].__class__(**MODELS[mname].get_params())
        m.fit(Xb[tr], y[tr])
        prob_sum[te] += m.predict_proba(Xb[te])[:, 1]
        cnt[te] += 1
    prob = prob_sum / np.maximum(cnt, 1)
    prec_m, rec_m, thr_m = precision_recall_curve(y, prob)
    f1_m = 2 * prec_m * rec_m / (prec_m + rec_m + 1e-12)
    mi = int(np.argmax(f1_m[:-1]))

    lines.append("| 方法 | 最佳F1 | 对应阈值 | 精确率 | 召回率 | 平均精确率(AP) |")
    lines.append("|---|---|---|---|---|---|")
    lines.append(f"| 规则打分器 | {rule_f1:.3f} | {thr_r[bi]:.1f} | "
                 f"{prec_r[bi]:.3f} | {rec_r[bi]:.3f} | {average_precision_score(y, rule):.3f} |")
    lines.append(f"| {mname}（{sname}） | {f1_m[mi]:.3f} | {thr_m[mi]:.3f} | "
                 f"{prec_m[mi]:.3f} | {rec_m[mi]:.3f} | {average_precision_score(y, prob):.3f} |")
    print(f"  规则基线 最佳F1={rule_f1:.3f}  AP={average_precision_score(y, rule):.3f}")
    print(f"  最佳模型 最佳F1={f1_m[mi]:.3f}  AP={average_precision_score(y, prob):.3f}")

    # ---------- 4.4 阈值扫描 ----------
    lines.append("\n## 四、阈值扫描（最佳模型）\n")
    lines.append("| 阈值 | 精确率 | 召回率 | F1 | 预测为AI数 |")
    lines.append("|---|---|---|---|---|")
    for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pred = (prob >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * p * r / (p + r) if p + r else 0
        lines.append(f"| {t} | {p:.3f} | {r:.3f} | {f1:.3f} | {int(pred.sum())} |")

    # ---------- 4.5 Bootstrap CI ----------
    pred_best = (prob >= thr_m[mi]).astype(int)
    lo, hi = bootstrap_f1(y, pred_best)
    lines.append(f"\n**Bootstrap 95% 置信区间**（最佳阈值 {thr_m[mi]:.2f}）："
                 f"F1 = {f1_m[mi]:.3f}，CI = [{lo:.3f}, {hi:.3f}]")
    print(f"  Bootstrap CI: [{lo:.3f}, {hi:.3f}]")

    # ---------- 4.6 图 ----------
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec_m, prec_m, label=f"{mname}（AP={average_precision_score(y, prob):.3f}）", lw=2)
    ax.plot(rec_r, prec_r, label=f"规则打分器（AP={average_precision_score(y, rule):.3f}）",
            lw=2, ls="--")
    ax.set_xlabel("召回率 Recall"); ax.set_ylabel("精确率 Precision")
    ax.set_title("PR 曲线")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_pr_curve.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for nm, sc in ((mname, prob), ("规则打分器", rule / max(1, rule.max()))):
        fpr, tpr, _ = roc_curve(y, sc)
        ax.plot(fpr, tpr, label=f"{nm}（AUC={auc(fpr, tpr):.3f}）", lw=2)
    ax.plot([0, 1], [0, 1], ls=":", c="gray")
    ax.set_xlabel("假正率 FPR"); ax.set_ylabel("真正率 TPR")
    ax.set_title("ROC 曲线"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_roc_curve.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    gs = [r[0] for r in rows]
    f1s = [r[2]["f1"][0] for r in rows]
    errs = [r[2]["f1"][1] for r in rows]
    ax.errorbar(range(len(gs)), f1s, yerr=errs, marker="o", capsize=4, lw=2, color="#4a7ab5")
    ax.set_xticks(range(len(gs)))
    ax.set_xticklabels([g.split()[0] for g in gs])
    ax.set_xlabel("累积加入的特征组")
    ax.set_ylabel("F1（5折×10次）")
    ax.set_title("累积消融：每组加入后的变化")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_ablation.png", dpi=130); plt.close(fig)

    cm = confusion_matrix(y, pred_best)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["真人", "AI"]); ax.set_yticklabels(["真人", "AI"])
    ax.set_xlabel("预测"); ax.set_ylabel("实际")
    ax.set_title(f"混淆矩阵（{mname} @ {thr_m[mi]:.2f}）")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(OUT / "fig_confusion_v2.png", dpi=130); plt.close(fig)

    # ---------- 4.7 结论 ----------
    lines.append("\n## 五、结论\n")
    lines.append(f"1. **符号类特征（5 个）单独就有 {rows[0][2]['f1'][0]:.3f} 的 F1**，"
                 "是整组特征里性价比最高的。")
    lines.append(f"2. **结构类（含长度）单独 F1 明显更低**，证实长度不是有效判据。")
    lines.append(f"3. 最佳组合：**{sname} + {mname}**，F1 = {f1_m[mi]:.3f}，"
                 f"AP = {average_precision_score(y, prob):.3f}。")
    lines.append(f"4. 与规则打分器对比：规则基线 F1 = {rule_f1:.3f}，"
                 f"学习模型 F1 = {f1_m[mi]:.3f}。")
    lines.append("\n### 局限")
    lines.append("- 正样本 31 条，F1 的 95% 置信区间很宽（见上）")
    lines.append("- 仅 2 个视频，话题偏置")
    lines.append("")
    lines.append("### 注意：必须交代的循环性（写进论文的局限性）")
    lines.append("")
    lines.append("**本文的 AI 标签是由规则打分器生成的，而规则打分器用的正是"
                 "「符号类 + 句式类 + 人设类 + 账号类」这些特征。**")
    lines.append("")
    lines.append("因此上面 0.8–0.9 的 F1 衡量的是「**模型能否复现规则的决策边界**」，"
                 "**不是**「模型能否识别真实 AI 评论」。")
    lines.append("")
    lines.append("这是标签循环（label leakage）——特征 → 标签 → 用同一批特征训练模型。")
    lines.append("")
    lines.append("**要得到有效的性能数字，必须换取独立的标签来源**，二选一：")
    lines.append("")
    lines.append("1. 人工标注 100–200 条（不看规则分数，盲标）")
    lines.append("2. 用已知 bot 账号发的评论作为 ground truth")
    lines.append("")
    lines.append("在此之前，本文可报告的**有效结论**只有：")
    lines.append(f"- 特征可分性对比（如：仅长度 F1≈0.33–0.39，远低于符号类）")
    lines.append(f"- 规则基线本身的表现（F1 = {rule_f1:.3f}）")
    lines.append("- 各特征组对规则边界的贡献度")

    (OUT / "v2_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告：{OUT / 'v2_report.md'}")


if __name__ == "__main__":
    main()
