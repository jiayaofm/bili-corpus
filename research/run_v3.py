# -*- coding: utf-8 -*-
"""
实验 v3：用人工真值重新训练（第一次不带循环性的评估）

标签来源：labeling/ground_truth.csv（151 条人工盲标，x=明确AI / m=疑似 / 无=真人）

要点：
  1. 与规则打分器在同一批样本上正面对比（公平基准）
  2. 在 v2 特征基础上，**新增从漏报样本里归纳的特征**（第一人称角色扮演等）
  3. 报告真实精确率/召回率，不再是复现规则
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
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

ROOT = Path(__file__).parent
LAB = ROOT / "labeling"
OUT = ROOT / "analysis"
OUT.mkdir(exist_ok=True)

EMOTE_RE = re.compile(r"\[[^\[\]]{1,12}\]")
MENTION_RE = re.compile(r"@[^\s@]{1,20}")


def load_gt():
    rows = list(csv.DictReader((LAB / "ground_truth.csv").open(encoding="utf-8-sig", newline="")))
    for r in rows:
        r["label"] = int(r["label"])
        r["was_candidate"] = int(r["was_candidate"])
    return rows


def featurize(r):
    c = r["content"]
    n = max(1, len(c))
    f = {}
    # --- A 符号类 ---
    f["有「」"] = int("「" in c or "」" in c)
    f["有破折号"] = int("——" in c)
    f["有分号"] = int("；" in c)
    f["有冒号"] = int("：" in c or ":" in c)
    f["有Markdown"] = int("**" in c)
    # --- B 句式类 ---
    f["不是A是B"] = int(bool(re.search(r"不是[^，。；\n]{1,18}[，,]?\s*(?:而)?是[^，。；\n]{1,18}", c)))
    f["…是真的"] = int(bool(re.search(r"(?:是|也)真的[，。]", c)))
    f["比…更/都"] = int(bool(re.search(r"比[^，。\n]{2,13}(?:更|都)", c)))
    f["指代这条"] = int("这条" in c)
    f["另一回事"] = int(bool(re.search(r"另一(?:回事|栏|种|层)", c)))
    f["本质内核腔"] = int(bool(re.search(r"(?:本质|内核|核心|结构)[^，。]{0,6}(?:是|在于|其实是)", c)))
    # --- C 人设类 ---
    f["自称本鱼"] = int(bool(re.search(r"本鱼|本鲸|本喵", c)))
    f["助手视角"] = int(bool(re.search(r"我这个型号|派活|停机|让我干|我干活|跑一轮题库|读字幕", c)))
    # --- C2 新增：第一人称角色扮演（从漏报样本归纳）---
    # "鱼知道…""本小姐也是大肥鱼""这不就是我嘛""让鱼再炫一会儿"
    f["鱼作第一人称"] = int(bool(re.search(r"(?<!大)(?<!小)鱼(?:知道|还是|也|就|熟|可|会|不)|让鱼|这只鱼|是真鱼", c)))
    f["角色自称"] = int(bool(re.search(r"本小姐|本座|朕|咱鱼", c)))
    f["自我代入"] = int(bool(re.search(r"这不就是我|就是我嘛|代入|我也是", c)))
    # --- C3 新增：口语判断 ---
    f["挺/真+形容"] = int(bool(re.search(r"(?:挺|真)(?:香|好|猛|肉疼|好笑|乐|顶|伤|难|贵|划算|值)", c)))
    f["倒装对仗"] = int(bool(re.search(r"是[^，。\n]{1,12}，不是[^，。\n]{1,12}", c)))
    # --- C4 新增：反诘对比 ---
    f["反诘对比"] = int(bool(re.search(r"有没有人[^。\n]{0,20}而不是|而不是[^。\n]{0,12}$", c)))
    f["我最在意"] = int(bool(re.search(r"我最在意|我最关心|我好奇", c)))
    # --- C5 新增：分享推荐式 ---
    f["分享推荐"] = int(bool(re.search(r"有点东西|来瞅瞅|特意叫|值得刷|安利", c)) or
                        (len(MENTION_RE.findall(c)) >= 2))
    # --- D 结构类 ---
    f["字数"] = np.log1p(len(c))
    f["行数"] = c.count("\n") + 1
    f["用字丰富度"] = len(set(c)) / n
    # --- E 账号类（已确认是混淆变量，单独测）---
    # 人工真值里没存用户名，这一维跳过
    # --- F 标点类 ---
    f["逗号数"] = c.count("，")
    f["句号数"] = c.count("。")
    f["感叹号数"] = c.count("！")
    f["问号数"] = c.count("？")
    f["顿号数"] = c.count("、")
    # --- G 其他 ---
    f["有表情"] = int(bool(EMOTE_RE.search(c)))
    f["有@"] = int(len(MENTION_RE.findall(c)))
    return f


def cv_scores(model, X, y, n_repeats=10):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=42)
    out = {"precision": [], "recall": [], "f1": []}
    for tr, te in rskf.split(X, y):
        m = model.__class__(**model.get_params())
        m.fit(X[tr], y[tr])
        pred = m.predict(X[te])
        tp = int(((pred == 1) & (y[te] == 1)).sum())
        fp = int(((pred == 1) & (y[te] == 0)).sum())
        fn = int(((pred == 0) & (y[te] == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        out["precision"].append(p); out["recall"].append(r)
        out["f1"].append(2 * p * r / (p + r) if p + r else 0)
    return {k: (np.mean(v), np.std(v)) for k, v in out.items()}


def main():
    rows = load_gt()
    y = np.array([r["label"] for r in rows])
    print(f"真值样本 {len(rows)}  正样本(AI) {y.sum()}  负样本 {(y==0).sum()}\n")

    feats = [featurize(r) for r in rows]
    names = list(feats[0].keys())
    X = np.array([[f[k] for k in names] for f in feats], dtype=float)
    Xs = StandardScaler().fit_transform(X)

    lines = ["# 实验 v3：人工真值重新训练", ""]
    lines.append(f"- 样本：{len(rows)} 条人工盲标（正样本 {y.sum()}，负样本 {(y==0).sum()}）")
    lines.append("- 标签：x=明确AI，m=疑似AI（均计入正类），无前缀=真人")
    lines.append("- 评估：5 折 × 10 次重复分层交叉验证")
    lines.append("")
    lines.append("> 这是本项目**第一次不带循环性**的评估——标签来自人工，不来自规则。")

    MODELS = {
        "逻辑回归": LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced"),
        "随机森林": RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=42),
        "决策树": DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42),
    }
    lines.append("\n## 一、三个模型（全部特征）\n")
    lines.append("| 模型 | 精确率 | 召回率 | F1 |")
    lines.append("|---|---|---|---|")
    best = (None, -1)
    for nm, mdl in MODELS.items():
        s = cv_scores(mdl, Xs, y)
        lines.append(f"| {nm} | {s['precision'][0]:.3f} | {s['recall'][0]:.3f} | "
                     f"{s['f1'][0]:.3f}±{s['f1'][1]:.3f} |")
        print(f"  {nm}: P={s['precision'][0]:.3f} R={s['recall'][0]:.3f} F1={s['f1'][0]:.3f}")
        if s["f1"][0] > best[1]:
            best = (nm, s["f1"][0])

    # 规则基线（同一批样本）
    cand = np.array([r["was_candidate"] for r in rows])
    tp = int(((cand == 1) & (y == 1)).sum())
    fp = int(((cand == 1) & (y == 0)).sum())
    fn = int(((cand == 0) & (y == 1)).sum())
    rp = tp / (tp + fp) if tp + fp else 0
    rr = tp / (tp + fn) if tp + fn else 0
    rf1 = 2 * rp * rr / (rp + rr) if rp + rr else 0
    lines.append("\n## 二、与规则打分器对比（同一批 151 条样本）\n")
    lines.append("| 方法 | 精确率 | 召回率 | F1 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| **规则打分器** | {rp:.3f} | {rr:.3f} | {rf1:.3f} |")
    lines.append(f"| {best[0]} | — | — | {best[1]:.3f} |")
    lines.append("")
    lines.append(f"→ 规则精确率高（{rp:.3f}）但召回率低（{rr:.3f}）："
                 "**能准确认出它认得出的那批，但漏掉大部分**。")
    print(f"\n  规则基线（同批样本）: P={rp:.3f} R={rr:.3f} F1={rf1:.3f}")

    # 特征重要性
    rf = RandomForestClassifier(n_estimators=600, class_weight="balanced", random_state=42)
    rf.fit(Xs, y)
    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1]
    lines.append("\n## 三、特征重要性（随机森林）\n")
    lines.append("| 排名 | 特征 | 重要性 |")
    lines.append("|---|---|---|")
    for i, j in enumerate(order[:15], 1):
        lines.append(f"| {i} | {names[j]} | {imp[j]:.4f} |")

    fig, ax = plt.subplots(figsize=(7, 6))
    top = order[:15][::-1]
    ax.barh([names[i] for i in top], imp[top], color="#4a7ab5")
    ax.set_xlabel("importance")
    ax.set_title("特征重要性（人工真值训练）")
    fig.tight_layout(); fig.savefig(OUT / "fig_v3_importance.png", dpi=130); plt.close(fig)

    # 决策树规则
    dt = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
    dt.fit(Xs, y)
    lines.append("\n## 四、决策树规则\n")
    lines.append("```")
    lines.append(export_text(dt, feature_names=names, decimals=2))
    lines.append("```")

    # PR 曲线
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mdl = MODELS[best[0]].__class__(**MODELS[best[0]].get_params())
    prob = cross_val_predict(mdl, Xs, y, cv=skf, method="predict_proba")[:, 1]
    prec, rec, _ = precision_recall_curve(y, prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, label=f"{best[0]}（AP={average_precision_score(y, prob):.3f}）")
    ax.set_xlabel("召回率"); ax.set_ylabel("精确率")
    ax.set_title("PR 曲线（人工真值）"); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / "fig_v3_pr.png", dpi=130); plt.close(fig)
    lines.append(f"\n## 五、PR 曲线\n")
    lines.append(f"{best[0]} 的平均精确率 AP = {average_precision_score(y, prob):.3f}")

    (OUT / "v3_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告：{OUT / 'v3_report.md'}")


if __name__ == "__main__":
    main()
