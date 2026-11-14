# -*- coding: utf-8 -*-
"""
实验 v1：聚类 + 分类

数据：368 条 B站评论（2 个 AI 主题视频）
标签：high+mid 视为 AI（银标签），其余视为真人

诚实声明：
  · 银标签来自规则打分器，不是 ground truth
  · 只有 2 个视频，聚类结果会被话题污染
  · AI 正样本仅 31 条，交叉验证波动会很大

输出：analysis/v1_report.md + 4 张图
"""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
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
# 1. 载入与标注
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

    ai_ids, conf_map = set(), {}
    with (LAB / "candidates.csv").open(encoding="utf-8-sig", newline="") as f:
        for d in csv.DictReader(f):
            conf_map[d["rpid"]] = d["conf"]
            if d["conf"] in ("high", "mid") and d["label"] != "0":
                ai_ids.add(d["rpid"])

    human_confirmed = {x["rpid"] for x in
                       json.loads((LAB / "human_marked.json").read_text(encoding="utf-8"))}

    for r in recs:
        r["is_ai"] = 1 if r["rpid"] in ai_ids else 0
        r["conf"] = conf_map.get(r["rpid"], "low")
        r["human_confirmed"] = 1 if r["rpid"] in human_confirmed else 0
    return recs


# ============================================================
# 2. 手工特征
# ============================================================
FEATURE_NAMES = [
    "字数", "逗号数", "句号数", "感叹号数", "问号数", "顿号数",
    "有「」", "有破折号", "有分号", "有冒号", "有表情", "有Markdown",
    "自称本鱼", "有@", "有BV号", "有链接", "行数", "用字丰富度",
    "名字含鱼", "名字长度", "名字含数字",
]


def featurize(r):
    c = r["content"]
    n = max(1, len(c))
    uname = r.get("user") or ""
    return [
        np.log1p(len(c)),
        c.count("，"),
        c.count("。"),
        c.count("！"),
        c.count("？"),
        c.count("、"),
        int("「" in c or "」" in c),
        int("——" in c),
        int("；" in c),
        int("：" in c or ":" in c),
        int(bool(EMOTE_RE.search(c))),
        int("**" in c),
        int(bool(re.search(r"本鱼|本鲸|本喵", c))),
        int(bool(MENTION_RE.search(c))),
        int(bool(BV_RE.search(c))),
        int(bool(URL_RE.search(c))),
        c.count("\n") + 1,
        len(set(c)) / n,
        int(bool(re.search(r"鱼|鲸", uname))),
        len(uname),
        int(bool(re.search(r"\d", uname))),
    ]


# ============================================================
# 3. 聚类
# ============================================================
def run_clustering(recs, lines):
    texts = []
    for r in recs:
        t = EMOTE_RE.sub("", r["content"])
        t = URL_RE.sub("", t)
        t = BV_RE.sub("", t)
        t = MENTION_RE.sub("", t)
        t = re.sub(r"\s+", "", t)
        texts.append(t)

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                          min_df=2, max_features=4000)
    X = vec.fit_transform(texts)
    print(f"TF-IDF 矩阵：{X.shape}")

    ks, sils = list(range(2, 9)), []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lab = km.fit_predict(X)
        s = silhouette_score(X, lab, sample_size=min(300, X.shape[0]), random_state=42)
        sils.append(s)
        print(f"  K={k}  silhouette={s:.4f}")

    best_k = ks[int(np.argmax(sils))]
    print(f"最佳 K = {best_k}（silhouette={max(sils):.4f}）")

    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # 各簇的 AI 比例与代表词
    terms = np.array(vec.get_feature_names_out())
    lines.append(f"\n## 聚类结果（K={best_k}，TF-IDF char 2-4gram）\n")
    lines.append(f"轮廓系数：{max(sils):.4f}\n")
    lines.append("| 簇 | 样本数 | AI数 | AI占比 | 主要来源视频 | 代表词 |")
    lines.append("|---|---|---|---|---|---|")

    ai_cluster_hit = []
    for cid in range(best_k):
        idx = np.where(labels == cid)[0]
        sub = [recs[i] for i in idx]
        n_ai = sum(r["is_ai"] for r in sub)
        vid = Counter(r["video_title"][:12] for r in sub).most_common(1)[0]
        centroid = km.cluster_centers_[cid]
        top = terms[np.argsort(centroid)[-8:]][::-1]
        ai_cluster_hit.append(n_ai)
        lines.append(f"| {cid} | {len(sub)} | {n_ai} | {n_ai/len(sub)*100:.0f}% | "
                     f"{vid[0]} ({vid[1]}) | {'、'.join(top[:6])} |")

    # PCA 图
    pca = PCA(n_components=2, random_state=42)
    P = pca.fit_transform(X.toarray())
    fig, ax = plt.subplots(figsize=(8, 6))
    for cid in range(best_k):
        m = labels == cid
        ax.scatter(P[m, 0], P[m, 1], s=18, alpha=0.6, label=f"簇{cid}")
    ai_m = np.array([r["is_ai"] for r in recs]) == 1
    ax.scatter(P[ai_m, 0], P[ai_m, 1], s=70, facecolors="none",
               edgecolors="red", linewidths=1.4, label="判定为AI")
    ax.set_title(f"评论聚类 PCA 投影（K={best_k}）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_clusters.png", dpi=130)
    plt.close(fig)

    # 轮廓曲线
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, sils, "o-")
    ax.axvline(best_k, ls="--", c="r", alpha=0.6)
    ax.set_xlabel("K"); ax.set_ylabel("silhouette")
    ax.set_title("聚类数选择")
    fig.tight_layout()
    fig.savefig(OUT / "fig_silhouette.png", dpi=130)
    plt.close(fig)

    # AI 是否集中 —— 必须结合 silhouette 判断，否则簇大小会造成假象
    total_ai = sum(r["is_ai"] for r in recs)
    max_conc = max(ai_cluster_hit) / max(1, total_ai)
    lines.append(f"\nAI 评论在单一簇中的最大集中度：**{max_conc*100:.0f}%**"
                 f"（{max(ai_cluster_hit)}/{total_ai}）")

    best_sil = max(sils)
    if best_sil < 0.10:
        lines.append(f"\n**结论：聚类失败。** 轮廓系数 {best_sil:.4f} 远低于 0.1，"
                     "说明样本本身没有可分的簇结构。")
        lines.append("\n该集中度是**簇大小造成的假象**——AI 评论掉进了最大的那个"
                     "「通用文本」簇，不是被风格聚拢。")
        lines.append("\n原因：全部 368 条来自 2 个同题材视频，"
                     "**话题方差远大于风格方差**，TF-IDF 只能分出刷屏文本和专名。")
    elif max_conc >= 0.7:
        lines.append("\n→ 轮廓系数可接受且 AI 显著聚集，聚类捕捉到了风格差异。")
    else:
        lines.append("\n→ AI 未显著聚集，聚类主要按话题分化，不是按说话方式。")
    return labels, max_conc


# ============================================================
# 4. 分类
# ============================================================
def run_classification(recs, lines):
    X = np.array([featurize(r) for r in recs], dtype=float)
    y = np.array([r["is_ai"] for r in recs])

    print(f"\n特征矩阵 {X.shape}   正样本 {y.sum()} / 负样本 {(y==0).sum()}")

    Xs = StandardScaler().fit_transform(X)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "逻辑回归": LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"),
        "决策树": DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42),
        "随机森林": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                            random_state=42),
    }

    lines.append("\n## 分类结果（手工特征，5 折分层交叉验证）\n")
    lines.append("| 模型 | 精确率 | 召回率 | F1 | 正类F1 |")
    lines.append("|---|---|---|---|---|")

    best_name, best_f1, best_pred = None, -1, None
    for name, model in models.items():
        pred = cross_val_predict(model, Xs, y, cv=cv)
        rep = classification_report(y, pred, output_dict=True, zero_division=0)
        p, r, f = rep["1"]["precision"], rep["1"]["recall"], rep["1"]["f1-score"]
        lines.append(f"| {name} | {p:.3f} | {r:.3f} | {f:.3f} | {rep['weighted avg']['f1-score']:.3f} |")
        print(f"  {name}: P={p:.3f} R={r:.3f} F1={f:.3f}")
        if f > best_f1:
            best_name, best_f1, best_pred = name, f, pred

    # 混淆矩阵
    cm = confusion_matrix(y, best_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["真人", "AI"]); ax.set_yticklabels(["真人", "AI"])
    ax.set_xlabel("预测"); ax.set_ylabel("实际")
    ax.set_title(f"混淆矩阵（{best_name}）")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(OUT / "fig_confusion.png", dpi=130)
    plt.close(fig)

    # 特征重要性
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=42)
    rf.fit(Xs, y)
    imp = rf.feature_importances_
    order = np.argsort(imp)[::-1]
    lines.append("\n## 特征重要性（随机森林）\n")
    lines.append("| 排名 | 特征 | 重要性 |")
    lines.append("|---|---|---|")
    for i, j in enumerate(order[:12], 1):
        lines.append(f"| {i} | {FEATURE_NAMES[j]} | {imp[j]:.4f} |")

    fig, ax = plt.subplots(figsize=(7, 5))
    top = order[:14][::-1]
    ax.barh([FEATURE_NAMES[i] for i in top], imp[top], color="#4a7ab5")
    ax.set_xlabel("importance")
    ax.set_title("特征重要性 Top 14")
    fig.tight_layout()
    fig.savefig(OUT / "fig_importance.png", dpi=130)
    plt.close(fig)

    # 决策树规则（可解释性）
    dt = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
    dt.fit(Xs, y)
    tree_txt = export_text(dt, feature_names=FEATURE_NAMES, decimals=2)
    lines.append("\n## 决策树规则（深度 3，可直接写进论文）\n")
    lines.append("```")
    lines.append(tree_txt)
    lines.append("```")

    lines.append(f"\n→ 最佳模型：**{best_name}**（正类 F1 = {best_f1:.3f}）")

    # ================= 消融实验 =================
    lines.append("\n## 消融实验（验证模型是否只靠长度/账号名）\n")
    IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
    GROUPS = {
        "全部特征": list(range(len(FEATURE_NAMES))),
        "去掉 长度+账号名": [i for i, n in enumerate(FEATURE_NAMES)
                        if n not in ("字数", "名字含鱼", "名字长度", "名字含数字")],
        "只保留 符号类": [IDX[n] for n in ("有「」", "有破折号", "有分号", "有冒号")],
        "只保留 标点类": [IDX[n] for n in ("逗号数", "句号数", "感叹号数", "问号数", "顿号数")],
        "符号+人设": [IDX[n] for n in ("有「」", "有破折号", "有分号", "有冒号",
                                    "自称本鱼", "有Markdown")],
        "只保留 账号名": [IDX[n] for n in ("名字含鱼", "名字长度", "名字含数字")],
        "只保留 长度": [IDX["字数"]],
    }

    lines.append("| 特征子集 | 精确率 | 召回率 | 正类F1 |")
    lines.append("|---|---|---|---|")
    ablation = {}
    for gname, cols in GROUPS.items():
        Xg = StandardScaler().fit_transform(X[:, cols])
        pred = cross_val_predict(RandomForestClassifier(n_estimators=300,
                                                        class_weight="balanced",
                                                        random_state=42),
                                 Xg, y, cv=cv)
        rep = classification_report(y, pred, output_dict=True, zero_division=0)
        f1 = rep["1"]["f1-score"]
        ablation[gname] = f1
        lines.append(f"| {gname} | {rep['1']['precision']:.3f} | {rep['1']['recall']:.3f} | {f1:.3f} |")
        print(f"  [消融] {gname}: F1={f1:.3f}")

    drop = ablation["全部特征"] - ablation["去掉 长度+账号名"]
    lines.append("")
    if drop > 0.15:
        lines.append(f"**关键发现**：去掉长度和账号名后，F1 从 "
                     f"{ablation['全部特征']:.3f} 掉到 {ablation['去掉 长度+账号名']:.3f}"
                     f"（降 {drop:.3f}）。")
        lines.append("\n说明模型**主要依赖长度和账号名**，而不是语言风格。"
                     "这与你此前的判断一致：不能靠长度判 AI。")
    else:
        lines.append(f"**关键发现**：去掉长度和账号名后 F1 仅变化 {drop:+.3f}，"
                     "说明风格特征本身也有判别力。")

    lines.append(f"\n单看**符号类特征**（「」破折号分号冒号）F1 = "
                 f"{ablation['只保留 符号类']:.3f}；"
                 f"单看**账号名** F1 = {ablation['只保留 账号名']:.3f}；"
                 f"单看**长度** F1 = {ablation['只保留 长度']:.3f}。")
    return best_f1


def main():
    recs = load()
    print(f"语料 {len(recs)} 条，其中判定为 AI {sum(r['is_ai'] for r in recs)} 条\n")

    lines = ["# 实验 v1 结果", ""]
    lines.append(f"- 语料：{len(recs)} 条 B站评论（2 个视频）")
    lines.append(f"- AI 标签：{sum(r['is_ai'] for r in recs)} 条（规则打分器 silver label）")
    lines.append(f"- 人工复核：{sum(r['human_confirmed'] for r in recs)} 条确认为真人")
    lines.append("")
    lines.append("> 注意：标签来自规则打分器，不是人工 ground truth；仅 2 个视频，结果有话题偏置。")

    print("=" * 60)
    print("一、聚类")
    print("=" * 60)
    run_clustering(recs, lines)

    print("\n" + "=" * 60)
    print("二、分类")
    print("=" * 60)
    run_classification(recs, lines)

    (OUT / "v1_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告：{OUT / 'v1_report.md'}")
    print(f"图：fig_clusters.png / fig_silhouette.png / fig_confusion.png / fig_importance.png")


if __name__ == "__main__":
    main()
