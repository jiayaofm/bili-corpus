# -*- coding: utf-8 -*-
"""
实验 v4：跨领域聚类

问题：换到多领域语料后，无监督聚类能否分出"体裁/分区"，而不是只分话题？

评估：
  · 轮廓系数 silhouette（簇的紧凑度）
  · AMI 调整互信息（聚类结果与真实分区的吻合度，0=随机，1=完全一致）
  · 簇 × 分区 列联表（看每个簇的成分）
  · PCA 可视化（按分区着色）

如果 AMI 明显 > 0，说明聚类学到了分区差异。
"""
import csv
import json
import re
import sys
from collections import Counter, defaultdict
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUT = ROOT / "analysis"
OUT.mkdir(exist_ok=True)

EMOTE_RE = re.compile(r"\[[^\[\]]{1,12}\]")
URL_RE = re.compile(r"https?://\S+")
BV_RE = re.compile(r"BV[0-9A-Za-z]{10}")
MENTION_RE = re.compile(r"@[^\s@]{1,20}")

# 分区码 → 可读名
TID_NAME = {
    "216": "鬼畜", "95": "娱乐", "76": "美食", "201": "美食·厨艺",
    "231": "科技", "173": "游戏",
}


def load():
    recs, seen = [], set()
    for fp in sorted(DATA.glob("*_like.jsonl")):
        for l in fp.open(encoding="utf-8"):
            if not l.strip():
                continue
            d = json.loads(l)
            if d["rpid"] in seen:
                continue
            seen.add(d["rpid"])
            tid = str(d.get("video_tid", ""))
            d["domain"] = TID_NAME.get(tid, f"tid{tid}")
            recs.append(d)
    return recs


def clean(c):
    t = EMOTE_RE.sub("", c)
    t = URL_RE.sub("", t)
    t = BV_RE.sub("", t)
    t = MENTION_RE.sub("", t)
    return re.sub(r"\s+", "", t)


def main():
    recs = load()
    print(f"语料 {len(recs)} 条\n")

    doms = [r["domain"] for r in recs]
    print("分区分布：")
    for d, n in Counter(doms).most_common():
        print(f"  {d:<10} {n} 条")
    print()

    texts = [clean(r["content"]) for r in recs]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                          min_df=2, max_features=6000)
    X = vec.fit_transform(texts)
    print(f"TF-IDF 矩阵：{X.shape}\n")

    kmin, kmax = 3, 12
    rows = []
    for k in range(kmin, kmax + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lab = km.fit_predict(X)
        sil = silhouette_score(X, lab, sample_size=min(500, X.shape[0]), random_state=42)
        ami = adjusted_mutual_info_score(doms, lab)
        rows.append((k, sil, ami, lab))
        print(f"  K={k:<2} silhouette={sil:.4f}  AMI={ami:.4f}")

    # 选 K：用 AMI 最大（本实验关心"能否恢复分区"）
    best = max(rows, key=lambda r: r[2])
    bk, bsil, bami, blab = best
    print(f"\n按 AMI 选 K={bk}（silhouette={bsil:.4f}, AMI={bami:.4f}）")

    # 列联表
    table = defaultdict(Counter)
    for r, c in zip(recs, blab):
        table[f"簇{c}"][r["domain"]] += 1

    lines = ["# 实验 v4：跨领域聚类", ""]
    lines.append(f"- 语料 {len(recs)} 条，来自 {len(set(doms))} 个分区、6 个视频")
    lines.append(f"- 文本表示：TF-IDF char 2–4gram（{X.shape[1]} 维）")
    lines.append("")

    lines.append("## 一、聚类数选择\n")
    lines.append("| K | silhouette | AMI（与真实分区吻合度） |")
    lines.append("|---|---|---|")
    for k, sil, ami, _ in rows:
        mark = " ←" if k == bk else ""
        lines.append(f"| {k} | {sil:.4f} | {ami:.4f}{mark} |")

    lines.append(f"\n选定 K = {bk}")
    lines.append("")

    lines.append("## 二、簇 × 分区 列联表\n")
    all_doms = [d for d, _ in Counter(doms).most_common()]
    header = "| 簇 | 总数 | " + " | ".join(all_doms) + " | 主导分区 |"
    lines.append(header)
    lines.append("|---" * (len(all_doms) + 3) + "|")
    for cid in sorted(table, key=lambda x: -sum(table[x].values())):
        cnt = table[cid]
        tot = sum(cnt.values())
        top = cnt.most_common(1)[0]
        cells = " | ".join(str(cnt.get(d, 0)) for d in all_doms)
        lines.append(f"| {cid} | {tot} | {cells} | "
                     f"{top[0]}（{top[1]/tot*100:.0f}%） |")

    # 簇纯净度
    purity = sum(max(table[c].values()) for c in table) / len(recs)
    lines.append("")
    lines.append(f"**簇纯净度**（每簇主导分区占比的加权均值）：**{purity*100:.1f}%**")

    # 各分区在聚类中是否被打散
    lines.append("\n## 三、各分区的分布分散度\n")
    lines.append("| 分区 | 条数 | 落在几个簇 | 最大单簇占比 |")
    lines.append("|---|---|---|---|")
    for d in all_doms:
        cnts = [table[c].get(d, 0) for c in table]
        nz = [x for x in cnts if x]
        lines.append(f"| {d} | {sum(cnts)} | {len(nz)} | "
                     f"{max(cnts)/sum(cnts)*100:.0f}% |")

    # 每簇代表词
    terms = np.array(vec.get_feature_names_out())
    km_best = KMeans(n_clusters=bk, random_state=42, n_init=10)
    lab2 = km_best.fit_predict(X)
    lines.append("\n## 四、各簇代表词\n")
    lines.append("| 簇 | 代表词 |")
    lines.append("|---|---|")
    for cid in range(bk):
        centroid = km_best.cluster_centers_[cid]
        top = terms[np.argsort(centroid)[-10:]][::-1]
        lines.append(f"| 簇{cid} | {'、'.join(top[:8])} |")

    lines.append("\n## 五、结论\n")
    if bami >= 0.3:
        lines.append(f"AMI = {bami:.3f}，**聚类能较好恢复分区结构**。")
        lines.append("说明多领域语料下，评论的说话方式确实存在可分的体裁差异。")
    elif bami >= 0.15:
        lines.append(f"AMI = {bami:.3f}，**聚类与分区有中等相关**。")
        lines.append("部分分区被识别出来，但仍有混淆。")
    else:
        lines.append(f"AMI = {bami:.3f}，**聚类基本不能恢复分区**。")
        lines.append("即使多领域语料，纯文本 TF-IDF 聚类也没找到体裁结构。")

    if bsil < 0.10:
        lines.append(f"\n注意：轮廓系数仅 {bsil:.4f}，簇内仍然松散，"
                     "聚类结果的绝对质量不高，AMI 的意义是「比随机好多少」。")

    (OUT / "v4_report.md").write_text("\n".join(lines), encoding="utf-8")

    # ---------- 图 ----------
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ks = [r[0] for r in rows]
    ax.plot(ks, [r[1] for r in rows], "o-", label="silhouette")
    ax2 = ax.twinx()
    ax2.plot(ks, [r[2] for r in rows], "s--", color="#d05a3a", label="AMI")
    ax.set_xlabel("K")
    ax.set_ylabel("silhouette")
    ax2.set_ylabel("AMI")
    ax.set_title("聚类数选择")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_v4_k.png", dpi=130)
    plt.close(fig)

    pca = PCA(n_components=2, random_state=42)
    P = pca.fit_transform(X.toarray())
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    markers = {"鬼畜": "o", "娱乐": "s", "美食": "^", "美食·厨艺": "v",
               "科技": "D", "游戏": "*"}
    for d in all_doms:
        m = np.array([x == d for x in doms])
        ax.scatter(P[m, 0], P[m, 1], s=22, alpha=0.6,
                   marker=markers.get(d, "o"), label=d)
    ax.set_title("跨领域评论聚类 PCA 投影（按分区着色）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "fig_v4_pca.png", dpi=130)
    plt.close(fig)

    print(f"\n报告：{OUT / 'v4_report.md'}")


if __name__ == "__main__":
    main()
