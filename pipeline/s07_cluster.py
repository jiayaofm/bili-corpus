# -*- coding: utf-8 -*-
"""
s07 聚类

两组对照：
  A. 文字聚类（TF-IDF char 2-4gram）—— 看能否恢复分区
  B. 风格聚类（s05 的风格矩阵）     —— 看簇是否紧凑、是否对应分区

指标：silhouette（簇紧凑度）+ AMI（与真实分区的吻合度）
输出：analysis/v5_cluster.md + fig_v5_k.png / fig_v5_pca.png / fig_v5_style.png
"""
import json
import re
from collections import Counter, defaultdict

import numpy as np

import common
from common import CFG, P, banner, info, read_jsonl, record

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

STAGE = "s07_cluster"
MD_REL = "analysis/v5_cluster.md"

EMOTE_RE = re.compile(r"\[[^\[\]]{1,12}\]")
URL_RE = re.compile(r"https?://\S+")
BV_RE = re.compile(r"BV[0-9A-Za-z]{10}")
MENTION_RE = re.compile(r"@[^\s@]{1,20}")


def clean(c):
    t = EMOTE_RE.sub("", c)
    t = URL_RE.sub("", t)
    t = BV_RE.sub("", t)
    t = MENTION_RE.sub("", t)
    return re.sub(r"\s+", "", t)


def sweep(X, doms, kmin, kmax, seed):
    rows = []
    for k in range(kmin, kmax + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        lab = km.fit_predict(X)
        sil = silhouette_score(X, lab, sample_size=min(500, X.shape[0]), random_state=seed)
        ami = adjusted_mutual_info_score(doms, lab)
        rows.append((k, sil, ami, lab, km))
    return rows


def contingency(labels, doms):
    t = defaultdict(Counter)
    for c, d in zip(labels, doms):
        t[f"簇{c}"][d] += 1
    purity = sum(max(t[c].values()) for c in t) / len(doms)
    return t, purity


def main(force=False):
    banner("s07 聚类")

    corpus_path = common.ROOT / "data/corpus.jsonl"
    style_path = P["state"] / "X_style.npy"
    for p, hint in ((corpus_path, "s02"), (style_path, "s05")):
        if not p.exists():
            info(f"[失败] 缺少 {p}，请先跑 {hint}")
            return 1

    recs = read_jsonl(corpus_path)
    doms = [r["domain"] for r in recs]
    info(f"语料 {len(recs)} 条，{len(set(doms))} 个分区")

    kmin = CFG["cluster"]["k_min"]
    kmax = min(CFG["cluster"]["k_max"], max(kmin + 1, len(recs) // 20))
    seed = CFG["cluster"]["seed"]

    lines = ["# 实验 v5：聚类", ""]
    lines.append(f"- 语料 {len(recs)} 条，来自 {len(set(doms))} 个分区")
    lines.append("")
    lines.append("> 说明：当前每个分区只采 1 个视频，"
                 "「分区」与「话题」完全共线，"
                 "AMI 偏高可能来自话题词而非体裁。")

    # ================= A. 文字聚类 =================
    texts = [clean(r["content"]) for r in recs]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                          min_df=2, max_features=6000)
    Xt = vec.fit_transform(texts)
    info(f"TF-IDF {Xt.shape}")

    rows_t = sweep(Xt, doms, kmin, kmax, seed)
    best_t = max(rows_t, key=lambda r: r[2])
    bk, bsil, bami, blab, bkm = best_t

    lines.append("\n## 一、文字聚类（TF-IDF）\n")
    lines.append("| K | silhouette | AMI（与分区吻合） |")
    lines.append("|---|---|---|")
    for k, sil, ami, _, _ in rows_t:
        mark = " ←" if k == bk else ""
        lines.append(f"| {k} | {sil:.4f} | {ami:.4f}{mark} |")
    lines.append(f"\n按 AMI 选 K = {bk}（silhouette={bsil:.4f}, AMI={bami:.4f}）")

    t, purity = contingency(blab, doms)
    all_doms = [d for d, _ in Counter(doms).most_common()]
    lines.append("\n### 簇 × 分区\n")
    lines.append("| 簇 | 总数 | " + " | ".join(all_doms) + " | 主导 |")
    lines.append("|---" * (len(all_doms) + 3) + "|")
    for cid in sorted(t, key=lambda x: -sum(t[x].values())):
        cnt = t[cid]
        tot = sum(cnt.values())
        top = cnt.most_common(1)[0]
        cells = " | ".join(str(cnt.get(d, 0)) for d in all_doms)
        lines.append(f"| {cid} | {tot} | {cells} | {top[0]}（{top[1]/tot*100:.0f}%） |")
    lines.append(f"\n**簇纯净度 {purity*100:.1f}%**")

    terms = np.array(vec.get_feature_names_out())
    lines.append("\n### 各簇代表词\n")
    lines.append("| 簇 | 代表词 |")
    lines.append("|---|---|")
    for cid in range(bk):
        cen = bkm.cluster_centers_[cid]
        top = terms[np.argsort(cen)[-8:]][::-1]
        lines.append(f"| 簇{cid} | {'、'.join(top)} |")

    # 图：K 选择
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ks = [r[0] for r in rows_t]
    ax.plot(ks, [r[1] for r in rows_t], "o-", label="silhouette")
    ax2 = ax.twinx()
    ax2.plot(ks, [r[2] for r in rows_t], "s--", color="#d05a3a", label="AMI")
    ax.set_xlabel("K"); ax.set_ylabel("silhouette"); ax2.set_ylabel("AMI")
    ax.set_title("文字聚类：K 选择")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(P["analysis"] / "fig_v5_k.png", dpi=130)
    plt.close(fig)

    # 图：PCA
    Pc = PCA(n_components=2, random_state=seed).fit_transform(Xt.toarray())
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for d in all_doms:
        m = np.array([x == d for x in doms])
        ax.scatter(Pc[m, 0], Pc[m, 1], s=20, alpha=0.6, label=d)
    ax.set_title("文字聚类 PCA 投影（按分区着色）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(P["analysis"] / "fig_v5_pca.png", dpi=130)
    plt.close(fig)

    # ================= B. 风格聚类 =================
    # 原始矩阵里标点是"计数"，会被长度主导。转成"每百字密度"再聚类，
    # 这样簇反映的是表达形态而不是篇幅。
    meta_style = json.loads((P["state"] / "meta_style.json").read_text(encoding="utf-8"))
    scols = meta_style["cols"]
    Xr = np.load(style_path).astype(float)
    li = scols.index("字数")
    base = np.maximum(Xr[:, li], 1.0)
    for name in ("逗号数", "句号数", "感叹号数", "问号数", "顿号数"):
        if name in scols:
            j = scols.index(name)
            Xr[:, j] = Xr[:, j] / base * 100.0
    Xr[:, li] = np.log1p(Xr[:, li])
    Xs = StandardScaler().fit_transform(Xr)
    info("风格特征已转为密度口径（每百字），长度取对数")

    rows_s = sweep(Xs, doms, kmin, min(kmax, 8), seed)

    # 退化解检查：主导簇超过 80% 或存在 <5 条的小簇的 K 不选，
    # 否则 silhouette 会因为"把少数离群点单独切出去"而虚高
    def ok(lab):
        sizes = np.bincount(lab)
        return sizes.min() >= 5 and sizes.max() <= len(lab) * 0.80

    valid_s = [r for r in rows_s if ok(r[3])]
    rejected = [r[0] for r in rows_s if not ok(r[3])]
    fallback = not valid_s
    best_s = (max(valid_s, key=lambda r: r[1]) if valid_s
              else max(rows_s, key=lambda r: r[1]))
    sk, ssil, sami, slab, skm = best_s

    lines.append("\n## 二、风格聚类（标点密度 / 长度 / 形态）\n")
    lines.append("> 标点已转为「每百字密度」，长度取对数，避免篇幅主导。")
    lines.append("> K 上限设为 8：风格空间的 silhouette 随 K 单调上升，"
                 "不设上限会选出无意义的过分割。")
    lines.append("")
    lines.append("| K | silhouette | AMI（与分区吻合） | 备注 |")
    lines.append("|---|---|---|---|")
    for k, sil, ami, lab, _ in rows_s:
        marks = []
        if k == sk:
            marks.append("选中")
        if not ok(lab):
            marks.append("退化解")
        lines.append(f"| {k} | {sil:.4f} | {ami:.4f} | {'，'.join(marks)} |")
    lines.append(f"\n入选 K = {sk}（silhouette={ssil:.4f}, AMI={sami:.4f}）")
    if rejected:
        lines.append(f"\n判定为退化解而排除：K = {', '.join(map(str, rejected))}")
    if fallback:
        lines.append("\n注意：没有合格解，退化为取最大 silhouette。")

    st, spurity = contingency(slab, doms)
    lines.append("\n### 风格簇 × 分区\n")
    lines.append("| 簇 | 总数 | " + " | ".join(all_doms) + " |")
    lines.append("|---" * (len(all_doms) + 2) + "|")
    for cid in sorted(st, key=lambda x: -sum(st[x].values())):
        cnt = st[cid]
        cells = " | ".join(str(cnt.get(d, 0)) for d in all_doms)
        lines.append(f"| {cid} | {sum(cnt.values())} | {cells} |")
    lines.append(f"\n**风格簇纯净度 {spurity*100:.1f}%**")

    # 形态画像
    lens = np.array([len(r["content"]) for r in recs])
    emote = np.array([1 if EMOTE_RE.search(r["content"]) else 0 for r in recs])
    lines.append("\n### 风格簇的形态画像\n")
    lines.append("| 簇 | 条数 | 均长 | 表情率 | 多行率 |")
    lines.append("|---|---|---|---|---|")
    order_s = sorted(set(slab.tolist()), key=lambda c: -int((slab == c).sum()))
    for cid in order_s:
        m = slab == cid
        n = int(m.sum())
        if n == 0:
            continue
        multiline = np.mean([1.0 if "\n" in recs[i]["content"] else 0.0
                             for i in np.where(m)[0]])
        lines.append(f"| 簇{cid} | {n} | {lens[m].mean():.1f} | "
                     f"{emote[m].mean()*100:.0f}% | {multiline*100:.0f}% |")

    # 图：风格 PCA
    Ps = PCA(n_components=2, random_state=seed).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for cid in sorted(set(slab)):
        m = slab == cid
        ax.scatter(Ps[m, 0], Ps[m, 1], s=18, alpha=0.6, label=f"簇{cid}")
    ax.set_title(f"风格聚类 PCA 投影（K={sk}）")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(P["analysis"] / "fig_v5_style.png", dpi=130)
    plt.close(fig)

    # ================= 结论 =================
    lines.append("\n## 三、结论\n")
    lines.append("| 表示 | silhouette | AMI | 纯净度 |")
    lines.append("|---|---|---|---|")
    lines.append(f"| 文字 TF-IDF | {bsil:.4f} | {bami:.4f} | {purity*100:.1f}% |")
    lines.append(f"| 风格特征 | {ssil:.4f} | {sami:.4f} | {spurity*100:.1f}% |")
    lines.append("")
    if ssil > bsil and sami < bami:
        lines.append("**核心发现：风格聚类簇更紧凑（silhouette 更高），"
                     "但与分区的吻合度更低（AMI 更低）。**")
        lines.append("")
        # 按实际画像给簇起名，避免写死
        names = []
        for cid in order_s:
            m = slab == cid
            if m.sum() == 0:
                continue
            names.append(f"簇{cid}（{int(m.sum())} 条，均长 {lens[m].mean():.0f} 字，"
                         f"表情率 {emote[m].mean()*100:.0f}%）")
        lines.append("风格簇的实际形态：" + "；".join(names) + "。")
        lines.append("")
        lines.append("也就是说，风格聚类切出来的是**篇幅/形态**上的分层，"
                     "而不是分区。")
        lines.append("")
        lines.append("结论：**体裁 ≠ 分区，两者近似正交。**")
        lines.append("文字聚类看起来「恢复了分区」，实际靠的是话题词；"
                     "风格聚类反映的是表达形态。")
    else:
        lines.append(f"文字 AMI={bami:.4f}，风格 AMI={sami:.4f}。")

    lines.append("\n### 局限\n")
    lines.append("- 每个分区仅 1 个视频，分区与话题共线，无法分离话题效应")
    lines.append("- 若要严格论证「体裁聚类」，需每分区 2–3 个不同话题的视频")

    (common.ROOT / MD_REL).write_text("\n".join(lines), encoding="utf-8")
    info(f"报告 {MD_REL}")

    record(STAGE, {
        MD_REL: len(lines),
        "analysis/fig_v5_k.png": 1,
        "analysis/fig_v5_pca.png": 1,
        "analysis/fig_v5_style.png": 1,
    }, extra={"text_ami": round(bami, 4), "style_sil": round(ssil, 4),
              "style_ami": round(sami, 4)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
