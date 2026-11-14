# -*- coding: utf-8 -*-
"""
实验 v4b：用「风格特征」而不是「文字内容」做聚类

对照 v4（TF-IDF 文字聚类）：
  如果风格特征聚类的 AMI 更高，说明文字聚类主要学的是话题，
  风格信号被话题淹没了。

风格特征 = 标点密度、表情率、长度、句式标记等——都不含话题词。
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent
DATA = ROOT / "data"

EMOTE_RE = re.compile(r"\[[^\[\]]{1,12}\]")
URL_RE = re.compile(r"https?://\S+")
BV_RE = re.compile(r"BV[0-9A-Za-z]{10}")
MENTION_RE = re.compile(r"@[^\s@]{1,20}")

TID_NAME = {"216": "鬼畜", "95": "娱乐", "76": "美食",
            "201": "美食·厨艺", "231": "科技", "173": "游戏"}


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
            d["domain"] = TID_NAME.get(str(d.get("video_tid", "")), "其他")
            recs.append(d)
    return recs


def style_vec(c):
    n = max(1, len(c))
    return [
        np.log1p(len(c)),
        c.count("，") / n * 100,
        c.count("。") / n * 100,
        c.count("！") / n * 100,
        c.count("？") / n * 100,
        c.count("、") / n * 100,
        c.count("~") / n * 100,
        c.count("\n"),
        len(set(c)) / n,
        int(bool(EMOTE_RE.search(c))),
        len(EMOTE_RE.findall(c)),
        int("「" in c or "」" in c),
        int("——" in c),
        int("；" in c),
        int(bool(MENTION_RE.search(c))),
        int(bool(URL_RE.search(c))),
        int(bool(BV_RE.search(c))),
        int(bool(re.search(r"[哈草笑]{2,}", c))),
    ]


recs = load()
doms = [r["domain"] for r in recs]
X = StandardScaler().fit_transform(np.array([style_vec(r["content"]) for r in recs]))

print(f"语料 {len(recs)}  风格维度 {X.shape[1]}\n")
print(f"{'K':>3}{'silhouette':>13}{'AMI(风格)':>12}")
print("-" * 30)
best = None
for k in range(3, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    lab = km.fit_predict(X)
    sil = silhouette_score(X, lab, sample_size=min(500, len(X)), random_state=42)
    ami = adjusted_mutual_info_score(doms, lab)
    print(f"{k:>3}{sil:>13.4f}{ami:>12.4f}")
    if best is None or ami > best[2]:
        best = (k, sil, ami, lab)

bk, bsil, bami, blab = best
print(f"\n最佳（按AMI）K={bk}  silhouette={bsil:.4f}  AMI={bami:.4f}")

table = defaultdict(Counter)
for r, c in zip(recs, blab):
    table[f"簇{c}"][r["domain"]] += 1

all_doms = [d for d, _ in Counter(doms).most_common()]
print(f"\n{'簇':<5}{'总数':>6}  " + "  ".join(f"{d:<8}" for d in all_doms))
print("-" * (20 + 10 * len(all_doms)))
for cid in sorted(table, key=lambda x: -sum(table[x].values())):
    cnt = table[cid]
    tot = sum(cnt.values())
    cells = "  ".join(f"{cnt.get(d,0):<8}" for d in all_doms)
    print(f"{cid:<5}{tot:>6}  {cells}")

purity = sum(max(table[c].values()) for c in table) / len(recs)
print(f"\n簇纯净度：{purity*100:.1f}%")
