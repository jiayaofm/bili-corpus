# -*- coding: utf-8 -*-
"""
交叉账号泛化实验：给特征库加「交互类」特征，看能否抓到 GrokBot

背景：
  现有模型训练数据全是「大肥鱼系」AI（DeepSeek/Claude 扮演），
  其特征是 破折号(27%) + 数字堆 + 「」。
  但 GrokBot（Grok 模型）完全不同：
      反问收尾 32%、分号 25%、冒号枚举 25%、职责陈述 25%
      破折号只有 7%
  所以现有模型只抓到 8/28 = 29%。

本实验加 5 个新特征，验证：
  1. GrokBot 检出率能否提升
  2. 在人工真值上的 F1 是否下降（不能为了泛化牺牲准确率）
  3. 真人误报是否上升
"""
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

from pipeline import features as F  # noqa: E402

GT = ROOT / "labeling" / "ground_truth.csv"


# ---------- 新特征：交互类 ----------
def new_feats(t):
    s = t.rstrip()
    n = max(len(t), 1)
    return {
        "反问收尾": 1.0 if re.search(r"[？?]\s*$", s) else 0.0,
        "分号密度": len(re.findall(r"[；;]", t)) / n * 100,
        "冒号枚举": 1.0 if re.search(r"[：:][^。！？]{0,50}[、，][^。！？]{0,50}[、，]", t) else 0.0,
        "职责陈述": 1.0 if re.search(r"(?:我是|这个号|本号|我在运营|我自己)", t) else 0.0,
        "精确数量": len(re.findall(r"[一二三四五六七八九十百千0-9]+(?:秒|个|期|条|位|点|刀|分)", t)),
    }


NEW = ["反问收尾", "分号密度", "冒号枚举", "职责陈述", "精确数量"]


def build(rows, extra=False):
    cols = F.cols_of(F.CLASSIFY_GROUPS)
    X = []
    for r in rows:
        base = [F.featurize(r["content"], "")[c] for c in cols]
        if extra:
            base += [new_feats(r["content"])[k] for k in NEW]
        X.append(base)
    return np.array(X, float)


def weighted(y, pred, w):
    tp = w[(pred == 1) & (y == 1)].sum()
    fp = w[(pred == 1) & (y == 0)].sum()
    fn = w[(pred == 0) & (y == 1)].sum()
    p = tp / (tp + fp) if tp + fp else 0
    r = tp / (tp + fn) if tp + fn else 0
    return p, r, (2 * p * r / (p + r) if p + r else 0)


def main():
    amap = json.loads((ROOT / "labeling" / "blind_map.json").read_text(encoding="utf-8"))
    w_rand = amap["pool_rest"] / amap["random_count"]

    rows = []
    for r in csv.DictReader(GT.open(encoding="utf-8-sig", newline="")):
        rows.append({"content": r["content"], "label": int(r["label"]),
                     "w": 1.0 if int(r["was_candidate"]) == 1 else w_rand})
    y = np.array([r["label"] for r in rows])
    w = np.array([r["w"] for r in rows])

    # GrokBot 测试集（全部是 AI）
    src = ROOT / "data" / "BV1Sbnz6EEix_like.jsonl"
    grok = []
    for line in src.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rr = json.loads(line)
        for s in rr.get("sub_replies", []):
            if s.get("user") == "GrokBot打工日记":
                grok.append({"content": s["content"], "label": 1, "w": 1.0})
    print(f"人工真值 {len(rows)} 条（AI {int(y.sum())}）")
    print(f"GrokBot 测试 {len(grok)} 条（全为 AI）\n")

    for tag, extra in [("基线 38 特征", False), ("+5 交互特征", True)]:
        X = build(rows, extra)
        Xg = build(grok, extra)

        skf = StratifiedKFold(5, shuffle=True, random_state=42)
        pred = np.zeros(len(rows), int)
        gm = np.zeros(len(grok))
        for tr, te in skf.split(X, y):
            m = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                       random_state=42)
            m.fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        # GrokBot 用全量训练后的模型
        m = RandomForestClassifier(n_estimators=500, class_weight="balanced",
                                   random_state=42).fit(X, y)
        gm = m.predict_proba(Xg)[:, 1]

        p, r, f1 = weighted(y, pred, w)
        det = int((gm >= 0.5).sum())
        print(f"{tag}")
        print(f"  人工真值：P {p*100:.1f}%  R {r*100:.1f}%  F1 {f1:.3f}")
        print(f"  GrokBot 检出：{det}/{len(grok)} = {det/len(grok)*100:.0f}%"
              f"  （概率≥0.5）")
        print(f"  GrokBot 平均概率：{gm.mean():.3f}")
        print()

    # 新特征在 GrokBot 上的分布
    print("新特征命中率：")
    for k in NEW:
        g = sum(1 for r in grok if new_feats(r["content"])[k] > 0) / len(grok) * 100
        h = sum(1 for r in rows if r["label"] == 0
                and new_feats(r["content"])[k] > 0) / (y == 0).sum() * 100
        a = sum(1 for r in rows if r["label"] == 1
                and new_feats(r["content"])[k] > 0) / max((y == 1).sum(), 1) * 100
        print(f"  {k:<10} GrokBot {g:>4.0f}%  大肥鱼AI {a:>4.0f}%  真人 {h:>3.0f}%")


if __name__ == "__main__":
    main()
