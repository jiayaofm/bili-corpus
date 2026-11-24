# -*- coding: utf-8 -*-
"""用鲸鱼数据把打分器的正样本从 3 个视频扩到 46 个视频。

## 为什么要这么做

现有 63 条 AI 正样本来自 3 个视频，263 条标注总共 6 个视频。
留一视频交叉验证下，九个模型全挤在 AP 0.53~0.63——瓶颈是数据不是模型。

但鲸鱼数据里，同一个人的评论覆盞 46 个视频：
    whale_20261121.jsonl   自己 221 条 / 真人 1020 条 / 50 视频
    whale_v4_20261122.jsonl 自己 159 条 / 真人 1954 条 / 46 视频

## 一个必须说清的事实

这 63 条正样本来自 40 多个不同账号，但都是同一个人的小号
（鲸汣本汣、冒泡的鲨鱼的大肥鱼、喵八大肥鱼…）。所以这个打分器
本质上是「识别这个大肥鱼人设」，不是「识别任意 AI」。
文档和前端文案都得写清楚，别让使用者以为它能抓所有 AI。

## 做法

正样本 = 鲸鱼数据里大肥鱼自己的评论 + 现有 63 条标注
负样本 = 鲸鱼数据里的真人评论 + 现有 200 条标注
评测 = 按视频分组，留一视频交叉验证
"""
import csv
import glob
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import pipeline.features as F  # noqa: E402

OUT = ROOT / "state" / "scorer_whale.npz"
MIN_LEN, MAX_LEN = 8, 400


def load_whale():
    """返回 [(text, label, bvid, source)]"""
    items = []
    for fp in sorted(glob.glob(str(ROOT / "data" / "whale_comments" / "*.jsonl"))):
        tag = Path(fp).stem
        for line in open(fp, encoding="utf-8"):
            r = json.loads(line)
            if r.get("skipped"):
                continue
            bv = r.get("bvid") or "?"
            for c in (r.get("comments") or []):
                t = (c if isinstance(c, str) else c.get("text") or "").strip()
                if MIN_LEN <= len(t) <= MAX_LEN:
                    items.append((t, 1, bv, tag))
            for c in (r.get("real_comments") or []):
                t = (c if isinstance(c, str) else c.get("text") or "").strip()
                if MIN_LEN <= len(t) <= MAX_LEN:
                    items.append((t, 0, bv, tag))
    return items


def load_labels():
    corp = {}
    for line in open(ROOT / "data" / "corpus.jsonl", encoding="utf-8"):
        r = json.loads(line)
        corp[str(r.get("rpid") or "")] = r.get("bvid", "")
    out = []
    for r in csv.DictReader(open(ROOT / "data" / "ground_truth_merged.csv",
                                 encoding="utf-8-sig", newline="")):
        t = (r.get("content") or "").strip()
        if not (MIN_LEN <= len(t) <= MAX_LEN):
            continue
        out.append((t, 1 if r["label"] == "1" else 0,
                    corp.get(str(r["rpid"]), "?"), "manual"))
    return out


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


def lgvo(X, y, g, C=0.03):
    pred = np.zeros(len(y))
    folds = []
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        clf = LogisticRegression(max_iter=5000, C=C, class_weight="balanced")
        clf.fit((X[tr] - mu) / sd, y[tr])
        p = clf.predict_proba((X[te] - mu) / sd)[:, 1]
        pred[te] = p
        if len(set(y[te])) == 2:
            folds.append((len(te), int(y[te].sum()), roc_auc_score(y[te], p)))
        else:
            folds.append((len(te), int(y[te].sum()), float("nan")))
    return pred, folds


def main():
    whale = load_whale()
    manual = load_labels()
    print(f"  鲸鱼数据 {len(whale)} 条（正 {sum(1 for x in whale if x[1] == 1)}）")
    print(f"  人工标注 {len(manual)} 条（正 {sum(1 for x in manual if x[1] == 1)}）")

    # 负样本下采样，别让 2000+ 真人压过 400 条正样本
    rng = np.random.default_rng(0)
    pos = [x for x in whale if x[1] == 1]
    neg = [x for x in whale if x[1] == 0]
    keep_neg = min(len(neg), len(pos) * 3)
    idx = rng.choice(len(neg), keep_neg, replace=False)
    data = pos + [neg[i] for i in idx] + manual
    print(f"  合并 {len(data)} 条（正 {sum(1 for x in data if x[1] == 1)}"
          f" / 负 {sum(1 for x in data if x[1] == 0)}）")

    texts = [x[0] for x in data]
    y = np.array([x[1] for x in data])
    g = np.array([x[2] for x in data])
    src = np.array([x[3] for x in data])
    print(f"  视频 {len(set(g))} 个，其中含正样本 {len(set(g[y == 1]))} 个")
    print()

    print("  计算手工特征 …", flush=True)
    cols = list(F.featurize(texts[0], "").keys())
    Xh = np.array([[F.featurize(t, "")[c] for c in cols] for t in texts], float)
    print("  计算句向量 …", flush=True)
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    mdl = AutoModel.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float16,
                                    device_map="auto").eval()
    E = embed(texts, tok, mdl)
    X = np.hstack([Xh, E])

    np.savez(OUT, Xh=Xh, E=E, y=y, g=g, src=src, cols=np.array(cols))
    print(f"  特征已存 → {OUT}")
    print()

    print("  【全量：留一视频交叉验证】")
    pred, folds = lgvo(X, y, g)
    ap, auc = average_precision_score(y, pred), roc_auc_score(y, pred)
    print(f"    AP={ap:.3f}  AUC={auc:.3f}")
    print()
    print("    对照：只用人工标注（3 个正视频）  AP=0.627  AUC=0.850")
    print()

    # 关键检验：能不能迁移到「没见过的视频」
    print("  【只看人工标注那 6 个视频上的表现】")
    m = np.isin(src, ["manual"])
    if len(set(y[m])) == 2:
        ap2 = average_precision_score(y[m], pred[m])
        auc2 = roc_auc_score(y[m], pred[m])
        print(f"    这 6 个视频上：AP={ap2:.3f}  AUC={auc2:.3f}"
              f"（样本 {m.sum()}，正 {y[m].sum()}）")
    print()
    print("  按来源分组的 AUC：")
    for s in sorted(set(src)):
        mm = src == s
        if len(set(y[mm])) == 2:
            print(f"    {s:<18} n={mm.sum():<5} AUC={roc_auc_score(y[mm], pred[mm]):.3f}")
    print()
    print("  每折明细（前 12）：")
    vids = sorted(set(g), key=lambda v: -sum(1 for x in g if x == v))
    for v, (nn, npos, a) in list(zip(vids, folds))[:12]:
        print(f"    {v[:20]:<22} {nn:>4} 条  正 {npos:>3}  "
              f"{'AUC=%.3f' % a if a == a else 'AUC=—'}")


if __name__ == "__main__":
    main()
