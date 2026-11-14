# -*- coding: utf-8 -*-
"""
s05 特征矩阵

输入：labeling/ground_truth.csv（人工真值）+ data/corpus.jsonl（语料）
输出：
  state/X_classify.npy   分类特征矩阵（按 config 的组，默认不含账号类）
  state/y_classify.npy   标签
  state/meta_classify.json  rpid 列表 / 列名 / 分组
  state/X_style.npy      聚类用风格矩阵（整个语料）
  state/meta_style.json
"""
import csv
import json

import numpy as np

import common
import features as F
from common import P, banner, info, read_jsonl, record

STAGE = "s05_features"


def _read_gt(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main(force=False):
    banner("s05 特征矩阵")

    corpus_path = common.ROOT / "data/corpus.jsonl"
    gt_path = P["labeling"] / "ground_truth.csv"
    for p, hint in ((corpus_path, "s02"), (gt_path, "s04")):
        if not p.exists():
            info(f"[失败] 缺少 {p}，请先跑 {hint}")
            return 1

    corpus = read_jsonl(corpus_path)
    by_rpid = {r["rpid"]: r for r in corpus}
    gt = _read_gt(gt_path)
    info(f"语料 {len(corpus)} 条，真值 {len(gt)} 条")

    # ---- 分类矩阵：只取有真值的样本 ----
    groups = F.CLASSIFY_GROUPS
    cols = F.cols_of(groups)
    X, y, rpids, was_cand = [], [], [], []
    missing = 0
    for r in gt:
        rpid = r["rpid"]
        rec = by_rpid.get(rpid)
        if rec is None:
            # 真值里的样本不在当前语料（旧批次），用真值自带的 content 兜底
            if not r.get("content"):
                missing += 1
                continue
            rec = {"content": r["content"], "user": ""}
        X.append(F.vector(rec, cols))
        y.append(int(r["label"]))
        rpids.append(rpid)
        was_cand.append(int(r.get("was_candidate") or 0))

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=int)
    info(f"分类矩阵 {X.shape}   正样本 {int(y.sum())} / 负样本 {int((y == 0).sum())}")
    if missing:
        info(f"注意：{missing} 条真值无法匹配语料且无 content，已跳过")

    np.save(P["state"] / "X_classify.npy", X)
    np.save(P["state"] / "y_classify.npy", y)
    meta = {
        "rpids": rpids,
        "cols": cols,
        "groups": groups,
        "group_of": {c: g for g, names in F.GROUPS.items() for c in names},
        "was_candidate": was_cand,
        "n_pos": int(y.sum()),
        "n_neg": int((y == 0).sum()),
    }
    (P["state"] / "meta_classify.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 风格矩阵：整个语料（聚类用）----
    scols = F.STYLE_COLS
    Xs = np.array([F.vector(r, scols) for r in corpus], dtype=float)
    info(f"风格矩阵 {Xs.shape}")
    np.save(P["state"] / "X_style.npy", Xs)
    (P["state"] / "meta_style.json").write_text(
        json.dumps({"cols": scols, "rpids": [r["rpid"] for r in corpus]},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    record(STAGE, {
        "state/X_classify.npy": int(X.shape[0]),
        "state/y_classify.npy": int(y.shape[0]),
        "state/meta_classify.json": len(cols),
        "state/X_style.npy": int(Xs.shape[0]),
        "state/meta_style.json": len(scols),
    }, extra={"n_pos": int(y.sum()), "n_neg": int((y == 0).sum())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
