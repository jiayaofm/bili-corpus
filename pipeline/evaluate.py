# -*- coding: utf-8 -*-
"""
加权评估工具

为什么需要加权：
  盲标采用分层抽样 —— 候选层（规则判为 AI 的）全取，随机层只抽一部分。
  因此样本里 AI 占 23.6%，而全库真实只有约 7.9%。
  在这个偏斜样本上直接算召回率会**明显虚高**。

加权还原：
  候选层每条样本权重 1（全取，代表自己）
  随机层每条样本权重 pool_rest / random_count（当前 963/120 ≈ 8.03）

  加权后，TP/FP/FN 代表的是「全库量级」的计数，
  精确率与召回率回到总体口径。

注意：
  · 精确率 = TP/(TP+FP)，条件于「预测为正」，
    抽样不改变正预测内部的构成，所以精确率通常不受影响；
  · 召回率 = TP/(TP+FN)，分母是全库所有 AI，
    样本里的 AI 被过采样了，所以未加权召回率会虚高；
  · 因此加权主要修正召回率（以及由此连带的 F1）。
"""
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------- 权重
def load_blind_map(path):
    """读取盲标抽样参数。缺失时返回 None，调用方退化为不加权。"""
    import json
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def stratum_weights(was_candidate, blind_map):
    """按分层抽样比给每条样本赋权。

    was_candidate: 0/1 序列，1 表示该样本属于「规则候选层」
    blind_map:     labeling/blind_map.json 的内容
    """
    n = len(was_candidate)
    if not blind_map:
        return np.ones(n, dtype=float)

    pool = blind_map.get("pool_rest")
    n_rand = blind_map.get("random_count") or 0
    if not pool or not n_rand:
        return np.ones(n, dtype=float)

    w_rand = float(pool) / float(n_rand)
    return np.array([1.0 if int(c) == 1 else w_rand for c in was_candidate],
                    dtype=float)


# ---------------------------------------------------------------- 指标
def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def sample_metrics(y_true, pred):
    """未加权指标（样本内口径）"""
    y_true = np.asarray(y_true)
    pred = np.asarray(pred)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    p, r, f1 = _prf(tp, fp, fn)
    return {"precision": p, "recall": r, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def weighted_metrics(y_true, pred, weights):
    """加权指标（总体口径）"""
    y_true = np.asarray(y_true)
    pred = np.asarray(pred)
    w = np.asarray(weights, dtype=float)
    tp = float(w[(pred == 1) & (y_true == 1)].sum())
    fp = float(w[(pred == 1) & (y_true == 0)].sum())
    fn = float(w[(pred == 0) & (y_true == 1)].sum())
    p, r, f1 = _prf(tp, fp, fn)
    return {"precision": p, "recall": r, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


def effective_n(weights):
    """加权后的等效样本量，用于自检（应接近全库条数）"""
    return float(np.asarray(weights, dtype=float).sum())
