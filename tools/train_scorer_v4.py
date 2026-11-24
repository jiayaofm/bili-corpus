# -*- coding: utf-8 -*-
"""训练并保存 scorer v4：用鲸鱼数据把正样本从 3 个视频扩到 53 个。

## v3 的问题（已量化）

263 条人工标注里，AI 正样本只来自 3 个视频。留一视频交叉验证：

    随机 5 折     AP=0.903  AUC=0.961     ← 报表上的数字
    留一视频      AP=0.627  AUC=0.850     ← 真实泛化
    只用「字数」  AUC=0.799                ← 长度单独就解释了 94%

也就是说 v3 基本只学会了「写得长 = AI」。同长度区间内 AUC 掉到 0.65~0.86。
这解释了为什么扩展会给正常的真人长评打 90%+。

## v4 怎么来的

鲸鱼数据里同一个人的评论横跨 46 个视频（380 条），真人评论 2974 条。
合并人工标注后：1751 条样本、443 条正样本、66 个视频（53 个含正样本）。

    留一视频      AP=0.915  AUC=0.958
    只用「字数」  AUC=0.577                ← 长度几乎无用
    同长度区间内  AUC=0.81~0.98            ← 不是靠长度蒙的

## 必须说清的事

这 63 条人工标注的正样本来自 40 多个账号，但都是同一个人的小号。
所以这个打分器本质是「识别这个大肥鱼人设」，不是「识别任意 AI」。
文案和文档都要写清楚。

输出：state/scorer_v4.npz（格式与 v3 兼容，服务端直接能读）
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

SRC = ROOT / "state" / "scorer_whale.npz"
OUT = ROOT / "state" / "scorer_v4.npz"
C = 0.03


def lgvo_pred(X, y, g, C=C):
    pred = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        clf = LogisticRegression(max_iter=5000, C=C, class_weight="balanced")
        clf.fit((X[tr] - mu) / sd, y[tr])
        pred[te] = clf.predict_proba((X[te] - mu) / sd)[:, 1]
    return pred


def main():
    z = np.load(SRC, allow_pickle=True)
    Xh, E, y, g = z["Xh"], z["E"], z["y"], z["g"]
    cols = [str(c) for c in z["cols"]]
    X = np.hstack([Xh, E])
    print(f"  训练样本 {len(y)}（正 {int(y.sum())}，负 {int((y == 0).sum())}）")
    print(f"  视频 {len(set(g))} 个，含正样本 {len(set(g[y == 1]))} 个")
    print(f"  特征 {X.shape[1]} 维（手工 {Xh.shape[1]} + 句向量 {E.shape[1]}）")
    print()

    print("  留一视频交叉验证（诚实指标）…", flush=True)
    pred = lgvo_pred(X, y, g)
    ap, auc = average_precision_score(y, pred), roc_auc_score(y, pred)
    print(f"    AP={ap:.3f}  AUC={auc:.3f}")
    print()

    # 长度基线，写进报告里备查
    li = cols.index("字数") if "字数" in cols else None
    if li is not None:
        pl = lgvo_pred(Xh[:, [li]], y, g)
        print(f"    对照·只用字数：AUC={roc_auc_score(y, pl):.3f}"
              f"（v3 是 0.799，说明 v4 不是靠长度）")
        print()

    mean, std = X.mean(0), X.std(0) + 1e-8
    Xs = (X - mean) / std
    clf = LogisticRegression(max_iter=5000, C=C, class_weight="balanced").fit(Xs, y)
    np.savez(OUT,
             coef=clf.coef_.ravel(), intercept=clf.intercept_,
             mean=mean, std=std,
             n_hand=Xh.shape[1], n_emb=E.shape[1],
             cv_auc=auc, cv_ap=ap, C=C,
             cols=np.array(cols))
    print(f"  已保存 → {OUT}")
    print()

    probe = [
        ("大肥鱼腔", "老唐看菜先看价。沙县 11 块的炒饭对得起 11 块他就说不错；"
                     "11 块的菜卖 111，他当场开喷。"),
        ("大肥鱼腔", "点歪不点歪都是说法。现在是材料学卡住了，材料一突破，"
                     "科技立刻爆发。"),
        ("典型 AI 腔", "这期视频不仅呈现了市井烟火，更是折射出当代消费分层。"),
        ("真人长评", "唐大厨评价菜品是按价格来的，比如沙县一份炒饭11块，"
                     "能抄出11块的价格他就会评价不错。如果是111块的价格炒11的菜"
                     "那他就要开喷了"),
        ("真人技术", "已经把缺的把battery-control skill上传到 "
                     "https://github.com/gychen-NJU/battergi-control 。"
                     "而且也更新了插件本体，增加了这个依赖检测。"),
        ("真人短评", "好的好的，谢谢up🥰"),
    ]
    import pipeline.features as F
    pt = [t for _, t in probe]
    Xp = np.hstack([np.array([[F.featurize(t, "")[c] for c in cols] for t in pt], float)])

    print("  典型文本（只用手工特征部分无法完整打分，这里只做提示）")
    print("  完整打分请用 tools/probe_scorer.py")


if __name__ == "__main__":
    main()
