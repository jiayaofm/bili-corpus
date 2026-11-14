# -*- coding: utf-8 -*-
"""v3 补充：用人工真值做特征组消融，检验模型是否又靠长度取胜"""
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

import importlib.util
spec = importlib.util.spec_from_file_location("v3", Path(__file__).parent / "run_v3.py")
v3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v3)

rows = v3.load_gt()
y = np.array([r["label"] for r in rows])
feats = [v3.featurize(r) for r in rows]
names = list(feats[0].keys())
X = np.array([[f[k] for k in names] for f in feats], dtype=float)

GROUPS = {
    "只保留 长度结构": ["字数", "行数", "用字丰富度"],
    "只保留 标点": ["逗号数", "句号数", "感叹号数", "问号数", "顿号数"],
    "只保留 符号": ["有「」", "有破折号", "有分号", "有冒号", "有Markdown"],
    "只保留 句式": ["不是A是B", "…是真的", "比…更/都", "指代这条", "另一回事", "本质内核腔"],
    "只保留 人设": ["自称本鱼", "助手视角", "鱼作第一人称", "角色自称", "自我代入"],
    "只保留 口语判断": ["挺/真+形容", "倒装对仗", "反诘对比", "我最在意", "分享推荐"],
    "去掉 长度结构": None,   # 特殊
    "全部": None,
}

drop = set(GROUPS["只保留 长度结构"])


def evaluate(cols):
    Xg = StandardScaler().fit_transform(X[:, cols])
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    P, R, F = [], [], []
    for tr, te in rskf.split(Xg, y):
        m = RandomForestClassifier(n_estimators=150, class_weight="balanced", random_state=42)
        m.fit(Xg[tr], y[tr])
        pred = m.predict(Xg[te])
        tp = int(((pred == 1) & (y[te] == 1)).sum())
        fp = int(((pred == 1) & (y[te] == 0)).sum())
        fn = int(((pred == 0) & (y[te] == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0
        r = tp / (tp + fn) if tp + fn else 0
        P.append(p); R.append(r); F.append(2 * p * r / (p + r) if p + r else 0)
    return np.mean(P), np.mean(R), np.mean(F)


print(f"{'特征子集':<18}{'维度':>6}{'精确率':>9}{'召回率':>9}{'F1':>9}")
print("-" * 52)
for gname, cols_names in GROUPS.items():
    if gname == "去掉 长度结构":
        cols = [i for i, n in enumerate(names) if n not in drop]
    elif gname == "全部":
        cols = list(range(len(names)))
    else:
        cols = [names.index(n) for n in cols_names if n in names]
    p, r, f = evaluate(cols)
    print(f"{gname:<18}{len(cols):>6}{p:>9.3f}{r:>9.3f}{f:>9.3f}")
