# -*- coding: utf-8 -*-
"""量新分类器（scorer_v3）在真实语料上的误伤率。

浏览器里看到的是「随机评论流」，和标注集的 200 条真人不是一回事。
所以必须单独在全库 9169 条上量。
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import pipeline.features as F  # noqa: E402

MODEL = ROOT / "state" / "scorer_v3.npz"
z = np.load(MODEL, allow_pickle=True)
coef, intercept = z["coef"], float(z["intercept"][0])
mean, std = z["mean"], z["std"]
n_hand = int(z["n_hand"])
cols = list(z["cols"])

rows = [json.loads(l) for l in open(ROOT / "data" / "corpus.jsonl", encoding="utf-8")]
texts = [(r.get("content") or "").strip()[:400] for r in rows]
texts = [t for t in texts if t and not t.startswith("@")]
random.seed(42)
texts = random.sample(texts, 1500)
print(f"  全库真人评论抽样 {len(texts)} 条")

import torch
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
mdl = AutoModel.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float16,
                                device_map="auto").eval()

Xh = np.array([[F.featurize(t, "")[c] for c in cols] for t in texts], float)
emb = []
with torch.inference_mode():
    for i in range(0, len(texts), 24):
        enc = tok(texts[i:i + 24], return_tensors="pt", padding=True,
                  truncation=True, max_length=256).to(mdl.device)
        h = mdl(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1)
        emb.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
E = np.vstack(emb)
X = np.hstack([Xh, E])
p = 1.0 / (1.0 + np.exp(-(((X - mean) / std) @ coef + intercept)))

print()
print("  新分类器在真人评论上的误伤率：")
for t in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9):
    print(f"    阈值 {t:.1f} → {(p >= t).mean()*100:5.1f}% 被误抓")

print()
print("  分位数：P50=%.3f  P90=%.3f  P95=%.3f  P99=%.3f"
      % tuple(np.percentile(p, [50, 90, 95, 99])))

print()
print("  被误抓最高的 5 条（看看是什么类型的真人评论）：")
idx = np.argsort(-p)[:5]
for i in idx:
    print(f"    {p[i]*100:5.1f}%  {texts[i][:60]}")
