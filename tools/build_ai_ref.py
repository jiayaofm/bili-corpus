# -*- coding: utf-8 -*-
"""预先算好「真人评论的 AI 分分布」，供打分时换算百分位。

## 为什么离线算

原来服务启动时现场跑 2000 次 `Scorer.describe()`，每次都要过一遍随机森林
和 joblib 调度，实测把启动拖到 200 秒以上（faulthandler 抓栈确认卡在这里）。
存成文件后启动只要几毫秒。

## 为什么用新分类器算

现在是 scorer_v4（鲸鱼数据扩样本后重训）。百分位必须和实际打分的模型一致，
否则「比 91% 的真人更高」这句话就是错的。
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

OUT = ROOT / "state" / "ai_ref.json"
N = 1500

MODEL = ROOT / "state" / "scorer_v4.npz"
if not MODEL.exists():
    MODEL = ROOT / "state" / "scorer_v3.npz"
print(f"  用 {MODEL.name} 计算参照分布")
z = np.load(MODEL, allow_pickle=True)
coef, b = z["coef"], float(z["intercept"][0])
mean, std, cols = z["mean"], z["std"], [str(c) for c in z["cols"]]

rows = [json.loads(l) for l in open(ROOT / "data" / "corpus.jsonl", encoding="utf-8")]
texts = [(r.get("content") or "").strip()[:400] for r in rows]
texts = [t for t in texts if t and not t.startswith("@")]
random.seed(42)
texts = random.sample(texts, min(N, len(texts)))
print(f"  语料抽样 {len(texts)} 条真人评论")

print("  计算手工特征 …", flush=True)
Xh = np.array([[F.featurize(t, "")[c] for c in cols] for t in texts], float)

print("  计算句向量 …", flush=True)
import torch
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
mdl = AutoModel.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float16,
                                device_map="auto").eval()
emb = []
with torch.inference_mode():
    for i in range(0, len(texts), 24):
        enc = tok(texts[i:i + 24], return_tensors="pt", padding=True,
                  truncation=True, max_length=400).to(mdl.device)
        h = mdl(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1)
        emb.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
E = np.vstack(emb)

zlogit = ((np.hstack([Xh, E]) - mean) / std) @ coef + b
p = 1.0 / (1.0 + np.exp(-zlogit))

OUT.write_text(json.dumps({
    "_note": "真人评论的 AI 分（scorer_v3）排序后分布，用于换算百分位",
    "n": len(p),
    "values": sorted(float(v) for v in p),
}, ensure_ascii=False), encoding="utf-8")

print(f"  已保存 → {OUT}")
print("  分布：P50=%.3f P90=%.3f P95=%.3f P99=%.3f"
      % tuple(np.percentile(p, [50, 90, 95, 99])))
