# -*- coding: utf-8 -*-
"""候选重排实验：生成 N 条，用「语法像不像人话」挑最好的那条。

## 为什么这么干

dev 上测出来模型已经学会了句号（6.64→1.54，真人 1.46），
但仍卡在分句长度（10.4 vs 真人 13.6）和逗号过多（7.0 vs 5.1）。

这些是**可以打分的**。所以不用重训，先生成多条候选，
再用「语法接近真人 + 内容保留」挑一条——把模型偶然写好的那条捞出来。

## 打分

score = 0.6 × 语法像人话(0~1) + 0.4 × 内容保留(0~1)

内容保留用输入 x 的 bigram 覆盖率，防止挑到跑题的漂亮句子。
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.stdout.reconfigure(encoding="utf-8")

from syntax_stats import human_score  # noqa: E402

ADAPTER = r"D:\dsh-ai\lora\out\fish-a2"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4
CJK = re.compile(r"[^\u4e00-\u9fff0-9A-Za-z]")

SYS = "你把 AI 写的 B站 评论改成像真人网友写的。"
TMPL = "这条评论：{x}\n\n改一下。"


def bg(t):
    t = CJK.sub("", t)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def content_recall(src, out):
    """输入里的字面内容被保留了多少。"""
    A, B = bg(src), bg(out)
    if not A:
        return 1.0
    return len(A & B) / len(A)


def rerank_score(src, cand):
    return 0.6 * human_score(cand) + 0.4 * content_recall(src, cand)


dev = [json.loads(l) for l in open(ROOT / "data" / "rewrite" / "a_dev.jsonl",
                                   encoding="utf-8")]
xs = [r["_meta"]["x"] for r in dev if r["_meta"]["x"]]
ys = [r["_meta"]["y"] for r in dev if r["_meta"]["x"]]
print(f"  dev {len(xs)} 条，每条约生成 {N} 个候选")

print("  加载模型 …", flush=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float16,
                                            device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER).eval()
tok.padding_side = "left"


def gen_batch(texts, n_per):
    """对每条输入生成 n_per 个候选，返回 [[cand,...], ...]。"""
    prompts = []
    for t in texts:
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": TMPL.format(x=t)}]
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
        prompts.extend([p] * n_per)
    outs = []
    B = 16
    for i in range(0, len(prompts), B):
        enc = tok(prompts[i:i + B], return_tensors="pt", padding=True).to(model.device)
        mn = min(200, max(64, int(max(len(p) for p in prompts[i:i + B]) * 1.2) + 40))
        with torch.inference_mode():
            o = model.generate(**enc, max_new_tokens=mn, do_sample=True,
                               temperature=0.95, top_p=0.95,
                               pad_token_id=tok.eos_token_id)
        g = o[:, enc["input_ids"].shape[1]:]
        outs.extend(s.strip().replace("\n", " ") for s in
                    tok.batch_decode(g, skip_special_tokens=True))
    return [outs[i * n_per:(i + 1) * n_per] for i in range(len(texts))]


print("  生成中 …", flush=True)
cands = gen_batch(xs, N)

first = [c[0] for c in cands]
picked = [max(c, key=lambda s: rerank_score(x, s)) for x, c in zip(xs, cands)]


def report(tag, texts):
    syn = st.mean(human_score(t) for t in texts)
    rec = st.mean(content_recall(x, t) for x, t in zip(xs, texts))
    comb = st.mean(rerank_score(x, t) for x, t in zip(xs, texts))
    print("  %-14s 语法像人话 %.3f   内容保留 %.3f   综合 %.3f"
          % (tag, syn, rec, comb))
    return syn, rec, comb


print()
print("  结果：")
report("单条（基线）", first)
report(f"{N} 选 1（重排）", picked)
report("真人原文", ys)

oracle = st.mean(max(human_score(s) for s in c) for c in cands)
print("  %-14s 语法像人话 %.3f   ← 候选池的理论上限" % ("上限(oracle)", oracle))

print()
print("  重排实际换掉了几条：%d / %d" % (sum(1 for a, b in zip(first, picked) if a != b), len(xs)))
print()
print("  例子（前 3 条）：")
for i in range(min(3, len(xs))):
    print(f"    输入  {xs[i][:44]}")
    print(f"    基线  {first[i][:44]}   语法人话 {human_score(first[i]):.2f}")
    print(f"    重排  {picked[i][:44]}   语法人话 {human_score(picked[i]):.2f}")
    print(f"    真人  {ys[i][:44]}")
    print()
