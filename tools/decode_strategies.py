# -*- coding: utf-8 -*-
"""对比四种推理策略，看哪个提升更大。

A  基线：单条，temp 0.95
B  逗号抑制（轻）：解码时压低「，」的 logit
C  逗号抑制（重）
D  候选重排：N 条挑最像人话的
E  重排 + 逗号抑制（轻）

为什么盯逗号：模型输出逗号密度 7.02，真人 5.05，多用了 39%，
结果分句被切碎（平均 10.4 字 vs 真人 13.6）。这不是语法问题，
是标点问题——解码期压一下就行，不用重训。
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.stdout.reconfigure(encoding="utf-8")

from syntax_stats import human_score, stats  # noqa: E402

ADAPTER = r"D:\dsh-ai\lora\out\fish-a2"
SYS = "你把 AI 写的 B站 评论改成像真人网友写的。"
TMPL = "这条评论：{x}\n\n改一下。"
CJK = re.compile(r"[^\u4e00-\u9fff0-9A-Za-z]")


def bg(t):
    t = CJK.sub("", t)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def content_recall(src, out):
    A, B = bg(src), bg(out)
    return 1.0 if not A else len(A & B) / len(A)


def rerank_score(src, cand):
    return 0.6 * human_score(cand) + 0.4 * content_recall(src, cand)


class Demote(LogitsProcessor):
    """压低指定 token 的 logit。"""

    def __init__(self, ids, bias):
        self.ids = ids
        self.bias = bias

    def __call__(self, input_ids, scores):
        for i in self.ids:
            scores[:, i] -= self.bias
        return scores


dev = [json.loads(l) for l in open(ROOT / "data" / "rewrite" / "a_dev.jsonl",
                                   encoding="utf-8")]
xs = [r["_meta"]["x"] for r in dev if r["_meta"]["x"]]
ys = [r["_meta"]["y"] for r in dev if r["_meta"]["x"]]

print(f"  dev {len(xs)} 条", flush=True)
print("  加载模型 …", flush=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float16,
                                            device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER).eval()
tok.padding_side = "left"

COMMA_IDS = sorted(set(
    tok.encode("，", add_special_tokens=False) +
    tok.encode("、", add_special_tokens=False) +
    tok.encode("；", add_special_tokens=False)
))
print(f"  逗号类 token id: {COMMA_IDS}", flush=True)


def gen(texts, n_per=1, comma_bias=0.0):
    prompts = []
    for t in texts:
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": TMPL.format(x=t)}]
        p = tok.apply_chat_template(msgs, tokenize=False,
                                    add_generation_prompt=True,
                                    enable_thinking=False)
        prompts.extend([p] * n_per)
    procs = [Demote(COMMA_IDS, comma_bias)] if comma_bias else None
    outs = []
    B = 16
    for i in range(0, len(prompts), B):
        chunk = prompts[i:i + B]
        enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
        mn = min(200, max(64, int(max(len(p) for p in chunk) * 1.4) + 40))
        with torch.inference_mode():
            o = model.generate(**enc, max_new_tokens=mn, do_sample=True,
                               temperature=0.95, top_p=0.95,
                               logits_processor=procs,
                               pad_token_id=tok.eos_token_id)
        g = o[:, enc["input_ids"].shape[1]:]
        outs.extend(s.strip().replace("\n", " ") for s in
                    tok.batch_decode(g, skip_special_tokens=True))
    return [outs[i * n_per:(i + 1) * n_per] for i in range(len(texts))]


def report(tag, texts):
    syn = st.mean(human_score(t) for t in texts)
    rec = st.mean(content_recall(x, t) for x, t in zip(xs, texts))
    comb = st.mean(rerank_score(x, t) for x, t in zip(xs, texts))
    ss = [stats(t) for t in texts]
    comma = st.mean(s["逗号密度"] for s in ss if s)
    clen = st.mean(s["平均分句长"] for s in ss if s)
    period = st.mean(s["句号密度"] for s in ss if s)
    print("  %-22s 语法 %.3f  内容 %.3f  综合 %.3f | 逗号 %.2f  分句长 %.2f  句号 %.2f"
          % (tag, syn, rec, comb, comma, clen, period))
    return comb


print()
print("  数据里真人的基准：", end="")
ss = [stats(t) for t in ys]
print("语法 %.3f  逗号 %.2f  分句长 %.2f  句号 %.2f"
      % (st.mean(human_score(t) for t in ys),
         st.mean(s["逗号密度"] for s in ss if s),
         st.mean(s["平均分句长"] for s in ss if s),
         st.mean(s["句号密度"] for s in ss if s)))
print()

results = {}


def flat(cands):
    """gen(n_per=1) 返回 [[候选]]，摊平成一维。"""
    return [c[0] for c in cands]


print("  A 基线（2 次采样取平均）…", flush=True)
a = flat(gen(xs, 1, 0.0)) + flat(gen(xs, 1, 0.0))
results["A"] = report("A 基线", a)

print("  B 逗号抑制 2.0 …", flush=True)
b = flat(gen(xs, 1, 2.0)) + flat(gen(xs, 1, 2.0))
results["B"] = report("B 抑制 2.0", b)

print("  C 逗号抑制 4.0 …", flush=True)
c = flat(gen(xs, 1, 4.0)) + flat(gen(xs, 1, 4.0))
results["C"] = report("C 抑制 4.0", c)

print("  D 重排 4 选 1 …", flush=True)
d = gen(xs, 4, 0.0)
d_pick = [max(cands, key=lambda s: rerank_score(x, s)) for x, cands in zip(xs, d)]
results["D"] = report("D 重排 4 选 1", d_pick)

print("  E 重排 + 抑制 2.0 …", flush=True)
e = gen(xs, 4, 2.0)
e_pick = [max(cands, key=lambda s: rerank_score(x, s)) for x, cands in zip(xs, e)]
results["E"] = report("E 重排 + 抑制 2.0", e_pick)

print()
best = max(results, key=results.get)
print(f"  最佳：{best}（综合 {results[best]:.3f}）")
print()
print("  相对基线的提升：")
base_v = results["A"]
for k in ("B", "C", "D", "E"):
    print("    %s  %+.3f  (%.1f%%)" % (k, results[k] - base_v,
                                       (results[k] / base_v - 1) * 100))
