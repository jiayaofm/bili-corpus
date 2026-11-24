# -*- coding: utf-8 -*-
"""真实场景评测：拿大肥鱼自己在 B站 发的评论，看模型到底改了多少。

为什么不用 v4_dev：那个测试集的输入是「大肥鱼改写真人评论」的产物，
短、口语。而用户在扩展里点的是它自己写的长评论，偏书面、带论证。
这是两种分布，dev 上的 0.35 相似度不能代表实际体验。

同时回答一个关键问题：**模型偶尔能改好，还是永远不会？**
如果能，那用「多生成几条挑一条」就能解决；如果不能，就得重训。
"""
import glob
import json
import re
import statistics as st
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.stdout.reconfigure(encoding="utf-8")

from syntax_stats import human_score  # noqa: E402

CJK = re.compile(r"[^\u4e00-\u9fff0-9A-Za-z]")
ADAPTER = r"D:\dsh-ai\lora\out\fish-v4"
SYS = "你把 AI 写的 B站 评论改成像真人网友写的。"
TMPL = "这条评论：{x}\n\n改一下。"
N = 6


def bg(t):
    t = CJK.sub("", t)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def sim(a, b):
    A, B = bg(a), bg(b)
    return len(A & B) / len(A | B) if (A | B) else 1.0


def recall(src, out):
    A, B = bg(src), bg(out)
    return 1.0 if not A else len(A & B) / len(A)


# ---- 收集大肥鱼自己发的评论
texts = []
for fp in glob.glob(str(ROOT / "data" / "whale_comments" / "*.jsonl")):
    for line in open(fp, encoding="utf-8"):
        r = json.loads(line)
        if r.get("skipped"):
            continue
        for c in (r.get("comments") or []):
            t = c if isinstance(c, str) else (c.get("text") or "")
            t = (t or "").strip()
            if len(t) >= 25 and not re.search(r"https?://", t):
                texts.append(t)
texts = list(dict.fromkeys(texts))[:20]
print(f"  真实样本 {len(texts)} 条（大肥鱼自己在 B站 发的评论，长度≥25）")
print(f"  长度中位 {st.median([len(t) for t in texts]):.0f} 字")
print()

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float16,
                                            device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER).eval()
tok.padding_side = "left"
print("  生成中（每条 6 个候选，预计 10 分钟）…", flush=True)

prompts = []
for t in texts:
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": TMPL.format(x=t)}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    prompts.extend([p] * N)

outs = []
B = 12
for i in range(0, len(prompts), B):
    chunk = prompts[i:i + B]
    enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
    mn = min(200, max(80, int(max(len(p) for p in chunk) * 1.2) + 40))
    with torch.inference_mode():
        o = model.generate(**enc, max_new_tokens=mn, do_sample=True,
                           temperature=0.95, top_p=0.95,
                           pad_token_id=tok.eos_token_id)
    g = o[:, enc["input_ids"].shape[1]:]
    outs.extend(s.strip().replace("\n", " ") for s in
                tok.batch_decode(g, skip_special_tokens=True))

cands = [outs[i * N:(i + 1) * N] for i in range(len(texts))]

# 存下来，后面比较不同「选择策略」时不用重新生成
dump = [{"src": t, "cands": cs} for t, cs in zip(texts, cands)]
(ROOT / "state" / "real_cands.json").write_text(
    json.dumps(dump, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"  候选池已存 state/real_cands.json（{len(dump)} 条 x {N} 候选）")


best_sims, med_sims, any_good = [], [], 0
print()
print("  %-3s %6s %8s %8s   %s" % ("#", "字数", "最像的", "最不像", "有合格候选(相似<0.5且保留>0.45)"))
print("  " + "-" * 74)
for t, cs in zip(texts, cands):
    ss = [sim(t, c) for c in cs]
    ok = [c for c in cs if sim(t, c) < 0.5 and recall(t, c) >= 0.45]
    if ok:
        any_good += 1
    best_sims.append(min(ss))
    med_sims.append(st.median(ss))
    print("  %-3d %6d %8.2f %8.2f   %s" % (
        len(best_sims), len(t), st.median(ss), min(ss), "✓" if ok else "✗"))

print()
print("  【结论】")
print("    单条生成时，输出与输入相似度中位（跨样本）%.2f" %
      st.median([st.median([sim(t, c) for c in cs]) for t, cs in zip(texts, cands)]))
print("    6 条里最不像的那条，相似度中位 %.2f" % st.median(best_sims))
print("    有合格候选的样本：%d/%d（%.0f%%）" % (any_good, len(texts),
                                              any_good / len(texts) * 100))
print()
print("  → 如果这个比例高，说明模型「能改好，只是不保证」，用多选一就能解决；")
print("    如果很低，说明得重训。")
