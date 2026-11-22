# -*- coding: utf-8 -*-
"""验证：模型是「不会改」，还是「被喂了不属于它的输入」。

用户反馈「很多基本没变化」。但之前 dev 上的样本明明改得很实在。
怀疑是域外输入——训练用的 x 是大肥鱼的改写（中位 41 字），
而扩展里喂进去的是真人原话（中位可能 60 字以上）。

这个脚本把两类输入分别跑一遍，量「改动幅度」和「内容保留」。
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

ADAPTER = sys.argv[1] if len(sys.argv) > 1 else r"D:\dsh-ai\lora\out\fish-a2"
CJK = re.compile(r"[^\u4e00-\u9fff0-9A-Za-z]")


def bg(t):
    t = CJK.sub("", t)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def jac(a, b):
    A, B = bg(a), bg(b)
    return len(A & B) / len(A | B) if (A | B) else 1.0


def terms(t):
    """内容词：数字、字母词、汉字 bigram。"""
    t2 = CJK.sub(" ", t)
    out = set(re.findall(r"\d+|[A-Za-z]{2,}", t2))
    han = re.sub(r"[^\u4e00-\u9fff]", "", t)
    out |= {han[i:i + 2] for i in range(len(han) - 1)}
    return out


def recall(src, out):
    T = terms(src)
    if not T:
        return 1.0
    O = terms(out)
    return len(T & O) / len(T)


dev = [json.loads(l) for l in open(ROOT / "data" / "rewrite" / "a_dev.jsonl", encoding="utf-8")]
in_dom = [r["_meta"]["x"] for r in dev[:5]]

out_dom = [
    "插个队，顺便我来帮你写",
    "哇塞，你会回复所有人的评论么",
    "唐大厨评价菜品是按价格来的，比如沙县一份炒饭11块，能抄出11块的价格他就会评价不错。如果是111块的价格炒11的菜那他就要开喷了",
    "陪玩模式最值钱的不是解说水平。我住在这台电脑底边，主人打游戏的时候我就在旁边坐着，不会算伤害也不会报技能，最多在他被怪追着跑的时候笑一声",
]

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYS = "你把 AI 写的 B站 评论改成像真人网友写的。"
TMPL = "这条评论：{x}\n\n改一下。"

print(f"  加载 {ADAPTER} …", flush=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float16,
                                            device_map="auto")
model = PeftModel.from_pretrained(base, ADAPTER).eval()


def gen(text):
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": TMPL.format(x=text)}]
    p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    ids = tok(p, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**ids, max_new_tokens=96, do_sample=True,
                             temperature=0.9, top_p=0.95,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip().replace("\n", " ")


for tag, items in (("训练域内（鱼腔输入）", in_dom), ("训练域外（真人输入）", out_dom)):
    print()
    print("=" * 66)
    print(f"  {tag}   共 {len(items)} 条")
    print("=" * 66)
    chg, rec = [], []
    for t in items:
        o = gen(t)
        d = 1 - jac(t, o)
        r = recall(t, o)
        chg.append(d)
        rec.append(r)
        print(f"  输入({len(t)}字) {t[:52]}")
        print(f"  输出({len(o)}字) {o[:52]}")
        print(f"  改动 {d*100:3.0f}%   内容保留 {r*100:3.0f}%")
        print()
    print(f"  ▶ 平均改动 {np.mean(chg)*100:.0f}%   平均内容保留 {np.mean(rec)*100:.0f}%")
