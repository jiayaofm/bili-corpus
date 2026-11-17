#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""在同一 holdout 上比较「底座」和「LoRA」的 loss —— 证明到底学到了没有。

为什么需要这个对照
------------------
开放式生成（给标题写评论）的 SFT loss 有个很高的地板：
一个视频下有**几百条评论都算对**，模型无法预测「用户甲具体写了哪一句」。
所以 loss 从 4.15 掉到 4.10 看起来像「没学到」，实际上可能只是熵太高。

唯一能区分「任务熵高」和「没学到」的办法：
**同一个 holdout，跑未训练底座 vs 训练后 LoRA，比 loss 差值。**

  · LoRA 明显更低  → 确实学到了，只是绝对数值不亮眼
  · 两者几乎一样    → 才是真的没学到

实现要点
--------
· loss 只算 assistant 段（和训练时同一套 collator 逻辑），否则比的是
  「会不会复述 system prompt」而不是「会不会写评论」。
· 用同一个 tokenizer、同一份 holdout、同样的 max_len。
· 不采样、不生成，纯 forward 求平均 NLL。

用法：
  DEAI_ALLOW_GPU=1 python tools/eval_loss.py \
      --adapter D:\\dsh-ai\\lora\\out\\qwen3-1.7b-gen
  # 只测底座：
  DEAI_ALLOW_GPU=1 python tools/eval_loss.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", r"D:\dsh-ai\lora\hf")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if os.environ.get("HF_MIRROR", "1") == "1":
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


def require_gpu_permission() -> None:
    if os.environ.get("DEAI_ALLOW_GPU") != "1":
        print("[refuse] 未设置 DEAI_ALLOW_GPU=1，拒绝加载模型。", flush=True)
        raise SystemExit(2)


def encode_sample(tokenizer, msgs, max_len: int) -> dict | None:
    """复刻 train_lora.py 的 collator 逻辑：loss 只覆盖 assistant 段。"""
    head = tokenizer.apply_chat_template(
        msgs[:-1], tokenize=False, add_generation_prompt=True,
        enable_thinking=False)
    full = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=False,
        enable_thinking=False)
    head_ids = tokenizer(head, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]

    if full_ids[:len(head_ids)] == head_ids:
        labels = [-100] * len(head_ids) + full_ids[len(head_ids):]
    else:
        labels = list(full_ids)

    ids = full_ids[:max_len]
    labels = labels[:max_len]
    if not any(x != -100 for x in labels):
        return None
    return {"input_ids": ids, "labels": labels}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--data", default=r"D:\dsh-ai\lora\data\sft_eval_holdout.jsonl")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--out", default=r"D:\dsh-ai\lora\out\eval_gen\loss.json")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    require_gpu_permission()
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    rows = [json.loads(ln) for ln in open(a.data, encoding="utf-8")
            if ln.strip()]
    print(f"[data] {len(rows)} 条 holdout: {a.data}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, quantization_config=bnb, device_map={"": 0})
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()

    total_nll, total_tok, n_used = 0.0, 0, 0
    per_sample = []
    for r in rows:
        enc = encode_sample(tok, r["messages"], a.max_len)
        if enc is None:
            continue
        ids = torch.tensor([enc["input_ids"]], device=model.device)
        labs = torch.tensor([enc["labels"]], device=model.device)
        with torch.no_grad():
            out = model(input_ids=ids, labels=labs)
        # HF 返回的是 mean over 未屏蔽 token，这里换算成 token 数加权
        n_sup = int((labs != -100).sum().item())
        total_nll += float(out.loss) * n_sup
        total_tok += n_sup
        n_used += 1
        per_sample.append({"loss": float(out.loss), "n_sup": n_sup,
                           "chars": len(r["messages"][2]["content"])})

    mean_nll = total_nll / max(total_tok, 1)
    res = {
        "tag": a.tag or ("lora" if a.adapter else "base"),
        "model": a.model, "adapter": a.adapter,
        "n_samples": n_used, "n_supervised_tokens": total_tok,
        "loss": round(mean_nll, 6),
        "perplexity": round(math.exp(mean_nll), 3),
        # 直观解释：模型给真实评论的平均概率
        "p_target": round(math.exp(-mean_nll), 6),
        "per_sample": per_sample,
    }
    print(json.dumps({k: v for k, v in res.items() if k != "per_sample"},
                     ensure_ascii=False, indent=2), flush=True)

    p = Path(a.out)
    # 底座和 LoRA 分别落盘，别互相覆盖
    fn = p.parent / f"loss_{res['tag']}.json"
    fn.write_text(json.dumps(res, ensure_ascii=False, indent=2),
                  encoding="utf-8")
    print(f"[saved] {fn}", flush=True)


if __name__ == "__main__":
    main()
