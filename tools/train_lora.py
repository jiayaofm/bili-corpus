#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Qwen3-1.7B 的 QLoRA 训练（去除 AI 味 / 学 B 站评论语域）。

设计说明
--------
· 4bit nf4 + double quant 加载底座，LoRA 只训 attention 四个投影层。
· 用 transformers.Trainer + 自写 collator，不用 TRL 的 SFTTrainer。
  原因：TRL 的 SFTTrainer 在 0.17~0.3 之间 API 反复变（dataset_text_field /
  max_seq_length / SFTConfig 各种改名），为了可复现性这里不依赖它。
· loss 只算 assistant 段（system+user 位置打 -100），这是 SFT 的标准做法。
· Qwen3 是思考模型，训练和推理都关掉 thinking（enable_thinking=False），
  否则会把 token 全花在 2000 字的思考上。

用法：
  python tools/train_lora.py --data D:\\dsh-ai\\lora\\data\\sft_train.jsonl \
      --out D:\\dsh-ai\\lora\\out\\qwen3-1.7b-deai
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# HF 缓存放 D 盘，别把 C 盘（系统盘，只剩几个 G）撑爆。
# 必须在 import transformers 之前设置。
os.environ.setdefault("HF_HOME", r"D:\dsh-ai\lora\hf")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
if os.environ.get("HF_MIRROR", "1") == "1":
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--data", default=r"D:\dsh-ai\lora\data\sft_train.jsonl")
    ap.add_argument("--out", default=r"D:\dsh-ai\lora\out\qwen3-1.7b-deai")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help=">0 时覆盖 epochs，用于卡死时间预算")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只取前 N 条")
    ap.add_argument("--no-4bit", action="store_true", help="调试用：关掉量化")
    return ap.parse_args()


# ---------------------------------------------------------------- 数据

def build_collator(tokenizer, max_len: int):
    """把 messages 样本编成 input_ids/labels；loss 只覆盖 assistant 段。"""
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    def encode(sample):
        msgs = sample["messages"]
        # 分隔点：system+user 的渲染文本（assistant 之前）
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
            # 模板不吃 add_generation_prompt 时退化为「整句都算 loss」
            labels = list(full_ids)

        ids = full_ids[:max_len]
        labels = labels[:max_len]
        if not any(x != -100 for x in labels):
            return None
        return {"input_ids": ids, "labels": labels}

    def collate(batch):
        feats = [f for f in (encode(b) for b in batch) if f]
        if not feats:
            return None
        width = max(len(f["input_ids"]) for f in feats)
        ids, labs, att = [], [], []
        for f in feats:
            pad = width - len(f["input_ids"])
            ids.append(f["input_ids"] + [pad_id] * pad)
            labs.append(f["labels"] + [-100] * pad)
            att.append([1] * len(f["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "labels": torch.tensor(labs, dtype=torch.long),
            "attention_mask": torch.tensor(att, dtype=torch.long),
        }

    return collate


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[env] gpu={torch.cuda.get_device_name(0)} "
              f"cap={torch.cuda.get_device_capability(0)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- 4bit 量化加载
    if args.no_4bit:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map={"": 0})
    else:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model, quantization_config=bnb, device_map={"": 0})
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)

    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[lora] trainable={trainable/1e6:.2f}M / total~{total/1e6:.1f}M "
          f"({100*trainable/max(total,1):.2f}%)", flush=True)

    # ---- 数据
    ds = load_dataset("json", data_files=args.data, split="train")
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"[data] {len(ds)} samples", flush=True)

    collator = build_collator(tokenizer, args.max_len)

    targs = TrainingArguments(
        output_dir=str(out / "ckpt"),
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        logging_steps=5,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=collator,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    result = trainer.train()
    elapsed = time.time() - t0

    peak = 0.0
    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**2
        peak_res = torch.cuda.max_memory_reserved() / 1024**2
    else:
        peak_res = 0.0

    model.save_pretrained(str(out))
    tokenizer.save_pretrained(str(out))

    summary = {
        "model": args.model,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "n_train": len(ds),
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "bs": args.bs,
        "grad_accum": args.grad_accum,
        "eff_batch": args.bs * args.grad_accum,
        "max_len": args.max_len,
        "lr": args.lr,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha,
                 "dropout": args.lora_dropout,
                 "targets": ["q_proj", "k_proj", "v_proj", "o_proj"]},
        "trainable_params": trainable,
        "steps": int(result.global_step) if hasattr(result, "global_step") else None,
        "train_loss": getattr(result, "training_loss", None),
        "elapsed_sec": round(elapsed, 1),
        "peak_vram_mib": round(peak, 1),
        "peak_vram_reserved_mib": round(peak_res, 1),
        "out": str(out),
    }
    (out / "train_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[done] " + json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
