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
# 减少碎片化，避免显存被 allocator 越吃越大
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if os.environ.get("HF_MIRROR", "1") == "1":
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 🔒 物理门禁：没有显式授权就**在 import torch 之前**退出。
# 这台机器是用户在用的（2026-11-17 有过两次抢显存导致游戏卡死的记录），
# 所以训练脚本必须「不可能因为一次误操作就自己跑起来」。
if os.environ.get("DEAI_ALLOW_GPU") != "1":
    import sys as _sys
    _sys.stderr.write(
        "[refuse] 未设置 DEAI_ALLOW_GPU=1，拒绝启动训练。\n"
        "         确认可以用 GPU 后再：$env:DEAI_ALLOW_GPU='1'\n")
    raise SystemExit(2)

import subprocess
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
    ap.add_argument("--bs", type=int, default=2,
                    help="默认 2：实测 peak 3.19GB。bs=4 会把显存推高，"
                         "在共用机器上不要用")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-targets", default="attn",
                    choices=["attn", "attn_mlp", "all_linear"],
                    help="挂 LoRA 的层：attn=只挂注意力（旧默认）；"
                         "attn_mlp=加 gate/up/down，风格迁移收益大；"
                         "all_linear=再加 lm_head/embed_tokens")
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help=">0 时覆盖 epochs，用于卡死时间预算")
    ap.add_argument("--limit", type=int, default=0, help=">0 时只取前 N 条")
    ap.add_argument("--no-4bit", action="store_true", help="调试用：关掉量化")
    # ---- 安全阀：这台机器是用户在用的，不是专用训练机 ----
    ap.add_argument("--min-free-gb", type=float, default=6.0,
                    help="开始前要求的最小空闲显存(GB)，不够就拒绝启动")
    ap.add_argument("--vram-frac", type=float, default=0.6,
                    help="本进程显存上限(占总额比例)，0 表示不设上限")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="跳过显存预检（仅在明确知道自己在干什么时用）")
    return ap.parse_args()


def free_vram_gb() -> tuple[float, float] | None:
    """返回 (已用 MiB, 总 MiB)；查询失败返回 None。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        used, total = [float(x) for x in out.splitlines()[0].split(",")]
        return used, total
    except Exception:
        return None


class VramLogger:
    """Trainer 回调：定期把当前/峰值显存打到日志里，方便回看。"""

    def __init__(self, every: int = 10, base: int = 0):
        self.every = every
        self.base = base

    def __call__(self, step: int, logs=None) -> None:
        if step % self.every == 0:
            cur = torch.cuda.memory_allocated() / 1024 ** 2
            peak = torch.cuda.max_memory_allocated() / 1024 ** 2
            print(f"[vram] step={step} cur={cur:.0f}MiB "
                  f"peak_alloc={peak:.0f}MiB", flush=True)


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


# 挂 LoRA 的层。
# 旧配置只有注意力层，但它管的是「抽什么信息」，不管「怎么措辞」。
# 风格改写真正要动的是 MLP（gate/up/down）——措辞和句式的落脚点在那里。
LORA_TARGET_MAP = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "attn_mlp": ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"],
    "all_linear": ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj",
                   "lm_head", "embed_tokens"],
}


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        print(f"[env] gpu={torch.cuda.get_device_name(0)} "
              f"cap={torch.cuda.get_device_capability(0)}", flush=True)

    # ---- 安全阀 1：开跑前查空闲显存。机器是用户在用的，不够就直接退出。
    v = free_vram_gb()
    if v is not None:
        used, total = v
        free = (total - used) / 1024
        print(f"[preflight] vram used={used:.0f}MiB total={total:.0f}MiB "
              f"free={free:.2f}GB (要求 >= {args.min_free_gb}GB)", flush=True)
        if not args.skip_preflight and free < args.min_free_gb:
            print(f"[preflight] 拒绝启动：空闲显存 {free:.2f}GB "
                  f"< {args.min_free_gb}GB。用户的机器还在用，先别抢。",
                  flush=True)
            raise SystemExit(3)

    # ---- 安全阀 2：给本进程设显存上限，别把整张卡吃干净。
    if torch.cuda.is_available() and args.vram_frac > 0:
        torch.cuda.set_per_process_memory_fraction(args.vram_frac)
        print(f"[preflight] 本进程显存上限 {args.vram_frac*100:.0f}% "
              f"of {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f}GB",
              flush=True)

    # ---- 安全阀 3：把梯度检查点先打开，降低激活占用
    torch.backends.cuda.matmul.allow_tf32 = True

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
        target_modules=LORA_TARGET_MAP[args.lora_targets],
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

    # transformers 5.x 去掉了 warmup_ratio，只剩 warmup_steps —— 自己算。
    if args.max_steps > 0:
        total_steps = args.max_steps
    else:
        steps_per_epoch = max(1, -(-len(ds) // (args.bs * args.grad_accum)))
        total_steps = int(steps_per_epoch * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    print(f"[sched] total_steps={total_steps} warmup_steps={warmup_steps}", flush=True)

    targs = TrainingArguments(
        output_dir=str(out / "ckpt"),
        per_device_train_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        logging_steps=5,
        # 每个 epoch 都存一次：上次 save_strategy="no" 被杀后零产物，
        # 这次就算中途断电，至少有上一轮的 adapter。
        save_strategy="epoch",
        save_total_limit=2,
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

    # 定期打显存，方便回看是不是在悄悄涨
    from transformers import TrainerCallback

    class _VramCB(TrainerCallback):
        def on_log(self, a, state, control, logs=None, **kw):
            if state.global_step and state.global_step % 10 == 0:
                cur = torch.cuda.memory_allocated() / 1024 ** 2
                peak = torch.cuda.max_memory_allocated() / 1024 ** 2
                print(f"[vram] step={state.global_step} cur={cur:.0f}MiB "
                      f"peak_alloc={peak:.0f}MiB", flush=True)

    trainer.add_callback(_VramCB())

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
                 "targets": LORA_TARGET_MAP[args.lora_targets]},
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
