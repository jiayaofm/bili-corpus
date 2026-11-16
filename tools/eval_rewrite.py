#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""三方对比：Qwen3-1.7B 原始 / +LoRA / qwen3:8b（Ollama）。

用项目自己的 GEN_SYSTEM（pipeline/llm.py 的十条硬规则）+ 项目自己的
第 1 轮改写指令（pipeline/rewrite.py 的 build_instruction），
保证三条基线看到的是同一个 prompt —— 只换模型，不换题。

生成方式说明
------------
GEN_USER 原本请求「写 {n} 个版本，每行一个」，但 1.7B 这种小模型基本不遵守，
会把「3 个版本」写成 1 段。为了保证「换模型不换题」的公平性，这里统一
**独立采样 3 次**（每次 prompt 里 n=1），拿 3 个候选。
这样对比的是「模型的改写能力」，不是「能不能遵守多行输出格式」。

用法：
  python eval_rewrite.py --tag base            # 只跑底座
  python eval_rewrite.py --tag lora --adapter <dir>
  python eval_rewrite.py --tag qwen3-8b --ollama qwen3:8b
  python eval_rewrite.py --all                 # 三种一起跑
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

os.environ.setdefault("HF_HOME", r"D:\dsh-ai\lora\hf")
if os.environ.get("HF_MIRROR", "1") == "1":
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(r"E:\DSH\bili-corpus")
sys.path.insert(0, str(ROOT))

from pipeline.llm import GEN_SYSTEM, GEN_USER  # noqa: E402  只读引用，不改动


_SCORER = None


def _scorer():
    global _SCORER
    if _SCORER is None:
        from pipeline.rewrite import get_scorer
        _SCORER = get_scorer()
    return _SCORER


def project_instruction(src: str) -> str:
    """复现 rewrite.py 第 1 轮的指令拼装（只读调用，不修改项目代码）。

    rewrite.py 里是 sc = get_scorer()，然后 orig_score = sc.describe(text)。
    """
    from pipeline.rewrite import build_instruction, describe_problems
    d = _scorer().describe(src)
    p = describe_problems(d)
    return build_instruction(1, src, described=d, problems=p)


def build_messages(src: str, n: int = 1) -> list[dict]:
    ins = project_instruction(src)
    user = GEN_USER.format(original=src, instruction=ins, n=n)
    return [{"role": "system", "content": GEN_SYSTEM},
            {"role": "user", "content": user}]


def clean_out(text: str) -> str:
    """去掉模型可能加的行号/引号/前后缀。"""
    t = (text or "").strip()
    for pre in ("改写：", "改写:", "输出：", "输出:"):
        if t.startswith(pre):
            t = t[len(pre):].strip()
    t = t.strip().strip('"').strip("“”").strip()
    # 去行首序号 "1. " / "10. " / "1、" / "1) "。
    # `[.)]` 后面必须跟空白，否则「1.5倍速看完的」会被啃成「5倍速看完的」。
    t = re.sub(r"^\d{1,2}(?:[.)]\s+|、\s*)", "", t)
    return t.strip()


def parse_multi(raw: str) -> list[str]:
    """从「要求一次吐 n 个版本」的输出里切出候选。

    只按行切 —— 这是最保守的口径：如果模型把 3 个版本写成一段，
    这里就只能切出 1 条，那正好说明它不满足 pipeline 的多输出要求。
    """
    out = []
    for line in (raw or "").splitlines():
        c = clean_out(line)
        if not c:
            continue
        # 去掉 markdown 列表符号
        if c.startswith("- ") or c.startswith("* "):
            c = c[2:].strip()
        if c:
            out.append(c)
    # 去掉完全重复的
    uniq = []
    for c in out:
        if c not in uniq:
            uniq.append(c)
    return uniq


# ---------------------------------------------------------------- Ollama

def ollama_chat(model: str, messages: list[dict], temperature: float) -> str:
    # Ollama 同样占显存（qwen3:8b Q4 权重 5.2GB + KV cache），一样要授权
    if os.environ.get("DEAI_ALLOW_GPU") != "1":
        raise SystemExit(
            "[refuse] 未设置 DEAI_ALLOW_GPU=1，拒绝调用 Ollama（会占显存）。")
    body = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"temperature": temperature, "num_predict": 400},
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read().decode("utf-8"))
    return (d.get("message") or {}).get("content", "")


# ---------------------------------------------------------------- HF 本地

def load_hf(model_name: str, adapter: str | None, four_bit: bool = True):
    # 🔒 物理门禁：真要加载模型了才检查（--dry-run 不走这里，所以能离线验证）
    if os.environ.get("DEAI_ALLOW_GPU") != "1":
        raise SystemExit(
            "[refuse] 未设置 DEAI_ALLOW_GPU=1，拒绝加载模型。\n"
            "         离线验证请用 --dry-run（不加载模型）。\n"
            "         确认可以用 GPU 后再：$env:DEAI_ALLOW_GPU='1'")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    kwargs = {"device_map": {"": 0}, "torch_dtype": torch.bfloat16}
    if four_bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def hf_generate(tok, model, messages, temperature=0.9, max_new_tokens=256) -> str:
    import torch
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=False)
    enc = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=True,
            temperature=temperature, top_p=0.9, repetition_penalty=1.05,
            pad_token_id=tok.pad_token_id)
    new = out[0][enc["input_ids"].shape[1]:]
    return tok.decode(new, skip_special_tokens=True)


# ---------------------------------------------------------------- 主流程

def load_samples() -> list[dict]:
    d = json.loads((ROOT / "state" / "rewrite_abc.json").read_text(encoding="utf-8"))
    items = d["items"]
    want = [0, 3]
    out = []
    for i in want:
        it = items[i]
        out.append({"id": it.get("id", str(i)), "index": i, "src": it["src"],
                    "label": "短评" if i == 0 else "长评"})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="base / lora / qwen3-8b 之类")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--ollama", default=None, help="给定时走 Ollama")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--outdir", default=r"D:\dsh-ai\lora\out\eval")
    ap.add_argument("--label", default=None,
                    help="报告里显示的名字，如 '原始 1.7B' / '训后 1.7B'")
    ap.add_argument("--with-n3", action="store_true",
                    help="额外跑一次 n=3 的「多输出合规性」检查（不计入质量对比）")
    ap.add_argument("--dry-run", action="store_true",
                    help="不加载任何模型，只验证 prompt 拼装和落盘（不碰 GPU）")
    a = ap.parse_args()
    display = a.label or a.tag

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    samples = load_samples()
    env_info = {"torch": None, "gpu": None, "capability": None}
    try:
        import torch
        env_info["torch"] = torch.__version__
        if torch.cuda.is_available():
            env_info["gpu"] = torch.cuda.get_device_name(0)
            env_info["capability"] = list(torch.cuda.get_device_capability(0))
    except Exception:
        pass

    results = {
        "tag": a.tag,
        "label": display,              # 报告里显示的名字
        "model": a.model,
        "adapter": a.adapter,
        "ollama": a.ollama,
        "temperature": a.temperature,
        "n": a.n,
        "sampling": "independent n=1 x3",   # 主口径：独立采样
        "with_n3": a.with_n3,
        "env": env_info,
        "items": [],
    }

    tok = model = None
    if not a.dry_run and not a.ollama:
        tok, model = load_hf(a.model, a.adapter)

    def gen(msgs: list[dict], dry: str) -> str:
        if a.dry_run:
            return dry
        if a.ollama:
            return ollama_chat(a.ollama, msgs, a.temperature)
        return hf_generate(tok, model, msgs, a.temperature)

    t_all = time.time()
    for s in samples:
        msgs = build_messages(s["src"], n=1)
        cands = []
        for k in range(a.n):
            t0 = time.time()
            raw = gen(msgs, f"[DRY-RUN candidate {k}]")
            cands.append({"k": k, "raw": raw, "text": clean_out(raw),
                          "sec": round(time.time() - t0, 1)})
            print(f"[{a.tag}] item{s['index']} cand{k}: {clean_out(raw)[:70]}",
                  flush=True)

        item = {
            "index": s["index"], "id": s["id"], "label": s["label"],
            "src": s["src"],
            "instruction": project_instruction(s["src"]),
            "candidates": cands,
        }

        # ---- 附加：一次请求 n=3 的「多输出合规性」（工程可用性，不计入质量分）
        if a.with_n3:
            msgs3 = build_messages(s["src"], n=3)
            raw3 = gen(msgs3, "[DRY-RUN n3] a\n[DRY-RUN n3] b\n[DRY-RUN n3] c")
            parsed = parse_multi(raw3)
            item["n3"] = {
                "requested": 3,
                "raw": raw3,
                "parsed": parsed,
                "distinct_count": len(parsed),
                # 能不能按 pipeline 的用法吐出 3 个独立候选
                "compliant": len(parsed) >= 3,
                "note": "退化成 1 段或多段粘连" if len(parsed) < 3 else "",
            }
            print(f"[{a.tag}] item{s['index']} n3 -> {len(parsed)} 条 "
                  f"compliant={len(parsed) >= 3}", flush=True)

        results["items"].append(item)

    results["elapsed_sec"] = round(time.time() - t_all, 1)
    results["peak_vram_mib"] = None
    try:
        import torch
        if torch.cuda.is_available():
            results["peak_vram_mib"] = round(
                torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
    except Exception:
        pass

    p = outdir / f"{a.tag}.json"
    p.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {p}", flush=True)


if __name__ == "__main__":
    main()
