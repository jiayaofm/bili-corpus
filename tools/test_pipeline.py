# -*- coding: utf-8 -*-
"""
验证「8B 写初稿 → 1.7B LoRA 改 AI 味」这条两段式流水线。

## 为什么要测

8B 在内容上略好（事实复现 9.3% vs 7.1%），但在语域上一塌糊涂
（骨架重复 65.7%、表情 0%）。LoRA 反过来：语域好（4.3%）、内容差（7.1%）。

**两者分工是合理的，拼起来理论上能互补。**

但有个结构性障碍：我们那个 LoRA 是**生成器**，只会一种输入形状

    system: 你在 B 站评论区发一条真实的短评。
    user  : 视频《{title}》讲的是：{摘要}
    → 一条评论

它从没见过「把这段文字改得像人写的」这种任务。所以本脚本测的是最朴素的做法：
**把 8B 的草稿塞进摘要位**，看 LoRA 会怎么处理。

## 三种可能的结果

    ① LoRA 无视草稿，重新生成  → 等于没串，白做
    ② LoRA 被草稿带偏，输出四不像 → 需要重新训练改写能力
    ③ LoRA 把草稿压缩成短评，且保留内容 → 流水线成立，赚了

## 用法

    python tools/test_pipeline.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

SRC = ROOT / "data" / "eval_ollama8b" / "ollama8b.json"      # 8B 的 140 条草稿
OUT = ROOT / "data" / "pipeline_8b_gen4"
ADAPTER = r"D:\dsh-ai\lora\out\qwen3-1.7b-gen4"

SYSTEM = "你在 B 站评论区发一条真实的短评。"
USER = "视频《{title}》讲的是：{summary}\n\n就这条视频，说一句你的看法。"


def load_summaries() -> dict[str, dict]:
    out = {}
    for p in sorted((ROOT / "data" / "summaries").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        out[d["bvid"]] = d
    return out


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    os.environ.setdefault("DEAI_ALLOW_GPU", "1")

    if not SRC.exists():
        print(f"[error] 找不到 8B 产物 {SRC}，先跑 tools/eval_ollama.py")
        return 1
    if not Path(ADAPTER).exists():
        print(f"[error] 找不到 adapter {ADAPTER}")
        return 1

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    drafts = json.loads(SRC.read_text(encoding="utf-8"))
    summ = load_summaries()

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-1.7B", torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    print(f"[load] {ADAPTER}\n")

    OUT.mkdir(parents=True, exist_ok=True)
    results = {}
    total = 0

    for i, (bv, v) in enumerate(drafts["per_video"].items(), 1):
        title = v.get("title") or summ.get(bv, {}).get("title", "")
        gens = []
        for draft in (v.get("samples") or [])[:10]:      # 每视频取 10 条草稿
            user = USER.format(title=title, summary=draft)
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user}]
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)  # Qwen3 必须关思考
            ids = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=120,
                                     do_sample=True, temperature=1.25,
                                     top_p=0.95, pad_token_id=tok.eos_token_id)
            g = tok.decode(out[0][ids["input_ids"].shape[1]:],
                           skip_special_tokens=True).strip()
            g = g.replace("\n", " ").split("。")[0]
            if g:
                gens.append(g)
        results[bv] = {"title": title,
                       "drafts": (v.get("samples") or [])[:10],
                       "rewrites": gens}
        total += len(gens)
        print(f"  [{i}/{len(drafts['per_video'])}] {title[:24]:<26} "
              f"{len(gens)} 条", flush=True)

    (OUT / "pipeline.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  共 {total} 条，产物 {OUT / 'pipeline.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
