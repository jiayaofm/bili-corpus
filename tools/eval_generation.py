#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成任务评估：给「视频标题 + 分区」，让本地模型写评论，和真人评论比分布。

为什么不自己写一套指标
----------------------
`tools/generate_comments.py` 里已经有 `profile()` / `template_repetition()`，
真人侧的「实测值」（中位 24.7 字、10 字以下 14.8% …）就是用它们算的。
这里**直接 import 复用它**，保证「模型侧」和「真人侧」用同一套定义 ——
各算各的会让表格没法比。

三个模型对比
------------
  base   Qwen3-1.7B 底座（不训）
  lora   Qwen3-1.7B + 本次 LoRA
  (可选) Qwen3-4B

判定标准：哪一列的分布最接近「真人实测值」。

用法
----
  # 真人侧基线（不加载模型，CPU 即可）
  python tools/eval_generation.py --human-only

  # 训后模型
  DEAI_ALLOW_GPU=1 python tools/eval_generation.py --tag lora \
      --adapter D:\\dsh-ai\\lora\\out\\qwen3-1.7b-gen

  # 底座
  DEAI_ALLOW_GPU=1 python tools/eval_generation.py --tag base

  # 先小样本试水（每视频 3 条，很快）
  DEAI_ALLOW_GPU=1 python tools/eval_generation.py --tag lora --adapter ... --n 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HOME", r"D:\dsh-ai\lora\hf")
if os.environ.get("HF_MIRROR", "1") == "1":
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

ROOT = Path(r"E:\DSH\bili-corpus")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# 复用项目里现成的指标实现（只 import，不修改）
from generate_comments import (  # noqa: E402
    DOMAIN,
    GEN_SYSTEM_V1,
    GEN_SYSTEM_V2,
    load_corpus,
    profile,
    template_repetition,
)

PROMPTS = {"v1": GEN_SYSTEM_V1, "v2": GEN_SYSTEM_V2}

# 🔴 和训练完全一致的格式（B 组主口径）。
# 训练样本是 build_dataset.py 造的：
#   system: 你在 B 站评论区发一条真实的短评。
#   user  : 关于「{video_title}」这个话题，说一句你的看法。
# 评估必须用同一个格式，否则是分布外输入 —— LoRA 从没见过 v2 那种长指令，
# 用它评估会把「评估方式错了」误读成「训练把模型带坏了」。
GEN_SYSTEM_TRAIN = "你在 B 站评论区发一条真实的短评。"
PROMPTS["train"] = GEN_SYSTEM_TRAIN

# 每种 prompt 对应的 user 模板（train 不加分区、不加任何分布要求）
USER_TEMPLATES = {
    "train": "关于「{title}」这个话题，说一句你的看法。",
    "v1": "《{title}》，分区：{domain}",
    "v2": "《{title}》，分区：{domain}",
}


def build_user_msg(prompt: str, title: str, domain: str) -> str:
    return USER_TEMPLATES[prompt].format(title=title, domain=domain)

# 用户/lead 给的真人实测值（**按视频等权平均**算的），用于最后对照
REFERENCE_HUMAN = {
    "len_median": 24.7,
    "short_lt10": 14.8,
    "long_gt80": 11.4,
    "emoji_rate": 34.8,
}

# 「短评」门槛**不自己定**：项目的 profile() 已内置 len<30 的 short_* 指标
# （short_n / short_distinct / short_distinct_ratio），直接用它，
# 避免出现「两套短评定义」导致表格不可比。


def span_stats(per_video: dict, key: str = "len_median") -> dict:
    """逐视频的跨度：max - min。

    用来抓「压平」这个失败模式：真人每视频中位跨度 17~36（跨度 19），
    模型如果无论什么话题都写差不多长度（17~22，跨度 5），
    即使全池中位数对上了，也是没学会「按话题调整长度」。
    """
    vals = [p[key] for p in per_video.values()
            if isinstance(p, dict) and key in p
            and isinstance(p[key], (int, float))
            and p[key] == p[key]]
    if not vals:
        return {"min": float("nan"), "max": float("nan"),
                "span": float("nan"), "n_videos": 0}
    return {"min": min(vals), "max": max(vals),
            "span": max(vals) - min(vals), "n_videos": len(vals)}


def equal_weight(per_video: dict, key: str) -> float:
    """按视频等权平均（每个视频权重相同）。

    和「全池合并」是两个不同的问题：
      · 等权    → 一个**典型视频**的指标（lead 的参考值就是按这个算的）
      · 全池    → 随机抽一条**评论**的指标（评论多的视频权重更大）
    如果模型在评论多的视频上表现差，等权平均会掩盖它，所以两个都要报。
    """
    vals = [p[key] for p in per_video.values()
            if isinstance(p, dict) and key in p
            and isinstance(p[key], (int, float))
            and p[key] == p[key]]          # 排除 nan
    return sum(vals) / len(vals) if vals else float("nan")


def clean_comment(text: str) -> str:
    """把模型输出收成「一条评论」。"""
    t = (text or "").strip()
    if not t:
        return ""
    for pre in ("评论：", "评论:", "输出：", "输出:"):
        if t.startswith(pre):
            t = t[len(pre):].strip()
    # 只要第一条非空行（我们要求只输出一条）
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return ""
    t = lines[0]
    t = t.strip().strip('"').strip("“”").strip("「」")
    t = re.sub(r"^\d{1,2}[.)、]\s+", "", t)
    if t.startswith("- ") or t.startswith("* "):
        t = t[2:].strip()
    return t.strip()


# ---------------------------------------------------------------- 真人侧

def human_profile(only: str | None = None) -> dict:
    corpus = load_corpus()
    items = [(bv, v) for bv, v in corpus.items() if v["comments"]]
    if only:
        items = [(bv, v) for bv, v in items if bv == only]
    items.sort(key=lambda kv: -len(kv[1]["comments"]))

    pooled = [c for _, v in items for c in v["comments"]]
    per_video = {}
    for bv, v in items:
        p = profile(v["comments"])
        p["title"] = v["title"]
        p["domain"] = DOMAIN.get(bv, "其他")
        per_video[bv] = p
    out = {
        "n_videos": len(items),
        "pooled": profile(pooled),
        "pooled_templates": template_repetition(pooled),
        "per_video": per_video,
        # 按视频等权（lead 的口径），和全池合并并列给出
        "equal_weight": {
            k: equal_weight(per_video, k)
            for k in ("len_median", "len_mean", "short_lt10",
                      "long_gt80", "emoji_rate", "short_distinct_ratio")
        },
        "len_median_span": span_stats(per_video, "len_median"),
        "videos": [(bv, v["title"], DOMAIN.get(bv, "其他"),
                    len(v["comments"])) for bv, v in items],
    }
    return out


# ---------------------------------------------------------------- 模型侧

def require_gpu_permission() -> None:
    if os.environ.get("DEAI_ALLOW_GPU") != "1":
        print("[refuse] 未设置 DEAI_ALLOW_GPU=1，拒绝加载模型。\n"
              "         只看真人侧基线请用 --human-only（不加载模型）。",
              flush=True)
        raise SystemExit(2)


def load_model(model_name: str, adapter: str | None):
    require_gpu_permission()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb, device_map={"": 0})
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return tok, model


def generate_many(tok, model, messages, n: int, temperature: float,
                  max_new_tokens: int) -> list[str]:
    import torch
    text = tok.apply_chat_template(messages, tokenize=False,
                                   add_generation_prompt=True,
                                   enable_thinking=False)
    enc = tok(text, return_tensors="pt").to(model.device)
    outs: list[str] = []
    for _ in range(n):
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=True,
                temperature=temperature, top_p=0.9,
                repetition_penalty=1.08,
                pad_token_id=tok.pad_token_id)
        new = gen[0][enc["input_ids"].shape[1]:]
        outs.append(clean_comment(
            tok.decode(new, skip_special_tokens=True)))
    return outs


# ---------------------------------------------------------------- 对照表

def row(name: str, p: dict, tpl: dict | None = None) -> str:
    cells = [
        f"{name:16s}",
        f"{p.get('n', 0):4d}",
        f"{p.get('len_median', 0):6.1f}",
        f"{p.get('short_lt10', 0):6.1f}%",
        f"{p.get('long_gt80', 0):6.1f}%",
        f"{p.get('emoji_rate', 0):6.1f}%",
        f"{p.get('len_mean', 0):6.1f}",
    ]
    if tpl is not None:
        cells.append(f"{tpl.get('dup_ratio', 0):5.1f}%")
    return " ".join(cells)


HEADER = (f"{'模型':16s} {'n':>4s} {'中位':>6s} {'<10字':>7s} "
          f"{'>80字':>7s} {'表情':>7s} {'均值':>6s}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="base / lora / qwen3-4b")
    ap.add_argument("--label", default=None, help="表格里显示的名字")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--n", type=int, default=20, help="每个视频生成几条")
    ap.add_argument("--temperature", type=float, default=1.25)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--prompt", default="v2", choices=sorted(PROMPTS))
    ap.add_argument("--only", default=None)
    ap.add_argument("--max-videos", type=int, default=0)
    ap.add_argument("--human-only", action="store_true",
                    help="只统计真人侧，不加载模型")
    ap.add_argument("--no-pai", action="store_true",
                    help="跳过 P(AI)（省时间）")
    ap.add_argument("--outdir", default=r"D:\dsh-ai\lora\out\eval_gen")
    a = ap.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    hp = human_profile(a.only)
    print(f"真人侧：{hp['n_videos']} 个视频，{hp['pooled']['n']} 条评论")
    p = hp["pooled"]
    print("  实测  中位%.1f  <10字%.1f%%  >80字%.1f%%  表情%.1f%%  均值%.1f"
          % (p["len_median"], p["short_lt10"], p["long_gt80"],
             p["emoji_rate"], p["len_mean"]))
    print("  lead 给的口径：中位%.1f  <10字%.1f%%  >80字%.1f%%  表情%.1f%%"
          % (REFERENCE_HUMAN["len_median"], REFERENCE_HUMAN["short_lt10"],
             REFERENCE_HUMAN["long_gt80"], REFERENCE_HUMAN["emoji_rate"]))

    (outdir / "human_baseline.json").write_text(
        json.dumps(hp, ensure_ascii=False, indent=2), encoding="utf-8")

    if a.human_only:
        print(f"\n[human-only] 已存 {(outdir / 'human_baseline.json')}")
        return

    label = a.label or a.tag or "model"
    items = hp["videos"]
    if a.max_videos:
        items = items[: a.max_videos]

    tok, model = load_model(a.model, a.adapter)
    print(f"\n[{label}] {a.model} adapter={a.adapter} n={a.n} "
          f"temp={a.temperature} prompt={a.prompt}", flush=True)

    t0 = time.time()
    all_gen: list[str] = []
    per_video = {}
    for vi, (bv, title, dom, nreal) in enumerate(items, 1):
        msgs = [
            {"role": "system", "content": PROMPTS[a.prompt]},
            {"role": "user", "content": build_user_msg(a.prompt, title, dom)},
        ]
        gens = generate_many(tok, model, msgs, a.n, a.temperature,
                             a.max_new_tokens)
        gens = [g for g in gens if g]
        all_gen.extend(gens)
        pv = profile(gens)
        pv["templates"] = template_repetition(gens)
        pv["samples"] = gens
        pv["title"] = title
        pv["domain"] = dom
        pv["n_real"] = nreal
        per_video[bv] = pv
        print("  [%d/%d] %-28s 中位%3.0f <10字%3.0f%% >80字%3.0f%% 表情%3.0f%%"
              % (vi, len(items), (title or "")[:26], pv["len_median"],
                 pv["short_lt10"], pv["long_gt80"], pv["emoji_rate"]),
              flush=True)

    gp = profile(all_gen)
    gtpl = template_repetition(all_gen)
    g_equal = {
        k: equal_weight(per_video, k)
        for k in ("len_median", "len_mean", "short_lt10",
                  "long_gt80", "emoji_rate", "short_distinct_ratio")
    }
    g_span = span_stats(per_video, "len_median")

    # P(AI)：用项目自己的分类器。注意 263 条真人评论参与过它训练，
    # 那部分真人的 P(AI) 偏乐观，不能当完全独立的判据。
    pai = None
    if not a.no_pai:
        try:
            from pipeline.scorer import Scorer
            sc = Scorer()
            pai = sum(sc.describe(c)["heldout_ai"] for c in all_gen) \
                / max(len(all_gen), 1)
        except Exception as e:
            print(f"  [P(AI)] 跳过：{type(e).__name__}: {e}")

    res = {
        "tag": a.tag, "label": label, "model": a.model,
        "adapter": a.adapter, "n_per_video": a.n,
        "temperature": a.temperature, "prompt": a.prompt,
        "max_new_tokens": a.max_new_tokens,
        "n_generated": len(all_gen),
        "pooled": gp, "pooled_templates": gtpl,
        "equal_weight": g_equal,
        "len_median_span": g_span,
        "p_ai_mean": pai,
        "per_video": per_video,
        "elapsed_sec": round(time.time() - t0, 1),
        "peak_vram_mib": None,
    }
    try:
        import torch
        if torch.cuda.is_available():
            res["peak_vram_mib"] = round(
                torch.cuda.max_memory_allocated() / 1024 ** 2, 1)
    except Exception:
        pass

    fn = a.tag or "model"
    (outdir / f"{fn}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    def line(name: str, med, mean, s10, l80, emo, dup, sdr) -> str:
        return (f"{name:22s} {med:6.1f} {mean:6.1f} {s10:5.1f}% {l80:5.1f}% "
                f"{emo:5.1f}%  {dup:5.1f}%   {sdr:5.1f}%")

    hp_p, h_eq = hp["pooled"], hp["equal_weight"]
    print("\n" + "=" * 100)
    print("  【全池合并】随机抽一条评论的口径（主口径：与模型侧同语料同 profile）")
    print(f"{'':22s} {'中位':>6s} {'均值':>6s} {'<10字':>6s} {'>80字':>6s} "
          f"{'表情':>6s} {'骨架重复':>8s} {'短评去重':>7s}")
    print("-" * 100)
    print(line("真人（实测）", hp_p["len_median"], hp_p["len_mean"],
               hp_p["short_lt10"], hp_p["long_gt80"], hp_p["emoji_rate"],
               hp["pooled_templates"]["dup_ratio"],
               hp_p["short_distinct_ratio"]))
    print(line(label, gp["len_median"], gp["len_mean"], gp["short_lt10"],
               gp["long_gt80"], gp["emoji_rate"], gtpl["dup_ratio"],
               gp["short_distinct_ratio"]))
    print("=" * 100)
    print(f"  【按视频等权】一个典型视频的口径（lead 的参考值就是按这个算的）")
    print(line("真人（实测）", h_eq["len_median"], h_eq["len_mean"],
               h_eq["short_lt10"], h_eq["long_gt80"], h_eq["emoji_rate"],
               float("nan"), h_eq["short_distinct_ratio"]))
    print(line(label, g_equal["len_median"], g_equal["len_mean"],
               g_equal["short_lt10"], g_equal["long_gt80"],
               g_equal["emoji_rate"], float("nan"),
               g_equal["short_distinct_ratio"]))
    print("=" * 100)
    print(f"  lead 参考（按视频等权）：中位 {REFERENCE_HUMAN['len_median']} "
          f"<10字 {REFERENCE_HUMAN['short_lt10']}% "
          f">80字 {REFERENCE_HUMAN['long_gt80']}% "
          f"表情 {REFERENCE_HUMAN['emoji_rate']}%")
    print(f"  短评定义用项目 profile() 的 len<30：真人 {hp_p.get('short_n')} 条，"
          f"模型 {gp.get('short_n')} 条")

    hs, gs = hp["len_median_span"], g_span
    print(f"\n  【逐视频压平检测】中位字数的跨视频跨度（真人 17~36，跨度 19）")
    print(f"    真人   min {hs['min']:.0f}  max {hs['max']:.0f}  "
          f"跨度 {hs['span']:.0f}")
    print(f"    {label:12s} min {gs['min']:.0f}  max {gs['max']:.0f}  "
          f"跨度 {gs['span']:.0f}")
    if hs["span"] and gs["span"] == gs["span"]:
        ratio = gs["span"] / hs["span"]
        verdict = "偏平（学不到按话题变长短）" if ratio < 0.5 else "跨度尚可"
        print(f"    跨度比 {ratio:.2f}× 真人 → {verdict}")
    print("\n  逐视频明细（中位字数）:")
    print(f"    {'视频':34s} {'真人':>6s} {'模型':>6s} {'差':>7s}")
    for bv, title, dom, nreal in items:
        r_med = hp["per_video"].get(bv, {}).get("len_median", float("nan"))
        m_med = per_video.get(bv, {}).get("len_median", float("nan"))
        d = m_med - r_med if (m_med == m_med and r_med == r_med) else float("nan")
        print(f"    {(title or '')[:32]:34s} {r_med:6.0f} {m_med:6.0f} "
              f"{d:+7.0f}")
    print(f"\n[meta] {label}: n={len(all_gen)} "
          f"P(AI)={pai if pai is None else round(pai, 4)} "
          f"峰值显存={res['peak_vram_mib']} MiB 耗时={res['elapsed_sec']}s")
    print(f"[saved] {outdir / (fn + '.json')}")


if __name__ == "__main__":
    main()
