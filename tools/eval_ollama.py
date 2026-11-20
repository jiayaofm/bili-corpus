# -*- coding: utf-8 -*-
"""
用本地 Ollama 的 Qwen3-8B 跑同一个「摘要 → 评论」任务，当对照组。

## 为什么要这一跑

现在有两组数据：

    Qwen3-1.7B 底座 + 摘要  →  中位 145 字，100% 超 80 字     （小作文）
    Qwen3-1.7B + LoRA + 摘要 →  中位 21 字，P(AI) 0.345        （像人）

缺的是中间那块：**参数大 4.7 倍的模型，会不会自己就写对？**

    如果 8B 也写小作文 → 说明不是参数量问题，是「没人教它 B站 语域」，LoRA 路线成立
    如果 8B 又短又咬内容 → 说明规模有用，那就值得想办法上大模型

**这个实验不用训练，不用 GPU 配额，只调 Ollama。**

## 为什么用 qwen3:8b-tool

DSH 的 qwen-preset 里写明这个 tag 是「num_ctx 8192、不带思考」的版本。
带 thinking 的型号会先输出一大段思考过程，浪费时间还污染结果。

## 用法

    python tools/eval_ollama.py                  # 每视频 20 条
    python tools/eval_ollama.py --n 10           # 少一点，快
    python tools/eval_ollama.py --model qwen3:8b
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from generate_comments import profile, template_repetition  # noqa: E402

OLLAMA = "http://127.0.0.1:11434"
SYSTEM = "你在 B 站评论区发一条真实的短评。"
USER = "视频《{title}》讲的是：{summary}\n\n就这条视频，说一句你的看法。"


def load_summaries() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for p in sorted(glob.glob(str(ROOT / "data" / "summaries" / "*.json"))):
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("bvid") and (d.get("summary") or "").strip():
            out[d["bvid"]] = d
    return out


def gen_one(model: str, user: str, temperature: float,
            num_predict: int = 200, timeout: int = 180) -> str:
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,                       # 不带思考
        "options": {"temperature": temperature,
                    "num_predict": num_predict,
                    "num_ctx": 8192},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA + "/api/chat", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode("utf-8"))
    txt = (d.get("message") or {}).get("content") or ""
    # 模型可能带引号或前缀，清一下
    txt = txt.strip().strip('"').strip()
    txt = re.sub(r"^(评论[:：]?\s*)", "", txt)
    return txt.replace("\n", " ").strip()


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3:8b-tool")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--temperature", type=float, default=1.25)
    ap.add_argument("--out", default=str(ROOT / "data" / "eval_ollama8b" / "ollama8b.json"),
                    help="产物路径。必须落在工作区内 —— Python 写不了工作区外面")
    a = ap.parse_args(argv)

    summ = load_summaries()
    if not summ:
        print("[error] data/summaries/ 是空的，先跑 fetch_materials + 写摘要")
        return 1

    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 先探一条，测延迟
    bv0 = next(iter(summ))
    t0 = time.time()
    try:
        first = gen_one(a.model, USER.format(title=summ[bv0]["title"],
                                            summary=summ[bv0]["summary"]),
                        a.temperature)
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"[error] 调 Ollama 失败: {e}\n        确认 `ollama serve` 在跑、模型名对")
        return 2
    dt = time.time() - t0
    print(f"[probe] 单条耗时 {dt:.1f}s，样例：{first[:40]}")
    print(f"[run] {a.model} × {len(summ)} 视频 × {a.n} 条，"
          f"预计 {dt * a.n * len(summ) / 60:.0f} 分钟\n")

    per_video: dict[str, dict] = {}
    all_gen: list[str] = []
    for i, (bv, s) in enumerate(summ.items(), 1):
        user = USER.format(title=s["title"], summary=s["summary"])
        gens: list[str] = []
        for k in range(a.n):
            try:
                g = gen_one(a.model, user, a.temperature)
            except Exception as e:
                print(f"    (第 {k+1} 条失败: {type(e).__name__})")
                continue
            if g:
                gens.append(g)
        all_gen.extend(gens)
        pv = profile(gens)
        pv["templates"] = template_repetition(gens)
        pv["samples"] = gens
        pv["title"] = s["title"]
        per_video[bv] = pv
        print(f"  [{i}/{len(summ)}] {s['title'][:26]:<28} "
              f"中位 {pv.get('len_median', 0):.0f} "
              f"<10字 {pv.get('short_lt10', 0)*100:.0f}% "
              f">80字 {pv.get('long_gt80', 0)*100:.0f}% "
              f"表情 {pv.get('emoji_rate', 0)*100:.0f}%", flush=True)

    pooled = profile(all_gen)
    pooled["n"] = len(all_gen)

    # 我们自己的分类器判它有多像 AI
    p_ai = None
    try:
        from pipeline.scorer import Scorer
        sc = Scorer()
        probs = [sc.describe(x)["heldout_ai"] for x in all_gen]
        p_ai = sum(probs) / len(probs) if probs else None
    except Exception as e:
        print(f"  (分类器跳过: {type(e).__name__})")

    out = {
        "tag": "ollama8b",
        "label": f"{a.model}·摘要",
        "model": a.model,
        "adapter": None,
        "n_per_video": a.n,
        "temperature": a.temperature,
        "prompt": "summary",
        "n_generated": len(all_gen),
        "pooled": pooled,
        "p_ai_mean": p_ai,
        "per_video": per_video,
    }
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"  {a.model}   中位 {pooled.get('len_median',0):.1f}  "
          f"均值 {pooled.get('len_mean',0):.1f}  "
          f"<10字 {pooled.get('short_lt10',0)*100:.1f}%  "
          f">80字 {pooled.get('long_gt80',0)*100:.1f}%  "
          f"表情 {pooled.get('emoji_rate',0)*100:.1f}%")
    if p_ai is not None:
        print(f"  分类器判 AI  P(AI) = {p_ai:.4f}")
    print(f"  产物: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
