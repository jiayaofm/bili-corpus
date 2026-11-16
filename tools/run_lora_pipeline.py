#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一条命令跑完整个流程（GPU 窗口一开就能用）。

🔒 **物理门禁**：本脚本会加载模型/起训练进程，所以必须显式设置
`DEAI_ALLOW_GPU=1` 才会执行，否则**直接退出**。这是为了确保
「不可能因为一次误操作就占掉用户的 GPU」。

设置方法（确认能占 GPU 后）：
    $env:DEAI_ALLOW_GPU='1'

判定逻辑本身在 `preflight.py` 里（纯函数、不 import torch），
用 `tools/test_preflight.py` 单测 —— **不要**用「设个不可能满足的门槛去跑真脚本」
来验证跳过逻辑：那正是在用待验证的机制证明它自己安全，属于循环论证
（2026-11-17 我就这么干过，结果评估跑完、训练也跑了 6 步，GPU 被占 77 秒）。

顺序设计的理由：窗口可能很短，所以**先跑最便宜、最能出东西的**，
万一被中断，至少有一列数据，而不是什么都没有。

  1. 预检：空闲显存 < 门槛就整体不启动
  2. 底座 Qwen3-1.7B 基线   （4bit ≈2.2GB，最便宜）
  3. QLoRA 训练             （bs=2 ≈3.2GB，~12 分钟）
  4. 训后 LoRA 评估
  5. qwen3:8b 基线（Ollama）（5.2GB+，**默认不跑**，要 --with-8b 显式打开）
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from preflight import effective_need, should_run  # noqa: E402

PY = sys.executable
LOG_DIR = Path(r"D:\dsh-ai\lora\out\logs")
GPU_GATE = "DEAI_ALLOW_GPU"


def require_gpu_permission() -> None:
    """物理门禁：没显式授权就地退出，一个模型都不加载。"""
    if os.environ.get(GPU_GATE) != "1":
        print(f"[refuse] 未设置 {GPU_GATE}=1，拒绝执行。\n"
              f"         这台机器是用户在用的；确认可以占 GPU 后再：\n"
              f"           $env:{GPU_GATE}='1'\n"
              f"         只验证判定逻辑请跑：python tools/test_preflight.py",
              flush=True)
        raise SystemExit(2)


def free_gb() -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip()
        used, total = [float(x) for x in out.splitlines()[0].split(",")]
        return (total - used) / 1024
    except Exception:
        return None


def run(name: str, cmd: list[str], min_free: float, timeout: int) -> dict:
    """跑一个步骤；先用纯函数判定，再把日志同时落盘（别再让输出丢掉）。"""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log = LOG_DIR / f"{name}.log"

    f = free_gb()
    ok, why = should_run(name, f, min_free)
    if not ok:
        print(f"[{name}] SKIP: {why}", flush=True)
        return {"step": name, "status": "skipped", "reason": why}
    print(f"[{name}] {why}，启动：{' '.join(cmd)}", flush=True)

    t0 = time.time()
    try:
        with log.open("w", encoding="utf-8") as lf:
            p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                               timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        rc = -1
        print(f"[{name}] TIMEOUT ({timeout}s)", flush=True)
    elapsed = round(time.time() - t0, 1)

    tail = ""
    if log.exists():
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-8:])
    print(f"[{name}] rc={rc} {elapsed}s  log={log}", flush=True)
    return {"step": name, "status": "ok" if rc == 0 else "failed",
            "rc": rc, "elapsed_sec": elapsed, "log": str(log), "tail": tail}


def main() -> None:
    require_gpu_permission()          # 🔒 物理门禁，必须在最前面

    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="base",
                    help="逗号分隔：base,train,lora,8b 或 all")
    ap.add_argument("--with-8b", action="store_true",
                    help="允许跑 qwen3:8b（5.2GB+，会抢显存）")
    ap.add_argument("--min-free-gb", type=float, default=6.0)
    ap.add_argument("--train-minutes", type=float, default=25.0)
    ap.add_argument("--out", default=r"D:\dsh-ai\lora\out")
    a = ap.parse_args()

    steps = ["base", "train", "lora", "8b"] if a.steps == "all" else \
            [s.strip() for s in a.steps.split(",") if s.strip()]

    data = Path(a.out) / "data"
    train_jsonl = Path(r"D:\dsh-ai\lora\data\sft_train.jsonl")
    adapter = Path(a.out) / "qwen3-1.7b-deai"
    evald = Path(a.out) / "eval"

    plan: list[tuple[str, list[str], float, int]] = []
    for s in steps:
        if s == "base":
            plan.append(("base", [
                PY, str(HERE / "eval_rewrite.py"),
                "--tag", "base", "--label", "原始 1.7B",
                "--model", "Qwen/Qwen3-1.7B",
                "--with-n3", "--outdir", str(evald)], 2.5, 1800))
        elif s == "train":
            plan.append(("train", [
                PY, str(HERE / "train_lora.py"),
                "--data", str(train_jsonl),
                "--bs", "2", "--grad-accum", "8", "--epochs", "3",
                "--max-len", "512", "--out", str(adapter)],
                4.0, int(a.train_minutes * 60)))
        elif s == "lora":
            plan.append(("lora", [
                PY, str(HERE / "eval_rewrite.py"),
                "--tag", "lora", "--label", "训后 1.7B",
                "--model", "Qwen/Qwen3-1.7B",
                "--adapter", str(adapter),
                "--with-n3", "--outdir", str(evald)], 2.5, 1800))
        elif s == "8b":
            if not a.with_8b:
                plan.append(("8b", [], 0, 0))   # 占位，下面跳过
            else:
                plan.append(("8b", [
                    PY, str(HERE / "eval_rewrite.py"),
                    "--tag", "qwen3-8b", "--label", "qwen3:8b",
                    "--model", "qwen3:8b", "--ollama", "qwen3:8b",
                    "--with-n3", "--outdir", str(evald)], 6.0, 3600))
        else:
            print(f"[warn] 未知步骤 {s}", flush=True)

    report = []
    for name, cmd, min_free, timeout in plan:
        if not cmd:
            print("[8b] SKIP: 未加 --with-8b（8B 会占 5.2GB+ 显存）", flush=True)
            report.append({"step": name, "status": "skipped",
                           "reason": "need --with-8b"})
            continue
        # 命令行的 --min-free-gb 是**下限**，只能抬高门槛、不能降低。
        # 老 bug：它只被 parse、没参与判定 → `--min-free-gb 999` 照样开跑，
        # 「安全测试」变成真跑。修法见 preflight.effective_need 的 docstring。
        eff_min = effective_need(min_free, a.min_free_gb)
        r = run(name, cmd, eff_min, timeout)
        report.append(r)
        if r["status"] == "skipped":
            print(f"[{name}] 跳过，后续步骤仍然尝试（各自会再查显存）", flush=True)

    out = Path(a.out) / "pipeline_run.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"[saved] {out}", flush=True)
    for r in report:
        print(f"  {r['step']:6s} {r['status']:8s} "
              f"{r.get('reason') or r.get('elapsed_sec','')}", flush=True)


if __name__ == "__main__":
    main()
