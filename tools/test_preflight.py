#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`preflight.py` 的单测 —— 不加载模型、不碰 GPU、不起子进程。

跑法：python tools/test_preflight.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preflight import effective_need, should_run  # noqa: E402

FAILS: list[str] = []


def check(name: str, got, want) -> None:
    ok = got == want
    print(("  PASS " if ok else "  FAIL ") + name +
          ("" if ok else f"\n        got={got!r} want={want!r}"))
    if not ok:
        FAILS.append(name)


def test_should_run() -> None:
    print("should_run:")
    check("查不到显存 -> 不跑", should_run("base", None, 2.5)[0], False)
    check("显存刚好够 -> 跑", should_run("base", 2.5, 2.5)[0], True)
    check("显存差一点 -> 不跑", should_run("base", 2.49, 2.5)[0], False)
    check("显存远超 -> 跑", should_run("base", 8.0, 2.5)[0], True)
    check("门槛为 0 -> 跑", should_run("x", 0.0, 0)[0], True)
    # 关键回归：正是这次事故的形状
    check("阈值 999GB -> 必须不跑", should_run("base", 6.03, 999.0)[0], False)


def test_effective_need() -> None:
    print("effective_need（命令行值只能抬高门槛）:")
    check("步内 2.5, 命令行 6 -> 6", effective_need(2.5, 6.0), 6.0)
    check("步内 4.0, 命令行 6 -> 6", effective_need(4.0, 6.0), 6.0)
    check("步内 4.0, 命令行 0 -> 4", effective_need(4.0, 0.0), 4.0)
    check("步内 2.5, 命令行 999 -> 999", effective_need(2.5, 999.0), 999.0)


def test_the_bug_shape() -> None:
    """复现那次事故：修好之后 999 必须让每一步都 SKIP。"""
    print("事故形状回归（999 必须全跳过）:")
    for step, step_need in [("base", 2.5), ("train", 4.0), ("lora", 2.5)]:
        need = effective_need(step_need, 999.0)
        ok, why = should_run(step, 6.03, need)
        check(f"{step} 必须不跑", ok, False)


def main() -> None:
    test_should_run()
    test_effective_need()
    test_the_bug_shape()
    print()
    if FAILS:
        print(f"FAILED: {len(FAILS)} -> {FAILS}")
        raise SystemExit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
