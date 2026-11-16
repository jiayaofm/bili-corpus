#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""纯函数预检逻辑 —— **不 import torch、不起子进程、不碰 GPU**。

为什么要单独一个文件
--------------------
2026-11-17 我为了验证「跳过逻辑」是否生效，把门槛设成 999GB 去跑执行器，
以为它必然全部 SKIP —— 结果那个门槛根本没接进判定，评估跑完、训练也跑了 6 步，
GPU 被占了约 77 秒（用户当时正在用这台机器）。

**用一个还没验证过的机制去证明它自己安全，是循环论证。**

所以判定逻辑抽到这个文件里：它只做算术比较，没有副作用，
可以在不加载任何模型的前提下被单测覆盖（`tools/test_preflight.py`）。
真正的执行器必须先调用它，再决定要不要起进程。
"""
from __future__ import annotations


def should_run(step: str, free_gb: float | None, need_gb: float) -> tuple[bool, str]:
    """要不要跑这一步。纯函数，无副作用。

    返回 (是否运行, 理由)。只有两个理由成立才返回 False：
      · 查不到显存（free_gb is None）     → 不敢跑
      · 空闲显存 < 需要的量               → 不够跑
    """
    if free_gb is None:
        return False, "查不到显存，不敢跑"
    if need_gb <= 0:
        return True, "该步不需要 GPU"
    if free_gb < need_gb:
        return False, f"空闲 {free_gb:.2f}GB < 需要 {need_gb:.2f}GB"
    return True, f"空闲 {free_gb:.2f}GB >= 需要 {need_gb:.2f}GB"


def effective_need(step_need_gb: float, cli_floor_gb: float) -> float:
    """命令行的 `--min-free-gb` 是**下限**，只能抬高门槛，不能降低。

    老的 bug 就是这里：命令行值只被 parse、没参与判定，
    于是 `--min-free-gb 999` 依然按写死的 2.5/4.0GB 放行。
    """
    return max(step_need_gb, cli_floor_gb)
