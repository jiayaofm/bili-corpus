# -*- coding: utf-8 -*-
"""区分「真的被截断」和「真人本来就不加句号」。

max_new=96 token，中文大约 70~96 字。如果输出长度大量堆在 90 字以上，
说明是被上限卡住的；如果分布正常，那结尾没句号是模型学到了真人习惯。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")

rows = [json.loads(l) for l in open(ROOT / "state" / "preds" / "fish-a2.jsonl",
                                    encoding="utf-8")]
L = sorted(len(r["pred"]) for r in rows)
Y = sorted(len(r["y"]) for r in rows)
print(f"  dev {len(rows)} 条")
print(f"  输出长度  中位 {L[len(L)//2]}  P90 {L[int(len(L)*0.9)]}  最大 {L[-1]}")
print(f"  真人目标  中位 {Y[len(Y)//2]}  P90 {Y[int(len(Y)*0.9)]}  最大 {Y[-1]}")
print()
near96 = sum(1 for v in L if v >= 90)
print(f"  输出 ≥90 字的：{near96} 条（{near96/len(L)*100:.0f}%）← 这些可能被 96 token 卡住")
print()
print("  最长的 5 条输出：")
for r in sorted(rows, key=lambda r: -len(r["pred"]))[:5]:
    p = r["pred"]
    tail = p[-14:]
    print(f"    {len(p):3d}字  结尾「{tail}」")
