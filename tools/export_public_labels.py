# -*- coding: utf-8 -*-
"""
导出去敏的公开标注集

目的：让仓库的读者能在不采集、不重标的前提下复现分类实验。

脱敏规则：
  · @用户名        → @用户       （保留 @ 结构，不破坏「有@」特征）
  · http(s) 链接   → <链接>      （避免暴露个人主页）
  · 去除 user / mid / rpid       （B站昵称与用户ID）
  · 保留 bvid（视频公开编号）、domain、label、strength

不发布全量 1008 条原始语料的理由见 README「伦理与隐私声明」。
"""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "data" / "corpus.jsonl"
GT = ROOT / "labeling" / "ground_truth.csv"
OUT_DIR = ROOT / "data" / "sample"
OUT = OUT_DIR / "labeled_public.csv"

MENTION_RE = re.compile(r"@[^\s@]{1,20}")
URL_RE = re.compile(r"https?://\S+")
# B站表情标签 [xxx] 不是 PII，保留


def desensitize(t: str) -> str:
    t = URL_RE.sub("<链接>", t)
    t = MENTION_RE.sub("@用户", t)
    return t


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus = {}
    for line in CORPUS.open(encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            corpus[r["rpid"]] = r

    rows = list(csv.DictReader(GT.open(encoding="utf-8-sig", newline="")))
    out, missing, masked_m, masked_u = [], 0, 0, 0
    for r in rows:
        c = corpus.get(r["rpid"])
        if not c:
            missing += 1
            continue
        raw = c["content"]
        clean = desensitize(raw)
        masked_m += len(MENTION_RE.findall(raw))
        masked_u += len(URL_RE.findall(raw))
        out.append({
            "id": len(out) + 1,
            "bvid": c.get("bvid", ""),
            "domain": c.get("domain", ""),
            "label": r["label"],
            "strength": r["strength"],
            "content": clean,
        })

    with OUT.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["id", "bvid", "domain", "label",
                                          "strength", "content"])
        w.writeheader()
        for row in out:
            w.writerow(row)

    pos = sum(1 for r in out if int(r["label"]) == 1)
    print(f"输出 {OUT}  共 {len(out)} 条")
    print(f"  正类 {pos} / 负类 {len(out)-pos}")
    print(f"  脱敏：@用户名 {masked_m} 处 → @用户；链接 {masked_u} 处 → <链接>")
    if missing:
        print(f"  [警告] {missing} 条在语料中找不到")
    print("  分区分布：", dict(Counter(r["domain"] for r in out)))

    # 校验：确认没有残留的 @真实用户名 或链接（@用户 是脱敏后的占位，不算残留）
    residual_re = re.compile(r"@(?!用户)[^\s@]{1,20}")
    left_m = sum(len(residual_re.findall(r["content"])) for r in out)
    left_u = sum(len(URL_RE.findall(r["content"])) for r in out)
    print(f"\n  残留检查：@真实用户名 {left_m} 处，链接 {left_u} 处"
          f"  → {'通过' if left_m == 0 and left_u == 0 else '仍有残留'}")


if __name__ == "__main__":
    main()
