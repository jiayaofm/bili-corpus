# -*- coding: utf-8 -*-
"""
把大肥鱼产出的「改写对」转成训练数据（两个变体）。

## 输入

`data/whale_comments/*.jsonl`，每行一条视频：

```json
{
  "bvid": "BV...", "title": "...", "summary": "...",
  "comments": ["任务一：它自己写的新评论"],
  "real_comments": ["真人原话"],
  "rewrites": [{"origin": "真人原话（逐字）", "rewrite": "鱼腔改写"}]
}
```

## 输出

```
data/rewrite/a_train.jsonl   data/rewrite/a_dev.jsonl
data/rewrite/b_train.jsonl   data/rewrite/b_dev.jsonl
```

每行 `{"messages": [{role, content}...], "_meta": {...}}`，
和 `train_lora.py` 直接对接（它用 `load_dataset("json", ...)` 读）。

## 两个变体的区别（只有 user 内容不同）

```
A（短）  这条评论：{鱼腔改写}
         改一下。

B（长）  视频《{title}》讲的是：{summary}
         有人这样评论：{鱼腔改写}
         按真人网友的口吻重写，保留它说的具体内容。
```

A 省 token、纯风格迁移；B 给上下文、改写时不容易把内容改飞。
两个都训，对比着看哪个好——这是待验证的问题，不预设答案。

**注意方向**：输入是 `rewrite`（鱼腔），输出是 `origin`（真人原话）。
不是反过来——我们要学的是「把鱼腔压回真人腔」。

## 校验（脏数据不进训练集）

| 检查 | 处理 |
|---|---|
| `origin` 在 `real_comments` 里逐字存在 | 查不到 → 丢弃并计数 |
| `rewrite` 非空、`origin` 非空 | 空 → 丢弃 |
| 长度比 `len(rewrite)/len(origin)` 在 0.5~1.3 | 超出 → 丢弃（改写没压缩/没膨胀到位） |
| 完全重复的 (origin, rewrite) | 去重 |
| `rewrite == origin`（没改） | 丢弃（那不是改写，是复读） |

所有丢弃都记在报告里，**不静默吞掉**。

## 切分

**按视频切**，不是按样本随机切。同一个视频的样本不能同时出现在 train 和 dev，
否则 dev 上的数字是自欺欺人（这个坑项目里踩过一次）。

默认按视频 80/20 切，seed 固定。
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "rewrite"

SYS = "你把 AI 写的 B站 评论改成像真人网友写的。"

TMPL_A = "这条评论：{x}\n\n改一下。"
TMPL_B = ("视频《{title}》讲的是：{summary}\n\n"
          "有人这样评论：{x}\n\n"
          "按真人网友的口吻重写，保留它说的具体内容。")

# 长度比只用来剔掉极端值，不做严格筛选——压缩得狠但内容没丢，那是好样本。
# 0.5 会把「56 字压到 26 字」这种合理改写错杀（实测发生过）。
# 长度比。v4 任务书要求「只换说法、不换信息量」，所以改写前后字数应当接近。
#
# 旧值是 0.35~1.6，等于放行「压成三分之一」的样本。那种数据教出来的是
# 「把短话变长」，而线上输入本来就是长评论，模型就无事可做——实测 49% 的
# 输出只是小改。新值按 v4 的 80%~120% 收紧，留一点余量。
MIN_RATIO, MAX_RATIO = 0.7, 1.35


def load_videos(src_glob: str | None = None) -> list[dict]:
    pattern = src_glob or str(ROOT / "data" / "whale_comments" / "*.jsonl")
    files = sorted(glob.glob(pattern))
    vids = []
    skip: Counter = Counter()
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("skipped"):
                    skip["该视频标了 skipped"] += 1
                    continue
                if not r.get("bvid"):
                    skip["没有 bvid"] += 1
                    continue
                if not (r.get("rewrites") or []):
                    skip["没有 rewrites 字段"] += 1
                    continue
                vids.append(r)
    return vids, skip


def _bigrams(text: str) -> set[str]:
    """去掉标点后的字符 bigram 集合，用来量「改了没有」。"""
    t = re.sub(r"[^\u4e00-\u9fff0-9A-Za-z]", "", text)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def _cjk_ratio(text: str) -> float:
    n = len(re.sub(r"\s", "", text))
    return len(re.findall(r"[\u4e00-\u9fff]", text)) / max(1, n)


def build_pairs(vids: list[dict], min_len: int = 8,
                max_jaccard: float = 0.5) -> tuple[list[dict], Counter, list[float]]:
    """校验、过滤、去重。

    过滤这件事是必须的。第一版没过滤，跑出来的模型学成了「把句号换成逗号」——
    因为数据里 52% 的配对相似度 ≥0.5，等于一半样本在教「加个句号」。
    那一版内容保留 0.845、鱼腥味 0.256，看着像成功，实际上一个字都没真正改写。
    """
    drop: Counter = Counter()
    seen: set[tuple[str, str]] = set()
    out: list[dict] = []
    ratios: list[float] = []

    for v in vids:
        bv = v.get("bvid") or ""
        title = (v.get("title") or "").strip().strip("《》")
        summ = (v.get("summary") or "").strip()
        real = [(c or "").strip() for c in (v.get("real_comments") or [])]
        real_set = set(real)

        if not summ:
            drop["概要为空（B 变体没法用）"] += 1
            continue

        for rw in v.get("rewrites") or []:
            origin = (rw.get("origin") or "").strip()
            rewrite = (rw.get("rewrite") or "").strip()

            if not origin or not rewrite:
                drop["原文或改写为空"] += 1
                continue
            if origin not in real_set:
                drop["origin 在 real_comments 里查无此句"] += 1
                continue
            if origin == rewrite:
                drop["改写成原文本身（没改）"] += 1
                continue
            if "http" in origin or "www." in origin:
                drop["原文是链接（不是评论）"] += 1
                continue
            if _cjk_ratio(origin) < 0.7:
                drop["原文不是中文（多半是翻译）"] += 1
                continue
            if len(origin) < min_len:
                drop["原文太短，学不到东西"] += 1
                continue

            A, B = _bigrams(origin), _bigrams(rewrite)
            jac = len(A & B) / len(A | B) if (A | B) else 1.0
            if jac >= max_jaccard:
                drop[f"改得不够（相似度 ≥{max_jaccard}，基本只动标点）"] += 1
                continue

            ratio = len(rewrite) / max(1, len(origin))
            if not (MIN_RATIO <= ratio <= MAX_RATIO):
                drop[f"长度比超出 {MIN_RATIO}~{MAX_RATIO}（极端）"] += 1
                continue
            ratios.append(ratio)

            key = (origin, rewrite)
            if key in seen:
                drop["重复对"] += 1
                continue
            seen.add(key)

            out.append({"bvid": bv, "title": title, "summary": summ,
                        "x": rewrite, "y": origin, "jac": jac})
    return out, drop, ratios


def make_sample(kind: str, p: dict) -> dict:
    if kind == "a":
        user = TMPL_A.format(x=p["x"])
    else:
        user = TMPL_B.format(title=p["title"], summary=p["summary"], x=p["x"])
    return {
        "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": user},
            {"role": "assistant", "content": p["y"]},
        ],
        "_meta": {
            "bvid": p["bvid"], "variant": kind,
            "chars_x": len(p["x"]), "chars_y": len(p["y"]),
            "title": p["title"],
            # 干净的鱼腔原文 / 真人目标，供评测拟合鱼腥味检测器用
            # （不要从 messages 里抠：B 变体的 user 混了视频摘要，会污染检测器）
            "x": p["x"], "y": p["y"],
        },
    }


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-ratio", type=float, default=0.2, help="按视频留出的比例")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-per-video", type=int, default=0,
                    help=">0 时每个视频最多取 N 条，防单个视频主导")
    ap.add_argument("--min-len", type=int, default=8, help="原文最少字数")
    ap.add_argument("--max-jaccard", type=float, default=0.5,
                    help="相似度上限，超过就认为「基本只改了标点」")
    ap.add_argument("--src", default=None, help="指定输入 glob（默认 data/whale_comments/*.jsonl）")
    ap.add_argument("--dry-run", action="store_true", help="只出报告不写文件")
    a = ap.parse_args()

    vids, skipped = load_videos(a.src)
    print("=" * 66)
    print("  构造改写配对数据集")
    print("=" * 66)
    if not vids:
        print("  ❌ 没有可用数据：data/whale_comments/*.jsonl 里没有带 rewrites 的记录")
        print("     → 等大肥鱼跑完 v3 任务书后重跑本脚本")
        return 2
    print(f"  视频 {len(vids)} 个")
    if skipped:
        print("  跳过的记录：")
        for k, n in skipped.most_common():
            print(f"    {n:>5} 个视频  {k}")

    pairs, drop, ratios = build_pairs(vids, a.min_len, a.max_jaccard)
    print(f"  可用配对 {len(pairs)} 条")
    if ratios:
        rs = sorted(ratios)
        print(f"  长度比  中位 {rs[len(rs)//2]:.2f}  范围 {rs[0]:.2f}~{rs[-1]:.2f}"
              f"（1.0 = 改写和原文一样长）")
    if drop:
        print("  丢弃明细：")
        for k, n in drop.most_common():
            print(f"    {n:>5} 条  {k}")
    if not pairs:
        print("  ❌ 一条可用配对都没有，检查 rewrites 的 origin 是否逐字复制了 real_comments")
        return 2

    # 每个视频限流
    if a.max_per_video > 0:
        by: dict[str, list[dict]] = {}
        for p in pairs:
            by.setdefault(p["bvid"], []).append(p)
        pairs = [p for g in by.values() for p in g[:a.max_per_video]]
        print(f"  每视频限 {a.max_per_video} 条后 {len(pairs)} 条")

    # 按视频切分
    bvids = sorted({p["bvid"] for p in pairs})
    rng = random.Random(a.seed)
    rng.shuffle(bvids)
    n_dev = max(1, round(len(bvids) * a.dev_ratio)) if len(bvids) >= 3 else 0
    dev_set = set(bvids[:n_dev]) if n_dev else set()

    if len(bvids) < 5:
        print(f"  ⚠️ 只有 {len(bvids)} 个视频，dev 切分会很不稳；"
              f"建议视频数 ≥10 再看指标")
    if n_dev == 0:
        print("  ⚠️ 视频太少，不做 dev 切分，全部进 train")

    train = [p for p in pairs if p["bvid"] not in dev_set]
    dev = [p for p in pairs if p["bvid"] in dev_set]

    print()
    print(f"  按视频切分（seed={a.seed}）：")
    print(f"    train  {len(train):>5} 条 / {len({p['bvid'] for p in train})} 个视频")
    print(f"    dev    {len(dev):>5} 条 / {len(dev_set)} 个视频")
    print(f"    正样本长度  中位 {sorted(len(p['x']) for p in pairs)[len(pairs)//2]} 字（鱼腔）")
    print(f"    目标长度    中位 {sorted(len(p['y']) for p in pairs)[len(pairs)//2]} 字（真人）")

    if a.dry_run:
        print()
        print("  --dry-run：不写文件")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    for kind in ("a", "b"):
        for name, subset in (("train", train), ("dev", dev)):
            fp = OUT / f"{kind}_{name}.jsonl"
            with fp.open("w", encoding="utf-8") as f:
                for p in subset:
                    f.write(json.dumps(make_sample(kind, p), ensure_ascii=False) + "\n")
            print(f"  → {fp.relative_to(ROOT)}  {len(subset)} 条")

    # 抽一条给人看
    print()
    print("  样例（B 变体）：")
    s = make_sample("b", pairs[0])
    for m in s["messages"]:
        body = m["content"].replace("\n", "\n      ")
        print(f"    [{m['role']}] {body}")
    print()
    print("  下一步：")
    print("    python tools/train_lora.py --data data/rewrite/b_train.jsonl \\")
    print("        --out D:\\dsh-ai\\lora\\out\\qwen3-1.7b-fish --max-len 512")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
