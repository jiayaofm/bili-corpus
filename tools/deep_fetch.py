# -*- coding: utf-8 -*-
"""
深采评论：把现有 7 个视频的评论从「只取 4.2%」扩到尽可能全。

## 为什么扩

实测这 7 个视频的真实评论总量是 **25188 条**，我们手上只有 **1053 条（4.2%）**。
而且原来那批是 `--max-comments 160` 按热度取的，**只保留了高赞**，
对分类和生成都是系统性偏差。

## 两点设计

**① 同时采 hot 和 time**

    --sort hot   热度序（原来是这个，有偏差）
    --sort time  时间序（更接近真实分布）

两份都采，就能量化「只看高赞」到底偏了多少 —— 这个数字写进报告很关键。

**② 断点续爬**

每页落一个文件 `data/deep/<BV>-<sort>/p<NNNN>.json`，已存在就跳过。
中断后重跑不会重复请求。

## 限额

插件有读预算：600/时、3000/天（按**请求**计，不是条数）。
`ps=20` → 一页 20 条 → 600 页/时 = 12000 条/时。
脚本会在接近预算时报错退出，此时隔一小时再跑即可续上。

## 用法

    python tools/deep_fetch.py                    # 全部视频，hot 序
    python tools/deep_fetch.py --sort both        # hot 和 time 都采
    python tools/deep_fetch.py --max-pages 50     # 每个视频最多 50 页（1000 条）
    python tools/deep_fetch.py --merge            # 采完后合并成 data/deep/<BV>.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BILI = Path(r"E:\DSH\dsh-bili-whale\tools\bili.mjs")
NODE = Path(r"C:\Program Files\nodejs\node.exe")
DEEP = ROOT / "data" / "deep"

sys.path.insert(0, str(ROOT / "tools"))
from fetch_materials import ensure_fake_home  # 复用假 home（沙箱限制）

PS = 20                    # B站 接口上限，实测 50 无效
GAP_SEC = 0.7              # 请求间隔，别顶风控


def safe(s: str) -> str:
    """清掉孤立代理字符。

    B站 返回的评论里 emoji 有时被截成半个（比如 \\ud83d 单独出现），
    json 能解析成 str，但往 UTF-8 文件写就抛 UnicodeEncodeError。
    """
    return (s or "").encode("utf-8", "replace").decode("utf-8")


def call(env: dict, bv: str, pn: int, sort: str) -> dict | None:
    cmd = [str(NODE), str(BILI), "comments", bv,
           "--ps", str(PS), "--pn", str(pn), "--sort", sort, "--json"]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=120, env=env)
    out = (p.stdout or "").strip()
    if not out.startswith("{"):
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def fetch_video(bv: str, sort: str, env: dict, max_pages: int | None,
                budget: dict) -> dict:
    """采一个视频的一个排序。返回统计。"""
    outdir = DEEP / f"{bv}-{sort}"
    outdir.mkdir(parents=True, exist_ok=True)

    got = 0
    requests = 0
    total = None
    failed = 0

    # 先探第一页拿 total
    first_path = outdir / "p0001.json"
    if first_path.exists():
        d = json.loads(first_path.read_text(encoding="utf-8"))
    else:
        d = call(env, bv, 1, sort)
        requests += 1
        if d and d.get("ok"):
            first_path.write_text(json.dumps(d, ensure_ascii=True), encoding="utf-8")

    if not d or not d.get("ok"):
        return {"bv": bv, "sort": sort, "ok": False, "got": 0, "requests": requests,
                "total": 0, "pages": 0}

    total = int(d.get("total") or 0)
    pages_needed = math.ceil(total / PS) if total else 1
    if max_pages:
        pages_needed = min(pages_needed, max_pages)

    got += len(d.get("items") or [])

    for pn in range(2, pages_needed + 1):
        if budget["left"] <= 0:
            break
        path = outdir / f"p{pn:04d}.json"
        if path.exists():
            try:
                got += len(json.loads(path.read_text(encoding="utf-8")).get("items") or [])
            except json.JSONDecodeError:
                pass
            continue

        d = call(env, bv, pn, sort)
        requests += 1
        budget["left"] -= 1
        if not d or not d.get("ok"):
            failed += 1
            if failed >= 3:      # 连续三次失败基本是风控了，停
                break
            time.sleep(3)
            continue
        failed = 0
        items = d.get("items") or []
        if not items:
            break
        path.write_text(json.dumps(d, ensure_ascii=True), encoding="utf-8")
        got += len(items)
        time.sleep(GAP_SEC)

    return {"bv": bv, "sort": sort, "ok": True, "got": got,
            "requests": requests, "total": total, "pages": pages_needed}


def merge(bv: str, sort: str) -> int:
    """把分页文件合成一个 jsonl。"""
    outdir = DEEP / f"{bv}-{sort}"
    if not outdir.exists():
        return 0
    seen: set[str] = set()
    rows: list[dict] = []
    for p in sorted(outdir.glob("p*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        for it in d.get("items") or []:
            rp = str(it.get("rpid") or "")
            if not rp or rp in seen:
                continue
            seen.add(rp)
            rows.append({
                "rpid": rp,
                "content": safe((it.get("text") or "").strip()),
                "user": safe(it.get("user") or ""),
                "mid": it.get("mid") or "",
                "like": it.get("like") or 0,
                "reply_count": it.get("replies") or 0,
                "ctime": it.get("time") or "",
                "video_title": safe(d.get("title") or ""),
                "sort": sort,
                "sub_replies": [],
            })
    out = DEEP / f"{bv}.jsonl" if sort == "hot" else DEEP / f"{bv}-{sort}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--sort", choices=["hot", "time", "both"], default="hot")
    ap.add_argument("--max-pages", type=int, default=None, help="每视频每排序最多几页")
    ap.add_argument("--budget", type=int, default=500, help="本次最多发多少请求")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--merge", action="store_true", help="采完后合并")
    ap.add_argument("bvs", nargs="*")
    a = ap.parse_args(argv)

    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    bvs = a.bvs or [v["bvid"] for v in cfg["videos"]]
    if "BV1Sbnz6EEix" not in bvs and (ROOT / "data" / "BV1Sbnz6EEix_like.jsonl").exists():
        bvs.append("BV1Sbnz6EEix")

    sorts = ["hot", "time"] if a.sort == "both" else [a.sort]
    DEEP.mkdir(parents=True, exist_ok=True)

    if not a.merge_only:
        env = ensure_fake_home()
        budget = {"left": a.budget}
        print(f"视频 {len(bvs)} 个 × 排序 {sorts}，本次请求预算 {a.budget}\n")
        print(f"{'BV':<14} {'排序':<5} {'总评论':>7} {'已采':>7} {'请求':>5} {'页':>5}")
        print("-" * 52)
        for bv in bvs:
            for s in sorts:
                if budget["left"] <= 0:
                    print(f"{bv:<14} {s:<5} {'':>7} {'':>7} {'预算用完，停止':>14}")
                    continue
                r = fetch_video(bv, s, env, a.max_pages, budget)
                print(f"{r['bv']:<14} {r['sort']:<5} {r['total']:>7} {r['got']:>7} "
                      f"{r['requests']:>5} {r['pages']:>5}")
        print("-" * 52)
        print(f"剩余预算 {budget['left']}")

    if a.merge or a.merge_only:
        print("\n合并中…")
        for bv in bvs:
            for s in sorts:
                n = merge(bv, s)
                if n:
                    print(f"  {bv} {s}: {n} 条")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
