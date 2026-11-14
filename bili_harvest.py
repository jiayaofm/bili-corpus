#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B站评论语料采集器 v2 —— 带断点续传与 412 风控处理

v2 相对 v1 的改动：
  1. 断点续传：每页成功后把进度写进 .state.json，中断后重跑从下一页继续，
     不会重头再抓（重头抓是触发 412 的主因之一）。
  2. 412 专门处理：识别安全风控，做三级长冷却（60s / 180s / 600s）后再试，
     仍失败则保存进度并优雅退出，而不是把进度丢掉。
  3. 节奏可调：默认每页间隔 2.5–4.5 秒，比 v1 温和得多。
  4. 优先用 curl_cffi（如果装了）：它能模拟浏览器 TLS 指纹，显著降低 412 概率。

依赖：
    pip install bilibili-api-python aiohttp
    pip install curl_cffi        # 可选但强烈建议

用法：
    python bili_harvest.py fetch --bvid BV1YxWx69Ens --order time --max-comments 2000
    python bili_harvest.py fetch --bvid-list bvids.txt --order like
    python bili_harvest.py stats
"""

import argparse
import asyncio
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


def _preflight():
    have_api = True
    try:
        import bilibili_api  # noqa: F401
    except ImportError:
        have_api = False
    http_lib = None
    for m in ("curl_cffi", "httpx", "aiohttp"):
        try:
            __import__(m)
            http_lib = m
            break
        except ImportError:
            continue
    if have_api and http_lib:
        print(f"[依赖] bilibili-api OK, HTTP 客户端 = {http_lib}")
        return
    pkgs = ([] if have_api else ["bilibili-api-python"]) + ([] if http_lib else ["aiohttp"])
    sys.exit(f"缺少依赖：pip install {' '.join(pkgs)}")


_preflight()

from bilibili_api import comment, video, Credential  # noqa: E402

CRED_CANDIDATES = [
    ROOT / "bili_credential.json",
    Path.home() / ".dsh" / "bili-duty" / "bili_credential.json",
]

ORDER_MAP = {"like": comment.OrderType.LIKE, "time": comment.OrderType.TIME}

# 412 风控的三级冷却（秒）
COOLDOWN_LADDER = (60, 180, 600)


def load_credential() -> Credential:
    for p in CRED_CANDIDATES:
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            if d.get("sessdata"):
                return Credential(
                    sessdata=d["sessdata"],
                    bili_jct=d.get("bili_jct", ""),
                    buvid3=d.get("buvid3", "") or "",
                    dedeuserid=d.get("dedeuserid", ""),
                )
    sys.exit("未找到凭证，请先运行 bili_login_manual.py 扫码登录。")


def is_rate_limited(err: Exception) -> bool:
    s = str(err)
    return "412" in s or "429" in s or "风控" in s or "security control" in s


async def get_video_meta(bvid: str, cred: Credential) -> dict:
    info = await video.Video(bvid=bvid, credential=cred).get_info()
    st = info.get("stat", {})
    return {
        "bvid": info.get("bvid"), "aid": info.get("aid"),
        "title": info.get("title"), "owner": info.get("owner", {}).get("name"),
        "tid": info.get("tid"), "duration": info.get("duration"),
        "pubdate": info.get("pubdate"),
        "reply": st.get("reply", 0), "view": st.get("view", 0),
        "desc": (info.get("desc") or "")[:300],
    }


def normalize(r: dict, meta: dict, order: str) -> dict:
    member = r.get("member", {}) or {}
    content = r.get("content", {}) or {}
    subs = []
    for s in (r.get("replies") or [])[:5]:
        sm = s.get("member", {}) or {}
        sc = s.get("content", {}) or {}
        subs.append({"user": sm.get("uname", ""), "content": sc.get("message", ""),
                     "like": s.get("like", 0)})
    return {
        "content": content.get("message", ""),
        "bvid": meta["bvid"], "aid": meta["aid"],
        "video_title": meta["title"], "video_tid": meta["tid"],
        "video_owner": meta["owner"], "order": order,
        "like": r.get("like", 0), "reply_count": r.get("rcount", 0),
        "rpid": str(r.get("rpid", 0)),
        "user": member.get("uname", ""), "mid": str(member.get("mid", "")),
        # 等级：可能存在 level_info 为 null 的情况，逐层兜底
        "level": ((member.get("level_info") or {}).get("current_level")
                  if isinstance(member.get("level_info"), dict) else None),
        "ctime": r.get("ctime", 0),
        "sub_replies": subs,
    }


async def harvest_one(bvid: str, order: str, max_comments: int, max_pages: int,
                      delay: tuple, cred: Credential, strip_user: bool) -> int:
    out_path = DATA / f"{bvid}_{order}.jsonl"
    state_path = DATA / f"{bvid}_{order}.state.json"
    meta_path = DATA / f"{bvid}_meta.json"

    # ---- 断点读取（用 utf-8-sig 容忍 BOM：PowerShell / 记事本保存的文件常带 BOM）----
    state = {"last_page": 0, "count": 0, "done": False}
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8-sig"))
        except Exception as e:
            print(f"[警告] state 文件损坏（{e}），从第 1 页重新开始")
            state = {"last_page": 0, "count": 0, "done": False}
    if state.get("done"):
        print(f"[已完成] {bvid} {order} 共 {state['count']} 条，跳过")
        return state["count"]

    meta = None
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8-sig"))
        except Exception:
            meta = None
    if meta is None:
        print(f"[视频] {bvid} 取元数据 ...")
        try:
            meta = await get_video_meta(bvid, cred)
        except Exception as e:
            print(f"  [失败] {e}")
            return state["count"]
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"  {meta['title'][:56]}")
    print(f"  分区={meta['tid']}  评论总数={meta['reply']}")

    seen = set()
    if out_path.exists():
        for line in out_path.open(encoding="utf-8"):
            if line.strip():
                try:
                    seen.add(json.loads(line)["rpid"])
                except Exception:
                    pass

    got = state["count"]
    page = state["last_page"] + 1
    if page > 1:
        print(f"  [续传] 已有 {got} 条，从第 {page} 页继续")

    cooldown_idx = 0
    with out_path.open("a" if page > 1 else "w", encoding="utf-8") as f:
        while got < max_comments and page <= max_pages:
            replies = None
            for attempt in range(3):
                try:
                    resp = await comment.get_comments(
                        oid=meta["aid"],
                        type_=comment.CommentResourceType.VIDEO,
                        page_index=page,
                        order=ORDER_MAP[order],
                        credential=cred,
                    )
                    replies = resp.get("replies") or []
                    cooldown_idx = 0
                    break
                except Exception as e:
                    if is_rate_limited(e):
                        wait = COOLDOWN_LADDER[min(cooldown_idx, len(COOLDOWN_LADDER) - 1)]
                        cooldown_idx += 1
                        print(f"    [412 风控] 冷却 {wait}s 后重试 ({attempt+1}/3)")
                        await asyncio.sleep(wait)
                    else:
                        wait = (2 ** attempt) + random.uniform(0, 1.5)
                        print(f"    [重试 {attempt+1}/3] {type(e).__name__}: {str(e)[:80]} 等待 {wait:.1f}s")
                        await asyncio.sleep(wait)

            if replies is None:
                print(f"  page={page} 连续失败，保存进度退出")
                state_path.write_text(json.dumps({**state, "last_page": page - 1, "count": got},
                                                 ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[中断] {bvid} {order} 已存 {got} 条，下次重跑本命令会从这里继续")
                return got

            if not replies:
                print(f"  page={page} 无数据，到达末尾")
                state["done"] = True
                break

            new = 0
            for r in replies:
                rec = normalize(r, meta, order)
                if rec["rpid"] in seen:
                    continue
                seen.add(rec["rpid"])
                if strip_user:
                    rec["user"] = ""; rec["mid"] = ""
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                got += 1
                new += 1
                if got >= max_comments:
                    break
            f.flush()

            print(f"  page={page:>3} +{new:>2} 累计 {got}")
            state.update(last_page=page, count=got)
            state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

            if new == 0:
                print("  本页全是重复，停止")
                state["done"] = True
                break
            page += 1
            await asyncio.sleep(random.uniform(*delay))

    if got >= max_comments:
        state["done"] = True
    state.update(count=got, last_page=page - 1)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[完成] {bvid} {order} 共 {got} 条 -> {out_path.name}")
    return got


def stats():
    files = sorted(DATA.glob("*_*.jsonl"))
    if not files:
        print("data/ 下还没有语料")
        return
    all_recs = []
    for fp in files:
        recs = [json.loads(l) for l in fp.open(encoding="utf-8") if l.strip()]
        all_recs.extend(recs)
        if not recs:
            continue
        lens = [len(r["content"]) for r in recs]
        print(f"\n{fp.name}: {len(recs)} 条  平均 {sum(lens)/len(lens):.1f} 字  "
              f"中位 {sorted(lens)[len(lens)//2]} 字")
    if not all_recs:
        return
    print("\n" + "=" * 56)
    print(f"总计 {len(all_recs)} 条")
    print("=" * 56)
    for lo, hi in [(1, 5), (6, 15), (16, 30), (31, 60), (61, 120), (121, 10**9)]:
        n = sum(1 for r in all_recs if lo <= len(r["content"]) <= hi)
        pct = n / len(all_recs) * 100
        label = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
        print(f"  {label:>8} 字  {n:>5} ({pct:>5.1f}%) {'█' * int(pct / 2)}")


def main():
    ap = argparse.ArgumentParser(description="B站评论语料采集 v2")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch")
    f.add_argument("--bvid", action="append", default=[])
    f.add_argument("--bvid-list")
    f.add_argument("--order", choices=["like", "time", "both"], default="both")
    f.add_argument("--max-comments", type=int, default=2000)
    f.add_argument("--max-pages", type=int, default=120)
    f.add_argument("--delay", type=float, nargs=2, default=[2.5, 4.5])
    f.add_argument("--strip-user", action="store_true")

    sub.add_parser("stats")

    args = ap.parse_args()
    if args.cmd == "stats":
        stats()
        return

    bvids = list(args.bvid)
    if args.bvid_list:
        bvids += [l.strip() for l in open(args.bvid_list, encoding="utf-8")
                  if l.strip() and not l.startswith("#")]
    if not bvids:
        sys.exit("需要 --bvid 或 --bvid-list")

    orders = ["like", "time"] if args.order == "both" else [args.order]
    cred = load_credential()

    async def run():
        for bv in bvids:
            for od in orders:
                await harvest_one(bv, od, args.max_comments, args.max_pages,
                                  tuple(args.delay), cred, args.strip_user)
                await asyncio.sleep(random.uniform(5, 12))

    asyncio.run(run())
    print()
    stats()


if __name__ == "__main__":
    main()
