# -*- coding: utf-8 -*-
"""
s02 语料构建

合并 data/*_{order}.jsonl，按 rpid 去重，按 bvid 从 config 补 domain 字段，
输出 data/corpus.jsonl。

corpus.jsonl 每行：
  rpid, bvid, domain, video_title, content, user, like, ctime, level
"""
import json

import common
from common import CFG, P, banner, info, read_jsonl, record, write_jsonl

STAGE = "s02_corpus"
OUT_REL = "data/corpus.jsonl"


def main(force=False):
    banner("s02 语料构建")

    bvid2domain = {v["bvid"]: v["domain"] for v in CFG["videos"]}
    bvid2note = {v["bvid"]: v.get("note", "") for v in CFG["videos"]}
    order = CFG["harvest"]["order"]

    files = sorted(P["data"].glob(f"*_{order}.jsonl"))
    # 深采（tools/deep_fetch.py）的产物：data/deep/<BV>.jsonl
    # 早期每个视频只取 160 条，深采后能到 2000 条。两路合并、按 rpid 去重。
    files += sorted((P["data"] / "deep").glob("*.jsonl")) if (P["data"] / "deep").is_dir() else []
    if not files:
        info("[失败] data/ 下没有抓到任何 jsonl，请先跑 s01")
        return 1

    seen = set()
    rows = []
    per_video = {}
    skipped_dup = 0

    for fp in files:
        # data/BV1xxx_like.jsonl → BV1xxx ；data/deep/BV1xxx.jsonl → BV1xxx
        # 深采还有 <BV>-time.jsonl（时间序对照样本），不并入主语料，避免和热度序重复
        if fp.name.endswith("-time.jsonl"):
            continue
        bvid = fp.name.split("_")[0]
        if bvid.endswith(".jsonl"):
            bvid = bvid[:-6]
        n = 0
        for r in read_jsonl(fp):
            rpid = str(r.get("rpid", ""))
            if not rpid or rpid in seen:
                skipped_dup += 1
                continue
            seen.add(rpid)
            rows.append({
                "rpid": rpid,
                "bvid": r.get("bvid", bvid),
                "domain": bvid2domain.get(bvid, "未知"),
                "video_title": r.get("video_title", ""),
                "video_note": bvid2note.get(bvid, ""),
                "content": r.get("content", ""),
                "user": r.get("user", ""),
                "like": r.get("like", 0),
                "ctime": r.get("ctime", 0),
                "level": r.get("level"),
            })
            n += 1
        per_video[bvid] = n
        info(f"{fp.name:<28} {n:>4} 条")

    out = common.ROOT / OUT_REL
    write_jsonl(out, rows)

    info(f"去重丢弃 {skipped_dup} 条")
    info(f"输出 {out}  共 {len(rows)} 条")

    # 分区统计
    by_domain = {}
    for r in rows:
        by_domain[r["domain"]] = by_domain.get(r["domain"], 0) + 1
    for d, n in sorted(by_domain.items(), key=lambda x: -x[1]):
        info(f"  {d:<10} {n:>4}")

    record(STAGE, {OUT_REL: len(rows)},
           extra={"per_video": per_video, "by_domain": by_domain})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
