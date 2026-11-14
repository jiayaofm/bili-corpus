# -*- coding: utf-8 -*-
"""
s01 采集

读 config 的 videos，调现成的 bili_harvest.py 抓评论。
已有 jsonl 的视频直接跳过（幂等）；`--force` 强制重抓。
"""
import subprocess
import sys
from pathlib import Path

import common
from common import CFG, P, banner, info, record

STAGE = "s01_fetch"
HARVEST = common.ROOT / "bili_harvest.py"


def count_jsonl(path: Path):
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main(force=False):
    banner("s01 采集")

    if not HARVEST.exists():
        info(f"[跳过] 未找到 {HARVEST.name}，假定数据已就绪")
        record(STAGE, {}, extra={"skipped": "no harvest script"})
        return 0

    order = CFG["harvest"]["order"]
    results = {}
    failed = []

    for v in CFG["videos"]:
        bvid = v["bvid"]
        jsonl = P["data"] / f"{bvid}_{order}.jsonl"
        done_n = count_jsonl(jsonl)

        if done_n > 0 and not force:
            info(f"{bvid} 已有 {done_n:>4} 条，跳过")
            results[bvid] = done_n
            continue

        info(f"{bvid} 开始采集 ...")
        delay = CFG["harvest"]["delay_sec"]
        cmd = [
            sys.executable, "-X", "utf8", str(HARVEST), "fetch",
            "--bvid", bvid,
            "--order", order,
            "--max-comments", str(CFG["harvest"]["max_comments"]),
            "--max-pages", str(CFG["harvest"]["max_pages"]),
            "--delay", str(delay[0]), str(delay[1]),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace")
        n = count_jsonl(jsonl)
        results[bvid] = n
        if n > 0:
            info(f"{bvid} 完成 {n} 条")
        else:
            failed.append(bvid)
            info(f"{bvid} 采集失败（可能触发风控）")
            tail = (r.stdout or "")[-300:]
            if tail.strip():
                info("  末尾输出: " + tail.strip().replace("\n", " | ")[:200])

    total = sum(results.values())
    info(f"合计 {total} 条")
    if failed:
        info(f"失败视频 {len(failed)}: {', '.join(failed)}")

    record(STAGE, {
        f"data/{v['bvid']}_{order}.jsonl": n
        for v, n in ((v, results.get(v["bvid"], 0)) for v in CFG["videos"])
        if n >= 0
    }, extra={"total": total, "failed": failed})
    return 0 if total > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(force="--force" in sys.argv))
