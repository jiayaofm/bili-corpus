# -*- coding: utf-8 -*-
"""快速预览已抓取的评论语料"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(__file__).parent / "data"

for name in ("BV1YxWx69Ens_like.jsonl", "BV1YxWx69Ens_time.jsonl"):
    p = DATA / name
    if not p.exists():
        continue
    recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    print("=" * 64)
    print(f"{name}   {len(recs)} 条")
    print("-" * 64)

    print("【原始样貌 · 前 12 条】")
    for r in recs[:12]:
        print(f"  [{r['like']:>5}赞] {r['content'][:70]!r}")

    print("\n【点赞最高 5 条】")
    for r in sorted(recs, key=lambda x: -x["like"])[:5]:
        print(f"  [{r['like']}赞] {r['content'][:110]!r}")

    print("\n【最短 10 条】")
    for r in sorted(recs, key=lambda x: len(x["content"]))[:10]:
        print(f"  ({len(r['content'])}字) {r['content']!r}")

    # 按点赞分层看长度，验证"高赞是不是更短/更长"
    if len(recs) >= 100:
        srt = sorted(recs, key=lambda x: -x["like"])
        top = srt[: len(srt) // 4]
        bottom = srt[-len(srt) // 4 :]
        avg_t = sum(len(r["content"]) for r in top) / len(top)
        avg_b = sum(len(r["content"]) for r in bottom) / len(bottom)
        print(f"\n【高赞 25% 平均 {avg_t:.1f} 字】 vs 【低赞 25% 平均 {avg_b:.1f} 字】")
    print()
