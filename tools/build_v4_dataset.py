# -*- coding: utf-8 -*-
"""把 v4 数据建成训练/验证集。

关键点：
  ① 按 bvid 划分，不按对划分——同一个视频的改写共享话题和词汇，
     按对随机分会把同一视频的样本漏进验证集，指标虚高。
  ② 用 v4 阈值过滤（长度比 0.7~1.35、相似度 <0.5），跟任务书一致。
  ③ 输出格式跟 a_train.jsonl 一样（_meta.x / _meta.y），旧脚本直接能读。
"""
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")

SRC = ROOT / "data" / "whale_comments" / "whale_v4_20261122.jsonl"
OUT = ROOT / "data" / "rewrite"
CJK = re.compile(r"[^\u4e00-\u9fff0-9A-Za-z]")
LINK = re.compile(r"https?://|BV[0-9A-Za-z]{10}|b23\.tv")

SEED = 20261124
DEV_VIDEO_RATIO = 0.12      # 约 12% 的视频做验证


def bg(t):
    t = CJK.sub("", t)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def jac(a, b):
    A, B = bg(a), bg(b)
    return len(A & B) / len(A | B) if (A | B) else 1.0


def cjk_ratio(t):
    return len(re.findall(r"[\u4e00-\u9fff]", t)) / len(t) if t else 0.0


def usable(o, w):
    """返回 (是否可用, 原因)。过滤规则跟任务书 v4 保持一致。"""
    if not o or not w:
        return False, "空"
    if o == w:
        return False, "一字未改"
    if LINK.search(o) or LINK.search(w):
        return False, "含链接"
    if len(o) < 8:
        return False, "原文太短"
    if cjk_ratio(o) < 0.7:
        return False, "原文不是中文"
    r = len(w) / max(1, len(o))
    if not (0.7 <= r <= 1.35):
        return False, "长度比越界"
    if jac(o, w) >= 0.5:
        return False, "改得不够"
    return True, ""


rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
by_video = []
reasons = {}
for r in rows:
    bv = r.get("bvid", "")
    keep = []
    for p in (r.get("pairs") or []):
        o = (p.get("origin") or "").strip()
        w = (p.get("rewrite") or "").strip()
        ok, why = usable(o, w)
        if ok:
            # 方向：输入是鱼的改写（有鱼腥味），目标是真人原话。
            # 2026-11-25 修：原来写成 x=origin, y=rewrite，等于教模型
            # 「人话 → 鱼话」，和「去鱼腥味」的目标正好相反。
            # 当时没发现，是因为只盯着「改写幅度」这个指标，方向错了看不出来。
            keep.append({"x": w, "y": o, "bvid": bv})
        else:
            reasons[why] = reasons.get(why, 0) + 1
    if keep:
        by_video.append((bv, keep))

print(f"  可用视频 {len(by_video)} 个，样本 {sum(len(k) for _, k in by_video)} 对")
print(f"  淘汰：{reasons}")

rng = random.Random(SEED)
vids = [bv for bv, _ in by_video]
rng.shuffle(vids)
n_dev = max(3, int(len(vids) * DEV_VIDEO_RATIO))
dev_vids = set(vids[:n_dev])

train, dev = [], []
for bv, keep in by_video:
    (dev if bv in dev_vids else train).extend(keep)

rng.shuffle(train)
rng.shuffle(dev)


def write(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            rec = {
                "_meta": {"bvid": it["bvid"], "x": it["x"], "y": it["y"],
                          "source": "whale_v4_20261122"},
                "messages": [
                    {"role": "system",
                     "content": "你把 AI 写的 B站 评论改成像真人网友写的。"},
                    {"role": "user", "content": f"这条评论：{it['x']}\n\n改一下。"},
                    {"role": "assistant", "content": it["y"]},
                ],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


write(OUT / "v4_train.jsonl", train)
write(OUT / "v4_dev.jsonl", dev)

print()
print(f"  训练集 {len(train)} 对（{len(by_video) - len(dev_vids)} 个视频）→ data/rewrite/v4_train.jsonl")
print(f"  验证集 {len(dev)} 对（{len(dev_vids)} 个视频）→ data/rewrite/v4_dev.jsonl")
print()
print("  验证集视频：")
for bv in sorted(dev_vids):
    n = sum(len(k) for b, k in by_video if b == bv)
    print(f"    {bv}  {n} 对")
