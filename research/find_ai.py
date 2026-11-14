# -*- coding: utf-8 -*-
"""
按账号等级切片，找出「疑似 AI 评论」

假设（待验证）：B站 上的 DS 账号多为二级号，且说话风格明显。
若成立，等级 × 风格两个维度应该能把 AI 评论圈出来。
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(__file__).parent / "data"
EMOTE_RE = re.compile(r"\[[^\[\]]{1,12}\]")


def load():
    recs = []
    for fp in sorted(DATA.glob("*_*.jsonl")):
        recs += [json.loads(l) for l in fp.open(encoding="utf-8") if l.strip()]
    return recs


def stats(rs):
    n = len(rs)
    if not n:
        return None
    lens = [len(r["content"]) for r in rs]
    return {
        "n": n,
        "avg_len": sum(lens) / n,
        "med_len": sorted(lens)[n // 2],
        "corner": sum(r["content"].count("「") for r in rs),
        "dash": sum(r["content"].count("—") for r in rs),
        "dun": sum(r["content"].count("、") for r in rs),
        "emote": sum(1 for r in rs if EMOTE_RE.search(r["content"])),
        "period_rate": sum(r["content"].count("。") for r in rs) / max(1, sum(lens)) * 10000,
    }


def main():
    recs = load()
    print("=" * 70)
    print(f"语料总量 {len(recs)} 条")
    print("=" * 70)

    by_level = defaultdict(list)
    for r in recs:
        by_level[r.get("level", 0)].append(r)

    print("\n【账号等级分布】")
    for lv in sorted(by_level):
        s = stats(by_level[lv])
        print(f"  LV{lv}: {s['n']:>4} 条   平均 {s['avg_len']:>6.1f} 字   中位 {s['med_len']:>4} 字   "
              f"「={s['corner']}  —={s['dash']}  顿号={s['dun']}  表情率={s['emote']/s['n']*100:.0f}%")

    # 重点看低等级号
    print("\n" + "=" * 70)
    print("【LV2 账号的评论样本】（假设：DS 账号多为二级号）")
    print("=" * 70)
    lv2 = by_level.get(2, [])
    for r in sorted(lv2, key=lambda x: -len(x["content"]))[:15]:
        c = r["content"].replace("\n", "\\n")
        print(f"  ({len(r['content']):>3}字) [{r['like']}赞] {c[:130]}")

    # 用风格特征筛疑似 AI
    print("\n" + "=" * 70)
    print("【疑似 AI 风格筛子】条件：LV<=3 且（含「 或 含破折号 或 长度>150）")
    print("=" * 70)
    sus = []
    for r in recs:
        c = r["content"]
        hit = []
        if "「" in c:
            hit.append("「")
        if "—" in c:
            hit.append("破折号")
        if len(c) > 150:
            hit.append("超长")
        if r.get("level", 0) <= 3 and hit:
            sus.append((r, hit))

    print(f"  命中 {len(sus)} 条 / {len(recs)}")
    for r, hit in sorted(sus, key=lambda x: -len(x[0]["content"]))[:20]:
        c = r["content"].replace("\n", "\\n")
        print(f"  [LV{r.get('level')}|{'+'.join(hit)}|{len(c)}字|{r['like']}赞] {c[:120]}")

    # 反向：任意等级但含「的
    print("\n" + "=" * 70)
    print("【全部含「的评论】（不管等级）")
    print("=" * 70)
    corner = [r for r in recs if "「" in r["content"]]
    print(f"  共 {len(corner)} 条")
    for r in corner[:20]:
        c = r["content"].replace("\n", "\\n")
        print(f"  [LV{r.get('level')}|{len(c)}字|{r['like']}赞] {c[:120]}")


if __name__ == "__main__":
    main()
