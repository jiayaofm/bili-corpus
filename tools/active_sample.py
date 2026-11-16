# -*- coding: utf-8 -*-
"""
主动学习抽样：只让你标「模型最不确定」的样本

为什么这样省力：
  全库 AI 密度只有 7.9%，随机抽样标 100 条大约只有 8 条正样本。
  而模型打分在中间地带（0.10~0.37）的样本，正样本比例高得多，
  标同样的条数能拿到更多有效信息。

同时保留一部分**纯随机样本**——它们用于无偏评估。
如果全部按主动学习抽，训练集会偏，指标就不可信了。

用法：
    python tools/active_sample.py                  # 默认 80 不确定 + 60 随机
    python tools/active_sample.py --n-uncertain 100 --n-random 40
    python tools/active_sample.py --seed 123

输出：
    labeling/blind_active.csv    给你标（无分数、无账号名，盲标）
    labeling/blind_active_map.json   编号 ↔ rpid 对照
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research"))

from audit import Auditor  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="主动学习抽样")
    ap.add_argument("--n-uncertain", type=int, default=80,
                    help="不确定地带抽多少条（默认 80）")
    ap.add_argument("--n-random", type=int, default=60,
                    help="纯随机抽多少条，用于无偏评估（默认 60）")
    ap.add_argument("--seed", type=int, default=20261114)
    args = ap.parse_args()

    corpus = []
    for line in (ROOT / "data" / "corpus.jsonl").open(encoding="utf-8"):
        line = line.strip()
        if line:
            corpus.append(json.loads(line))

    # 已标过的排除掉，避免重复劳动
    gt_path = ROOT / "labeling" / "ground_truth.csv"
    labeled = set()
    if gt_path.exists():
        for r in csv.DictReader(gt_path.open(encoding="utf-8-sig", newline="")):
            labeled.add(r["rpid"])

    remaining = [c for c in corpus if c["rpid"] not in labeled]
    print(f"语料 {len(corpus)} 条，已标 {len(labeled)} 条，剩余 {len(remaining)} 条")

    auditor = Auditor(0.95)
    lo, hi = auditor.auto_thr * 0.28, auditor.auto_thr
    print(f"自动阈值 {auditor.auto_thr:.3f}，不确定地带取 [{lo:.3f}, {hi:.3f})")

    scored = []
    for c in remaining:
        fused, _r, _p = auditor._fuse_raw(c["content"])
        scored.append((c, fused))

    uncertain = [c for c, s in scored if lo <= s < hi]
    clear_human = [c for c, s in scored if s < lo]
    clear_ai = [c for c, s in scored if s >= hi]
    print(f"  不确定地带 {len(uncertain)} 条")
    print(f"  明显真人   {len(clear_human)} 条")
    print(f"  明显 AI    {len(clear_ai)} 条（这些工具已能判，不需要你标）")

    rng = random.Random(args.seed)

    # ---- 按分区加权的主动学习抽样 ----
    # 纯不确定度抽样会把名额平均铺到所有分区，但 AI 几乎只集中在
    # 少数分区（实测：科技 12.9%，游戏/美食 0%）。
    # 用各分区「工具已能明确判为 AI」的条数作为密度信号，
    # 给每个分区配额，避免把标注预算浪费在几乎不可能有 AI 的地方。
    by_dom = {}
    for c, s in scored:
        if lo <= s < hi:
            by_dom.setdefault(c.get("domain", "?"), []).append(c)

    clear_ai_dom = {}
    for c, s in scored:
        if s >= hi:
            d = c.get("domain", "?")
            clear_ai_dom[d] = clear_ai_dom.get(d, 0) + 1

    weights = {d: clear_ai_dom.get(d, 0) + 2 for d in by_dom}
    wsum = sum(weights.values())
    quotas = {d: int(args.n_uncertain * w / wsum) for d, w in weights.items()}
    # 余数补给权重最高的分区
    rest = args.n_uncertain - sum(quotas.values())
    for d in sorted(weights, key=lambda x: -weights[x]):
        if rest <= 0:
            break
        quotas[d] += 1
        rest -= 1

    picked_unc = []
    for d, q in quotas.items():
        pool = by_dom.get(d, [])
        k = min(q, len(pool))
        if k:
            picked_unc.extend(rng.sample(pool, k))

    print("\n不确定地带的分区配额：")
    for d in sorted(weights, key=lambda x: -weights[x]):
        avail = len(by_dom.get(d, []))
        got = sum(1 for c in picked_unc if c.get("domain", "?") == d)
        print(f"  {d:<10} 可抽 {avail:>3}  配额 {quotas[d]:>3}  实抽 {got:>3}")

    picked_ids = {c["rpid"] for c in picked_unc}
    n_unc = len(picked_unc)

    # 随机样本从「未选中的剩余」里抽，避免重复
    pool_rand = [c for c in remaining if c["rpid"] not in picked_ids]
    n_rand = min(args.n_random, len(pool_rand))
    picked_rand = rng.sample(pool_rand, n_rand)

    picked = picked_unc + picked_rand
    rng.shuffle(picked)
    print(f"\n本次抽样 {len(picked)} 条（不确定 {n_unc} + 随机 {n_rand}）")

    # 写表格（保持盲标：无分数、无账号名）
    out_csv = ROOT / "labeling" / "blind_active.csv"
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["编号", "评论内容", "判断(1=AI 0=真人 ?=不确定)", "备注"])
        for i, c in enumerate(picked, 1):
            w.writerow([f"{i:03d}", c["content"], "", ""])

    amap = {
        "seed": args.seed,
        "total": len(picked),
        "uncertain_count": n_unc,
        "random_count": n_rand,
        "auto_threshold": round(auditor.auto_thr, 4),
        "band": [round(lo, 4), round(hi, 4)],
        "items": [
            {
                "编号": f"{i:03d}",
                "编号_int": i,
                "rpid": c["rpid"],
                "stratum": "uncertain" if c in picked_unc else "random",
                "domain": c.get("domain", ""),
            }
            for i, c in enumerate(picked, 1)
        ],
    }
    (ROOT / "labeling" / "blind_active_map.json").write_text(
        json.dumps(amap, ensure_ascii=False, indent=2), encoding="utf-8")

    from collections import Counter
    dom = Counter(c.get("domain", "?") for c in picked)
    print(f"  分区分布: {dict(dom)}")
    print(f"\n表格: {out_csv}")
    print(f"对照: {ROOT / 'labeling' / 'blind_active_map.json'}")
    print("\n标注方式：判断列填 1 / 0 / ?，或在编号前加 x（明确AI）/ m（疑似）")
    print("注意：这批有意偏向模型拿不准的样本，AI 比例会比之前高，不必因此觉得奇怪。")


if __name__ == "__main__":
    main()
