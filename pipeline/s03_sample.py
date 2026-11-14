# -*- coding: utf-8 -*-
"""
s03 盲标抽样

分层抽样：
  · 候选层 —— 规则打分器判为 high/mid 的全部取出（信息量大）
  · 随机层 —— 其余评论里随机抽 N 条（覆盖真实分布）

输出：
  labeling/blind_sample.csv   编号 / 评论内容 / 判断(1=AI 0=真人 ?=不确定) / 备注
  labeling/blind_map.json     编号(int) -> rpid / was_candidate

注意：编号是纯 3 位数字，不带任何前缀。Excel 会吃掉前导零，
所以 map 里同时存 int 形式，s04 按 int 匹配。
"""
import csv
import importlib.util
import json
import random
from pathlib import Path

import common
from common import CFG, P, banner, info, read_jsonl, record

STAGE = "s03_sample"
OUT_CSV_REL = "labeling/blind_sample.csv"
OUT_MAP_REL = "labeling/blind_map.json"


def load_rule():
    """从现有 label_ai.py 借用规则打分，保证与已有效率一致"""
    path = common.ROOT / "label_ai.py"
    spec = importlib.util.spec_from_file_location("_rule", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.score, mod.conf_of


def main(force=False):
    banner("s03 盲标抽样")

    corpus_path = common.ROOT / "data/corpus.jsonl"
    if not corpus_path.exists():
        info("[失败] data/corpus.jsonl 不存在，请先跑 s02")
        return 1

    recs = read_jsonl(corpus_path)
    info(f"语料 {len(recs)} 条")

    score, conf_of = load_rule()

    # 已人工确认的真人（历史标注）排除，避免重复劳动
    known = set()
    hm = P["labeling"] / "human_marked.json"
    if hm.exists():
        try:
            known = {x["rpid"] for x in json.loads(hm.read_text(encoding="utf-8"))}
        except Exception:
            known = set()

    candidates, pool = [], []
    for r in recs:
        if r["rpid"] in known:
            continue
        s, hits = score(r)
        if conf_of(s, hits) in ("high", "mid"):
            candidates.append(r)
        else:
            pool.append(r)

    n_random = CFG["sample"]["n_random"]
    seed = CFG["sample"]["seed"]
    rng = random.Random(seed)
    picked_random = rng.sample(pool, min(n_random, len(pool)))

    info(f"候选层 {len(candidates)} 条（全取）")
    info(f"随机层 {len(picked_random)} / {len(pool)} 条")

    picked = candidates + picked_random
    rng.shuffle(picked)

    rows, items = [], []
    for i, r in enumerate(picked, 1):
        code = f"{i:03d}"
        rows.append({
            "编号": code,
            "评论内容": r["content"],
            "判断(1=AI 0=真人 ?=不确定)": "",
            "备注": "",
        })
        items.append({
            "编号": code,
            "编号_int": i,
            "rpid": r["rpid"],
            "was_candidate": 1 if r in candidates else 0,
        })

    out_csv = common.ROOT / OUT_CSV_REL
    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    amap = {
        "seed": seed,
        "total": len(rows),
        "candidate_count": len(candidates),
        "random_count": len(picked_random),
        "pool_rest": len(pool),
        "excluded_known_human": len(known),
        "items": items,
    }
    (common.ROOT / OUT_MAP_REL).write_text(
        json.dumps(amap, ensure_ascii=False, indent=2), encoding="utf-8")

    info(f"输出 {out_csv}  共 {len(rows)} 条")
    info("下一步：打开该 CSV 在「判断」列填 1 / 0 / ?，或直接在编号前加 x / m")

    record(STAGE, {OUT_CSV_REL: len(rows), OUT_MAP_REL: len(items)},
           extra={"candidates": len(candidates), "random": len(picked_random)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
