# -*- coding: utf-8 -*-
"""
s04 回收标注（人工 gate）

兼容两种标注写法：
  (a) 判断列填 1 / 0 / ?
  (b) 在「编号」前加前缀：x=明确AI，m=疑似AI，无前缀=真人

优先级：
  1. blind_sample.csv 有有效标注   → 解析它（旧 ground_truth.csv 先备份到 state/）
  2. ground_truth.csv 已存在       → 直接复用
  3. blind_151.csv 有有效标注      → 解析它
  4. 都没有                        → 提示并退出码 2，让 run_all 停下

输出 labeling/ground_truth.csv：rpid,label,strength,was_candidate,content
"""
import csv
import json
import re
import shutil
import sys
from datetime import datetime

import common
from common import P, banner, info, record

STAGE = "s04_labels"
OUT_REL = "labeling/ground_truth.csv"
PREFIX_RE = re.compile(r"^\s*([A-Za-z]*)\s*(\d+)\s*$")


def _read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _parse(rows, judge_col):
    """返回 (certain, maybe, human) 三个编号 int 集合；None 表示无有效标注"""
    # 先判断是哪种格式
    prefix_mode = False
    for r in rows:
        m = PREFIX_RE.match((r.get("编号") or "").strip())
        if m and m.group(1).lower() in ("x", "m"):
            prefix_mode = True
            break

    certain, maybe, human = set(), set(), set()
    for r in rows:
        raw = (r.get("编号") or "").strip()
        m = PREFIX_RE.match(raw)
        if not m:
            continue
        pre, num = m.group(1).lower(), int(m.group(2))

        if pre == "x":
            certain.add(num)
        elif pre == "m":
            maybe.add(num)
        elif pre == "":
            if prefix_mode:
                human.add(num)
            else:
                v = (r.get(judge_col) or "").strip()
                if v in ("1", "x", "X"):
                    certain.add(num)
                elif v in ("0", "o", "O"):
                    human.add(num)
                elif v in ("?", "？", "m", "M"):
                    maybe.add(num)
                # 空白 = 未标
    if not (certain or maybe or human):
        return None
    return certain, maybe, human


def _load_map(path):
    amap = json.loads(path.read_text(encoding="utf-8"))
    return {int(it["编号_int"]): it for it in amap["items"]}


def _write_gt(pairs, source):
    """pairs: [(rpid, label, strength, was_candidate, content)]"""
    with (common.ROOT / OUT_REL).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rpid", "label", "strength", "was_candidate", "content"])
        for row in pairs:
            w.writerow(row)
    info(f"写入 {OUT_REL}  共 {len(pairs)} 条（来源：{source}）")


def _backup_existing():
    dst = common.ROOT / OUT_REL
    if not dst.exists():
        return None
    bak = P["state"] / f"ground_truth_backup_{datetime.now():%Y%m%d_%H%M%S}.csv"
    shutil.copy2(dst, bak)
    info(f"已备份旧真值到 state/{bak.name}")
    return bak


def main(force=False):
    banner("s04 回收标注")

    sample_csv = P["labeling"] / "blind_sample.csv"
    sample_map = P["labeling"] / "blind_map.json"
    legacy_csv = P["labeling"] / "blind_151.csv"
    legacy_map = P["labeling"] / "blind_151_map.json"
    gt_path = common.ROOT / OUT_REL

    judged = 0

    # ---- 1) 新的 blind_sample.csv 有标注 ----
    if sample_csv.exists() and sample_map.exists():
        rows = _read_csv(sample_csv)
        col = next((k for k in rows[0].keys() if k.startswith("判断")), None) if rows else None
        parsed = _parse(rows, col or "")
        if parsed:
            certain, maybe, human = parsed
            _backup_existing()
            amap = _load_map(sample_map)
            content = {}
            for r in rows:
                _m = PREFIX_RE.match((r.get("编号") or "").strip())
                if _m:
                    content[int(_m.group(2))] = r["评论内容"]
            pairs = []
            for num in sorted(certain | maybe | human):
                it = amap.get(num)
                if not it:
                    continue
                if num in certain:
                    lab, st = 1, "x"
                elif num in maybe:
                    lab, st = 1, "m"
                else:
                    lab, st = 0, ""
                pairs.append([it["rpid"], lab, st, it["was_candidate"],
                              content.get(num, "")])
            _write_gt(pairs, "blind_sample.csv")
            judged = len(certain) + len(maybe) + len(human)
            info(f"明确AI {len(certain)} / 疑似 {len(maybe)} / 真人 {len(human)}")
        else:
            info("[提示] blind_sample.csv 还没有任何标注")

    # ---- 2) 复用已有真值（blind_sample 未标注时回退）----
    if not judged and gt_path.exists():
        rows = _read_csv(gt_path)
        if rows:
            info(f"复用已有真值 {len(rows)} 条（{OUT_REL}）")
            judged = len(rows)
            info(f"明确AI {sum(1 for r in rows if r['strength'] == 'x')} / "
                 f"疑似 {sum(1 for r in rows if r['strength'] == 'm')} / "
                 f"真人 {sum(1 for r in rows if r['strength'] == '')}")

    # ---- 3) 历史标注表 blind_151.csv ----
    if not judged and legacy_csv.exists() and legacy_map.exists():
        rows = _read_csv(legacy_csv)
        col = next((k for k in rows[0].keys() if k.startswith("判断")), None) if rows else None
        parsed = _parse(rows, col or "")
        if parsed:
            certain, maybe, human = parsed
            amap = _load_map(legacy_map)
            pairs = []
            for num in sorted(certain | maybe | human):
                it = amap.get(num)
                if not it:
                    continue
                lab = 1 if (num in certain or num in maybe) else 0
                st = "x" if num in certain else ("m" if num in maybe else "")
                pairs.append([it["rpid"], lab, st, it["was_candidate"], ""])
            _backup_existing()
            _write_gt(pairs, "blind_151.csv")
            judged = len(pairs)

    # ---- gate ----
    awaiting_sample = sample_csv.exists() and sample_map.exists()
    if awaiting_sample and not judged:
        info("")
        info("=" * 56)
        info("  需要人工标注，流水线在此暂停")
        info("=" * 56)
        info(f"  请打开: {sample_csv}")
        info("  在「判断」列填 1=AI / 0=真人 / ?=不确定，或在编号前加 x / m")
        info("  保存后重跑 run_all.py")
        return 2

    if not judged:
        info("")
        info("[需要人工介入] 没有可用的标注数据")
        if not sample_csv.exists():
            info("  请先运行 s03 生成标注表")
        return 2

    record(STAGE, {OUT_REL: judged}, extra={"judged": judged})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
