# -*- coding: utf-8 -*-
"""
生成配对数据的第一批候选

用途：训练/示范需要 `(AI 原文 → 人写的改写)` 的配对，现在一条都没有。
      本脚本从已标注的 AI 评论里分层抽样，用现有 workflow 生成候选改写，
      输出成一个人工可以快速「挑一条 / 改几个字」的清单。

输出：
    state/pairs_candidates.json   结构化数据（含每条的分数与判断）
    state/pairs_candidates.md     人工审阅用（原文 + 候选 + 空行待改）

用法：
    python tools/build_pairs.py --n 20
    python tools/build_pairs.py --n 5 --dry-run     # 只抽样不出候选
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from pipeline.llm import LLMClient, LLMGenerator, make_llm_fidelity_check  # noqa: E402
from pipeline.rewrite import get_scorer, rewrite  # noqa: E402

GT = ROOT / "labeling" / "ground_truth.csv"
OUT_JSON = ROOT / "state" / "pairs_candidates.json"
OUT_MD = ROOT / "state" / "pairs_candidates.md"


def load_ai_rows():
    rows = []
    for r in csv.DictReader(GT.open(encoding="utf-8-sig", newline="")):
        if r["label"] != "1":
            continue
        c = (r["content"] or "").strip()
        if len(c) < 8:
            continue
        rows.append({"rpid": r["rpid"], "content": c, "n": len(c)})
    return rows


def stratified_pick(rows, n):
    """按长度分档抽样：短(<30) / 中(30-80) / 长(>80)，比例尽量接近 4:12:4。"""
    short = [r for r in rows if r["n"] < 30]
    mid = [r for r in rows if 30 <= r["n"] <= 80]
    long_ = [r for r in rows if r["n"] > 80]

    want_short = max(1, round(n * 0.2))
    want_long = max(1, round(n * 0.2))
    want_mid = max(1, n - want_short - want_long)

    picked = []
    for pool, k in ((short, want_short), (mid, want_mid), (long_, want_long)):
        step = max(1, len(pool) // k) if k else 1
        picked.extend(pool[::step][:k])
    # 去重后不足则从 mid 补
    seen, out = set(), []
    for r in picked:
        if r["rpid"] not in seen:
            seen.add(r["rpid"])
            out.append(r)
    for r in mid:
        if len(out) >= n:
            break
        if r["rpid"] not in seen:
            seen.add(r["rpid"])
            out.append(r)
    return sorted(out, key=lambda r: r["n"])[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--cand", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true", help="只抽样，不调模型")
    a = ap.parse_args()

    rows = load_ai_rows()
    picked = stratified_pick(rows, a.n)
    print("AI 评论 %d 条，分层抽 %d 条：" % (len(rows), len(picked)))
    for r in picked:
        print("  [%3d字] %s" % (r["n"], r["content"][:60].replace("\n", " ")))

    if a.dry_run:
        print("\n[dry-run] 不调模型，退出")
        return

    client = LLMClient(verbose=True)
    gen = LLMGenerator(client=client, temperature=1.1)
    fid = make_llm_fidelity_check(client=client)
    sc = get_scorer()
    print("\n模型 %s  打分器就绪（%d 条训练）\n" % (client.model, sc.n_train))

    out = []
    t0 = time.time()
    for i, r in enumerate(picked, 1):
        base = sc.describe(r["content"])
        print("[%d/%d] %d 字  原文验收 %.3f  %s"
              % (i, len(picked), r["n"], base["heldout_ai"], r["content"][:40]))
        t1 = time.time()
        res = rewrite(r["content"], gen, max_rounds=a.rounds, n_per_round=a.cand,
                      scorer=sc, fidelity_check=fid)
        best = res.get("best")
        dt = time.time() - t1
        if best:
            after = sc.describe(best)
            drop = base["heldout_ai"] - after["heldout_ai"]
            print("        -> 收敛=%s  验收 %.3f (降 %.3f)  %.0fs"
                  % (res.get("converged"), after["heldout_ai"], drop, dt))
            print("           %s" % best[:70])
        else:
            drop = None
            print("        -> 未收敛  %.0fs" % dt)

        out.append({
            "index": i,
            "rpid": r["rpid"],
            "src": r["content"],
            "n_src": r["n"],
            "candidate": best,
            "converged": bool(res.get("converged")),
            "rule_before": base["ai_rule"],
            "heldout_before": base["heldout_ai"],
            "heldout_after": (sc.describe(best)["heldout_ai"] if best else None),
            "drop": drop,
            "human_edit": None,     # ← 留给人工填写
            "seconds": round(dt, 1),
        })

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "model": client.model,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "human_edit 字段留空，由人工填写或直接改 candidate",
        "items": out,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = [x for x in out if x["candidate"]]
    lines = [
        "# 配对数据候选（第一批）",
        "",
        "> 每一条：上面是 AI 原文，下面是模型给出的改写候选。",
        "> **你要做的**：觉得哪条凑合就留着；不对就直接在上面改几个字。",
        "> 改完告诉我，我把 `human_edit` 填进去当训练配对。",
        "",
        "生成模型：`%s`　共 %d 条，其中 %d 条模型自己认为收敛。"
        % (client.model, len(out), len(ok)),
        "",
        "*凡是不行的、不用客气的，直接说「这条不行」我就删掉。*",
        "",
        "---",
        "",
    ]
    for x in out:
        lines.append("## %d. （原文 %d 字）" % (x["index"], x["n_src"]))
        lines.append("")
        lines.append("**原文**")
        lines.append("")
        lines.append("```")
        lines.append(x["src"])
        lines.append("```")
        lines.append("")
        if x["candidate"]:
            lines.append("**模型改写**（验收 %.2f → %.2f）"
                         % (x["heldout_before"], x["heldout_after"]))
            lines.append("")
            lines.append("```")
            lines.append(x["candidate"])
            lines.append("```")
        else:
            lines.append("**模型改写**：未产出合格候选")
        lines.append("")
        lines.append("**你改的版本**：")
        lines.append("")
        lines.append("```")
        lines.append("")
        lines.append("```")
        lines.append("")
        lines.append("---")
        lines.append("")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print("\n用时 %.1f 分钟" % ((time.time() - t0) / 60))
    print("写好：%s" % OUT_MD)
    print("写好：%s" % OUT_JSON)


if __name__ == "__main__":
    main()
