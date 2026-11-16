# -*- coding: utf-8 -*-
"""
用真实 LLM 跑改写 workflow

用法：
    python tools/run_rewrite_llm.py                # 跑全部 5 条样本
    python tools/run_rewrite_llm.py --id 4         # 只跑第 4 条
    python tools/run_rewrite_llm.py --rounds 2 --n 4
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from pipeline.llm import LLMClient, LLMGenerator, make_llm_fidelity_check  # noqa: E402
from pipeline.rewrite import get_scorer, rewrite  # noqa: E402

SRC = ROOT / "state" / "rewrite_abc.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", type=int, help="只跑指定样本编号")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    items = json.loads(SRC.read_text(encoding="utf-8"))["items"]
    if a.id:
        items = [x for x in items if x["id"] == a.id]

    t0 = time.time()
    client = LLMClient(verbose=True)
    gen = LLMGenerator(client=client, temperature=1.0, verbose=True)
    fid = make_llm_fidelity_check(client=client)
    sc = get_scorer()
    print("模型 %s  打分器就绪（%d 条训练）\n" % (client.model, sc.n_train))

    for it in items:
        print("=" * 76)
        print("#%d 原文（%d 字）" % (it["id"], len(it["src"])))
        print("  " + it["src"])
        base = sc.describe(it["src"])
        print("  原文：规则分 %.1f  P(AI) %.3f  验收 %.3f"
              % (base["ai_rule"], base["ai_prob"], base["heldout_ai"]))
        if base["hits"]:
            print("  命中：" + " / ".join(base["hits"]))
        print()

        t1 = time.time()
        r = rewrite(it["src"], gen, max_rounds=a.rounds, n_per_round=a.n,
                    scorer=sc, fidelity_check=fid, verbose=a.verbose)
        dt = time.time() - t1

        print("  结果：%s" % ("收敛" if r["converged"] else "未收敛"))
        b = r.get("best")
        if b:
            d = sc.describe(b)
            print("  改写：" + b)
            print("  改写后：规则分 %.1f  P(AI) %.3f  验收 %.3f  （原文验收 %.3f，降 %.3f）"
                  % (d["ai_rule"], d["ai_prob"], d["heldout_ai"],
                     base["heldout_ai"], base["heldout_ai"] - d["heldout_ai"]))
            f = r.get("fidelity") or {}
            print("  保真：%s" % f.get("ok"))
            for n in (f.get("notes") or [])[:3]:
                print("        " + str(n))
        else:
            print("  没有候选通过")
        st = r.get("stats") or {}
        print("  统计：%s" % json.dumps(st, ensure_ascii=False))
        print("  耗时：%.1fs（%d 轮）" % (dt, r.get("rounds", 0)))
        print()

    print("总耗时 %.1fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
