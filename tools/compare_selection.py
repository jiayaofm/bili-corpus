# -*- coding: utf-8 -*-
"""在已保存的候选池上比较不同「选择策略」，不用重新生成。

真人改写后与原文的字符 bigram 相似度：中位 0.29、P90 0.40（v4 数据实测）。
所以「改多少」是有客观目标值的，选择函数应该朝这个目标排，而不是
单纯奖励「像人话」和「保留内容」——那两个都不惩罚照抄。
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.stdout.reconfigure(encoding="utf-8")

from syntax_stats import human_score  # noqa: E402

CJK = re.compile(r"[^\u4e00-\u9fff0-9A-Za-z]")
TARGET_SIM = 0.29


def bg(t):
    t = CJK.sub("", t)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def sim(a, b):
    A, B = bg(a), bg(b)
    return len(A & B) / len(A | B) if (A | B) else 1.0


def recall(src, out):
    A, B = bg(src), bg(out)
    return 1.0 if not A else len(A & B) / len(A)


data = json.loads((ROOT / "state" / "real_cands.json").read_text(encoding="utf-8"))
print(f"  {len(data)} 条真实评论，每条 {len(data[0]['cands'])} 个候选")
print()

# 候选池里各候选的相似度分布
all_sims = [sim(d["src"], c) for d in data for c in d["cands"]]
all_rec = [recall(d["src"], c) for d in data for c in d["cands"]]
print("  候选池整体：相似度中位 %.2f  保留度中位 %.2f" % (
    st.median(all_sims), st.median(all_rec)))
print()

STRATS = {
    "旧：0.6像人话+0.4保留":
        lambda src, c: 0.6 * human_score(c) + 0.4 * recall(src, c),
    "新：0.4像人话+0.3保留+0.3贴近0.29":
        lambda src, c: (0.4 * human_score(c) + 0.3 * recall(src, c)
                        + 0.3 * (1 - min(1.0, abs(sim(src, c) - TARGET_SIM)
                                         / TARGET_SIM))),
    "新+硬过滤(sim<0.55 且 保留>0.40)":
        None,   # 特殊处理
}


def pick_new_guarded(src, cands):
    good = [c for c in cands if sim(src, c) < 0.55 and recall(src, c) > 0.40]
    pool = good or cands
    return max(pool, key=lambda c: (0.4 * human_score(c) + 0.3 * recall(src, c)
                                    + 0.3 * (1 - min(1.0, abs(sim(src, c)
                                                             - TARGET_SIM)
                                                     / TARGET_SIM))))


print("  %-34s %8s %8s %8s %8s" % ("策略 / 候选数", "相似度中位", "太多(≥0.6)",
                                   "合格率", "最像的"))
print("  " + "-" * 74)
for name, fn in STRATS.items():
    for n in (2, 3, 4, 6):
        ss, good_n, pick_ss = [], 0, []
        for d in data:
            src = d["src"]
            cands = d["cands"][:n]
            if fn is None:
                c = pick_new_guarded(src, cands)
            else:
                c = max(cands, key=lambda x: fn(src, x))
            s = sim(src, c)
            ss.append(s)
            pick_ss.append(s)
            if s < 0.5 and recall(src, c) >= 0.45:
                good_n += 1
        too_same = sum(1 for s in ss if s >= 0.6)
        print("  %-34s %8.2f %8d %7.0f%% %8.2f" % (
            f"{name[:22]} N={n}", st.median(ss), too_same,
            good_n / len(data) * 100, min(ss)))
    print()

print("  参考：v4 训练目标（真人原文）与输入的相似度中位 0.29")
print("       「合格」定义：相似度 <0.5 且内容保留 ≥0.45")
