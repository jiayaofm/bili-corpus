# -*- coding: utf-8 -*-
"""量一量改写结果的「语义完整性」缺陷有多频繁。

不靠人眼看，用几条可判定的规则：

  ① 中英混杂    输出里有英文单词，但输入里没有
  ② 句子没说完  以虚词/连接词结尾，或者末句缺谓语
  ③ 否定反转    输入的否定词和输出对不上
  ④ 程度丢失    「压根/根本/才/都」这类程度副词少了
  ⑤ 长度塌陷    输出比输入短太多（内容被砍掉）

这些规则会误报，但用来比较「哪类问题最多」够了。
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")

DATA = ROOT / "state" / "rewrite_eval_fish.json"

EN = re.compile(r"[A-Za-z]{2,}")
NEG = "不没别未毫无非莫勿"
DEGREE = ["压根", "根本", "才", "都", "就", "连", "甚至", "居然", "竟然"]
# 这些字结尾说明句子没说完
DANGLING = set("的而但因所如虽与及或况并且还有把被让给对向从为以于在")

# 常见英文词白名单（出现在输入里就不算混）
KNOWN_EN = {"up", "ai", "b站", "ok", "doge", "cpu", "gpu", "vtuber", "mbti",
            "cv", "op", "ed", "pv", "bgm", "dd", "yyds", "tiktok", "bilibili"}


def check(x: str, out: str, target: str) -> dict:
    bad = {}

    # ① 中英混杂：输出有英文，输入没有
    en_out = {w.lower() for w in EN.findall(out)} - KNOWN_EN
    en_in = {w.lower() for w in EN.findall(x)} - KNOWN_EN
    extra = en_out - en_in
    if extra:
        bad["中英混杂"] = sorted(extra)[:3]

    # ② 句末悬空
    tail = out.rstrip("。，！？…~ ")
    if tail and tail[-1] in DANGLING:
        bad["句子没说完"] = tail[-8:]

    # ③ 否定反转：输入的否定字数量明显多于输出
    n_in = sum(out.count(c) for c in NEG) if False else sum(x.count(c) for c in NEG)
    n_out = sum(out.count(c) for c in NEG)
    if n_in >= 2 and n_out < n_in * 0.4:
        bad["否定丢失"] = f"入{n_in}个→出{n_out}个"

    # ④ 程度丢失
    d_in = sum(1 for w in DEGREE if w in x)
    d_out = sum(1 for w in DEGREE if w in out)
    if d_in >= 2 and d_out == 0:
        bad["程度丢失"] = f"入{d_in}个→出0"

    # ⑤ 长度塌陷
    if len(out) < len(x) * 0.45:
        bad["内容被砍"] = f"{len(x)}→{len(out)}"

    return bad


def main():
    rows = json.loads(DATA.read_text(encoding="utf-8"))
    print(f"  检查 {len(rows)} 条改写结果")
    print()

    counter = {}
    flagged = []
    for i, r in enumerate(rows, 1):
        b = check(r["x"], r["out"], r["y"])
        if b:
            flagged.append((i, r, b))
        for k in b:
            counter[k] = counter.get(k, 0) + 1

    print("  各类问题出现次数：")
    for k, v in sorted(counter.items(), key=lambda kv: -kv[1]):
        print("    %-12s %2d 条（%.0f%%）" % (k, v, v / len(rows) * 100))
    print(f"\n  ≥1 个问题的：{len(flagged)}/{len(rows)}（{len(flagged) / len(rows) * 100:.0f}%）")
    print()

    for i, r, b in flagged[:8]:
        print(f"  [{i}] {' / '.join(b.keys())}")
        if "中英混杂" in b:
            print(f"       混入英文: {b['中英混杂']}")
        if "句子没说完" in b:
            print(f"       结尾: …{b['句子没说完']}")
        print(f"       入 {r['x'][:52]}")
        print(f"       出 {r['out'][:52]}")
        print()


if __name__ == "__main__":
    main()
