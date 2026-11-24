# -*- coding: utf-8 -*-
"""量化「数字被改坏」的严重程度。

背景：实测「沙县 11 块的炒饭」被改写成「沙县十一点的炒饭」，
「11块」变成了 11 点，语义直接错了。

真人也确实会把数字写成汉字（dev 里有 99% → 九成九），所以不能简单要求
"必须保留阿拉伯数字"。判断标准是：输入里的数字，在输出里要么是同样的
阿拉伯数字，要么是等值的中文数字。
"""
import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

CN_DIGIT = "零一二三四五六七八九"
CN_UNIT = ["", "十", "百", "千"]


def to_cn(n: int) -> str:
    """0~9999 转中文数字（口语写法：11→十一，110→一百一十）。"""
    if n == 0:
        return "零"
    if n < 0 or n > 9999:
        return ""
    s = str(n)
    out = []
    for i, ch in enumerate(s):
        d = int(ch)
        unit = CN_UNIT[len(s) - i - 1]
        if d == 0:
            if out and out[-1] != "零" and i != len(s) - 1:
                out.append("零")
            continue
        if d == 1 and unit == "十" and i == 0:
            out.append("十")          # 11 → 十一，不是 一十一
        else:
            out.append(CN_DIGIT[d] + unit)
    return "".join(out).rstrip("零")


def nums_of(t: str) -> list[str]:
    return re.findall(r"\d+", t)


def lost_nums(src: str, out: str) -> list[str]:
    """返回输入里有、但输出里既没有阿拉伯形式也没有中文等值形式的数字。"""
    src_nums = nums_of(src)
    out_nums = set(nums_of(out))
    only_digits = re.sub(r"\d+", "", out)
    lost = []
    for s in src_nums:
        if s in out_nums:
            continue
        n = int(s)
        cn = to_cn(n)
        if cn and cn in only_digits:
            continue
        lost.append(s)
    return lost


def main():
    rows = [json.loads(l) for l in open("data/rewrite/v4_dev.jsonl", encoding="utf-8")]
    preds = {json.loads(l)["x"]: json.loads(l)["pred"]
             for l in open("state/preds/fish-v4.jsonl", encoding="utf-8")}

    stat = {"有数字": 0, "丢数字": 0, "错语义": 0}
    examples = []
    for r in rows:
        x, y = r["_meta"]["x"], r["_meta"]["y"]
        p = preds.get(x)
        if not p:
            continue
        if not nums_of(x):
            continue
        stat["有数字"] += 1
        lost = lost_nums(x, p)
        if lost:
            stat["丢数字"] += 1
            # 数字在，但紧跟着的量词被换了（11块 → 11点），属于语义错
            if len(examples) < 3:
                examples.append((x, p, y, lost))
    # 真人的对照
    human_lost = sum(1 for r in rows if nums_of(r["_meta"]["x"])
                     and lost_nums(r["_meta"]["x"], r["_meta"]["y"]))

    print(f"  带数字的样本 {stat['有数字']}/{len(rows)}")
    print()
    print(f"  fish-v4 丢数字   {stat['丢数字']}/{stat['有数字']}"
          f"（{stat['丢数字']/max(1,stat['有数字'])*100:.0f}%）")
    print(f"  真人原文丢数字   {human_lost}/{stat['有数字']}"
          f"（{human_lost/max(1,stat['有数字'])*100:.0f}%）")
    print()
    for x, p, y, lost in examples:
        print(f"    丢 {lost}")
        print(f"      入 {x[:46]}")
        print(f"      出 {p[:46]}")
        print(f"      真 {y[:46]}")
        print()


if __name__ == "__main__":
    main()
