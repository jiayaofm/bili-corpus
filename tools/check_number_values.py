# -*- coding: utf-8 -*-
"""数值正确性检查：不管写成阿拉伯数字还是汉字，只看值对不对。

之前用「必须保留阿拉伯数字」当标准，结果真人原文 69% 不合格——
因为真人就爱写「八碗」「九成九」「十五分钟」。那是写法差异，不是错误。

真正要防的是改值：11 块→十点（11 变时间）、15 分钟→五分钟（15→5）、
111→一百一十（111→110）。
"""
import json
import re
import sys
import importlib.util

sys.stdout.reconfigure(encoding="utf-8")

spec = importlib.util.spec_from_file_location("cs", "tools/comment_server.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
R = m.Rewriter

CN = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
      "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
UNIT = {"十": 10, "百": 100, "千": 1000}


def parse_cn(s: str) -> int | None:
    """解析中文数字，支持 十一 / 九十 / 九十九 / 一百一十 / 一百一十一 这种。"""
    if not s:
        return None
    total, section, num = 0, 0, 0
    for ch in s:
        if ch in CN:
            num = CN[ch]
        elif ch in UNIT:
            u = UNIT[ch]
            if num == 0:
                num = 1
            section += num * u
            num = 0
        else:
            return None
    return section + num if (section or num or s) else None


def values_in(t: str) -> set:
    """文本里出现的所有数值（阿拉伯 + 中文数字）。"""
    vals = set()
    for n in re.findall(r"\d+", t):
        vals.add(int(n))
    t2 = re.sub(r"\d+", "", t)
    for run in re.findall(r"[零〇一二两三四五六七八九十百千]+", t2):
        v = parse_cn(run)
        if v is not None:
            vals.add(v)
    # 百分比口语：九成九 = 99，九成 = 90
    for tens, ones in re.findall(r"([一二三四五六七八九])成([一二三四五六七八九])?", t2):
        v = CN[tens] * 10 + (CN[ones] if ones else 0)
        vals.add(v)
    return vals


def wrong_values(src: str, out: str) -> list[int]:
    """源文里有、但输出里找不到等值的数字。"""
    bad = []
    out_vals = values_in(out)
    for n, nxt in re.findall(r"(\d+)\s*(.)?", src):
        if int(n) not in out_vals:
            bad.append(int(n))
    return bad


rows = [json.loads(l) for l in open("data/rewrite/v4_dev.jsonl", encoding="utf-8")]
preds = {json.loads(l)["x"]: json.loads(l)["pred"]
         for l in open("state/preds/fish-v4.jsonl", encoding="utf-8")}

tot = raw_bad = fix_bad = hum_bad = 0
examples = []
for r in rows:
    x, y = r["_meta"]["x"], r["_meta"]["y"]
    if not re.search(r"\d", x):
        continue
    tot += 1
    if wrong_values(x, y):
        hum_bad += 1
    p = preds.get(x)
    if not p:
        continue
    if wrong_values(x, p):
        raw_bad += 1
    p2 = R._restore_numbers(x, p)
    if wrong_values(x, p2):
        fix_bad += 1
    elif wrong_values(x, p) and len(examples) < 3:
        examples.append((x, p, p2))

print(f"  带数字的样本 {tot} 条")
print()
print(f"  模型（还原前）  数值出错  {raw_bad} 条（{raw_bad/max(1,tot)*100:.0f}%）")
print(f"  模型（还原后）  数值出错  {fix_bad} 条（{fix_bad/max(1,tot)*100:.0f}%）")
print(f"  真人原文       数值出错  {hum_bad} 条（{hum_bad/max(1,tot)*100:.0f}%）")
print()
if examples:
    print("  还原救回来的例子：")
    for x, p, p2 in examples:
        print(f"    模型 {p[:42]}")
        print(f"    还原 {p2[:42]}")
        print()
