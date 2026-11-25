# -*- coding: utf-8 -*-
"""评估改写模型：这次主指标是「鱼分位有没有降」，不是「改了多少」。

## 为什么重写这个脚本

上一版只盯「输出与输入的相似度」，用来判断「改得够不够多」。
但方向搞反的时候，一个把鱼话改得更鱼的模型，相似度照样可以很漂亮——
它确实改了，只是改错了方向。所以这个指标对方向完全不敏感。

这次把鱼分位放到第一位：

    input_fish   输入（大肥鱼写的）的鱼分位
    out_fish     模型输出的鱼分位
    target_fish  同一对里真人原话的鱼分位（参考天花板）

真正的成功标准是 `out_fish` 接近 `target_fish`，
而且明显低于 `input_fish`。

## 数据

用 v4_dev（113 对，5 个留出视频）。这些视频在训练时完全没见过，
所以测的是泛化，不是记忆。
"""
import json
import statistics as st
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")

DEV = ROOT / "data" / "rewrite" / "v4_dev.jsonl"
N = 24          # 太多会等很久（每条约 8~30 秒）


def score(texts):
    out = []
    for i in range(0, len(texts), 50):
        req = urllib.request.Request(
            "http://127.0.0.1:8849/score",
            data=json.dumps({"texts": texts[i:i + 50]}).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        out += json.loads(urllib.request.urlopen(req, timeout=600).read())["results"]
    return out


def rewrite(text):
    req = urllib.request.Request(
        "http://127.0.0.1:8849/rewrite",
        data=json.dumps({"text": text}).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=600).read())


def main():
    rows = [json.loads(l) for l in open(DEV, encoding="utf-8")][:N]
    xs = [r["_meta"]["x"] for r in rows]     # 鱼话
    ys = [r["_meta"]["y"] for r in rows]     # 真人原话

    print(f"  验证集取 {len(rows)} 对（都是训练时没见过的视频）")
    print()

    print("  给输入和目标打分 …", flush=True)
    rx, ry = score(xs), score(ys)

    print(f"  逐条改写 {len(xs)} 条（每条几秒到几十秒）…", flush=True)
    outs, t0 = [], time.time()
    for i, x in enumerate(xs, 1):
        try:
            outs.append(rewrite(x)["rewritten"])
        except Exception as e:                                  # noqa: BLE001
            print(f"    {i} 失败: {e}")
            outs.append("")
        if i % 10 == 0:
            print(f"    已完成 {i}/{len(xs)}  用时 {time.time() - t0:.0f}s", flush=True)

    print("  给输出打分 …", flush=True)
    ro = score(outs)

    fi = [r["fish_pct"] or 0 for r in rx]
    ft = [r["fish_pct"] or 0 for r in ry]
    fo = [r["fish_pct"] or 0 for r in ro]
    ai_i = [r["ai_pct"] for r in rx]
    ai_t = [r["ai_pct"] for r in ry]
    ai_o = [r["ai_pct"] for r in ro]

    print()
    print("=" * 70)
    print("  鱼分位（越低越不像大肥鱼）")
    print("  " + "-" * 66)
    print("  %-14s %8s %8s %8s" % ("", "中位", "P75", "≥90 占比"))
    for name, v in (("输入·鱼话", fi), ("输出·模型", fo), ("目标·真人", ft)):
        s = sorted(v)
        print("  %-14s %8.1f %8.1f %7.0f%%" % (
            name, st.median(v), s[int(len(s) * .75)],
            sum(1 for x in v if x >= 90) / len(v) * 100))

    print()
    print("  AI 分位")
    for name, v in (("输入", ai_i), ("输出", ai_o), ("目标", ai_t)):
        print("    %-8s 中位 %5.1f" % (name, st.median(v)))

    dec = sum(1 for a, b in zip(fi, fo) if b < a)
    big = sum(1 for a, b in zip(fi, fo) if b < a - 10)
    print()
    print("  【主指标】鱼分位下降：%d/%d（%.0f%%）  下降超过 10 分的：%d 条（%.0f%%）" % (
        dec, len(fi), dec / len(fi) * 100, big, big / len(fi) * 100))
    print("  离目标还有多远：输出鱼分位中位比真人目标高 %.1f 分" % (
        st.median(fo) - st.median(ft)))

    print()
    print("  抽样看几条（输入鱼分位 → 输出鱼分位 / 目标鱼分位）：")
    order = sorted(range(len(xs)), key=lambda i: fo[i] - fi[i])
    for i in order[:3]:
        print()
        print("    [降得最多] %5.1f → %5.1f（目标 %.1f）" % (fi[i], fo[i], ft[i]))
        print("      入 %s" % xs[i][:60])
        print("      出 %s" % outs[i][:60])
    for i in order[-2:]:
        print()
        print("    [降得最少/升高] %5.1f → %5.1f（目标 %.1f）" % (fi[i], fo[i], ft[i]))
        print("      入 %s" % xs[i][:60])
        print("      出 %s" % outs[i][:60])

    (ROOT / "state" / "rewrite_eval_fish.json").write_text(json.dumps(
        [{"x": xs[i], "y": ys[i], "out": outs[i], "fi": fi[i], "fo": fo[i],
          "ft": ft[i]} for i in range(len(xs))], ensure_ascii=False, indent=1),
        encoding="utf-8")
    print()
    print("  明细 → state/rewrite_eval_fish.json")


if __name__ == "__main__":
    main()
