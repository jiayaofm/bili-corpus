# -*- coding: utf-8 -*-
"""走完整服务链路评 40 条真实评论，量最终交付给用户的是什么。

跟 eval_real_comments.py 的区别：那个直接调模型，这个调 /rewrite 接口，
所以会经历完整的候选生成 → 硬过滤 → 打分选择 → 数字还原 → weak 标记。
也就是说，量到的就是用户在扩展里看到的效果。
"""
import glob
import json
import random
import re
import statistics as st
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")

CJK = re.compile(r"[^\u4e00-\u9fff0-9A-Za-z]")
N_SAMPLE = 40


def bg(t):
    t = CJK.sub("", t)
    return {t[i:i + 2] for i in range(len(t) - 1)}


def sim(a, b):
    A, B = bg(a), bg(b)
    return len(A & B) / len(A | B) if (A | B) else 1.0


def recall(src, out):
    A, B = bg(src), bg(out)
    return 1.0 if not A else len(A & B) / len(A)


def post(t, tmo=300):
    r = urllib.request.Request(
        "http://127.0.0.1:8849/rewrite",
        data=json.dumps({"text": t}).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=tmo).read())


# 收集真实评论
texts = []
for fp in glob.glob(str(ROOT / "data" / "whale_comments" / "*.jsonl")):
    for line in open(fp, encoding="utf-8"):
        r = json.loads(line)
        if r.get("skipped"):
            continue
        for c in (r.get("comments") or []):
            t = c if isinstance(c, str) else (c.get("text") or "")
            t = (t or "").strip()
            if len(t) >= 20 and not re.search(r"https?://", t):
                texts.append(t)
texts = list(dict.fromkeys(texts))

rng = random.Random(20261124)
at = [t for t in texts if "@" in t]
rest = [t for t in texts if "@" not in t]
rng.shuffle(rest)
sample = at[:8] + rest[:N_SAMPLE - min(8, len(at))]
rng.shuffle(sample)

print(f"  数据库里可用评论 {len(texts)} 条（含 @ 的 {len(at)} 条）")
print(f"  抽 {len(sample)} 条走完整服务（含 @ 的 {sum(1 for t in sample if '@' in t)} 条）")
print()

rows = []
for i, t in enumerate(sample, 1):
    t0 = time.time()
    try:
        r = post(t)
    except Exception as e:                                    # noqa: BLE001
        print(f"  {i:2d} 请求失败: {e}")
        continue
    dt = time.time() - t0
    o = r["rewritten"]
    rows.append({"src": t, "out": o, "s": sim(t, o), "rec": recall(t, o),
                 "weak": bool(r.get("weak")), "dt": dt,
                 "at": "@" in t})

ss = sorted(x["s"] for x in rows)
n = len(ss)
print("  改写幅度（相似度，越低改得越多）：")
print("    P10 %.2f  P25 %.2f  中位 %.2f  P75 %.2f  P90 %.2f  最大 %.2f" % (
    ss[n // 10], ss[n // 4], st.median(ss), ss[n * 3 // 4], ss[n * 9 // 10], ss[-1]))
print()
for name, fn in (("几乎没改 ≥0.85", lambda x: x >= 0.85),
                 ("小改 0.7~0.85", lambda x: 0.7 <= x < 0.85),
                 ("有改动 0.55~0.7", lambda x: 0.55 <= x < 0.7),
                 ("大改 <0.55", lambda x: x < 0.55)):
    k = sum(1 for x in ss if fn(x))
    print("    %-16s %2d 条 (%.0f%%)" % (name, k, k / n * 100))
print()
print("  weak 标记（服务自己承认没改到位）：%d/%d" % (
    sum(1 for x in rows if x["weak"]), len(rows)))
print("  延迟中位 %.1f 秒" % st.median([x["dt"] for x in rows]))
print()
print("  改得最少的三条：")
for x in sorted(rows, key=lambda r: -r["s"])[:3]:
    print("    相似度 %.2f  weak=%s  @=%s" % (x["s"], x["weak"], x["at"]))
    print("      原 %s" % x["src"][:52])
    print("      改 %s" % x["out"][:52])
print()
print("  改得最多的三条：")
for x in sorted(rows, key=lambda r: r["s"])[:3]:
    print("    相似度 %.2f  保留 %.2f" % (x["s"], x["rec"]))
    print("      原 %s" % x["src"][:52])
    print("      改 %s" % x["out"][:52])

(ROOT / "state" / "live_eval.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
print()
print("  明细已存 state/live_eval.json")
