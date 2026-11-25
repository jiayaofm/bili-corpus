# -*- coding: utf-8 -*-
"""对比 normal 和 conservative 两种改写，看「语义不通」是不是改太多导致的。

同一批输入各生成一次：
  normal       模型自由发挥（改动大，容易跑偏）
  conservative 明确要求「只换措辞，逻辑结构和连接词一律不动」

如果 conservative 明显更通顺，说明问题在改动幅度上，调参数能解决；
如果一样不通顺，说明是 1.7B 的能力上限，得加数据或换模型。
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(ROOT / "tools"))
from check_coherence import check  # noqa: E402

CASES = [
    "我看是歪了。要按我的打算，该去点生物技术，那样满街就不是机器人，是肉体复制人",
    "换十年前，我会觉得这剧纯鬼扯。可网上那些奇葩看多了，反倒觉得这种抽象人类大赏才真。"
    "我们这儿公安局年底正开着防盗会议，居然有贼挑这时候溜进局里偷东西，把局长气得够呛。",
    "不开玩笑，以后肯定有《刀剑神域》《头号玩家》那种游戏。让人消费，就跟让仓鼠蹬轮子一样，"
    "把多余的精力磨掉",
    "很多人说后面是机械降神。其实是编剧给万梓强开了挂：原型本来就好抓，压根没这么难",
    "我记得半个世纪前有三样发明没推广开，人类亏大了。一是拿水当燃料的汽车，"
    "二是一种治蛀牙一劳永逸的漱口水，三是不用电线就能全球供电的设备。"
    "细节我记不太清，但这三样的共同点是都挡了别人财路，最后被摁死了",
    "也就是密子君的视频，看的时候不用管进度条。八碗下去还战力满满，根本不用替她担心吃不下",
]


def call(path, body, tmo=600):
    req = urllib.request.Request(
        f"http://127.0.0.1:8849{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=tmo).read())


def sim(a, b):
    import re
    C = re.compile(r"[^\u4e00-\u9fff0-9A-Za-z]")
    A = C.sub("", a)
    B = C.sub("", b)
    sa = {A[i:i + 2] for i in range(len(A) - 1)}
    sb = {B[i:i + 2] for i in range(len(B) - 1)}
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


def main():
    rows = []
    for t in CASES:
        r = {}
        for mode in ("normal", "conservative"):
            d = call("/rewrite", {"text": t, "mode": mode})
            o = d["rewritten"]
            s = call("/score", {"texts": [o]})["results"][0]
            r[mode] = {"out": o, "fish": s["fish_pct"] or 0, "ai": s["ai_pct"],
                       "sim": sim(t, o), "bad": check(t, o, "")}
        rows.append({"x": t, **r})

    print()
    for i, r in enumerate(rows, 1):
        a, b = r["normal"], r["conservative"]
        print(f"  [{i}] 入 {r['x'][:66]}")
        for tag, v in (("A 自由", a), ("B 保守", b)):
            flag = " ".join(v["bad"].keys()) if v["bad"] else "✓"
            print(f"      {tag}  相似度{v['sim']:.2f} 鱼{v['fish']:5.1f} AI{v['ai']:5.1f}  {flag}")
            print(f"            {v['out'][:68]}")
        print()

    print("  " + "=" * 66)
    for tag, k in (("A 自由改动", "normal"), ("B 保守", "conservative")):
        n = len(rows)
        probs = sum(1 for r in rows if r[k]["bad"])
        print("  %-10s 检出问题 %d/%d  相似度中位 %.2f  鱼分位中位 %.1f" % (
            tag, probs, n,
            sorted(r[k]["sim"] for r in rows)[n // 2],
            sorted(r[k]["fish"] for r in rows)[n // 2]))

    (ROOT / "state" / "ab_conservative.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n  明细 → state/ab_conservative.json")


if __name__ == "__main__":
    main()
