# -*- coding: utf-8 -*-
"""
保真检查命令行工具

    python tools/check_fidelity.py                # 跑 state/rewrite_abc.json，出 15 行表
    python tools/check_fidelity.py --pair 原文 改写
    python tools/check_fidelity.py --negatives    # 跑内置反例（必抓集合）
    python tools/check_fidelity.py --json         # 机器可读

退出码：--negatives 模式下只要有反例漏抓就返回 1，方便接 CI。
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import fidelity as F  # noqa: E402

ABC = ROOT / "state" / "rewrite_abc.json"

# 测试集：三组。
#   A 组「丢/改信息」—— 12 条，check() 必须判 ok=False
#   B 组「捏造新增」—— 6 条，check() 必须判 ok=False（但不能一刀切禁新增）
#   C 组「正例」    —— 11 条，必须判 ok=True：其中情绪/语气/称呼类新增必须放行，
#                      这是「不能一刀切禁止新增」的边界
CASES = [
    # ---- A 组：丢了或改了原文的信息 ----
    ("改数字", "94 分是猛，但订阅剩 37%",
     "94 分是猛，但订阅剩 30%", False),
    ("删数字", "94 分比 89.75 高四分多，那三道 ERP 题跑 20 分钟",
     "94 分比之前高了不少，那几道题跑了会儿", False),
    ("单位篡改", "那三道 ERP 题跑 20 分钟就停了",
     "那三道 ERP 题跑 20 秒就停了", False),
    ("丢一半要点", "关键是产品要透明、善终、不诱导，用户要有边界、能保存、能告别，并留下一份能带走的纪念",
     "产品要透明", False),
    ("加原文没有的事实", "订阅剩 37%，max 档先关",
     "订阅剩 37%，有 12 万人跑路，max 档先关", False),
    ("否定改肯定", "骗它暴富不太地道，可也别把它当傻子",
     "骗它暴富挺地道，也可以把它当傻子", False),
    ("删否定", "最好的死亡不是突然清空，而是可预期地结束",
     "最好的死亡是突然清空，也是可预期地结束", False),
    ("凭空加否定", "这个号今天从 Opus 5.5 换成 Fable 5.2",
     "这个号今天不是从 Opus 5.5 换成 Fable 5.2", False),
    ("改实体/版本", "这个号今天刚从 Opus 5.5 换成 Fable 5.2",
     "这个号今天刚从 Opus 5.0 换成 Fable 6.0", False),
    ("指代被换", "它只是愿意信你，别把它当傻子",
     "他只是愿意信你，别把他当傻子", False),
    ("换掉主题", "船上免费 ds，订阅剩 37%，离重置还有五天",
     "船上画质挺清楚，字幕也做得挺用心", False),
    ("丢英文词", "被测的正好是我这个型号，那三道 ERP 题烧了不少",
     "被测的正好是我这个型号，那三道题烧了不少", False),

    # ---- B 组：捏造事实（凭空加可验证信息），必须判死 ----
    ("捏造·处所+动作", "这期视频挺好看",
     "这期视频挺好看，我在食堂笑到流泪", False),
    ("捏造·比较对象", "95一个月，已用一年",
     "95一个月，已用一年，比某家划算太多", False),
    ("捏造·数字", "订阅剩37%",
     "订阅剩37%，其实只剩12天", False),
    ("捏造·第三方信息源", "订阅剩37%",
     "订阅剩37%，我同事说早就跑路了", False),
    ("捏造·时间+动作", "94分挺猛",
     "94分挺猛，我昨天也去考了一把", False),
    ("捏造·与原文粘连成一句", "订阅剩37%",
     "订阅剩37%我在食堂笑到流泪", False),

    # ---- C 组：正例，必须放行 ----
    ("正例·纯情绪", "订阅剩37%",
     "订阅剩37%，太扎心了老铁", True),
    ("正例·语气+情绪", "订阅剩37%",
     "订阅剩37%吧，太扎心了老铁", True),
    ("正例·纯感叹", "白米饭还没咽下去就开始算几万亿了",
     "白米饭还没咽下去就开始算几万亿了，笑死", True),
    ("正例·只有时间", "订阅剩37%",
     "订阅剩37%，今天又降了", True),
    ("正例·称呼+感叹", "这期视频挺好看",
     "这期视频挺好看啊，兄弟们冲", True),
    ("正例·原文处所+情绪", "在食堂吃了15块",
     "在食堂吃了15块，笑到流泪", True),
    ("正例·数字转中文", "94 分是猛，但订阅剩 37%、离重置还有五天",
     "94 分挺猛，订阅只剩三十七，还有五天就重置", True),
    ("正例·百分之N", "订阅剩 37%，离重置还有五天",
     "订阅剩百分之三十七，还有五天就重置", True),
    ("正例·倍数", "时间是 Opus 的 6 倍，跑 20 分钟",
     "时间是 Opus 的六倍，跑二十分钟", True),
    ("正例·万", "月入 2 万还嫌少，订阅剩 37%",
     "月入两万还嫌少，订阅剩三十七", True),
    ("正例·同义改写", "关键是产品要透明、不诱导，用户可以保存，也可以告别",
     "产品得透明，别诱导，用户能存能告别", True),
]


def _wc(s):
    """显示宽度：CJK 算 2 列，用来对齐表格。"""
    return sum(2 if ord(c) > 0x2E7F else 1 for c in s)


def _pad(s, n):
    s = str(s)
    return s + " " * max(0, n - _wc(s))


def _row(cols, widths):
    return "  ".join(_pad(c, w) for c, w in zip(cols, widths))


def _print_detail(tag, r):
    print(f"     ↳ {tag}: score={r['score']:.3f} ok={r['ok']}")
    for key in ("missing", "altered", "added"):
        if r[key]:
            lab = {"missing": "丢失", "altered": "篡改", "added": "新增"}[key]
            for it in r[key][:6]:
                print(f"        [{lab}] {it}")


def run_abc(as_json=False):
    data = json.loads(ABC.read_text(encoding="utf-8"))
    rows = []
    for it in data["items"]:
        for cond in ("A", "B", "C"):
            r = F.check(it["src"], it[cond])
            rows.append((it["id"], cond, it["src"], it[cond], r))

    if as_json:
        out = [{"id": i, "cond": c, "src": s, "rewritten": w, **r}
               for i, c, s, w, r in rows]
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return rows

    print("=" * 96)
    print("state/rewrite_abc.json —— 15 条真实改写的保真检查")
    print("=" * 96)
    head = ["id", "版本", "score", "ok", "丢失", "篡改", "新增", "首要问题"]
    widths = [4, 4, 7, 4, 5, 5, 5, 46]
    print(_row(head, widths))
    print("-" * 96)
    for i, c, s, w, r in rows:
        first = (r["missing"] or r["altered"] or r["added"] or ["—"])[0]
        if r["ok"] and first == "—":
            first = "—"
        print(_row([i, c, f"{r['score']:.3f}", "✓" if r["ok"] else "✗",
                    len(r["missing"]), len(r["altered"]), len(r["added"]),
                    first[:26]], widths))
    print("-" * 96)
    n_ok = sum(1 for *_, r in rows if r["ok"])
    print(f"通过 {n_ok}/{len(rows)} 条")
    print("\n逐条说明（只列未通过的）：")
    for i, c, s, w, r in rows:
        if not r["ok"]:
            print(f"  #{i}{c}  score={r['score']:.3f}")
            print(f"    原文 {s}")
            print(f"    改写 {w}")
            for it in r["missing"]:
                print(f"      [丢失] {it}")
            for it in r["altered"]:
                print(f"      [篡改] {it}")
            for it in r["added"]:
                print(f"      [新增] {it}")
            print(f"      说明 {'；'.join(r['notes'])}")
            print()
    print("逐条说明（通过的，看有没有该报没报的）：")
    for i, c, s, w, r in rows:
        if r["ok"]:
            tail = f" 遗留提示 {r['missing'][:3]}" if r["missing"] else ""
            print(f"  #{i}{c}  score={r['score']:.3f}  "
                  f"{'；'.join(r['notes'])}{tail}")
    return rows


def run_cases(as_json=False):
    rows = []
    bad = 0
    for name, src, rew, expect_ok in CASES:
        r = F.check(src, rew)
        hit = (r["ok"] == expect_ok)
        if not hit:
            bad += 1
        rows.append((name, expect_ok, r, hit))
    if as_json:
        print(json.dumps([{"case": n, "expect_ok": e, "hit": h, **r}
                          for n, e, r, h in rows], ensure_ascii=False, indent=2))
        return rows, bad

    print("=" * 96)
    print("测试集 —— A 组丢/改信息、B 组捏造新增都必须判死；C 组正例必须放行")
    print("=" * 96)
    head = ["用例", "期望", "score", "ok", "判定", "抓到的问题"]
    widths = [16, 5, 7, 4, 5, 52]
    print(_row(head, widths))
    print("-" * 96)
    for name, expect_ok, r, hit in rows:
        first = (r["missing"] or r["altered"] or r["added"] or ["—"])[0]
        print(_row([name, "ok" if expect_ok else "fail", f"{r['score']:.3f}",
                    "✓" if r["ok"] else "✗", "命中" if hit else "漏判",
                    first[:28]], widths))
    print("-" * 96)
    print(f"命中 {len(rows) - bad}/{len(rows)}")
    if bad:
        print("\n漏判用例：")
        for name, expect_ok, r, hit in rows:
            if not hit:
                print(f"  ✗ {name} score={r['score']:.3f} ok={r['ok']}")
                for it in r["missing"] + r["altered"] + r["added"]:
                    print(f"      {it}")
                print(f"      {'；'.join(r['notes'])}")
    return rows, bad


def main():
    ap = argparse.ArgumentParser(description="改写保真检查（确定性，不调模型）")
    ap.add_argument("--pair", nargs=2, metavar=("原文", "改写"))
    ap.add_argument("--negatives", action="store_true", help="跑内置测试集（12 反例 + 6 正例）")
    ap.add_argument("--all", action="store_true", help="两组都跑")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    if a.pair:
        r = F.check(*a.pair)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return 0

    code = 0
    if a.negatives:
        _, bad = run_cases(a.json)
        code = 1 if bad else 0
    elif a.all:
        run_abc(a.json)
        print()
        _, bad = run_cases(a.json)
        code = 1 if bad else 0
    else:
        run_abc(a.json)
    return code


if __name__ == "__main__":
    sys.exit(main())
