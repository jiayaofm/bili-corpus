# -*- coding: utf-8 -*-
"""
给视频标题，让模型写评论，再和该视频下的真实评论比。

为什么是这个任务形态（而不是改写）
----------------------------------
改写需要 `(AI 原文 → 人写的改写)` 配对，本项目一条都没有；
生成只需要 `(视频标题 → 真人评论)`，手上有 1035 条。
而且训练形态和评估形态一致，LoRA 才有得训。

评估不靠人工逐条看，比四组可量化的东西：
    1. 长度分布 —— 真人评论是两极化的（一句话 / 一大段），模型容易全写中等长度
    2. 表情率   —— 真人约 35%
    3. P(AI)    —— 用项目自己的分类器判，越低越像人
    4. 模板重复 —— 生成的多条之间是不是同一个骨架换个词

用法：
    python tools/generate_comments.py --n 20
    python tools/generate_comments.py --n 5 --only BV1Qx4g6JEsr
    python tools/generate_comments.py --n 20 --dry-run      # 不调模型，只统计真人侧
"""
import argparse
import glob
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from pipeline.scorer import Scorer  # noqa: E402

OUT_MD = None   # 由 --prompt 决定，见 main()
OUT_JSON = None

# 分区映射：video_tid 不好认，直接用标题人工归类
DOMAIN = {
    "BV1w6ZEYrEsb": "美食",
    "BV1ArZnYUE9a": "游戏",
    "BV1H5Z4YhEsn": "游戏",
    "BV1stZAYvELe": "鬼畜",
    "BV1Qx4g6JEsr": "整活",
    "BV1edWM63Ewd": "科技",
    "BV1Sbnz6EEix": "科技",
}

GEN_SYSTEM_V1 = """你在刷 B 站。下面给你一个视频的标题和分区，请你像普通用户一样留一条评论。

规则：
1. 说一件具体的事，或者一句真实的感受/观点。不要面面俱到。
2. 不总结、不解释、不点评自己说过的话，不要「总结一下」「说白了就是」这类。
3. **长度要像真评论那样参差不齐**：有时候就一句「下饭这点是真的」，
   有时候写一整段百来字。不要每条都写成二十来字。
4. 可以用 B站 黑话、表情包、颜文字，但不是每条都要带表情。
5. 别写「大家怎么看」这种客套话。
6. 不要分点，不要小标题。

只输出一条评论本身，不要引号、不要编号、不要解释。"""

# v2：v1 实测的问题——中位 62 字（真人 25）、一条 10 字以下都没有、
# 80 字以上占 27%（真人 11%）、表情率 10%（真人 35%）。
# 所以 v2 把「分布」写成更硬的约束，压短、放长尾、加表情。
GEN_SYSTEM_V2 = """你在刷 B 站。下面给你一个视频的标题和分区，请你像普通用户一样留一条评论。

真人评论长什么样（这些是实测出的分布，照着写）：
· **绝大多数很短**。最常见的是一条十几二十个字的评论，说完就走。
· **经常只有几个字**：「笑死」「这也行」「真的有羊驼点赞」——这种大概占七分之一。
· **偶尔**（十次里一次）才会认真写一大段，上百字，那时候是真有话说。
· **大约三分之一的评论带表情或颜文字**，比如 😂 🤣 [doge] [笑哭] (´･ω･)。
· 不会被要求写完整，想到哪写到哪，句子可以断。

这样做会被判为不像人：
· 每次都写成一整段完整论述
· 每条都二十到六十字，长度全都一样
· 一条评论恨不得把想法说尽

还要注意：
· 不总结、不解释、不点评自己说过的话，不要「总结一下」「说白了就是」。
· 别写「大家怎么看」这种客套话。
· 不要分点、不要小标题。

只输出一条评论本身，不要引号、不要编号、不要解释。"""

PROMPTS = {"v1": GEN_SYSTEM_V1, "v2": GEN_SYSTEM_V2}

EMO_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]|\[[^\]]{1,12}\]")
PUNCT_RE = re.compile(r"[，。！？、；：…—~]")
NUM_RE = re.compile(r"\d")


def load_corpus():
    """返回 {bvid: {title, comments[...]}}"""
    out = {}
    for p in glob.glob(str(ROOT / "data" / "BV*_like.jsonl")):
        bvid = "BV" + Path(p).name.split("BV")[1][:10]
        rec = out.setdefault(bvid, {"title": None, "comments": []})
        for ln in open(p, encoding="utf-8"):
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if rec["title"] is None:
                rec["title"] = r.get("video_title")
            c = (r.get("content") or "").strip()
            if c:
                rec["comments"].append(c)
    return out


def profile(comments):
    """一组评论的统计画像。"""
    n = len(comments)
    if not n:
        return {}
    lens = [len(c) for c in comments]
    lens_sorted = sorted(lens)
    shorts = [c for c in comments if len(c) < 30]
    return {
        "n": n,
        "len_mean": sum(lens) / n,
        "len_median": lens_sorted[n // 2],
        "len_p10": lens_sorted[max(0, int(n * 0.10) - 1)],
        "len_p90": lens_sorted[min(n - 1, int(n * 0.90))],
        "short_lt10": 100.0 * sum(1 for L in lens if L < 10) / n,
        "long_gt80": 100.0 * sum(1 for L in lens if L > 80) / n,
        "emoji_rate": 100.0 * sum(1 for c in comments if EMO_RE.search(c)) / n,
        "punct_per100": 100.0 * sum(len(PUNCT_RE.findall(c)) for c in comments) / max(sum(lens), 1),
        "digit_rate": 100.0 * sum(1 for c in comments if NUM_RE.search(c)) / n,
        # 短评多样性：强行压短时，模型容易塌缩成几句固定套话。
        # 真人短评花样极多，所以这一列能直接暴露「压短就只剩套话」的问题。
        "short_n": len(shorts),
        "short_distinct": len(set(shorts)),
        "short_distinct_ratio": (100.0 * len(set(shorts)) / len(shorts)) if shorts else float("nan"),
    }


def template_repetition(comments):
    """粗测模板重复：取每条的前 6 个字做骨架，看有多少条共享骨架。"""
    if len(comments) < 2:
        return {"unique_prefix6": len(comments), "dup_ratio": 0.0, "worst": None}
    pres = [re.sub(r"[\s\W]|\[[^\]]*\]", "", c)[:6] for c in comments]
    cnt = Counter(p for p in pres if p)
    dup = sum(v - 1 for v in cnt.values() if v > 1)
    worst = cnt.most_common(1)[0] if cnt else (None, 0)
    return {
        "unique_prefix6": len(cnt),
        "dup_ratio": 100.0 * dup / len(comments),
        "worst": {"prefix": worst[0], "count": worst[1]},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="每个视频生成几条")
    ap.add_argument("--temperature", type=float, default=1.25)
    ap.add_argument("--only", help="只跑指定 bvid")
    ap.add_argument("--dry-run", action="store_true", help="不调模型，只统计真人侧")
    ap.add_argument("--max-videos", type=int, default=0, help="只跑前几个视频")
    ap.add_argument("--prompt", default="v2", choices=sorted(PROMPTS), help="提示词版本")
    a = ap.parse_args()

    corpus = load_corpus()
    items = [(bv, v) for bv, v in corpus.items() if v["comments"]]
    if a.only:
        items = [(bv, v) for bv, v in items if bv == a.only]
    items.sort(key=lambda kv: -len(kv[1]["comments"]))
    if a.max_videos:
        items = items[: a.max_videos]

    print("视频 %d 个\n" % len(items))
    for bv, v in items:
        print("  %-14s [%s] %-34s %3d 条" % (bv, DOMAIN.get(bv, "?"),
                                             (v["title"] or "?")[:32], len(v["comments"])))

    sc = Scorer()
    # 真人侧用同一个打分器判。注意：1035 条评论里有 263 条参与过分类器训练，
    # 这会让这部分真人评论的 P(AI) 偏乐观。未参与训练的 772 条不受影响。
    # 报告里要写明这一点，不能把这个 P(AI) 当成完全独立的判据。
    real_probs = {}
    print("\n统计真人侧 P(AI)…")
    for bv, v in items:
        real_probs[bv] = [sc.describe(c)["heldout_ai"] for c in v["comments"]]

    if a.dry_run:
        print("\n=== 真人侧画像（dry-run）===")
        for bv, v in items:
            p = profile(v["comments"])
            print("  %-14s 中位%3d 短句%3.0f%% 长句%3.0f%% 表情%3.0f%% P(AI)%.3f"
                  % (bv, p["len_median"], p["short_lt10"], p["long_gt80"],
                     p["emoji_rate"], sum(real_probs[bv]) / len(real_probs[bv])))
        return

    from pipeline.llm import LLMClient
    cli = LLMClient(verbose=True)
    print("\n模型 %s  每个视频生成 %d 条\n" % (cli.model, a.n))

    results = []
    t0 = time.time()
    for vi, (bv, v) in enumerate(items, 1):
        title = v["title"] or "?"
        dom = DOMAIN.get(bv, "其他")
        print("[%d/%d] %s [%s]" % (vi, len(items), title[:34], dom))
        gens = []
        for i in range(a.n):
            try:
                out = cli.chat(
                    [{"role": "system", "content": PROMPTS[a.prompt]},
                     {"role": "user", "content": "《%s》，分区：%s" % (title, dom)}],
                    temperature=a.temperature, max_tokens=700)
            except Exception as e:
                print("      调用失败：%s" % e)
                continue
            t = (out or "").strip().strip('"“”').strip()
            t = re.sub(r"^\s*(?:\d+[\.\)、]|[-*•])\s*", "", t)
            if len(t) >= 2:
                gens.append(t)
        if not gens:
            print("      全部生成失败")
            continue

        gp = profile(gens)
        gp["P(AI)"] = sum(sc.describe(g)["heldout_ai"] for g in gens) / len(gens)
        rp = profile(v["comments"])
        rp["P(AI)"] = sum(real_probs[bv]) / len(real_probs[bv])
        tr = template_repetition(gens)

        print("      中位%3d 短句%3.0f%% 长句%3.0f%% 表情%3.0f%% P(AI)%.3f 重复%3.0f%%"
              % (gp["len_median"], gp["short_lt10"], gp["long_gt80"],
                 gp["emoji_rate"], gp["P(AI)"], tr["dup_ratio"]))

        results.append({
            "bvid": bv, "title": title, "domain": dom,
            "n_generated": len(gens), "n_real": len(v["comments"]),
            "generated": gp, "real": rp,
            "template": tr, "samples": gens,
        })

    # ---- 汇总 ----
    def agg(key, side):
        vals = [r[side][key] for r in results if r.get(side) and key in r[side]]
        return sum(vals) / len(vals) if vals else float("nan")

    print("\n" + "=" * 78)
    print("%-14s %-8s %6s %6s %6s %6s %6s" % ("指标", "真人", "", "模型", "", "差", ""))
    print("-" * 78)
    rows_out = []
    for key, label, better in [
        ("len_median", "中位字数", None),
        ("len_mean", "平均字数", None),
        ("short_lt10", "10字以下%", "up"),
        ("long_gt80", "80字以上%", "up"),
        ("emoji_rate", "表情率%", None),
        ("punct_per100", "标点/百字", None),
        ("digit_rate", "含数字%", None),
        ("short_distinct_ratio", "短评去重%", "up"),
        ("P(AI)", "P(AI)", "down"),
    ]:
        rv, gv = agg(key, "real"), agg(key, "generated")
        diff = gv - rv
        print("%-14s %6.2f %-8s %6.2f %-8s %+6.2f" % (label, rv, "", gv, "", diff))
        rows_out.append({"metric": label, "real": rv, "gen": gv, "diff": diff})

    print("\n模板重复度（生成侧）")
    for r in results:
        print("  %-14s 骨架去重 %2d/%2d  重复 %3.0f%%  最常见「%s」×%d"
              % (r["bvid"], r["template"]["unique_prefix6"], r["n_generated"],
                 r["template"]["dup_ratio"],
                 (r["template"]["worst"] or {}).get("prefix", "?"),
                 (r["template"]["worst"] or {}).get("count", 0)))

    global OUT_MD, OUT_JSON
    OUT_MD = ROOT / "analysis" / ("v7_generation_%s.md" % a.prompt)
    OUT_JSON = ROOT / "state" / ("generation_eval_%s.json" % a.prompt)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "model": cli.model, "prompt": a.prompt, "n_per_video": a.n, "temperature": a.temperature,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "真人侧与生成侧都用同一个打分器判；1035 条真人评论中 263 条参与过分类器训练，"
                "这部分 P(AI) 偏乐观，其余 772 条未参与",
        "summary": rows_out, "per_video": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- markdown ----
    md = ["# v7 生成式评论 vs 真人评论", "",
          "> 任务：给视频标题 + 分区，让模型写一条评论，和该视频下的真实评论比较。",
          "> 模型：`%s`　每视频 %d 条　temperature %.2f" % (cli.model, a.n, a.temperature), "",
          "## 汇总", "",
          "| 指标 | 真人 | 模型 | 差 |", "|---|---|---|---|"]
    for x in rows_out:
        md.append("| %s | %.2f | %.2f | %+.2f |" % (x["metric"], x["real"], x["gen"], x["diff"]))

    md += ["", "## 分视频", "",
           "| 视频 | 分区 | 真人数 | 真人中位 | 模型中位 | 真人P(AI) | 模型P(AI) | 模板重复 |",
           "|---|---|---|---|---|---|---|---|"]
    for r in results:
        md.append("| %s | %s | %d | %.0f | %.0f | %.3f | %.3f | %.0f%% |"
                  % (r["bvid"], r["domain"], r["n_real"],
                     r["real"]["len_median"], r["generated"]["len_median"],
                     r["real"]["P(AI)"], r["generated"]["P(AI)"],
                     r["template"]["dup_ratio"]))

    md += ["", "## 生成样本（每个视频前 6 条）", ""]
    for r in results:
        md.append("### %s [%s]" % (r["title"], r["domain"]))
        md.append("")
        for s in r["samples"][:6]:
            md.append("- （%d 字）%s" % (len(s), s))
        md.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")

    print("\n用时 %.1f 分钟" % ((time.time() - t0) / 60))
    print("写好：%s" % OUT_MD)
    print("写好：%s" % OUT_JSON)


if __name__ == "__main__":
    main()
