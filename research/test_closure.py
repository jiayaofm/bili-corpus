# -*- coding: utf-8 -*-
"""
测试「句子闭合度」类特征

用户洞察：
  大肥鱼把整句话都说出来 —— 每个分句语义完整，末尾还给判断/转折。
  真人的幽默来自「说半句，后半句留给你脑补」。

可量化的代理指标：
  · 句末是否闭合（。！？ 结尾）
  · 分句数量、平均分句长度、短分句占比（"半句话"多 → 短分句多）
  · 末尾是否有转折词（不过/但是/只是…）—— AI 爱在结尾升华
  · 末尾是否为判断句（…才算数 / …不顶）
  · 数字密度（AI 爱堆数据）
  · 省略号结尾（真人留白）

输出每个候选在 AI / 真人 两侧的命中率，判断是否可用。
"""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent
CORPUS = ROOT / "data" / "corpus.jsonl"
GT = ROOT / "labeling" / "ground_truth.csv"

SPLIT_RE = re.compile(r"[，。！？；\n]+")
TURN_RE = re.compile(r"^(?:不过|但是|只是|可惜|然而|可|但)")
JUDGE_RE = re.compile(r"(?:才算|不顶|更值得|才是|才是真|都算|也就|谈不上|说到底)")
DIGIT_RE = re.compile(r"\d")


def load():
    corpus = {}
    for line in CORPUS.open(encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            corpus[r["rpid"]] = r
    rows = list(csv.DictReader(GT.open(encoding="utf-8-sig", newline="")))
    out = []
    for r in rows:
        c = corpus.get(r["rpid"])
        if not c:
            continue
        t = c["content"]
        out.append({"text": t, "label": int(r["label"]), "domain": c.get("domain")})
    return out


def segments(t):
    return [s for s in SPLIT_RE.split(t) if s.strip()]


def feats(t):
    segs = segments(t)
    n = max(1, len(t))
    lens = [len(s) for s in segs] or [0]
    last = segs[-1] if segs else ""
    return {
        "句末闭合": int(bool(re.search(r"[。！？~]\s*$", t))),
        "省略号结尾": int(bool(re.search(r"(?:…|\.{3,}|。。)\s*$", t))),
        "分句数": len(segs),
        "平均分句长度": sum(lens) / len(lens),
        "短分句占比": sum(1 for x in lens if x <= 4) / len(lens),
        "转折收尾": int(bool(TURN_RE.search(last))),
        "判断收尾": int(bool(JUDGE_RE.search(last))),
        "数字密度": len(DIGIT_RE.findall(t)) / n * 100,
        "数字个数": len(DIGIT_RE.findall(t)),
        "分句>=6": int(len(segs) >= 6),
        "破折号转折": int("——" in t),
    }


def main():
    data = load()
    ai = [d for d in data if d["label"] == 1]
    hu = [d for d in data if d["label"] == 0]
    print(f"AI {len(ai)} 条 / 真人 {len(hu)} 条\n")

    names = list(feats("测试").keys())
    print(f"{'特征':<14}{'AI均值':>9}{'真人均值':>10}{'差值':>9}   判定")
    print("-" * 62)
    for k in names:
        a = [feats(d["text"])[k] for d in ai]
        h = [feats(d["text"])[k] for d in hu]
        ma, mh = sum(a) / len(a), sum(h) / len(h)
        diff = ma - mh
        # 二值特征的"命中率"更有意义
        if k in ("句末闭合", "省略号结尾", "转折收尾", "判断收尾", "分句>=6", "破折号转折"):
            ra, rh = ma * 100, mh * 100
            verdict = "强" if ra - rh > 15 else ("偏强" if ra - rh > 8 else ("偏弱" if ra - rh < -8 else ""))
            print(f"{k:<14}{ra:>8.0f}%{rh:>9.0f}%{ra-rh:>8.0f}pt   {verdict}")
        else:
            verdict = "有效" if abs(diff) > 0 else ""
            print(f"{k:<14}{ma:>9.2f}{mh:>10.2f}{diff:>9.2f}   {verdict}")

    print("\n--- AI 样本的分句画像（前 5 条）---")
    for d in ai[:5]:
        f = feats(d["text"])
        print(f"  分句{f['分句数']:>2}  均长{f['平均分句长度']:>5.1f}  短句占比{f['短分句占比']*100:>4.0f}%  "
              f"转折{f['转折收尾']} 判断{f['判断收尾']} | {d['text'][:46]}")

    print("\n--- 真人样本的分句画像（前 5 条）---")
    for d in hu[:5]:
        f = feats(d["text"])
        print(f"  分句{f['分句数']:>2}  均长{f['平均分句长度']:>5.1f}  短句占比{f['短分句占比']*100:>4.0f}%  "
              f"转折{f['转折收尾']} 判断{f['判断收尾']} | {d['text'][:46]}")


if __name__ == "__main__":
    main()
