# -*- coding: utf-8 -*-
"""评一评新的人设写出来的评论。

对每条评论同时看三样：
  ① 打分器给的 AI 分位 / 鱼分位（代理指标，能被骗，只看趋势）
  ② 表层特征：句号密度、分句长、叠字、语气词——这些是提示词直接瞄准的
  ③ 人的判断：内容有没有说具体的、有没有编造

同时拿真人语料的分布做参照，否则「好还是不好」没有刻度。
"""
import json
import re
import statistics as st
import sys
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.stdout.reconfigure(encoding="utf-8")

import pipeline.features as F  # noqa: E402

SRC = Path(r"C:\Users\m1660\.dsh\whale-draft-comments.md")


def load_comments():
    """从草稿文件里抽出引用的评论正文。"""
    out = []
    for line in SRC.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("> "):
            out.append(s[2:].strip())
    return out


def score(texts):
    r = urllib.request.Request(
        "http://127.0.0.1:8849/score",
        data=json.dumps({"texts": texts}).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=300).read())["results"]


def ref_stats():
    """真人语料的表层特征分布，做刻度用。"""
    rows = [json.loads(l) for l in
            open(ROOT / "data" / "corpus.jsonl", encoding="utf-8")]
    texts = [(r.get("content") or "").strip() for r in rows]
    texts = [t for t in texts if 15 <= len(t) <= 200]
    per = [F.featurize(t, "") for t in texts]
    keys = ["句号数", "平均分句长", "逗号数", "分号数", "感叹号数", "问号数"]
    return {k: (st.median([p[k] for p in per]),
                float(np.percentile([p[k] for p in per], 90)))
            for k in keys if k in per[0]}


def main():
    cs = load_comments()
    print(f"  草稿里 {len(cs)} 条评论")
    print()

    res = score(cs)
    ref = ref_stats()

    print("  %-3s %4s %6s %7s %7s %6s %6s" % (
        "#", "字数", "AI分位", "鱼分位", "句号", "分句长", "叠字"))
    print("  " + "-" * 50)
    rows = []
    for i, (t, r) in enumerate(zip(cs, res), 1):
        f = F.featurize(t, "")
        rows.append((i, t, r, f))
        print("  %-3d %4d %6.1f %7s %6.1f %6.1f %6.1f" % (
            i, len(t), r["ai_pct"],
            ("%.1f" % r["fish_pct"]) if r.get("fish_pct") is not None else "—",
            f["句号数"], f["平均分句长"], f["用字丰富度"]))

    print()
    print("  真人语料参照（中位 / P90）：")
    for k, (m, p90) in ref.items():
        print("    %-8s 中位 %5.2f   P90 %5.2f" % (k, m, p90))

    ais = [r["ai_pct"] for _, _, r, _ in rows]
    fs = [r["fish_pct"] for _, _, r, _ in rows if r.get("fish_pct") is not None]
    print()
    print("  汇总：AI 分位 中位 %.1f  最高 %.1f ｜ 判红(≥90) %d/%d" % (
        st.median(ais), max(ais), sum(1 for a in ais if a >= 90), len(ais)))
    if fs:
        print("        鱼分位 中位 %.1f  最高 %.1f" % (st.median(fs), max(fs)))
    dots = [f["句号数"] for _, _, _, f in rows]
    print("        句号总数 %d（%d 条里）" % (sum(dots), len(dots)))
    print("        叠字密度中位 %.2f（真人 5.03）" % st.median(
        [f["用字丰富度"] for _, _, _, f in rows]))
    print()

    print("  " + "=" * 60)
    for i, t, r, f in rows:
        print()
        print(f"  [{i}] AI {r['ai_pct']:.1f}%  ({r['ai_level']})   "
              f"鱼 {r.get('fish_pct')}   句号 {f['句号数']:.0f}   "
              f"分句 {f['平均分句长']:.1f} 字")
        print(f"      {t}")


if __name__ == "__main__":
    main()
