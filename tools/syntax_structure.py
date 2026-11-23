# -*- coding: utf-8 -*-
"""句法结构分析：动词位置、宾语省略、谓语前置、连词位置。

跟 syntax_stats.py 的区别：那个只看表层统计（标点、字数、语气词），
这个用 jieba 的词性标注看**结构**——动词落在哪里、后面接不接宾语、
分句是不是以谓语开头、连词在句首还是句中。

对比两组：大肥鱼的改写 vs 对应的真人原话（同视频同话题）。
"""
import glob
import json
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import jieba
import jieba.posseg as pseg

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8")
jieba.setLogLevel(60)

CLAUSE_PUNCT = "，,。！？!?…~；;：:、"

# jieba 词性前缀
V_TAGS = ("v",)          # 动词
N_TAGS = ("n",)          # 名词
A_TAGS = ("a",)          # 形容词
R_TAGS = ("r",)          # 代词
D_TAGS = ("d",)          # 副词
C_TAGS = ("c",)          # 连词
U_TAGS = ("u",)          # 助词
Y_TAGS = ("y",)          # 语气词

CAUSAL = ["因为", "所以", "由于", "因此", "于是", "导致", "结果", "搞得", "以至于"]
CONTRAST = ["但是", "不过", "可是", "然而", "只是", "虽然", "却", "但", "反倒", "反而"]
ADD = ["而且", "并且", "甚至", "还有", "加上", "况且", "再说"]

ALL_CONN = set(CAUSAL + CONTRAST + ADD)


def tags_of(tag: str) -> str:
    return tag[0] if tag else "x"


def clauses_of(t: str) -> list[str]:
    return [c.strip() for c in re.split(f"[{CLAUSE_PUNCT}]", t) if c.strip()]


def analyze(t: str) -> dict | None:
    t = (t or "").strip()
    if not t:
        return None
    cl = clauses_of(t)
    if not cl:
        return None

    n_clause = 0
    first_is_pred = 0        # 分句以动词/形容词开头（谓语前置）
    verb_first_pos = []      # 首个动词的相对位置
    主語省略 = 0
    obj_omit = 0             # 动词后紧跟标点/助词/语气词（无宾语）
    obj_total = 0
    conn_at_head = 0         # 连词在分句首位
    conn_total = 0
    causal = contrast = add = 0
    verb_cnt = han_cnt = 0
    passives = 0
    verb_obj_pairs = []      # (动词, 是否带宾语)

    for c in cl:
        words = [(w.word, w.flag) for w in pseg.cut(c) if w.word.strip()]
        words = [(w, f) for w, f in words if not re.fullmatch(rf"[{CLAUSE_PUNCT}\s]+", w)]
        if not words:
            continue
        n_clause += 1
        kinds = [tags_of(f) for _, f in words]

        # 首个实词
        first_kind = kinds[0]
        if first_kind in V_TAGS or first_kind in A_TAGS:
            first_is_pred += 1

        # 首个动词位置
        vidx = next((i for i, k in enumerate(kinds) if k in V_TAGS), None)
        if vidx is not None:
            verb_first_pos.append(vidx / len(words))
            # 动词之前有没有主语（名词/代词）
            has_subj = any(k in N_TAGS or k in R_TAGS for k in kinds[:vidx])
            if not has_subj:
                主語省略 += 1

        # 动词后是否带宾语
        for i, (w, f) in enumerate(words):
            k = tags_of(f)
            if k in V_TAGS:
                verb_cnt += 1
                j = i + 1
                # 跳过副词「了/过/着」这类助词再看
                while j < len(words) and (
                        tags_of(words[j][1]) in U_TAGS or tags_of(words[j][1]) in D_TAGS):
                    j += 1
                if j >= len(words):
                    obj_omit += 1
                    obj_total += 1
                    verb_obj_pairs.append((w, 0))
                elif tags_of(words[j][1]) in N_TAGS or tags_of(words[j][1]) in R_TAGS:
                    obj_total += 1
                    verb_obj_pairs.append((w, 1))
                else:
                    obj_omit += 1
                    obj_total += 1
                    verb_obj_pairs.append((w, 0))

        # 连词位置
        for i, (w, f) in enumerate(words):
            if w in ALL_CONN:
                conn_total += 1
                if i == 0:
                    conn_at_head += 1
                if w in CAUSAL:
                    causal += 1
                elif w in CONTRAST:
                    contrast += 1
                elif w in ADD:
                    add += 1

        han_cnt += sum(1 for w, _ in words if re.search(r"[\u4e00-\u9fff]", w))
        if re.search(r"[把被]", c):
            passives += 1

    if not n_clause:
        return None
    han_cnt = han_cnt or 1
    return {
        "谓语前置率": first_is_pred / n_clause,
        "首动词位置": st.mean(verb_first_pos) if verb_first_pos else 0.0,
        "主语省略率": 主語省略 / max(1, len(verb_first_pos)),
        "宾语省略率": obj_omit / max(1, obj_total),
        "连词句首率": conn_at_head / max(1, conn_total),
        "因果连词密度": causal / han_cnt * 100,
        "转折连词密度": contrast / han_cnt * 100,
        "递进连词密度": add / han_cnt * 100,
        "动词密度": verb_cnt / han_cnt * 100,
        "把被字句率": passives / n_clause,
        "_verb_obj": verb_obj_pairs,
    }


def main():
    fish, human = [], []
    fish_v, human_v = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    fish_ex, human_ex = [], []

    for fp in glob.glob(str(ROOT / "data" / "whale_comments" / "*.jsonl")):
        for line in open(fp, encoding="utf-8"):
            r = json.loads(line)
            if r.get("skipped"):
                continue
            for rw in (r.get("rewrites") or []):
                o = (rw.get("origin") or "").strip()
                w = (rw.get("rewrite") or "").strip()
                if not o or not w or o == w or len(o) < 8:
                    continue
                so, sw = analyze(o), analyze(w)
                if not (so and sw):
                    continue
                human.append(so)
                fish.append(sw)
                for verb, has in sw["_verb_obj"]:
                    fish_v[verb][0] += has
                    fish_v[verb][1] += 1
                for verb, has in so["_verb_obj"]:
                    human_v[verb][0] += has
                    human_v[verb][1] += 1
                if len(fish_ex) < 6:
                    if so["谓语前置率"] > 0 and sw["谓语前置率"] == 0:
                        fish_ex.append(w)

    print(f"  配对样本 {len(fish)} 对")
    print()
    keys = [k for k in fish[0] if not k.startswith("_")]
    print("  %-12s %10s %10s %9s   %s" % ("结构维度", "鱼的改写", "真人原文", "差", "谁更高"))
    print("  " + "-" * 60)
    for k in keys:
        a = st.mean(x[k] for x in fish)
        b = st.mean(x[k] for x in human)
        d = a - b
        who = "鱼" if d > 1e-9 else ("人" if d < -1e-9 else "—")
        print("  %-12s %10.3f %10.3f %+9.3f   %s" % (k, a, b, d, who))

    print()
    print("  哪些动词最常在真人评论里省略宾语（出现≥8次，按省略率排序）：")
    rows = []
    for v, (no, n) in human_v.items():
        if n >= 8:
            rows.append((no / n, v, n, no))
    rows.sort(reverse=True)
    print("    %-8s %8s %6s" % ("动词", "省略率", "次数"))
    for rate, v, n, no in rows[:12]:
        f_no, f_n = fish_v.get(v, [0, 0])
        fr = (f_no / f_n) if f_n else None
        extra = f"   (鱼 {fr*100:.0f}%)" if fr is not None else ""
        print("    %-8s %7.0f%% %6d%s" % (v, rate * 100, n, extra))

    print()
    print("  哪些动词最常在鱼的改写里省略宾语：")
    rows2 = [(no / n, v, n) for v, (no, n) in fish_v.items() if n >= 8]
    rows2.sort(reverse=True)
    for rate, v, n in rows2[:8]:
        print("    %-8s %7.0f%% %6d" % (v, rate * 100, n))


if __name__ == "__main__":
    main()
