# -*- coding: utf-8 -*-
"""真人评论的语法统计量。

用途：
  ①  对比鱼的改写和真人原文（tools/syntax_contrast.py）
  ②  给改写模型的输出打分（tools/check_syntax_match.py）
  ③  候选重排（best-of-N）的打分函数

每个维度都是**密度**或**比例**，所以不同长度的文本可比。
"""
import re

SENT_PUNCT = "。！？!?…~"
CLAUSE_PUNCT = "，,。！？!?…~；;：:、"
PARTICLES = "啊吧呢嘛哦诶呀哈嘿咯啦哇嗯"

# 口语里常见的连接/填充词
DISCOURSE = ["就是", "反正", "其实", "那个", "然后", "所以", "但是", "因为",
             "真的", "感觉", "好像", "居然", "竟然", "直接", "根本"]

# 书面语/总结腔
FORMAL = ["而", "因此", "从而", "不仅", "更是", "以及", "然而", "进而",
          "本质", "层面", "维度", "方向上"]

RHETORIC = ["不是", "难道", "哪有", "何必", "怎么", "凭什么", "谁能"]

# 真人评论在这些维度上明显不同于 AI 腔（945 对配对样本实测）：
#   句号密度     鱼 7.25 → 人 1.19     鱼高 +509%
#   句末有标点   鱼 0.99 → 人 0.26     鱼高 +285%
#   叠字重复     鱼 1.14 → 人 5.03     人高 +77%
#   平均分句长   鱼 8.89 → 人 14.76    人高 +66%
#   语气词密度   鱼 0.36 → 人 0.78     人高 +54%
# 所以「像人话」不等于「像好文章」，而是句号少、句子长、有语气词和重复。
HUMAN_TARGET = {
    "句末有标点": 0.26,
    "句号密度": 1.19,
    "平均分句长": 14.76,
    "叠字重复": 5.03,
    "语气词密度": 0.78,
    "短分句占比": 0.10,
    "数字英文占比": 6.98,
}


def clauses(t: str) -> list[str]:
    return [c for c in re.split(f"[{CLAUSE_PUNCT}]", t) if c.strip()]


def stats(t: str) -> dict | None:
    """返回 16 个语法维度的统计量；空文本返回 None。"""
    t = (t or "").strip()
    n = len(t)
    if not n:
        return None
    cl = clauses(t)
    lens = [len(c) for c in cl] or [n]
    han = len(re.findall(r"[\u4e00-\u9fff]", t)) or 1
    return {
        "字数": n,
        "分句数": len(cl),
        "平均分句长": sum(lens) / len(lens),
        "最短分句": min(lens),
        "短分句占比": sum(1 for x in lens if x <= 4) / len(lens),
        "句末有标点": 1.0 if t[-1] in SENT_PUNCT else 0.0,
        "逗号密度": t.count("，") / han * 100,
        "句号密度": t.count("。") / han * 100,
        "语气词密度": sum(t.count(c) for c in PARTICLES) / han * 100,
        "口语连接词": sum(t.count(w) for w in DISCOURSE) / han * 100,
        "书面连接词": sum(t.count(w) for w in FORMAL) / han * 100,
        "反问标记": sum(t.count(w) for w in RHETORIC) / han * 100,
        "第一人称我": t.count("我") / han * 100,
        "数字英文占比": len(re.findall(r"[0-9A-Za-z]", t)) / n * 100,
        "叠字重复": len(re.findall(r"(.)\1", t)) / han * 100,
    }


def human_score(t: str) -> float:
    """0~1，越大越像真人评论的语法形态。

    对 HUMAN_TARGET 里的每个维度算相对误差，再做平均。
    只用于**同一批候选之间排序**，不是绝对判定。
    """
    s = stats(t)
    if not s:
        return 0.0
    errs = []
    for k, target in HUMAN_TARGET.items():
        v = s[k]
        scale = max(abs(target), 0.5)
        errs.append(min(1.0, abs(v - target) / scale))
    return 1.0 - sum(errs) / len(errs)
