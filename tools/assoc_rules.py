# -*- coding: utf-8 -*-
"""
v6 关联规则挖掘 —— 自实现 Apriori（不依赖 mlxtend）

为什么要做这个：课程四个方向里分类、聚类已经做过（v5）。本脚本补关联规则，
用来检验核心论点「AI 味来自特征组合，不是单一特征」：

  * 单特征就能高置信度判 AI 吗？
  * 有没有必须两个以上同时出现才判得准的组合？
  * 组合的增益是真的，还是仅仅因为这些特征本来就高度相关（都跟篇幅挂钩）？

设计要点（和诚实性直接相关，先读这段）：

1. 项集空间只由文本特征构造，不看标签。特征全部复用 pipeline/features.py
   （只读导入，不修改），标签只作为额外的一个项 `label=AI`。因此「不含 AI 的
   频繁项集」与标签无关，只需挖一次；任意标签划分下，{X} → {AI} 的
   support / confidence / lift 都能由位图交集直接算出。这与「把 AI 项一起丢进
   Apriori」在数学上等价（Apriori 单调性），`--self-check` 会用完整 Apriori 复核。

2. 离散化切点是代码常量（CUTS），由 263 条标注样本的**特征分布**（与标签无关）
   算出后固化，不在每次运行/每一折重新拟合分位。
   `--show-cuts` 打印切点、每档样本量、档内 AI 比例，可逐条核对。

3. 263 条、63 个正类，规则极容易挖出一堆偶然的漂亮数字。所以自带三道检验：
   - 每条规则一个超几何精确检验 p 值 + Bonferroni 修正；
   - 组合增益的条件检验（在父项集层内再检验新增项，「多这一项」到底有没有用）；
   - 标签置换检验（打乱 63 个 AI 标签 N 次，看「漂亮规则」数量的零分布）；
   - 再加上 5 折交叉验证（训练折挖规则，测试折量置信度）。

用法：
    python -X utf8 tools/assoc_rules.py                  # 主分析（全 263 条）
    python -X utf8 tools/assoc_rules.py --show-cuts       # 只核对离散化
    python -X utf8 tools/assoc_rules.py --quick           # 跳过置换/交叉验证
    python -X utf8 tools/assoc_rules.py --min-support 0.02
    python -X utf8 tools/assoc_rules.py --subset random   # 稳健性：只用非候选层
    python -X utf8 tools/assoc_rules.py --boot 30         # 对照：随机抽 32 个正类（等正类数）
    python -X utf8 tools/assoc_rules.py --include-account # 把 E 账号类也放进来
    python -X utf8 tools/assoc_rules.py --plot            # 出 support-confidence 散点
    python -X utf8 tools/assoc_rules.py --json out.json   # 机器可读输出
    python -X utf8 tools/assoc_rules.py --self-check      # 复核位图法与完整 Apriori 等价

输出是 markdown 表格，便于直接贴进 analysis/v6_assoc.md。
不修改 pipeline / labeling / analysis 下的任何文件（--plot 除外），不跑模型，不用 GPU。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import hypergeom

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import features as F  # noqa: E402

GT_PATH = ROOT / "labeling" / "ground_truth.csv"
CORPUS_PATH = ROOT / "data" / "corpus.jsonl"
FIG_PATH = ROOT / "analysis" / "fig_v6_support_conf.png"

AI_ITEM = "label=AI"

# ======================================================================
# 一、离散化方案（常量）
# ======================================================================
# 二值特征：原样，只建「出现」项（例 `有破折号`）。缺席是默认状态，不单独建项——
# 否则「无「」∧ 无Markdown ∧ 无BV号」这类恒真组合会淹没规则表。
# 缺席项在「单特征判别力」那张表里单独补算（见 single_feature_table），不会漏掉。
# 注：`感叹号数` 在本样本里只取 0/1（没有「！！」），按二值特征处理。
BINARY_FEATURES = [
    "有「」", "有破折号", "有分号", "有冒号", "有Markdown",
    "不是A是B", "…是真的", "比…更/都", "指代这条", "另一回事", "本质内核腔",
    "自称本鱼", "助手视角", "鱼作第一人称", "角色自称", "自我代入",
    "口语判断", "倒装对仗", "反诘对比", "我最在意", "分享推荐",
    "感叹号数", "有表情", "有BV号", "有链接", "句末闭合",
    "名字含鱼",          # E 账号类，默认不进项集空间，--include-account 才启用
]

PRESENCE_NAME = {"感叹号数": "有感叹号"}   # 项名比特征名好读

# 连续/计数特征的切点。统一判定「档号 = 满足 v > cut 的 cut 个数」：
#   cuts 长度 1 → 两档；长度 2 → 三档。
# 取值来源（写死，不随运行变化）：
#   * 分布不退化、真有中段的特征用三分位 p33 / p67；
#   * 退化特征（p33 == p67，即 >2/3 样本挤在 0；或中位数等于最小值）改用「有/无」二分。
# 字数 p67 = 35.67，取 35 可让分区与三分位完全一致（整数值特征向下取整）。
# 【重要】切点只用特征分布、不看标签，所以这一步不引入标签泄漏。
CUTS = {
    # —— 三分位三档（5 个）——
    "字数": (17, 35),            # p33=17.00  p50=24.00  p67=35.67 → ≤17 / 18–35 / ≥36
    "用字丰富度": (0.83, 0.95),    # p33=0.8333 p50=0.8913 p67=0.9516
    "逗号数": (0, 1),             # p33=0.00   p50=1       p67=1.00 → =0 / =1 / ≥2
    "分句数": (1, 3),             # p33=1.00   p50=2       p67=3.00 → =1 / 2–3 / ≥4
    "平均分句长": (9.22, 14.0),    # p33=9.2167 p50=11.80   p67=14.00
    # —— 退化，二分（9 个）——
    "行数": (1,),                 # 中位数=最小值=1（81% 单行）→ =1 / ≥2
    "句号数": (0,),               # p33=p67=0 → =0 / ≥1
    "问号数": (0,),               # p33=p67=0 → =0 / ≥1
    "顿号数": (0,),               # p33=p67=0 → =0 / ≥1
    "有@": (0,),                  # p33=p67=0 → =0 / ≥1
    "短分句占比": (0.0,),          # p33=p67=0 → =0 / >0
    "数字密度": (0.0,),            # p33=p67=0 → =0 / >0
    "语气词密度": (0.0,),          # p33=p67=0 → =0 / >0
    "提问密度": (0.0,),            # p33=p67=0 → =0 / >0
    # —— E 账号类（仅 --include-account）——
    "名字长度": (5, 7),            # p33=5 p50=6 p67=7
}

INTEGER_FEATURES = {"字数", "行数", "逗号数", "句号数", "问号数", "顿号数",
                    "有@", "分句数", "名字长度"}

# 篇幅/密度块：用于回答「规则是不是只在说『长评的特征一起出现』」
LENGTH_BLOCK = {"字数", "行数", "逗号数", "句号数", "顿号数", "分句数",
                "平均分句长", "用字丰富度"}

# 特征 → 所属组（本地表，不去改 features 模块）
FEATURE2GROUP = {n: g for g, names in F.GROUPS.items() for n in names}


# ======================================================================
# 二、离散化
# ======================================================================
def _label(name: str, lo, hi, integer: bool) -> str:
    """档位标签，区间语义 (lo, hi]。"""
    if lo is None and hi is None:
        return name
    if lo is None:
        if hi == 0:
            return f"{name}=0"
        return f"{name}≤{int(hi)}" if integer else f"{name}≤{hi:g}"
    if hi is None:
        return f"{name}≥{int(lo) + 1}" if integer else f"{name}>{lo:g}"
    if integer:
        lo_i, hi_i = int(lo) + 1, int(hi)
        return f"{name}={hi_i}" if lo_i == hi_i else f"{name}={lo_i}–{hi_i}"
    return f"{name}{lo:g}–{hi:g}"


def bin_labels(name: str) -> list[str]:
    cuts = CUTS[name]
    integer = name in INTEGER_FEATURES
    out = []
    for j in range(len(cuts) + 1):
        lo = cuts[j - 1] if j > 0 else None
        hi = cuts[j] if j < len(cuts) else None
        out.append(_label(name, lo, hi, integer))
    return out


def bin_index(name: str, v: float) -> int:
    b = 0
    for c in CUTS[name]:
        if v > c:
            b += 1
    return b


def build_vocab(cols: list[str]) -> list[str]:
    items: list[str] = []
    for name in cols:
        if name in BINARY_FEATURES:
            items.append(PRESENCE_NAME.get(name, name))
        elif name in CUTS:
            items.extend(bin_labels(name))
        else:
            raise SystemExit(f"特征 {name} 没有离散化定义（不在 BINARY_FEATURES / CUTS 里）")
    return items


def row_items(rec: dict, cols: list[str]) -> frozenset:
    out = set()
    for name in cols:
        v = rec[name]
        if name in BINARY_FEATURES:
            if float(v) > 0.5:
                out.add(PRESENCE_NAME.get(name, name))
        else:
            out.add(bin_labels(name)[bin_index(name, float(v))])
    return frozenset(out)


ITEM2FEATURE: dict[str, str] = {}


def build_item_map(cols: list[str]) -> None:
    ITEM2FEATURE.clear()
    for name in cols:
        if name in BINARY_FEATURES:
            ITEM2FEATURE[PRESENCE_NAME.get(name, name)] = name
        else:
            for lab in bin_labels(name):
                ITEM2FEATURE[lab] = name


def feature_of_item(item: str) -> str:
    return ITEM2FEATURE.get(item, item)


# ======================================================================
# 三、数据
# ======================================================================
@dataclass
class Dataset:
    rows: list[dict]
    y: np.ndarray
    users: list[str]
    feats: list[dict]
    items: list[frozenset]
    candidate: np.ndarray
    source: list[str]

    @property
    def n(self) -> int:
        return len(self.rows)


def load(cols: list[str], subset: str = "all") -> Dataset:
    rows = list(csv.DictReader(GT_PATH.open(encoding="utf-8-sig", newline="")))
    corpus = {}
    for line in CORPUS_PATH.open(encoding="utf-8"):
        line = line.strip()
        if line:
            r = json.loads(line)
            corpus[str(r["rpid"])] = r

    keep = []
    for r in rows:
        c = corpus.get(str(r["rpid"]).strip())
        if c is None or c["content"] != r["content"]:
            continue          # 内容和语料对不上就不用（用户名要算 E 组特征）
        if subset == "random" and r["was_candidate"] == "1":
            continue
        keep.append((r, c))

    y = np.array([int(r["label"]) for r, _ in keep], dtype=np.int8)
    return Dataset(
        rows=[r for r, _ in keep],
        y=y,
        users=[c["user"] for _, c in keep],
        feats=[F.featurize(r["content"], c["user"]) for r, c in keep],
        items=[row_items(F.featurize(r["content"], c["user"]), cols) for r, c in keep],
        candidate=np.array([int(r["was_candidate"]) for r, _ in keep], dtype=np.int8),
        source=[r["source"] for r, _ in keep],
    )


def row_mask(idx) -> int:
    m = 0
    for t in idx:
        m |= 1 << int(t)
    return m


# ======================================================================
# 四、Apriori（位图版）
# ======================================================================
class BitApriori:
    """频繁项集挖掘。

    263 行 × ~60 项，用 Python 大整数当位图：支持度 = popcount(AND of item bits)。
    比 dict/set 计数快一个量级，也让「置换标签后重算」几乎免费。
    """

    def __init__(self, names: list[str], bits: list[int], n: int):
        self.names = names
        self.bits = bits
        self.n = n
        self._idx = {s: i for i, s in enumerate(names)}

    @classmethod
    def from_transactions(cls, names: list[str], transactions: list[frozenset], n: int):
        idx = {s: i for i, s in enumerate(names)}
        bits = [0] * len(names)
        for t, tr in enumerate(transactions):
            bit = 1 << t
            for s in tr:
                bits[idx[s]] |= bit
        return cls(names, bits, n)

    def count_ids(self, ids, mask: int | None = None) -> int:
        """按项下标计数（fit 内部用）。"""
        b = mask
        for i in ids:
            bb = self.bits[i]
            b = bb if b is None else (b & bb)
            if b == 0:
                return 0
        return b.bit_count() if b is not None else self.n

    def support_count(self, itemset, mask: int | None = None) -> int:
        return self.count_ids([self._idx[s] for s in itemset], mask)

    def fit(self, min_count: int, max_len: int = 4,
            mask: int | None = None) -> dict[frozenset, int]:
        """返回 {频繁项集: 支持数}。mask 限定统计的行（交叉验证用）。"""
        freq: dict[frozenset, int] = {}
        level: dict[tuple, int] = {}
        for i, nm in enumerate(self.names):
            c = self.count_ids((i,), mask)
            if c >= min_count:
                level[(i,)] = c
                freq[frozenset((nm,))] = c
        k = 2
        while level and k <= max_len:
            keys = sorted(level)
            nxt: dict[tuple, int] = {}
            for a_i in range(len(keys)):
                a = keys[a_i]
                for b_i in range(a_i + 1, len(keys)):
                    b = keys[b_i]
                    if a[:-1] != b[:-1]:
                        continue
                    cand = a + (b[-1],)
                    ok = True
                    for j in range(k):       # Apriori 剪枝
                        if cand[:j] + cand[j + 1:] not in level:
                            ok = False
                            break
                    if not ok:
                        continue
                    c = self.count_ids(cand, mask)
                    if c >= min_count:
                        nxt[cand] = c
                        freq[frozenset(self.names[i] for i in cand)] = c
            level = nxt
            k += 1
        return freq


# ======================================================================
# 五、规则与统计
# ======================================================================
@dataclass
class Rule:
    ant: frozenset
    cons: frozenset
    n_ant: int
    n_cons: int
    n_both: int
    n: int
    conf: float
    lift: float
    p: float = 1.0
    p_bonf: float = 1.0
    parent: frozenset | None = None
    parent_conf: float = float("nan")
    p_cond: float = float("nan")

    @property
    def support(self) -> float:
        return self.n_both / self.n


def _p_exact(n: int, k_cons: int, n_ant: int, k_both: int) -> float:
    """单侧超几何上尾：P(交集 ≥ k_both | 独立)。"""
    if n_ant <= 0 or k_cons <= 0 or k_both <= 0:
        return 1.0
    return float(hypergeom.sf(k_both - 1, n, k_cons, n_ant))


def rules_from_freq(freq: dict[frozenset, int], n: int, min_conf: float) -> list[Rule]:
    """从频繁项集生成规则（前提、结论都允许是多项）。"""
    out = []
    for s, cnt in freq.items():
        if len(s) < 2:
            continue
        for r in range(1, len(s)):
            for cons in combinations(sorted(s), r):
                cons = frozenset(cons)
                ant = s - cons
                n_ant = freq.get(ant)
                n_cons = freq.get(cons)
                if not n_ant or not n_cons:
                    continue
                conf = cnt / n_ant
                if conf < min_conf:
                    continue
                out.append(Rule(ant, cons, n_ant, n_cons, cnt, n,
                                conf, conf / (n_cons / n)))
    return out


def n_candidate_rules(freq: dict[frozenset, int]) -> int:
    """所有可能被检验的规则数（多重比较的宇宙）：每个 k 项集贡献 2^k - 2 条。"""
    return sum((1 << len(s)) - 2 for s in freq if len(s) >= 2)


class Analyzer:
    """特征项集空间固定（与标签无关），标签只影响 AI 位图。"""

    def __init__(self, vocab: list[str], itemsets: list[frozenset], n: int):
        self.vocab = vocab
        self.n = n
        self.ap = BitApriori.from_transactions(vocab, itemsets, n)
        self.bits = self.ap.bits
        self.idx = self.ap._idx

    def ai_bits(self, y: np.ndarray) -> int:
        b = 0
        for t, v in enumerate(y):
            if v:
                b |= 1 << t
        return b

    def count(self, s, mask: int | None = None) -> int:
        b = mask
        for it in s:
            bb = self.bits[self.idx[it]]
            b = bb if b is None else (b & bb)
            if b == 0:
                return 0
        return b.bit_count() if b is not None else self.n

    def freq_features(self, min_count: int, max_len: int,
                      mask: int | None = None) -> dict[frozenset, int]:
        return self.ap.fit(min_count, max_len, mask)

    def rules_to_ai(self, y: np.ndarray, freq: dict[frozenset, int], min_count: int,
                    min_conf: float, mask: int | None = None) -> list[Rule]:
        """{X} → {AI}。

        与「把 AI 项一起丢进 Apriori」等价：
        项集 X∪{AI} 的支持数 = |X ∩ AI|，它必须 ≥ min_count 才算频繁。
        """
        aib = self.ai_bits(y) if mask is None else (self.ai_bits(y) & mask)
        n = self.n if mask is None else mask.bit_count()
        k_cons = aib.bit_count()
        if k_cons == 0:
            return []
        base = k_cons / n
        out = []
        for s, cnt in freq.items():
            k = self.count(s, aib)
            if k < min_count:
                continue
            conf = k / cnt
            if conf < min_conf:
                continue
            out.append(Rule(s, frozenset((AI_ITEM,)), cnt, k_cons, k, n,
                            conf, conf / base))
        return out

    def conditional_test(self, rule: Rule, freq: dict[frozenset, int],
                         y: np.ndarray) -> None:
        """组合增益的条件检验。

        取前提 S 中「去掉一项后置信度最高的子集」P 作父项集，再在 P 成立的行内部
        用超几何检验新增项 B 是否仍指向 AI。p_cond 不显著 ⇒ 「多这一项」没独立贡献。
        """
        aib = self.ai_bits(y)
        best = None
        for it in rule.ant:
            p = rule.ant - {it}
            n_p = freq.get(p)
            if not n_p:
                continue
            conf_p = self.count(p, aib) / n_p
            if best is None or conf_p > best[1]:
                best = (p, conf_p, it, n_p)
        if best is None:
            return
        p, conf_p, it, n_p = best
        pbits = None
        for x in p:
            bb = self.bits[self.idx[x]]
            pbits = bb if pbits is None else (pbits & bb)
        k_p = (pbits & aib).bit_count()
        n_x = (pbits & self.bits[self.idx[it]]).bit_count()
        k_x = (pbits & self.bits[self.idx[it]] & aib).bit_count()
        rule.parent = p
        rule.parent_conf = conf_p
        rule.p_cond = float(hypergeom.sf(k_x - 1, n_p, k_p, n_x)) if n_x and k_p else 1.0


def attach_pvalues(rules: list[Rule], m: int, n: int) -> None:
    for r in rules:
        r.p = _p_exact(n, r.n_cons, r.n_ant, r.n_both)
        r.p_bonf = min(1.0, r.p * max(1, m))


# ======================================================================
# 六、输出工具
# ======================================================================
def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def sec(title: str) -> str:
    return f"\n## {title}\n"


def sub(title: str) -> str:
    return f"\n### {title}\n"


def f3(x) -> str:
    return f"{x:.3f}"


def fmt_p(p: float) -> str:
    if p < 1e-4:
        return f"{p:.1e}"
    return f"{p:.4f}"


def rule_str(r: Rule, sep: str = " → ") -> str:
    return "{" + ", ".join(sorted(r.ant)) + "}" + sep + "{" + ", ".join(sorted(r.cons)) + "}"


def items_str(s) -> str:
    return "{" + ", ".join(sorted(s)) + "}"


# ======================================================================
# 七、各部分输出
# ======================================================================
def print_cuts(cols, ds: Dataset) -> None:
    print(sec("离散化切点（代码常量 CUTS，来自特征分布，与标签无关）"))
    rows = []
    for name in F.NAMES:
        if name not in cols:
            continue
        v = np.array([d[name] for d in ds.feats], float)
        if name in BINARY_FEATURES:
            on = v > 0.5
            rows.append([name, "二值", "原样（只建出现项）",
                         f"{int(on.sum())}（{on.mean()*100:.1f}%）",
                         f"{ds.y[on].mean()*100:.1f}%" if on.sum() else "—",
                         PRESENCE_NAME.get(name, name)])
        else:
            labels = bin_labels(name)
            bi = np.array([bin_index(name, x) for x in v])
            for b, lab in enumerate(labels):
                m = bi == b
                rows.append([name if b == 0 else "",
                             f"档 {b+1}/{len(labels)}",
                             "v > " + " / v > ".join(str(c) for c in CUTS[name]),
                             f"{int(m.sum())}（{m.mean()*100:.1f}%）",
                             f"{ds.y[m].mean()*100:.1f}%" if m.sum() else "—", lab])
    print(md_table(["特征", "档", "切点", "样本量（占比）", "档内 AI 率", "项名"], rows))


def describe_items(an: Analyzer, ds: Dataset, top: int) -> None:
    n = ds.n
    k_ai = int(ds.y.sum())
    base = k_ai / n
    aib = an.ai_bits(ds.y)
    data = []
    empty = []
    for name in an.vocab:
        cnt = an.count((name,))
        if cnt == 0:               # 子集分析时可能出现样本里完全没有的项
            empty.append(name)
            continue
        k = an.count((name,), aib)
        data.append([name, cnt, k, cnt / n, k / cnt, (k / cnt) / base,
                     _p_exact(n, k_ai, cnt, k)])
    data.sort(key=lambda r: (-r[5], -r[1]))
    print(sec("项分布与单特征 lift"))
    print(f"{len(an.vocab)} 个项，基础率 P(AI) = {base:.4f}（{k_ai}/{n}）。"
          f"按 lift 排序，前 {top} 个：")
    if empty:
        print()
        print(f"（有 {len(empty)} 个项在当前样本里一次都没出现，已跳过："
              f"{'、'.join(empty)}）")
    print()
    print(md_table(["项", "样本数", "support", "其中 AI", "P(AI｜项)", "lift", "p"],
                   [[r[0], str(r[1]), f3(r[3]), str(r[2]), f3(r[4]), f3(r[5]), fmt_p(r[6])]
                    for r in data[:top]]))
    print()


def print_freq_itemsets(freq: dict[frozenset, int], an: Analyzer, ds: Dataset,
                        top: int) -> None:
    n = ds.n
    k_ai = int(ds.y.sum())
    base = k_ai / n
    aib = an.ai_bits(ds.y)
    by: dict[int, int] = {}
    for s in freq:
        by[len(s)] = by.get(len(s), 0) + 1
    print(sec("频繁项集（特征之间，与标签无关）"))
    print(f"共挖掘到频繁项集 **{len(freq)}** 个：")
    print()
    print(md_table(["项集长度", "个数"],
                   [[str(k), str(by[k])] for k in sorted(by)]))
    print()
    print(sub(f"按支持度 Top {top}"))
    print(md_table(["项集", "支持数", "support", "含 AI 数", "P(AI｜项集)", "lift(AI)"],
                   [[items_str(s), str(c), f3(c / n), str(an.count(s, aib)),
                     f3(an.count(s, aib) / c),
                     f3((an.count(s, aib) / c) / base) if an.count(s, aib) else "—"]
                    for s, c in sorted(freq.items(), key=lambda kv: (-kv[1], -len(kv[0])))[:top]]))
    data = [(an.count(s, aib) / c, s, c, an.count(s, aib)) for s, c in freq.items()]
    data = [d for d in data if d[2] >= 5 and d[3] > 0]
    data.sort(key=lambda x: (-x[0], -x[2]))
    print(sub(f"按 P(AI｜项集) Top {top}（支持数 ≥ 5）"))
    print(md_table(["项集", "支持数", "含 AI 数", "P(AI｜项集)", "lift(AI)"],
                   [[items_str(s), str(c), str(k), f3(conf), f3(conf / base)]
                    for conf, s, c, k in data[:top]]))
    print()


def print_overview(all_rules: list[Rule], freq_all: dict[frozenset, int],
                   min_conf: float, min_lift: float) -> None:
    n_ai_rows = sum(1 for r in all_rules if r.cons == frozenset({AI_ITEM}))
    print(sec("关联规则总览（多重比较口径也在这里交代）"))
    rows = [
        ["频繁项集（含 AI 项）", str(len(freq_all))],
        ["可能被检验的规则总数（= Σ(2^k−2)，k = 项集长度）", str(n_candidate_rules(freq_all))],
        [f"实际满足 confidence ≥ {min_conf:.2f} 的规则数 m", str(len(all_rules))],
        [f"　其中 lift > {min_lift}", str(sum(1 for r in all_rules if r.lift > min_lift))],
        ["　其中结论 = {AI}", str(n_ai_rows)],
        [f"　其中结论 = {{AI}} 且 lift > {min_lift}",
         str(sum(1 for r in all_rules if r.cons == frozenset({AI_ITEM})
                 and r.lift > min_lift))],
        ["Bonferroni（乘以上面「可能被检验的规则总数」）后 p < 0.05",
         str(sum(1 for r in all_rules if r.p_bonf < 0.05))],
        ["　其中结论 = {AI}",
         str(sum(1 for r in all_rules if r.cons == frozenset({AI_ITEM})
                 and r.p_bonf < 0.05))],
    ]
    print(md_table(["计数", "值"], rows))
    print()


def print_general_rules(all_rules: list[Rule], min_lift: float, top: int) -> None:
    gen = [r for r in all_rules if AI_ITEM not in r.ant and AI_ITEM not in r.cons]
    gen.sort(key=lambda r: (-r.lift, -r.conf))
    print(sec("特征之间的关联规则（结论不含 label）"))
    print(f"「特征 → 特征」规则 {len(gen)} 条（lift > {min_lift} 的 "
          f"{sum(1 for r in gen if r.lift > min_lift)} 条）。"
          f"这张表是「特征本来就一起出现」的直接证据：")
    print()
    print(md_table(["规则", "n(前提)", "n(并集)", "confidence", "lift", "p", "p·m"],
                   [[rule_str(r), str(r.n_ant), str(r.n_both), f3(r.conf), f3(r.lift),
                     fmt_p(r.p), fmt_p(r.p_bonf)] for r in gen[:top]]))
    print()


def print_ai_rules(to_ai: list[Rule], ds: Dataset, min_lift: float, top: int) -> None:
    n = ds.n
    base = int(ds.y.sum()) / n
    to_ai = sorted(to_ai, key=lambda r: (-r.lift, -r.conf))
    print(sec("{特征组合} → {AI}"))
    print(f"P(AI) 基础率 = {base:.4f}。共 **{len(to_ai)}** 条 →AI 规则，"
          f"其中 lift > {min_lift} 的 **{sum(1 for r in to_ai if r.lift > min_lift)}** 条。")
    print()
    print(sub(f"按 lift 排序 Top {top}"))
    print(md_table(["规则", "n(前提)", "n(前提∧AI)", "support", "confidence", "lift",
                    "p", "p·m"],
                   [[rule_str(r), str(r.n_ant), str(r.n_both), f3(r.support), f3(r.conf),
                     f3(r.lift), fmt_p(r.p), fmt_p(r.p_bonf)] for r in to_ai[:top]]))
    multi = [r for r in to_ai if len(r.ant) >= 2 and not np.isnan(r.p_cond)]
    multi.sort(key=lambda r: -(r.conf - r.parent_conf))
    print(sub(f"多项前提（≥2 项）：组合增益检验（按 conf 增益排序，前 {top}）"))
    print("「父项集」= 前提去掉一项后置信度最高的子集；p_cond = 在父项集内部再检验"
          "新增项是否仍指向 AI（超几何，单侧）。")
    print()
    print(md_table(["规则", "n", "n(AI)", "confidence", "lift", "父项集", "父项集 conf",
                    "conf 增益", "p_cond"],
                   [[rule_str(r), str(r.n_ant), str(r.n_both), f3(r.conf), f3(r.lift),
                     items_str(r.parent), f3(r.parent_conf), f"{r.conf - r.parent_conf:+.3f}",
                     fmt_p(r.p_cond)] for r in multi[:top]]))
    print()


def single_feature_table(an: Analyzer, ds: Dataset, cols: list[str], top: int):
    """单特征（含二值特征的缺席项）对 AI 的判别力 —— 「单一特征够不够」的基线。"""
    n = ds.n
    k_ai = int(ds.y.sum())
    base = k_ai / n
    aib = an.ai_bits(ds.y)
    allbits = (1 << n) - 1
    rows = []
    for name in cols:
        if name in BINARY_FEATURES:
            b = an.bits[an.idx[PRESENCE_NAME.get(name, name)]]
            cand = [(PRESENCE_NAME.get(name, name), b), (f"无{name}", allbits ^ b)]
        else:
            cand = [(lab, an.bits[an.idx[lab]]) for lab in bin_labels(name)]
        for lab, b in cand:
            cnt = b.bit_count()
            if cnt < 5:
                continue
            k = (b & aib).bit_count()
            rows.append([lab, cnt, k, k / cnt, (k / cnt) / base,
                         _p_exact(n, k_ai, cnt, k)])
    rows.sort(key=lambda r: -r[4])
    m = max(1, len(rows))
    print(f"共 {len(rows)} 个单特征项（含 {sum(1 for r in rows if r[0].startswith('无'))} 个缺席项）；"
          f"按 lift 排序前 {top}（p·m 的 m = {m}，即单特征检验这一族的条数）：")
    print()
    print(md_table(["项", "n", "n(AI)", "confidence", "lift", "p", "p·m"],
                   [[r[0], str(r[1]), str(r[2]), f3(r[3]), f3(r[4]), fmt_p(r[5]),
                     fmt_p(min(1.0, r[5] * m))] for r in rows[:top]]))
    print()
    return rows


def single_vs_combo(to_ai: list[Rule], single_rows: list, ds: Dataset,
                    min_count: int, max_ant_len: int) -> None:
    n = ds.n
    base = int(ds.y.sum()) / n
    print(sec("单特征 vs 组合"))
    print(f"前提下至少覆盖 {min_count} 条（= min_support × n）才计入比较；"
          f"受 max_len 限制，前提最长 {max_ant_len} 项。")
    print()

    def best_of(k: int):
        cand = [r for r in to_ai if len(r.ant) == k and r.n_ant >= min_count]
        return max(cand, key=lambda r: (r.conf, r.lift)) if cand else None

    rows = []
    for k in range(1, max_ant_len + 1):
        r = best_of(k)
        if r is None:
            rows.append([f"{k} 项前提", "（无规则达标）", "—", "—", "—", "—", "—"])
        else:
            rows.append([f"{k} 项前提", items_str(r.ant), str(r.n_ant), str(r.n_both),
                         f3(r.conf), f3(r.lift), f"{r.conf - base:+.3f}"])
    print(md_table(["前提长度", "置信度最高的规则（前提）", "n", "n(AI)", "confidence",
                    "lift", "conf − 基础率"], rows))
    print()

    top_single = max(single_rows, key=lambda r: r[4])
    solid = [r for r in single_rows if r[1] >= min_count]
    top_solid = max(solid, key=lambda r: r[4]) if solid else None
    print(f"全部单特征项里 lift 最高的是 `{top_single[0]}`"
          f"（lift={f3(top_single[4])}, conf={f3(top_single[3])}, n={top_single[1]}），"
          f"但它只覆盖 {top_single[1]} 条。")
    if top_solid is not None:
        print(f"把支持度门槛压到 ≥ {min_count} 条后，最强单特征项是 `{top_solid[0]}`"
              f"（lift={f3(top_solid[4])}, conf={f3(top_solid[3])}, n={top_solid[1]}）。")
    print()
    multi = [r for r in to_ai if len(r.ant) >= 2 and r.n_ant >= min_count
             and not np.isnan(r.parent_conf)]
    multi.sort(key=lambda r: -(r.conf - r.parent_conf))
    print("多项规则相对「它的最强子集」还有多少增益：")
    print()
    print(md_table(["多项规则", "conf", "最强子集", "子集 conf", "conf 增益", "p_cond",
                    "增益显著（p<0.05）"],
                   [[rule_str(r), f3(r.conf), items_str(r.parent), f3(r.parent_conf),
                     f"{r.conf - r.parent_conf:+.3f}", fmt_p(r.p_cond),
                     "是" if r.p_cond < 0.05 else "否"] for r in multi[:15]]))
    print()


def redundant_dedup(to_ai: list[Rule], min_lift: float) -> list[Rule]:
    """非冗余规则：若存在前提更小的规则 conf 不更低，则大前提规则没有新信息。

    这是 Bayardo 那类「productive/非冗余规则」的朴素版本：
    R 冗余 ⇔ ∃R' 使 ant(R') ⊊ ant(R) 且 conf(R') ≥ conf(R)。
    """
    picked = [r for r in to_ai if r.lift > min_lift]
    picked.sort(key=lambda r: (len(r.ant), -r.conf))
    kept: list[Rule] = []
    for r in picked:
        redundant = False
        for q in kept:
            if q.ant < r.ant and q.conf >= r.conf - 1e-12:
                redundant = True
                break
        if not redundant:
            kept.append(r)
    return kept


def print_dedup(to_ai: list[Rule], min_lift: float, top: int) -> None:
    picked = [r for r in to_ai if r.lift > min_lift]
    kept = redundant_dedup(to_ai, min_lift)
    print(sec(f"规则去冗余（lift > {min_lift} 的规则里，有多少是「加了项但没更准」）"))
    print(f"原始 {len(picked)} 条；去掉「前提更大、置信度却没更高」的冗余规则后剩 "
          f"**{len(kept)}** 条（{len(kept)/max(1,len(picked))*100:.1f}%）。")
    print()
    if not kept:
        return
    kept.sort(key=lambda r: (-r.conf, -r.lift))
    print(md_table(["非冗余规则（按 conf）", "n", "n(AI)", "support", "confidence",
                    "lift", "p", "p·m"],
                   [[rule_str(r), str(r.n_ant), str(r.n_both), f3(r.support), f3(r.conf),
                     f3(r.lift), fmt_p(r.p), fmt_p(r.p_bonf)] for r in kept[:top]]))
    print()
    # 这些规则到底由几个不同的「信号」拼出来
    use: dict[str, int] = {}
    for r in kept:
        for it in r.ant:
            use[it] = use.get(it, 0) + 1
    print(f"这 {len(kept)} 条非冗余规则的前提里，一共只出现了 **{len(use)}** 个不同的项：")
    print()
    print(md_table(["项", "出现在多少条规则的前提里"],
                   [[k, str(v)] for k, v in sorted(use.items(), key=lambda kv: -kv[1])]))
    print()
    lens = {}
    for r in kept:
        lens[len(r.ant)] = lens.get(len(r.ant), 0) + 1
    print("非冗余规则的前提长度分布：" +
          "，".join(f"{k} 项 {v} 条" for k, v in sorted(lens.items())))
    print()


def collinearity(an: Analyzer, ds: Dataset, cols: list[str], top: int) -> int:
    n = ds.n
    items = []
    for name in cols:
        if name in BINARY_FEATURES:
            items.append(PRESENCE_NAME.get(name, name))
        else:
            items.extend(bin_labels(name))
    mat = {}
    for s in items:
        b = an.bits[an.idx[s]]
        mat[s] = np.array([(b >> t) & 1 for t in range(n)], dtype=np.float64)
    rows = []
    n_pairs = 0
    for a, b in combinations(items, 2):
        if feature_of_item(a) == feature_of_item(b):
            continue              # 同一特征的两档互斥，φ 必为 -1，属构造痕迹，不统计
        n_pairs += 1
        x, z = mat[a], mat[b]
        if x.std() < 1e-9 or z.std() < 1e-9:
            continue
        phi = float(np.corrcoef(x, z)[0, 1])
        if abs(phi) >= 0.5:
            rows.append([a, b, f"{phi:+.3f}", str(int(x.sum())), str(int(z.sum())),
                         str(int((x * z).sum()))])
    rows.sort(key=lambda r: -abs(float(r[2])))
    print(sec("特征共线性（|φ| ≥ 0.5 的项对，已排除同一特征的不同档）"))
    print(f"{len(items)} 个项、{n_pairs} 对跨特征组合，|φ| ≥ 0.5 的 **{len(rows)}** 对：")
    print()
    print(md_table(["项 A", "项 B", "φ", "n(A)", "n(B)", "n(A∧B)"], rows[:top]))
    print()
    print(f"（只显示 |φ| 最大 {top} 对。）")
    print()
    return len(rows)


def stratified_table(an: Analyzer, ds: Dataset, cols: list[str]) -> None:
    """控制篇幅后，各记号还剩下多少信息量。"""
    n = ds.n
    k_ai = int(ds.y.sum())
    base = k_ai / n
    aib = an.ai_bits(ds.y)
    cut = CUTS["字数"][1]
    long_mask = row_mask([t for t, d in enumerate(ds.feats) if d["字数"] > cut])
    n_long = long_mask.bit_count()
    k_long = (long_mask & aib).bit_count()
    base_long = k_long / n_long
    print(sec(f"控制篇幅：在长评（字数 ≥ {cut+1}）内部再看各记号"))
    print(f"长评 {n_long} 条，其中 AI {k_long} 条，档内基础率 = {base_long:.4f}"
          f"（全样本 {base:.4f}）。lift(长评内) = 档内 P(AI｜项) / 档内基础率。")
    print()
    items = []
    for name in cols:
        if name in BINARY_FEATURES:
            items.append(PRESENCE_NAME.get(name, name))
        elif name in CUTS:
            items.extend(bin_labels(name)[1:])     # 只看「有」的那一侧
    rows = []
    for it in items:
        b = an.bits[an.idx[it]]
        cnt = b.bit_count()
        k = (b & aib).bit_count()
        if cnt < 10:
            continue
        sub_b = b & long_mask
        cnt_l = sub_b.bit_count()
        k_l = (sub_b & aib).bit_count()
        if cnt_l < 6 or k == 0:
            continue
        conf_all = k / cnt
        conf_l = k_l / cnt_l
        rows.append([it, str(cnt), f3(conf_all), f3(conf_all / base),
                     str(cnt_l), str(k_l), f3(conf_l), f3(conf_l / base_long),
                     f"{(conf_all/base) / (conf_l/base_long):.2f}"])
    rows.sort(key=lambda r: -float(r[3]))
    print(md_table(["项", "n(全)", "P(AI｜项)", "lift(全)", "n(长评内)", "AI(长评内)",
                    "P(AI｜项,长评)", "lift(长评内)", "lift 衰减"], rows))
    print()


def length_block_share(to_ai: list[Rule], min_lift: float) -> None:
    picked = [r for r in to_ai if r.lift > min_lift]
    if not picked:
        print(sec("这些规则在说什么"))
        print("没有规则达到 lift 阈值。")
        return
    all_len = part_len = 0
    for r in picked:
        feats = [feature_of_item(it) for it in r.ant]
        nin = sum(1 for f in feats if f in LENGTH_BLOCK)
        if nin == len(feats):
            all_len += 1
        if nin >= 1:
            part_len += 1
    print(sec("这些规则在说什么：篇幅块占比"))
    print(f"lift > {min_lift} 的 →AI 规则 {len(picked)} 条，前提里篇幅/密度块项的出现情况：")
    print()
    print(md_table(["口径", "条数", "占比"], [
        ["前提**全部**是篇幅/密度块项", str(all_len), f"{all_len/len(picked)*100:.1f}%"],
        ["前提**至少含一个**篇幅/密度块项", str(part_len), f"{part_len/len(picked)*100:.1f}%"],
        ["前提**完全不含**篇幅/密度块项", str(len(picked) - part_len),
         f"{(len(picked)-part_len)/len(picked)*100:.1f}%"],
    ]))
    print()
    print("篇幅/密度块 = " + "、".join(sorted(LENGTH_BLOCK)))
    print()
    only_text = [r for r in picked
                 if all(feature_of_item(it) not in LENGTH_BLOCK for it in r.ant)]
    if only_text:
        only_text.sort(key=lambda r: -r.conf)
        print("完全不含篇幅块项的规则（也就是「纯文本记号」组合）：")
        print()
        print(md_table(["规则", "n", "n(AI)", "confidence", "lift", "p"],
                       [[rule_str(r), str(r.n_ant), str(r.n_both), f3(r.conf),
                         f3(r.lift), fmt_p(r.p)] for r in only_text[:12]]))
        print()


def account_probe(ds: Dataset) -> None:
    n = ds.n
    k_ai = int(ds.y.sum())
    base = k_ai / n
    fish = np.array([("鱼" in u) or ("鲸" in u) for u in ds.users])
    a = int((fish & (ds.y == 1)).sum())
    b = int((fish & (ds.y == 0)).sum())
    print(sec("为什么 E 账号类不能进项集空间"))
    print(md_table(["项", "n", "n(AI)", "confidence", "lift", "p"],
                   [["名字含鱼/鲸", str(a + b), str(a), f3(a / (a + b)),
                     f3((a / (a + b)) / base), fmt_p(_p_exact(n, k_ai, a + b, a))]]))
    print()
    print(f"这是主分析项集空间（min_support 门槛内）里 lift 最高的单特征：AI 侧 {a}/{k_ai} 条来自"
          f"「大肥鱼 / 小鲸鱼」这一族人格号。比它 lift 更高的只有 n=5~7 的稀有条目"
          f"（有「」/自称本鱼 等），那些在 Bonferroni 后不过关。")
    print(f"但它测的是**账号身份**，不是书写风格。放进项集空间后规则会被它吞掉，"
          f"只剩「这个号是机器人」这种同义反复，所以正表按 CLASSIFY_GROUPS 排除 E 组，"
          f"与 v5 分类保持一致。")
    print()
    n_ai_users = len({u for u, y in zip(ds.users, ds.y) if y == 1})
    print(f"附：63 条 AI 来自 {n_ai_users} 个不同账号（几乎一行一号），"
          f"所以「同一账号重复发言导致观测不独立」不是主要偏差来源。")
    print()


# ======================================================================
# 八、置换检验 / 交叉验证 / 绘图
# ======================================================================
def permutation_test(an: Analyzer, ds: Dataset, min_count: int, min_conf: float,
                     min_lift: float, n_perm: int, seed: int,
                     max_ant_len: int) -> dict:
    n = ds.n
    k_ai = int(ds.y.sum())
    aib = an.ai_bits(ds.y)
    base = k_ai / n
    freq = an.freq_features(min_count, max_ant_len)   # 与标签无关，只挖一次

    def stats(mask: int):
        hit = conf_hit = 0
        max_lift = max_conf = 0.0
        for s, c in freq.items():
            k = an.count(s, mask)
            if k < min_count:
                continue
            conf = k / c
            if conf < min_conf:
                continue
            conf_hit += 1
            lift = conf / base
            max_lift = max(max_lift, lift)
            max_conf = max(max_conf, conf)
            if lift > min_lift:
                hit += 1
        return hit, max_lift, conf_hit, max_conf

    obs = stats(aib)
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(n_perm):
        yy = np.zeros(n, dtype=np.int8)
        yy[rng.choice(n, size=k_ai, replace=False)] = 1
        null.append(stats(an.ai_bits(yy)))
    nc = np.array([x[0] for x in null], float)
    nl = np.array([x[1] for x in null], float)
    nf = np.array([x[2] for x in null], float)
    p_count = (1 + int((nc >= obs[0]).sum())) / (1 + n_perm)
    p_lift = (1 + int((nl >= obs[1]).sum())) / (1 + n_perm)

    print(sec("置换检验：这些数字有多少是硬挖出来的"))
    print(f"把 {k_ai} 个 AI 标签随机重排 {n_perm} 次（特征、离散化、频繁项集都不动，"
          f"只换标签），每次重新统计「漂亮规则」的数量和最大 lift。这就是零分布。")
    print()
    print(md_table(["统计量", "观测值", "零分布均值", "零分布 p95", "零分布最大", "经验 p"], [
        [f"lift > {min_lift} 且 conf ≥ {min_conf:.2f} 的规则数", str(obs[0]),
         f"{nc.mean():.1f}", f"{np.percentile(nc, 95):.0f}", f"{nc.max():.0f}", f3(p_count)],
        ["最大 lift", f3(obs[1]), f3(nl.mean()), f3(np.percentile(nl, 95)),
         f3(nl.max()), f3(p_lift)],
        [f"conf ≥ {min_conf:.2f} 且支持达标的规则数", str(obs[2]), f"{nf.mean():.1f}",
         f"{np.percentile(nf, 95):.0f}", f"{nf.max():.0f}", "—"],
    ]))
    print()
    print(f"经验 p = (1 + #{{零分布 ≥ 观测}})/(1 + {n_perm})；"
          f"{n_perm} 次置换能分辨的最小 p 是 {1/(1+n_perm):.4f}。")
    print()
    return {"observed": {"n_rules_lift": obs[0], "max_lift": obs[1],
                         "n_rules_conf": obs[2], "max_conf": obs[3]},
            "null": {"n_rules_lift_mean": float(nc.mean()),
                     "n_rules_lift_p95": float(np.percentile(nc, 95)),
                     "n_rules_lift_max": float(nc.max()),
                     "max_lift_mean": float(nl.mean()),
                     "max_lift_p95": float(np.percentile(nl, 95)),
                     "max_lift_max": float(nl.max())},
            "p_count": p_count, "p_lift": p_lift, "n_perm": n_perm}


def cross_validation(an: Analyzer, ds: Dataset, min_conf: float, min_lift: float,
                     folds: int, repeats: int, seed: int, max_ant_len: int) -> dict:
    """训练折挖规则 → 测试折量置信度。

    两种口径：
      (a) 全部通过阈值的规则（规则×折 汇总）—— 反映「整体规则集」的外推表现；
      (b) 每折按 conf 取前 K 条（K = 1/5/10/20）—— 反映「你照着报告选最好的几条用」
          时的真实表现，也是最能暴露选择偏差的口径。
    """
    from sklearn.model_selection import StratifiedKFold

    n = ds.n
    base = int(ds.y.sum()) / n
    aib = an.ai_bits(ds.y)
    print(sec("交叉验证：规则拿到没见过的折上还算数吗"))
    print(f"{folds} 折 × {repeats} 次重复分层划分。训练部分挖「→AI」规则"
          f"（conf ≥ {min_conf:.2f}、lift > {min_lift}、min_support 按训练折缩放、"
          f"前提最长 {max_ant_len} 项），再到测试折上直接量 confidence。")
    print()
    rows = []
    all_in, all_out, all_w = [], [], []
    rediscover: dict[frozenset, int] = {}
    ks = [1, 5, 10, 20]
    topk_in: dict[int, list] = {k: [] for k in ks}
    topk_out: dict[int, list] = {k: [] for k in ks}
    n_train_folds = folds * repeats
    for rep in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + rep)
        ins, outs, wts = [], [], []
        for tr, te in skf.split(np.zeros(n), ds.y):
            tr_mask = row_mask(tr)
            min_count = max(2, int(np.ceil(0.05 * len(tr))))
            freq = an.freq_features(min_count, max_ant_len, tr_mask)
            rules = an.rules_to_ai(ds.y, freq, min_count, min_conf, tr_mask)
            picked = [r for r in rules if r.lift > min_lift]
            if not picked:
                continue
            te_mask = row_mask(te)
            scored = []
            for r in picked:
                rediscover[r.ant] = rediscover.get(r.ant, 0) + 1
                b = te_mask
                for it in r.ant:
                    b &= an.bits[an.idx[it]]
                n_te = b.bit_count()
                if n_te == 0:
                    continue
                k_te = (b & aib).bit_count()
                oos = k_te / n_te
                ins.append(r.conf)
                outs.append(oos)
                wts.append(n_te)
                scored.append((r.conf, oos, n_te))
            scored.sort(key=lambda x: -x[0])
            for k in ks:
                sel = scored[:k]
                if sel:
                    topk_in[k].append(float(np.mean([s[0] for s in sel])))
                    topk_out[k].append(float(np.average([s[1] for s in sel],
                                                        weights=[s[2] for s in sel])))
        if ins:
            rows.append([f"第 {rep+1} 次", str(len(ins)), f3(np.mean(ins)),
                         f3(np.mean(outs)), f3(float(np.median(outs))),
                         f3(np.average(outs, weights=wts))])
            all_in += ins
            all_out += outs
            all_w += wts
    if not rows:
        print("没有规则通过阈值，跳过。")
        print()
        return {}
    print(sub("(a) 全部达标规则的汇总"))
    print(md_table(["重复", "规则×折 条数", "训练折 conf 均值", "测试折 conf 均值",
                    "测试折 conf 中位", "测试折按支持加权 conf"], rows))
    print()
    drop = float(np.mean(all_in) - np.mean(all_out))
    share_05 = float(np.mean([o > 0.5 for o in all_out]))
    share_base = float(np.mean([o > base for o in all_out]))
    print(f"合计 {len(all_in)} 条（规则×折）：训练折 mean conf = {np.mean(all_in):.3f}，"
          f"测试折 = {np.mean(all_out):.3f}，**衰减 {drop:+.3f}**；基础率 {base:.3f}。")
    print()
    print(f"测试折上 conf 仍 > 0.5：{share_05*100:.1f}%；仍 > 基础率 {base:.2f}："
          f"{share_base*100:.1f}%。")
    print()

    print(sub("(b) 每折按 conf 取前 K 条 —— 暴露选择偏差"))
    rows = []
    for k in ks:
        if not topk_in[k]:
            continue
        rows.append([f"前 {k} 条", str(len(topk_in[k])), f3(np.mean(topk_in[k])),
                     f3(np.mean(topk_out[k])), f"{np.mean(topk_in[k])-np.mean(topk_out[k]):+.3f}"])
    print(md_table(["每折取", "折数", "训练折 conf 均值(=表里那几行写的数)", "测试折 conf 均值",
                    "衰减"], rows))
    print()
    print(f"（训练折 conf 就是「在训练数据上挖出来时看到的数」，测试折是没进过挖掘的 "
          f"1/{folds} 数据。两者之差就是「报告里的漂亮数字」里含多少乐观偏差。）")
    print()

    top = sorted(rediscover.items(), key=lambda kv: -kv[1])[:10]
    print(f"稳定性：前提在多少个训练折里被重新挖到（共 {n_train_folds} 个训练折）")
    print()
    print(md_table(["前提", "被挖到的折数"],
                   [[items_str(k), str(v)] for k, v in top]))
    print()
    return {"mean_conf_in": float(np.mean(all_in)), "mean_conf_out": float(np.mean(all_out)),
            "drop": drop, "n_rule_fold": len(all_in), "base": base,
            "share_conf_above_0.5": share_05, "share_above_base": share_base,
            "n_train_folds": n_train_folds,
            "topk": {str(k): {"in": float(np.mean(topk_in[k])),
                              "out": float(np.mean(topk_out[k]))}
                     for k in ks if topk_in[k]}}


def positive_subsample_control(vocab: list[str], ds: Dataset, n_ai_keep: int,
                               n_boot: int, min_conf: float, min_lift: float,
                               max_ant_len: int, seed: int) -> dict:
    """对照实验：把正类数降到 n_ai_keep，看规则数是不是也会跟着塌。

    去候选层子集只剩 32 个 AI，规则数从 391 掉到 10。要判断这个塌方是
    「正类少了必然如此」还是「候选层本身就是循环证据」，就得在**未经去层**的
    全样本里随机抽同样多的正类做对照。
    """
    n = ds.n
    ai_idx = [i for i in range(n) if ds.y[i] == 1]
    hu_idx = [i for i in range(n) if ds.y[i] == 0]
    rng = np.random.default_rng(seed)
    counts, lifts = [], []
    for _ in range(n_boot):
        pick = set(rng.choice(ai_idx, size=n_ai_keep, replace=False).tolist())
        keep = hu_idx + [i for i in ai_idx if i in pick]
        keep.sort()
        sub = Dataset([ds.rows[i] for i in keep], ds.y[keep], [ds.users[i] for i in keep],
                      [ds.feats[i] for i in keep], [ds.items[i] for i in keep],
                      ds.candidate[keep], [ds.source[i] for i in keep])
        an = Analyzer(vocab, sub.items, sub.n)
        mc = max(2, int(np.ceil(0.05 * sub.n)))
        freq = an.freq_features(mc, max_ant_len)
        rules = an.rules_to_ai(sub.y, freq, mc, min_conf)
        picked = [r for r in rules if r.lift > min_lift]
        counts.append(len(picked))
        lifts.append(max((r.lift for r in picked), default=0.0))
    c = np.array(counts, float)
    l = np.array(lifts, float)
    print(sec(f"对照实验：把正类数压到 {n_ai_keep} 条（随机抽，不去候选层）"))
    print(f"全样本里随机抽 {n_ai_keep} 个 AI（保留全部真人），重复 {n_boot} 次，"
          f"每次重新挖频繁项集与规则。参数与主分析一致（按子集 n 缩放 min_support）。")
    print()
    print(md_table(["统计量", "值"], [
        [f"lift > {min_lift} 且 conf ≥ {min_conf:.2f} 的 →AI 规则数：均值", f"{c.mean():.1f}"],
        ["　中位数", f"{np.median(c):.0f}"],
        ["　最小 / 最大", f"{int(c.min())} / {int(c.max())}"],
        ["最大 lift：均值", f3(l.mean())],
    ]))
    print()
    print("这一栏要和「去候选层子集」的结果并排看：如果随机抽正类的规则数远高于"
          "去层子集，就说明塌方不是「正类变少」造成的，而是候选层被规则筛过带来的循环。")
    print()
    return {"n_ai_keep": n_ai_keep, "n_boot": n_boot,
            "rules_mean": float(c.mean()), "rules_median": float(np.median(c)),
            "rules_min": int(c.min()), "rules_max": int(c.max()),
            "max_lift_mean": float(l.mean())}


def plot_support_conf(to_ai: list[Rule], ds: Dataset, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    base = int(ds.y.sum()) / ds.n
    pos = [r for r in to_ai if r.n_both >= 5]
    if not pos:
        print("没有可画的规则。")
        return
    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=130)
    sc = ax.scatter([r.support for r in pos], [r.conf for r in pos],
                    c=[r.lift for r in pos], s=[30 + 3.4 * r.n_both for r in pos],
                    cmap="viridis", alpha=0.85, edgecolors="white", linewidths=0.6)
    ax.axhline(base, color="crimson", ls="--", lw=1.2)
    ax.text(0.012, base + 0.015, f"基础率 P(AI)={base:.3f}（lift=1）",
            color="crimson", fontsize=9)
    ax.axhline(0.5, color="gray", ls=":", lw=1.0)
    ax.text(ax.get_xlim()[0] + 0.001, 0.516, "confidence = 0.5（报告下限）", color="0.35",
            fontsize=8, ha="left",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
    # 标注：按 lift 取前几条，坐标相同的只标一次；用编号 + 右下角图例，避免标签打架
    seen = set()
    targets = []
    for r in sorted(pos, key=lambda r: -r.lift):
        key = (round(r.support, 3), round(r.conf, 3))
        if key in seen:
            continue
        seen.add(key)
        targets.append(r)
        if len(targets) >= 5:
            break
    lines = []
    for i, r in enumerate(targets, 1):
        ax.annotate(str(i), (r.support, r.conf), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, fontweight="bold",
                    color="black")
        lab = ", ".join(sorted(r.ant))
        if len(lab) > 34:
            lab = lab[:33] + "…"
        lines.append(f"{i}. {lab}  (conf={r.conf:.2f}, lift={r.lift:.2f})")
    ax.text(0.985, 0.03, "\n".join(lines), transform=ax.transAxes, ha="right",
            va="bottom", fontsize=7.2, linespacing=1.45,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", alpha=0.9))
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("lift", fontsize=9)
    ax.set_xlabel("support（前提 ∧ AI 的行占比）", fontsize=10)
    ax.set_ylabel("confidence（P(AI｜前提)）", fontsize=10)
    ax.set_title(f"关联规则 support–confidence（{len(pos)} 条 →AI 规则，点大小∝支持数）",
                 fontsize=11)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"图已保存：{path}")


# ======================================================================
# 九、main
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(description="v6 关联规则挖掘（自实现 Apriori）")
    p.add_argument("--min-support", type=float, default=0.05)
    p.add_argument("--min-confidence", type=float, default=0.50)
    p.add_argument("--min-lift", type=float, default=1.5)
    p.add_argument("--max-len", type=int, default=4)
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--perms", type=int, default=200)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=4)
    p.add_argument("--subset", choices=["all", "random"], default="all")
    p.add_argument("--include-account", action="store_true")
    p.add_argument("--show-cuts", action="store_true")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--json", type=str, default="")
    p.add_argument("--quick", action="store_true", help="跳过置换检验和交叉验证")
    p.add_argument("--self-check", action="store_true", help="完整 Apriori 复核等价性")
    p.add_argument("--boot", type=int, default=0,
                   help="对照实验：随机抽正类子样本，重复次数（0 = 关闭）")
    p.add_argument("--match-positives", type=int, default=32,
                   help="对照实验里保留多少个正类（默认 32，等于去候选层子集的正类数）")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    cols = list(F.cols_of(F.CLASSIFY_GROUPS))
    if args.include_account:
        cols += list(F.cols_of(["E 账号类"]))
    ds = load(cols, subset=args.subset)
    build_item_map(cols)
    vocab = build_vocab(cols)

    if args.show_cuts:
        print_cuts(cols, ds)
        return

    n = ds.n
    k_ai = int(ds.y.sum())
    base = k_ai / n
    min_count = max(2, int(np.ceil(args.min_support * n)))
    an = Analyzer(vocab, ds.items, n)

    print(f"# v6 关联规则挖掘 · 数据摘要\n")
    print(md_table(["项目", "值"], [
        ["样本", f"{n} 条（AI {k_ai} / 真人 {n - k_ai}），P(AI) = {base:.4f}"],
        ["数据子集", "全部标注行" if args.subset == "all" else "仅非候选层（was_candidate=0）"],
        ["特征", f"{len(cols)} 个（CLASSIFY_GROUPS"
                + ("+E 账号类" if args.include_account else "，不含 E 账号类") + "）"],
        ["离散化后项数", str(len(vocab))],
        ["min_support", f"{args.min_support}（支持数 ≥ {min_count} / {n}）"],
        ["min_confidence", str(args.min_confidence)],
        ["lift 报告下限", str(args.min_lift)],
        ["频繁项集最大长度", f"{args.max_len}（含 AI 项；纯特征项集最长 {args.max_len - 1}）"],
        ["Apriori", "自实现，位图（Python int）支持度计数"],
        ["随机种子", str(args.seed)],
    ]))
    print()

    print_cuts(cols, ds)
    describe_items(an, ds, args.top)

    freq_feat = an.freq_features(min_count, args.max_len - 1)
    aib = an.ai_bits(ds.y)
    freq_all = dict(freq_feat)
    # AI 单独也是频繁项（63 ≥ min_count），完整 Apriori 会保留它；
    # 规则 X→AI 的 lift 需要「AI 自身支持度」当分母，所以这里必须补上。
    freq_all[frozenset((AI_ITEM,))] = k_ai
    for s, c in freq_feat.items():
        k = an.count(s, aib)
        if k >= min_count:
            freq_all[s | {AI_ITEM}] = k
    print_freq_itemsets(freq_feat, an, ds, args.top)

    all_rules = rules_from_freq(freq_all, n, args.min_confidence)
    attach_pvalues(all_rules, n_candidate_rules(freq_all), n)
    print_overview(all_rules, freq_all, args.min_confidence, args.min_lift)
    print_general_rules(all_rules, args.min_lift, args.top)

    to_ai = [r for r in all_rules if r.cons == frozenset({AI_ITEM})]
    for r in to_ai:
        an.conditional_test(r, freq_feat, ds.y)
    print_ai_rules(to_ai, ds, args.min_lift, args.top)

    print(sec("单特征判别力（含缺席项）"))
    single_rows = single_feature_table(an, ds, cols, args.top)
    single_vs_combo(to_ai, single_rows, ds, min_count, args.max_len - 1)
    print_dedup(to_ai, args.min_lift, args.top)

    collinearity(an, ds, cols, args.top)
    stratified_table(an, ds, cols)
    length_block_share(to_ai, args.min_lift)
    account_probe(ds)

    if args.self_check:
        print(sec("自检：位图法与完整 Apriori 是否给出同一批 →AI 规则"))
        full = [frozenset(set(it) | ({AI_ITEM} if ds.y[i] else set()))
                for i, it in enumerate(ds.items)]
        ap2 = BitApriori.from_transactions(vocab + [AI_ITEM], full, n)
        f2 = ap2.fit(min_count, args.max_len)
        r2 = [r for r in rules_from_freq(f2, n, args.min_confidence)
              if r.cons == frozenset({AI_ITEM})]
        mine = {(r.ant, r.n_both) for r in to_ai}
        theirs = {(r.ant, r.n_both) for r in r2}
        print(f"位图法 {len(to_ai)} 条，完整 Apriori {len(r2)} 条；集合一致："
              f"**{mine == theirs}**")
        print()

    perm = permutation_test(an, ds, min_count, args.min_confidence, args.min_lift,
                            args.perms, args.seed, args.max_len - 1) \
        if (args.perms > 0 and not args.quick) else {}
    cv = cross_validation(an, ds, max(0.6, args.min_confidence), args.min_lift,
                          args.folds, args.repeats, args.seed,
                          args.max_len - 1) \
        if (not args.quick and args.folds >= 2) else {}
    boot = positive_subsample_control(vocab, ds, args.match_positives, args.boot,
                                      args.min_confidence, args.min_lift,
                                      args.max_len - 1, args.seed) if args.boot > 0 else {}

    if args.plot:
        print()
        plot_support_conf(to_ai, ds, FIG_PATH)

    if args.json:
        payload = {
            "params": vars(args),
            "n": n, "k_ai": k_ai, "base": base, "min_count": min_count,
            "n_items": len(vocab), "n_freq_itemsets": len(freq_all),
            "n_rules": len(all_rules), "n_rules_to_ai": len(to_ai),
            "n_rules_to_ai_lift": sum(1 for r in to_ai if r.lift > args.min_lift),
            "n_candidate_rules": n_candidate_rules(freq_all),
            "rules_to_ai": [
                {"ant": sorted(r.ant), "n_ant": r.n_ant, "n_both": r.n_both,
                 "support": r.support, "confidence": r.conf, "lift": r.lift,
                 "p": r.p, "p_bonf": r.p_bonf,
                 "parent": sorted(r.parent) if r.parent else None,
                 "parent_conf": None if np.isnan(r.parent_conf) else r.parent_conf,
                 "p_cond": None if np.isnan(r.p_cond) else r.p_cond}
                for r in sorted(to_ai, key=lambda r: -r.lift)],
            "general_rules": [
                {"ant": sorted(r.ant), "cons": sorted(r.cons), "n_ant": r.n_ant,
                 "n_both": r.n_both, "support": r.support, "confidence": r.conf,
                 "lift": r.lift, "p": r.p, "p_bonf": r.p_bonf}
                for r in sorted([r for r in all_rules if AI_ITEM not in r.cons],
                                key=lambda r: -r.lift)],
            "freq_itemsets": [{"items": sorted(s), "count": c, "support": c / n}
                              for s, c in sorted(freq_all.items(), key=lambda kv: -kv[1])],
            "permutation": perm, "cv": cv,
        }
        Path(args.json).write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                   encoding="utf-8")
        print(f"\nJSON 已写入：{args.json}")


if __name__ == "__main__":
    main()
