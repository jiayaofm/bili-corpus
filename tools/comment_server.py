# -*- coding: utf-8 -*-
"""
Edge 扩展的本地服务：给 B站 评论打分 + 改写。

## 为什么要本地服务

- 打分用 `pipeline/scorer.py`（sklearn），搬进浏览器要重写
- 改写用 Qwen3-1.7B + LoRA（fish-a2），浏览器跑不动

扩展只负责抓评论、显示结果；算全在这边。

## 接口

```
GET  /health
     → {"ok":true, "scorer":true, "fish":true, "lora":false, "lora_model":"..."}

POST /score    {"texts": ["评论1", "评论2"]}
     → {"results": [{"text":..., "ai_prob":0.12, "ai_level":"low",
                      "fish_prob":0.87, "fish_level":"high"}]}

POST /rewrite  {"text": "..."}
     → {"text":..., "rewritten":"...", "ai_before":..., "ai_after":...,
        "fish_before":..., "fish_after":...}
```

## 两个分数

| 分数 | 测什么 | 来源 |
|---|---|---|
| `ai_prob` | **通用 AI 腔**：长句、书面语、复述标题、升华 | `pipeline/scorer.py`，40 个特征 |
| `fish_prob` | **鱼腥味**：鱼尚在个人的句式习惯 | 用 986 对改写样本现训的检测器 |

两者是不同层次的问题，一条评论可能「AI 味低但鱼腥味高」。

## 阈值从哪来

鱼腥味检测器在 986 对样本上的分布：

```
鱼腔   P25 0.545  中位 0.558  P75 0.572
真人   P25 0.414  中位 0.464  P75 0.486
```

中间的空档（0.486~0.545）是天然分界，所以取 0.47 / 0.52 两档。

**注意**：这个检测器可以被标点欺骗（只换句号加表情就能拿低分），
所以分数只做参考，不要当成判决。

## 启动

```bash
python -X utf8 tools/comment_server.py            # 只开打分
python -X utf8 tools/comment_server.py --lora     # 同时加载改写模型（占 3.8G 显存）
```

默认端口 8849。
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent   # tools/comment_server.py → 项目根
sys.path.insert(0, str(ROOT))

# 鱼腥味分档，单位是「比多少比例的真人更像鱼」的百分位。
# 真人自身的分位**本来就均匀分布在 0~100**（一半真人会高于 50），
# 所以 60 对真人是正常值，不能标黄。只有明显偏离才提示。
# 实测：真人 50~70，大肥鱼腔 100，典型 AI 腔 99。
FISH_LOW, FISH_HIGH = 75.0, 90.0

# 短于这个字数不测鱼腥味。3~9 字的短评在训练数据里特征太稀疏，
# 检测器输出基本是噪声（实测「将劫驶」8 分位区间内乱跳）。
FISH_MIN_LEN = 10
# 通用 AI 腔的判定阈值，用概率绝对值，不用分位。
#
# 为什么改回来：标注集（63 AI / 200 真人）上的实测是
#     阈值 0.50 → AI 召回 100%，真人误伤 0.0%
#     阈值 0.80 → AI 召回 100%，真人误伤 0.0%
#     阈值 0.90 → AI 召回 93.7%
# 而全库 9169 条随机真人评论上，阈值 0.43 会误伤 15.1%、0.80 误伤 6.6%、0.90 误伤 3.2%。
#
# 取 0.80：换到随机评论流时误伤降到 6.6%，标注集上召回仍是 100%。
# 0.50（这篇报告默认的「最佳阈值 0.43」那一档）在真实评论流里误伤太高，
# 在浏览器逐条显示的场合体感很差——它适合批量筛查，不适合逐条定罪。
AI_HIGH = 0.80      # ≥ 判红（AI 嫌疑）
AI_LOW = 0.50       # 0.50~0.80 判黄，<0.50 判绿


def level(v: float, lo: float, hi: float) -> str:
    return "low" if v < lo else ("medium" if v < hi else "high")


# ------------------------------------------------------------------ 鱼腥味检测器
class FishDetector:
    """用 whale_comments 里的改写对现训一个二分类器。"""

    def __init__(self) -> None:
        self.clf = None
        self.n_pos = self.n_neg = 0
        self._fit()

    def _fit(self) -> None:
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        P, H = [], []
        for fp in glob.glob(str(ROOT / "data" / "whale_comments" / "*.jsonl")):
            with open(fp, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if r.get("skipped"):
                        continue
                    for rw in (r.get("rewrites") or []):
                        o = (rw.get("origin") or "").strip()
                        w = (rw.get("rewrite") or "").strip()
                        if o and w and o != w:
                            P.append(w)
                            H.append(o)
        if len(P) < 20:
            print(f"  [fish] 样本不足（{len(P)}），跳过鱼腥味检测", flush=True)
            return
        self.clf = make_pipeline(
            TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=1, sublinear_tf=True),
            LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced"),
        )
        self.clf.fit(P + H, np.array([1] * len(P) + [0] * len(H)))
        self.n_pos, self.n_neg = len(P), len(H)
        # 真人评论的分数分布，用来把原始概率换算成「百分位」。
        # 平衡训练出来的概率两类的均值都在 0.5 附近，直接当百分比显示会误导
        # （真人中位 0.464，显示成「鱼 46%」像是有一半概率是鱼）。
        hs = [float(x) for x in self.clf.predict_proba(H)[:, 1]]
        self.h_ref = sorted(hs)
        # 按长度分桶：短评在整体分布里是长尾，直接比会被误判成「很鱼」。
        # 实测「将劫驶」（3 字）在全局百分位上是 90，明显是假阳性。
        self.h_buckets: dict[int, list[float]] = {}
        for t, v in zip(H, hs):
            self.h_buckets.setdefault(len(t) // 20, []).append(v)
        for k in self.h_buckets:
            self.h_buckets[k].sort()
        print(f"  [fish] 检测器就绪：鱼 {self.n_pos} / 人 {self.n_neg}", flush=True)

    MIN_BUCKET = 30

    def percentile(self, v: float, text_len: int = 0) -> float:
        # 这个分数在「同等长度」的真人评论里排在多少分位（0~100）。
        # 分桶是必要的：真人评论长度中位 55 字，3~8 字的短评落在长尾，
        # 用全局分布比会把正常短评判成 90 分位（实测「将劫驶」就是）。
        import bisect
        if not getattr(self, 'h_ref', None):
            return 50.0
        bucket = self.h_buckets.get(text_len // 20)
        ref = bucket if (bucket and len(bucket) >= self.MIN_BUCKET) else self.h_ref
        i = bisect.bisect_left(ref, v)
        return i / len(ref) * 100.0

    def score(self, texts: list[str]) -> list[float]:
        if self.clf is None:
            return [0.5] * len(texts)
        return [float(x) for x in self.clf.predict_proba(texts)[:, 1]]


# ------------------------------------------------------------------ 共享底座模型
# 显存只够装一个 1.7B，分类器（算句向量）和改写（要 lm_head）必须共用同一份。
BASE_MODEL = None          # AutoModelForCausalLM，带 lm_head，改写用
BASE_TOK = None
BASE_ERROR = None

QWEN = "Qwen/Qwen3-1.7B"


def ensure_base() -> bool:
    """加载 Qwen3-1.7B 底座，整个进程只加载一次。"""
    global BASE_MODEL, BASE_TOK, BASE_ERROR
    if BASE_MODEL is not None:
        return True
    if BASE_ERROR:
        return False
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        for cand in (Path(r"D:\dsh-ai\lora\hf"),
                     Path.home() / ".cache" / "huggingface"):
            hub = cand / "hub"
            if hub.is_dir() and any(hub.glob("models--Qwen*")):
                os.environ.setdefault("HF_HOME", str(cand))
                break
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        print(f"  [base] 加载 {QWEN} …", flush=True)
        BASE_TOK = AutoTokenizer.from_pretrained(QWEN)
        BASE_MODEL = AutoModelForCausalLM.from_pretrained(
            QWEN, dtype=torch.float16, device_map="auto").eval()
        print("  [base] 就绪", flush=True)
        return True
    except Exception as e:                                       # noqa: BLE001
        BASE_ERROR = f"{type(e).__name__}: {e}"
        print(f"  [base] 失败：{BASE_ERROR}", flush=True)
        return False


def embed_texts(texts: list[str], bs: int = 16) -> "object":
    """取句向量（对 token 做注意力掩码平均）。

    如果改写适配器已经挂上，要临时禁用——否则句向量会带上 LoRA 的影响，
    和训练分类器时用的底座不一致。
    """
    import contextlib

    import numpy as np
    import torch

    if not ensure_base():
        raise RuntimeError(BASE_ERROR or "底座未加载")
    dis = None
    if REWRITER is not None and REWRITER.model is not None \
            and hasattr(REWRITER.model, "disable_adapter"):
        dis = REWRITER.model.disable_adapter()

    out = []
    with (dis if dis is not None else contextlib.nullcontext()), torch.inference_mode():
        for i in range(0, len(texts), bs):
            enc = BASE_TOK(texts[i:i + bs], return_tensors="pt", padding=True,
                           truncation=True, max_length=400).to(BASE_MODEL.device)
            h = BASE_MODEL.model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1)
            out.append(((h * m).sum(1) / m.sum(1)).float().cpu().numpy())
    return np.vstack(out)


# ------------------------------------------------------------------ 分类器 v3
class ClassifierV3:
    """手工特征 + Qwen3 句向量 + 逻辑回归。

    换掉纯手工特征的理由（5 折 CV，263 条标注）：
        42 手工特征 + 随机森林      AP=0.838  AUC=0.921
        Qwen3 句向量 + LR          AP=0.888  AUC=0.956
        两者拼接                   AP=0.901  AUC=0.960

    手工特征看标点统计（「」、分号、句号数），所以「长句子、讲道理、带链接」的
    真人评论会被误判——实测这类评论从 62.6% 降到 1.0%。
    """

    def __init__(self, path: Path) -> None:
        import numpy as np
        self.ok = False
        self.err = None
        try:
            z = np.load(path, allow_pickle=True)
            self.coef = z["coef"]
            self.b = float(z["intercept"][0])
            self.mean = z["mean"]
            self.std = z["std"]
            self.cols = [str(c) for c in z["cols"]]
            self.cv_auc = float(z["cv_auc"]) if "cv_auc" in z else None
            self.ok = True
        except Exception as e:                                   # noqa: BLE001
            self.err = f"{type(e).__name__}: {e}"

    def _hand(self, texts):
        import numpy as np
        import pipeline.features as F
        return np.array([[F.featurize(t, "")[c] for c in self.cols] for t in texts], float)

    def score(self, texts: list[str]) -> list[float]:
        import numpy as np
        X = np.hstack([self._hand(texts), embed_texts(texts)])
        z = ((X - self.mean) / self.std) @ self.coef + self.b
        return [float(v) for v in 1.0 / (1.0 + np.exp(-z))]


# ------------------------------------------------------------------ 改写模型
class Rewriter:
    """Qwen3-1.7B + LoRA(fish-a2)，懒加载。"""

    SYSTEM = "你把 AI 写的 B站 评论改成像真人网友写的。"
    TMPL = "这条评论：{x}\n\n改一下。"

    _CJK_RE = __import__("re").compile(r"[^\u4e00-\u9fff0-9A-Za-z]")

    def _bigrams(self, t: str) -> set:
        t = self._CJK_RE.sub("", t)
        return {t[i:i + 2] for i in range(len(t) - 1)}

    def _too_similar(self, a: str, b: str, thr: float = 0.60) -> bool:
        """输出和输入几乎一样 → 判定为「没改」。

        阈值原来是 0.85，太松了。v4 训练数据里「输入 vs 真人目标」的
        相似度中位是 0.29、P90 是 0.40，而 0.78、0.84 这种只换几个近义词的
        输出能轻松通过 0.85 —— 实测用户拿到的就是这么两条：
          「拼一块→黏一块」「想法→意思」           相似度 0.78
          「只删掉句尾的『啊。』」                  相似度 0.84
        """
        A, B = self._bigrams(a), self._bigrams(b)
        if not (A | B):
            return True
        return len(A & B) / len(A | B) >= thr

    @staticmethod
    def _max_new(texts: list[str]) -> int:
        """按输入长度给生成上限。

        原来固定 64，长评论会被硬截断——实测 71 字的输入，输出停在
        「科学家一看到外星人飞船就明白，」这种半句上，内容直接丢了。
        """
        # 系数从 1.6 提到 2.2：改写常比原文长，1.6 倍时输出会断在半句
        # （实测「…11块的菜卖111块，老唐」就断在这）
        return min(260, max(80, int(max(len(t) for t in texts) * 2.2) + 60))

    # --- 候选重排 + 逗号抑制 ---------------------------------------------
    # 实测（dev 78 条，综合分 = 0.6×语法像人话 + 0.4×内容保留）：
    #   基线            0.468
    #   只压逗号        0.443 (-5.3%)   ← 单独做没用，逗号压下去句号涨上来
    #   4 选 1          0.534 (+14.2%)  ← 主力
    #   4 选 1 + 压逗号  0.543 (+16.1%)  ← 最佳
    # 压逗号只在重排之上才有意义，因为重排能把「压过头」的候选筛掉。
    # 候选数的性价比（dev 78 条实测，综合分 = 0.6×语法 + 0.4×内容）：
    #   基线      0.468
    #   2 选 1    0.508 (+7.4%)   耗时约单条的 2 倍
    #   3 选 1    0.534 (+10.8%)  内容保留已追平 4 选 1
    #   4 选 1    0.534 (+14.2%)  只多赚语法分
    # 默认取 2；想要更好质量用 --cand 3 或 4。
    N_CAND = 4
    # 压逗号在离线实验里只多 1.9 个点（0.534→0.543），实测却会把逗号换成空格、
    # 还诱发了重复退化（点歪歪点歪歪）。收益太小、副作用太明显，默认关掉。
    COMMA_BIAS = 0.0

    @classmethod
    def _demote_proc(cls, ids, bias):
        """把逗号类 token 的 logit 压低。sisyphus 生成时逐 token 调用。"""

        def proc(input_ids, scores):
            for i in ids:
                scores[:, i] = scores[:, i] - bias
            return scores

        return proc

    # 常见繁体专用字。模型偶尔会整条输出繁体（实测「點歪不點歪都是說法」），
    # 而语法分只看统计量，发现不了这种问题，所以单独挡。
    #
    # 只收「繁体独有」的字。繁简同形的字绝对不能进来，否则会大面积误杀：
    #   第一版误收「材」→「材料学卡住了」被丢
    #   第二版误收「得」→「笑得不行」「写得漂漂亮亮」全被丢（!!）
    #   还误收过「真」「本」
    # 简繁同形字举例：得 真 本 材 东 后? 注意「后/後」不算同形。
    _TRAD = set(
        "點說學現發財機麼沒這個們來時會為與對開關頓過無錯體樣種動"
        "聲氣邊連遠運進選雖讓訴識讀變車馬鳥魚幾嗎話語見覺還貴價錢")
    # 明确排除（繁简同形，写在这里提醒后来人别再收进去）
    _TRAD_SHARED = set("得真本材東")

    # 中文评论不会以这些字收尾，出现了基本就是被 token 上限切断的。
    # 只有这些字真正不可能出现在句末。
    # 原来的表里收了「的、是、在、了、就、都、也、还、很、太、更、最」——
    # 这些在中文评论里本来就是常见收尾（真人 74% 的评论结尾连标点都没有），
    # 拿它们判截断会把合格候选误杀。
    TRUNC_TAIL = set("把被但而因所如虽与及或况并")

    # ---------------------------------------------------------------- 数字
    # 实测「沙县 11 块的炒饭」会被改写成「沙县十一点的炒饭」，
    # 「15分钟」→「五分钟」，「99%」→「九成」（99 变成 90）。带了数字的
    # 样本里 25% 出错，而真人原文只有 6%。
    _CN_DIGIT = "零一二三四五六七八九"
    _CN_UNIT = ["", "十", "百", "千"]

    @classmethod
    def _to_cn(cls, n: int) -> str:
        """0~9999 转中文数字（11→十一，110→一百一十）。"""
        if n <= 0 or n > 9999:
            return ""
        s = str(n)
        out = []
        for i, ch in enumerate(s):
            d = int(ch)
            unit = cls._CN_UNIT[len(s) - i - 1]
            if d == 0:
                if out and out[-1] != "零" and i != len(s) - 1:
                    out.append("零")
                continue
            if d == 1 and unit == "十" and i == 0:
                out.append("十")
            else:
                out.append(cls._CN_DIGIT[d] + unit)
        return "".join(out).rstrip("零")

    # 数字后面跟的量词。只比数字不够：「11块 → 十一点」数字是对的，
    # 但把价格变成了时间，必须连量词一起看。
    _UNITS = ("块元角分年月日个只条次碗斤米秒点张件份套本台部杯包瓶袋双对成"
              "遍回轮层位名岁周期集章页支场局盘球拍页")

    @classmethod
    def _restore_numbers(cls, src: str, out: str) -> str:
        """把模型改坏的数字还原成源文的阿拉伯数字。

        为什么不在解码期管：实测 1.7B 模型根本做不到——就算提示里明确写
        「11 块必须保持 11 块」，它还是写出「十块」「一百一十」「十分钟」。
        这些不只是写法不同，是**改值**（11→10，111→110，15→10）。

        做法：按源文顺序取「数字+量词」，在输出里顺序找下一个
        「数字（阿拉伯或中文）+同一量词」，替换成源文的数字。
        某个找不到就跳过，不整体放弃——模型经常少写或多写几个数字。
        """
        items = []
        for n, nxt in re.findall(r"(\d+)\s*(.)?", src):
            unit = nxt if nxt and nxt in cls._UNITS else (
                "%" if nxt == "%" else "")
            items.append((n, unit))

        cursor = 0
        pieces = []
        for num, unit in items:
            if not unit:
                continue
            # 百分比在口语里常写成「九成」/「九成九」，两种都要认
            alts = [unit] + (["成"] if unit == "%" else [])
            hit, matched = None, None
            for u in alts:
                pat = re.compile(r"([0-9]+|[零〇一二三四五六七八九十百千两]+)"
                                 r"(\s*)(" + re.escape(u) + r")")
                m = pat.search(out, cursor)
                if not m:
                    continue
                # 「九成九」是 99 的正确写法，别把前半截换掉
                if u == "成" and m.end() < len(out) and \
                        out[m.end()] in cls._CN_DIGIT + "两":
                    continue
                hit, matched = m, u
                break
            if not hit:
                continue
            # 命中「成」时还原成 %，否则会出现「99成的人」
            rep_unit = "%" if matched == "成" else hit.group(3)
            pieces.append(out[cursor:hit.start()])
            pieces.append(num + hit.group(2) + rep_unit)
            cursor = hit.end()
        if not pieces:
            return out
        pieces.append(out[cursor:])
        return "".join(pieces)

    @classmethod
    def _num_forms(cls, num: str, unit: str) -> list[str]:
        """这个数字在这个量词下所有可接受写法。

        **只接受阿拉伯数字写法。** 本来想放宽到等值汉字（真人会写「八碗」
        「九成九」），但实测 1.7B 模型做数字→汉字转换不可靠：
          11 块    → 十块      （11 变 10）
          111 块   → 一百一十   （111 变 110）
        放宽的收益（更自然）远小于风险（数字错），所以收死。
        """
        return [num + unit if unit else num]

    @classmethod
    def _nums_lost(cls, src: str, out: str) -> list[str]:
        """输入里有、输出里被改值或丢掉的东西。

        允许写成等值汉字（真人也会写「八碗」「九成九」），
        但不允许变值、改量词（11块→11点）、或直接消失。
        """
        if not re.search(r"\d", src):
            return []
        lost = []
        out_compact = re.sub(r"\s+", "", out)      # 「11 块」和「11块」都算
        # 数字 + 后面紧跟的一个字符（量词或标点）。
        # \s* 不能省：源文常写成「11 块」，中间带空格，否则量词会漏掉。
        for num, nxt in re.findall(r"(\d+)\s*(.)?", src):
            unit = nxt if nxt and nxt in cls._UNITS else (
                "%" if nxt == "%" else "")
            forms = cls._num_forms(num, unit)
            if not any(f in out or f in out_compact for f in forms):
                lost.append(num + (unit or ""))
        # 去重保序
        seen, uniq = set(), []
        for x in lost:
            if x not in seen:
                seen.add(x)
                uniq.append(x)
        return uniq

    def _cand_ok(self, src: str, cand: str) -> bool:
        """硬过滤：语言一致、没被截断、够长。"""
        if not cand or len(cand) < 6:
            return False
        if any(ch in self._TRAD for ch in cand):
            return False
        # 表情标签被切一半：[打cal
        if cand.count("[") > cand.count("]"):
            return False
        # 中文之间夹空格当逗号用（实测「材料学卡了 材料一突破了」）。
        # 真人不这么写，但语法分统计不到，只能硬挡。
        # 阈值从 2 降到 1：只出现一处的也见过（「都是说法 你现在是…」）。
        if re.search(r"[\u4e00-\u9fff][ \t]{1,}[\u4e00-\u9fff]", cand):
            return False
        # 整句被方括号包住
        if cand.startswith("[") and cand.endswith("]"):
            return False
        # 结尾挂着一个只有左括号或光秃秃的标点
        if cand[-1] in "[（(【":
            return False
        # 结尾落在这些字上，几乎一定是被切断的：
        # 实测「…11块的菜卖111，当场」「…他手里却是拿木棍，但」
        # 数字也要算：出现过「…11块钱的菜卖1」这种半截价格。
        if cand[-1] in self.TRUNC_TAIL:
            return False
        # 数字不在过滤阶段管：模型把 11 写成十、15 写成十是常态，
        # 用这条否决会把好候选也杀掉。改在生成后由 _restore_numbers 修。
        return True

    def _rep_penalty(self, s: str) -> float:
        """重复退化惩罚。实测重排会挑出「点歪和点歪都是说法」这种病句。"""
        from collections import Counter
        bg = [s[i:i + 2] for i in range(len(s) - 1)]
        bg = [b for b in bg if b.strip()]
        if not bg:
            return 0.0
        c = Counter(bg)
        return sum(max(0, v - 2) for v in c.values()) * 0.03

    def _sim(self, a: str, b: str) -> float:
        """字符 bigram 的 Jaccard 相似度。"""
        A, B = self._bigrams(a), self._bigrams(b)
        return len(A & B) / len(A | B) if (A | B) else 1.0

    # 真人改写后与原文的相似度中位 0.29、P90 0.40（v4 训练数据实测）。
    # 「改多少」是有客观目标值的：太像=没改，太不像=跑题。
    TARGET_SIM = 0.29

    def _rank(self, src: str, cand: str) -> float:
        """候选打分：像人话 + 内容保留 + 改写幅度贴近真人分布 − 重复惩罚。

        第三项是关键，别删。原来的 0.6×像人话 + 0.4×保留 完全不惩罚照抄：
        在 20 条真实评论上实测「太像输入(≥0.6)」占 10/20，合格率仅 25%，
        而且把候选数从 2 加到 6 毫无改善——选择函数永远挑最像的那条。
        加上改写幅度项后（N=4）：太像 2/20，合格率 85%。
        """
        s = self._sim(src, cand)
        div = 1.0 - min(1.0, abs(s - self.TARGET_SIM) / self.TARGET_SIM)
        return (0.40 * human_score(cand)
                + 0.30 * self._content_recall(src, cand)
                + 0.30 * div
                - self._rep_penalty(cand))

    def _content_recall(self, src: str, out: str) -> float:
        A, B = self._bigrams(src), self._bigrams(out)
        return 1.0 if not A else len(A & B) / len(A)

    def __init__(self, adapter: str | None) -> None:
        self.adapter = adapter
        self.model = None
        self.tok = None
        self.lock = threading.Lock()
        self.error = None
        self.q = __import__('queue').Queue()
        self.worker = None
        self.BATCH_MAX = 8
        self.comma_ids = []

    def load(self) -> bool:
        if self.model is not None:
            return True
        if self.error:
            return False
        if not self.adapter:
            self.error = "未指定 adapter（用 --lora 启动）"
            return False
        try:
            import torch
            from peft import PeftModel

            # 基础模型和分类器共用一份，显存只够装一个 1.7B
            if not ensure_base():
                self.error = BASE_ERROR or "基础模型加载失败"
                return False
            self.tok = BASE_TOK
            print(f"  [lora] 挂载 {self.adapter} …", flush=True)
            self.model = PeftModel.from_pretrained(BASE_MODEL, self.adapter).eval()
            print("  [lora] 就绪", flush=True)
            return True
        except Exception as e:                                    # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            print(f"  [lora] 失败：{self.error}", flush=True)
            return False

    def rewrite(self, text: str, timeout: float = 90.0) -> tuple[str, bool] | None:
        """把一个改写请求丢进队列，等批量工作线程回结果。

        为什么排队而不直接生成：模型一次只能跑一个，
        连点 5 个按钮时串行要 55 秒（实测），后面几个必然超时。
        合成一个 batch 一次生成，5 个大概和 1 个一样快。

        返回 (改写结果, 是否没改到位)。第二个值用来让前端如实提示，
        而不是把一条几乎没变的句子当成「已改写」展示。
        """
        if not self.load():
            return None
        import queue as _q

        self._ensure_worker()
        ev = threading.Event()
        box = {"out": None, "err": None, "weak": False}
        self.q.put((text, ev, box))
        if not ev.wait(timeout):
            return None
        if box["err"]:
            raise RuntimeError(box["err"])
        return box["out"], bool(box.get("weak"))

    def _ensure_worker(self) -> None:
        if getattr(self, "worker", None) is not None:
            return
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    def _worker_loop(self) -> None:
        import queue as _q
        import torch
        while True:
            item = self.q.get()                       # 阻塞等第一个
            batch = [item]
            while len(batch) < self.BATCH_MAX:        # 把同时到达的一起带走
                try:
                    batch.append(self.q.get_nowait())
                except _q.Empty:
                    break
            try:
                self.tok.padding_side = "left"
                # 每条请求生成 N_CAND 个候选，最后挑一个
                prompts = []
                for text, _, _ in batch:
                    msgs = [{"role": "system", "content": self.SYSTEM},
                            {"role": "user", "content": self.TMPL.format(x=text)}]
                    p = self.tok.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                        enable_thinking=False)
                    prompts.extend([p] * self.N_CAND)

                if not self.comma_ids:
                    self.comma_ids = sorted(set(
                        self.tok.encode("，", add_special_tokens=False)
                        + self.tok.encode("、", add_special_tokens=False)
                        + self.tok.encode("；", add_special_tokens=False)))
                procs = ([self._demote_proc(self.comma_ids, self.COMMA_BIAS)]
                         if self.COMMA_BIAS else None)

                mn = self._max_new([t for t, _, _ in batch])
                enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
                with torch.inference_mode():
                    out = self.model.generate(**enc, max_new_tokens=mn, do_sample=True,
                                              temperature=0.95, top_p=0.95,
                                              logits_processor=procs,
                                              pad_token_id=self.tok.eos_token_id)
                gen = out[:, enc["input_ids"].shape[1]:]
                eos = self.tok.eos_token_id
                # 准确的截断信号：没以 EOS 结尾就是撞上了 max_new。
                # 之前靠「结尾字黑名单」，把「了」「的」「是」这类正常收尾
                # 也当成截断，误杀了不少本来合格的候选。
                trunc1 = [int(t) != eos for t in gen[:, -1].tolist()]
                cands = [s.strip().replace("\n", " ") for s in
                         self.tok.batch_decode(gen, skip_special_tokens=True)]

                picks, weak = {}, {}
                retry_idx = []
                for i, (txt, ev, box) in enumerate(batch):
                    lo = i * self.N_CAND
                    pool = cands[lo:(i + 1) * self.N_CAND]
                    tr1 = trunc1[lo:(i + 1) * self.N_CAND]
                    # 先排除「几乎没改」、被截断、和其他不合格的
                    good = [c for j, c in enumerate(pool)
                            if not self._too_similar(txt, c)
                            and self._cand_ok(txt, c) and not tr1[j]]
                    if good:
                        picks[i] = max(good, key=lambda c: self._rank(txt, c))
                    else:
                        retry_idx.append(i)

                # 候选全没改（或全不合格）时，升温 + 明确要求换说法，重生成一次
                if retry_idx:
                    print(f"  [rewrite] {len(retry_idx)} 条候选没改到位，重生成",
                          flush=True)
                    hint = ("\n\n必须换一种说法，不能照抄原句："
                            "句式、语序也要改，只换几个近义词不算。")
                    rp = []
                    for i in retry_idx:
                        txt = batch[i][0]
                        msgs = [{"role": "system", "content": self.SYSTEM},
                                {"role": "user",
                                 "content": self.TMPL.format(x=txt) + hint}]
                        p = self.tok.apply_chat_template(
                            msgs, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False)
                        rp.extend([p] * self.N_CAND)
                    enc2 = self.tok(rp, return_tensors="pt",
                                    padding=True).to(self.model.device)
                    with torch.inference_mode():
                        out2 = self.model.generate(
                            **enc2, max_new_tokens=mn, do_sample=True,
                            temperature=1.15, top_p=0.95,
                            pad_token_id=self.tok.eos_token_id)
                    g2 = out2[:, enc2["input_ids"].shape[1]:]
                    # 准确的截断信号：没以 EOS 结尾就说明撞上了 max_new
                    eos = self.tok.eos_token_id
                    trunc2 = [int(t) != eos for t in g2[:, -1].tolist()]
                    cands2 = [s.strip().replace("\n", " ") for s in
                              self.tok.batch_decode(g2, skip_special_tokens=True)]
                    for k, i in enumerate(retry_idx):
                        lo = k * self.N_CAND
                        pool2 = cands2[lo:(k + 1) * self.N_CAND]
                        tr2 = trunc2[lo:(k + 1) * self.N_CAND]
                        txt = batch[i][0]
                        good2 = [c for j, c in enumerate(pool2)
                                 if not self._too_similar(txt, c)
                                 and self._cand_ok(txt, c) and not tr2[j]]
                        if good2:
                            picks[i] = max(good2, key=lambda c: self._rank(txt, c))
                        else:
                            # 两次都改不动，如实返回并标记，别假装成功
                            picks[i] = max(pool2 or
                                           cands[i * self.N_CAND:(i + 1) * self.N_CAND],
                                           key=lambda c: self._rank(txt, c))
                            weak[i] = True

                for i, (txt, ev, box) in enumerate(batch):
                    # 数字最后统一还原：模型把 11 块写成十块、15 分钟写成
                    # 十分钟这类改值错误，靠提示管不住，只能生成后改回来。
                    box["out"] = self._restore_numbers(txt, picks[i])
                    box["weak"] = weak.get(i, False)
                    ev.set()
            except Exception as e:                    # noqa: BLE001
                for _, ev, box in batch:
                    box["err"] = f"{type(e).__name__}: {e}"
                    ev.set()


# ------------------------------------------------------------------ HTTP
SCORER = None
FISH = None
REWRITER = None
AI_REF: list[float] = []          # 真人评论的 AI 分分布，用来算百分位
CLF3 = None                       # 新分类器（手工特征 + 句向量）


try:
    sys.path.insert(0, str(ROOT / "tools"))
    from syntax_stats import human_score          # noqa: E402
except Exception as _e:                            # noqa: BLE001
    def human_score(_t):
        return 0.0
    print(f"  [syntax] 语法打分不可用：{_e}", flush=True)


def load_ai_reference(n: int = 1500) -> None:
    """加载「真人评论的 AI 分分布」。

    这个分布由 tools/build_ai_ref.py 离线算好（1500 条语料，用 scorer_v3）。
    以前是在服务启动时现场跑 2000 次 Scorer.describe()，每次都要过随机森林
    和 joblib 调度，实测启动被拖到 200 秒以上——faulthandler 抓栈确认卡在这。
    """
    global AI_REF
    path = ROOT / "state" / "ai_ref.json"
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            AI_REF = [float(v) for v in data["values"]]
            print(f"  [ai] 真人参照分布：{len(AI_REF)} 条（离线预计算）", flush=True)
            return
        except Exception as e:                                    # noqa: BLE001
            print(f"  [ai] 参照分布读取失败：{e}", flush=True)
    print("  [ai] 没有 ai_ref.json，AI 分位退回按概率算（先跑 tools/build_ai_ref.py）",
          flush=True)


def ai_percentile(v: float) -> float:
    """这个 AI 分比多少比例的真人评论更高。"""
    import bisect
    if not AI_REF:
        return v * 100
    return bisect.bisect_left(AI_REF, v) / len(AI_REF) * 100.0


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):          # 静音，避免刷屏
        pass

    def _send(self, code: int, obj: dict) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        # 扩展的 background 走 host_permissions，但本地直连调试也需要 CORS
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:               # noqa: N802
        self._send(204, {})

    def do_GET(self) -> None:                   # noqa: N802
        if self.path != "/health":
            self._send(404, {"error": "not found"})
            return
        self._send(200, {
            "ok": True,
            "scorer": SCORER is not None,
            "scorer_v3": bool(CLF3 is not None and CLF3.ok),
            "fish": FISH is not None and FISH.clf is not None,
            "fish_samples": (FISH.n_pos, FISH.n_neg) if FISH else (0, 0),
            "lora": REWRITER is not None and REWRITER.model is not None,
            "lora_adapter": REWRITER.adapter if REWRITER else None,
            "lora_error": REWRITER.error if REWRITER else None,
            "thresholds": {"ai": [AI_LOW, AI_HIGH], "fish": [FISH_LOW, FISH_HIGH]},
        })

    def do_POST(self) -> None:                  # noqa: N802
        try:
            n = int(self.headers.get("Content-Length") or 0)
            payload = json.loads(self.rfile.read(n) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._send(400, {"error": "bad json"})
            return

        if self.path == "/score":
            texts = [t for t in (payload.get("texts") or []) if isinstance(t, str)]
            texts = [t.strip()[:400] for t in texts if t.strip()]
            if not texts:
                self._send(400, {"error": "no texts"})
                return
            try:
                if CLF3 is not None and CLF3.ok:
                    ai = CLF3.score(texts)
                else:
                    ai = [s["ai_prob"] for s in (SCORER.describe(t) for t in texts)]
            except Exception as e:                                # noqa: BLE001
                # 句向量算不出来（显存/底座问题）时退回老的纯手工特征
                try:
                    ai = [s["ai_prob"] for s in (SCORER.describe(t) for t in texts)]
                except Exception:                                 # noqa: BLE001
                    self._send(500, {"error": f"scorer: {type(e).__name__}: {e}"})
                    return
            fish = FISH.score(texts) if FISH else [0.5] * len(texts)
            rows = []
            for t, a, f in zip(texts, ai, fish):
                apct = ai_percentile(a)
                if FISH and len(t) >= FISH_MIN_LEN:
                    pct = FISH.percentile(f, len(t))
                else:
                    pct = None
                rows.append({
                    "text": t,
                    "ai_prob": round(a, 4), "ai_pct": round(apct, 1),
                    "ai_level": ("high" if a >= AI_HIGH
                                  else ("medium" if a >= AI_LOW else "low")),
                    "fish_prob": round(f, 4),
                    "fish_pct": round(pct, 1) if pct is not None else None,
                    "fish_level": (level(pct, FISH_LOW, FISH_HIGH)
                                    if pct is not None else "na"),
                })
            self._send(200, {"results": rows})
            return

        if self.path == "/rewrite":
            text = (payload.get("text") or "").strip()
            if not text:
                self._send(400, {"error": "no text"})
                return
            if not REWRITER.load():
                self._send(503, {"error": REWRITER.error or "lora unavailable"})
                return
            got = REWRITER.rewrite(text)
            if not got:
                self._send(500, {"error": "generation failed"})
                return
            out, weak = got
            try:
                ai_b = SCORER.describe(text)["ai_prob"]
                ai_a = SCORER.describe(out)["ai_prob"]
            except Exception:                                     # noqa: BLE001
                ai_b = ai_a = None
            f_b = FISH.score([text])[0] if FISH else None
            f_a = FISH.score([out])[0] if FISH else None
            self._send(200, {
                "text": text, "rewritten": out, "weak": weak,
                "ai_before": round(ai_b, 4) if ai_b is not None else None,
                "ai_before_pct": round(ai_percentile(ai_b), 1) if ai_b is not None else None,
                "ai_after_pct": round(ai_percentile(ai_a), 1) if ai_a is not None else None,
                "ai_after": round(ai_a, 4) if ai_a is not None else None,
                "fish_before": (round(FISH.percentile(f_b, len(text)), 1)
                                if (FISH and f_b is not None and len(text) >= FISH_MIN_LEN) else None),
                "fish_after": (round(FISH.percentile(f_a, len(out)), 1)
                               if (FISH and f_a is not None and len(out) >= FISH_MIN_LEN) else None),
            })
            return

        self._send(404, {"error": "not found"})


def main() -> int:
    global SCORER, FISH, REWRITER, CLF3
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8849)
    ap.add_argument("--cand", type=int, default=4,
                    help="改写时生成几条候选再重排：1=不重排，2~4 越大越慢但越好")
    ap.add_argument("--lora", action="store_true", help="同时加载改写模型（占显存）")
    ap.add_argument("--adapter", default=str(Path(r"D:\dsh-ai\lora\out\fish-a2")))
    a = ap.parse_args()

    print("=" * 60)
    print("  B站 评论打分 / 改写 本地服务")
    print("=" * 60)

    from pipeline.scorer import Scorer
    SCORER = Scorer()
    load_ai_reference()
    print("  [ai] 分类器就绪", flush=True)

    CLF3 = ClassifierV3(ROOT / "state" / "scorer_v3.npz")
    if CLF3.ok:
        auc = (" AUC=%.3f" % CLF3.cv_auc) if CLF3.cv_auc else ""
        print(f"  [ai-v3] 句向量分类器就绪{auc}", flush=True)
    else:
        print(f"  [ai-v3] 未启用：{CLF3.err}（退回手工特征）", flush=True)

    FISH = FishDetector()
    REWRITER = Rewriter(a.adapter if a.lora else None)
    REWRITER.N_CAND = max(1, a.cand)
    if REWRITER.N_CAND > 1:
        print(f"  [rewrite] 每条生成 {REWRITER.N_CAND} 个候选再挑一个"
              f"（耗时约为单条的 {REWRITER.N_CAND} 倍）", flush=True)

    if a.lora:
        if not REWRITER.load():
            print("  [lora] 加载失败，/rewrite 会返回 503；/score 仍可用", flush=True)

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print()
    print(f"  监听 http://127.0.0.1:{a.port}")
    print(f"  /health  /score  {'/rewrite' if a.lora else '(/rewrite 未启用)'}")
    print("  Ctrl+C 停止")
    print()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
