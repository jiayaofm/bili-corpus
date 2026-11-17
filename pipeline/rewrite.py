# -*- coding: utf-8 -*-
"""
改写 workflow 主干：指导 → 生成 → 打分 → 修改

目标：把 AI 写的评论改写成真人口吻（去 AI 味），但要防住「改标点骗分数」这类作弊。

--------------------------------------------------------------------------
一、循环结构
--------------------------------------------------------------------------
    指导  describe(原文)                     → hits + advice
    生成  generator(原文, 指令, n)           → 最多 n 个候选（外部注入，本模块不含任何 API 调用）
    打分  scorer.compare(原文, 候选)          → reward / verdict / 是否 gaming
          fidelity.check(原文, 候选)          → 信息保真（模块可能不存在，见下）
    选优  过滤 gaming + 过滤不保真，按 reward 降序
    修改  没达标就拿「上一轮剩下的问题」拼新指令，再来一轮，最多 max_rounds 轮

    收敛条件（严格）：某轮出现 verdict == "ok" 且保真通过的候选 → 取 reward 最高者收工。

--------------------------------------------------------------------------
二、为什么不直接用 scorer 的 verdict 就完事（实测踩到的坑）
--------------------------------------------------------------------------
1) **空串/单字是奖励黑洞**。scorer 对空串打 reward=0.999（规则分 -12 被归一成 0，
   模型也判它像人）。实测 compare(原文, "")：
       item1  d_reward +0.677  d_held +0.302  → verdict "ok"
   也就是说，只要 fidelity 模块缺席，workflow 会当场把**空字符串**选成最佳改写。
   所以本模块内置一道与 fidelity 无关的「退化候选」闸门（空 / 长度比越界），
   见 _degenerate_reason()。这是兜底，不是替代 fidelity。

2) **相似度不能当廉价改写的判据**。实测真实改写 A/B/C 与原文的 difflib 相似度
   低到 0.26，而「原文末尾加个 [笑哭]」这种廉价操作相似度 0.955~0.987。
   但 item5 的 A（真实改写）相似度 0.970，和廉价操作重叠 —— 阈值分不开。
   因此相似度只作为候选的**诊断字段**记录，不参与过滤。

3) **截断 30% 有时能过 verdict**（item2/item3 实测 verdict=ok）。真正能防住的
   是 fidelity；没有 fidelity 时只能用长度比粗挡（默认下限 0.25，实测真实
   改写最低 0.32，故 0.25 不会误杀）。这条局限写进返回值的 notes 里。

--------------------------------------------------------------------------
三、fidelity 模块（并行开发中，按接口编码 + 容错）
--------------------------------------------------------------------------
    fidelity.check(original, rewritten)
        -> {"score": 0-1, "ok": bool, "missing": [...], "altered": [...],
            "added": [...], "notes": [...]}

    做法：
      · import 失败 / 没有 check → 整轮跳过保真检查，结果里标 status="skipped"；
      · 开跑前用 (原文, 原文) 做一次**恒等自检**：恒等改写必须判 ok，
        否则认为模块没写好 → 本次运行整体降级为跳过（避免半成品把流程卡死）；
      · 单个候选上 check 抛异常 → 该候选记 status="error" 并**不当作通过**
        （宁可不选，也不要选一条没被验证过的）；
      · 返回值结构不对（不是 dict / ok 不是 bool）→ 归一到 error 或按 score 折算；
      · 真实模块多返回一个 "parts" 字段 —— 本模块忽略未知字段，不依赖它。

--------------------------------------------------------------------------
四、fidelity 严格档 vs 宽松档（实测踩到的坑）
--------------------------------------------------------------------------
    fidelity 上线后实测：它会把**项目自己的好改写**判成不保真 ——
        state/rewrite_abc.json  item2: A ok(0.93)  B FAIL(0.61)  C FAIL(0.55)
                                item4: A ok(1.00)  B FAIL(0.56)  C FAIL(0.48)
    原因是该模块 __doc_limits__ 第 1 条自己写了：「同义改写会被要点覆盖率误报」。
    于是严格档下，item2 / item4 只有「几乎没改」的 A 能过保真闸，而 A 又被
    gaming 闸挡掉 —— 这两条永远收敛不了。
    这是两个模块的验收口径冲突（scorer 要更大改动，fidelity 嫌改动太大），
    不是本模块能调和的事实，所以做成显式开关：

        fidelity_policy="strict"（默认，照 workflow 约定：不保真就过滤）
        fidelity_policy="soft"   （分数 >= soft_fidelity_min 时放行但打
                                  fidelity_warn 标记，且选优时永远让
                                  真正通过的候选优先）

    strict 下行为完全按约定；soft 只是把「冲突」摆到台面上，不改判定口径。

--------------------------------------------------------------------------
五、奖励是绝对分
--------------------------------------------------------------------------
    cand["reward"] 是候选自己的分数（与原文无关的绝对量），所以跨轮、跨不同
    base 的候选可以直接比大小。d_reward / d_held_human 才是相对原文的增量。
    这也是为什么第 2 轮可以「基于上一轮候选继续改」，而验收口径始终对着原文。

用法：
    python pipeline/rewrite.py --text "一条 AI 味很重的评论"
    python pipeline/rewrite.py                      # 跑 state/rewrite_abc.json 里的 src
"""
import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Protocol

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
for _p in (str(ROOT), str(ROOT / "research")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ============================================================================
# 常量与可调参数
# ============================================================================
DEFAULT_MAX_ROUNDS = 3
DEFAULT_N_PER_ROUND = 4

# 单轮最多给多少候选打分（防止生成器发疯返回 10000 条，把打分器打爆）
MAX_CANDIDATES_PER_ROUND = 16

# 退化闸门：候选长度 / 原文长度的允许区间
# 下限 0.25 的依据：state/rewrite_abc.json 里真实改写 A/B/C 的最低长度比是 0.32
MIN_LEN_RATIO = 0.25
MAX_LEN_RATIO = 3.0

# fidelity 只给 ok/score 时，score 低于此值视为不通过
FIDELITY_MIN_SCORE = 0.8

# difflib 相似度只算短文本（长文本 O(n^2) 会拖慢主流程）
SIMILARITY_MAX_CHARS = 2000

# 硬约束：每一轮指令都要带。
#
# 2026-11-16 两次修正之后定稿：
#   第一次以为「信息罗列本身就是 AI 味」，于是放开成允许取舍 —— 错了。
#   错在把「表述方式不对」理解成了「内容该删」。放开之后模型开始做 summary，
#   把「订阅剩37%」挖成「订阅剩那点」，用户一眼看出「整句话挖空，像做summary」。
#   正确的是：**信息一条不少，但要从「陈述」改成「暗示」**。
#     不说「94分是猛」，说「开max能跑94分」
#     不说「离重置还有五天」，说「还要撑五天」
#     不说「max档先关」，说「用不起了」
#   删掉的是评价和元评论，不是事实。
HARD_CONSTRAINT = (
    "信息一条也不许少：原文提到的数字、型号、人名、结论都要留着。"
    "但要把它们从「陈述」改成「暗示」，不要整句挖空做 summary。"
    "不许编造原文没有的事实。核心立场必须和原文一致。"
    "删掉的只能是评价句和元评论（比如「这话由我来说最没有利益冲突」）。"
)

# 这些 hit 不是「文风」问题，改文风解决不了，别塞进指令里误导生成器
NON_ACTIONABLE_HITS = {
    "账号名含鱼/鲸",   # 取决于用户名，不在正文里
    "提及鱼名",        # 是视频话题词，删掉反而丢信息
}

# scorer.ADVICE 没覆盖到的 hit → 本模块补的建议
# （scorer 的 ADVICE 只匹配了 12 条规则，而 label_ai.MARKERS 有 19 条，
#   不补的话「助手视角」这类高分 hit 拿不到任何可执行建议）
EXTRA_ADVICE = {
    "markdown粗体": "删掉 Markdown 粗体（B站不渲染 **xx** 这种写法）",
    "第一人称自称": "别用「本鱼/本鲸/本喵」这种角色自称，用普通说法",
    "…是真的": "别用「…是真的」下判断句，改成随口的说法",
    "指代「这条」": "别用「这条」指代被回复的评论，直接说事",
    "另一回事": "别用「另一回事」这种转折评判，换成普通口语转折",
    "冒号解释": "别用冒号去解释，把后半句拆成短句",
    "元评论腔": "删掉「这话由我来说」这种自我评论",
    "傲娇否认": "删掉「才没有」式的否认",
    "书面副词": "去掉「恰恰/反而/无非」这类书面副词",
    "柔化语气": "句末语气词别一条接一条地堆",
}

# 第 2 轮起「上一轮没消掉」时的换手法提示。同一项如果照第一种建议改不掉，
# 指令就得给出**另一种手法**，否则等于把同一串建议原样再说一遍。
RETRY_TECHNIQUE = {
    "markdown粗体": "整段抄成纯文本，别用 ** 包字",
    "破折号插入": "直接删掉插入的那半句，或者拆成两个独立短句（别只用逗号替换）",
    "第一人称自称": "把「本鱼」整个删掉，句子从第三人称说起",
    "助手视角": "把自述整句删掉，只留对事的评价",
    "不是A是B": "把对仗整句删掉，只留其中一半的意思",
    "…是真的": "删掉判断句尾，改成一个具体例子",
    "分号并列": "拆成两句，中间加点口语连接（「结果」「然后就」）",
    "总结腔": "整句删掉，别换同义词保留总结位置",
    "本质内核腔": "整句删掉",
    "指代「这条」": "把「这条」换成省略，直接说内容",
    "另一回事": "改成「这个不一样」「那个另说」",
    "冒号解释": "把冒号后那句单独成句",
    "元评论腔": "整句删掉",
    "傲娇否认": "直接说否定，不加「才」",
    "书面副词": "把副词整个删掉，不加替代词",
    "数字对比": "只留一个数，其它用「差不多」「小几分」代替",
    "超长": "砍成两三个短句，中间用逗号断开",
    "「」引号": "去掉引号直接把词说出来",
    "柔化语气": "语气词只留一个",
}

_HIT_RE = re.compile(r"^(?P<name>.*?)(?P<w>[+-]\d+)$")


def _log(verbose, msg):
    if verbose:
        print(msg)


# ============================================================================
# 惰性加载 scorer（模块导入要快；--help 不该去拟合随机森林）
# ============================================================================
_SCORER_MOD = None
_SCORER = None


def _scorer_module():
    global _SCORER_MOD
    if _SCORER_MOD is None:
        from pipeline import scorer as m
        _SCORER_MOD = m
    return _SCORER_MOD


def get_scorer():
    """进程内单例。Scorer() 要拟合两个模型（约 2 秒），别在循环里反复建。"""
    global _SCORER
    if _SCORER is None:
        _SCORER = _scorer_module().Scorer()
    return _SCORER


# ============================================================================
# hit 解析与「要修的问题」抽取
# ============================================================================
def split_hit(hit):
    """'破折号插入+16' -> ('破折号插入', 16)；'URL+刷屏-30' -> ('URL+刷屏', -30)

    注意名字里本身可能带 '+'（URL+刷屏），所以只能用「末尾带符号整数」做切分。
    """
    h = (hit or "").strip()
    m = _HIT_RE.match(h)
    if not m:
        return h, 0
    return m.group("name"), int(m.group("w"))


def _advice_for_hit(name, hit, described):
    """先认 scorer 自己算的建议（保持单一事实来源），再查本模块的补充表。"""
    tips = described.get("advice") or []
    for pat, tip in getattr(_scorer_module(), "ADVICE", []):
        if pat and re.search(pat, hit) and tip in tips:
            return tip
    if name in EXTRA_ADVICE:
        return EXTRA_ADVICE[name]
    return f"减少「{name}」这类痕迹"


def describe_problems(described):
    """从 describe() 结果里抽出「本轮要修的问题」，按扣分从高到低排。

    只保留**正分** hit（负分是真人信号，改了反而是倒退），
    并剔掉改文风解决不了的（见 NON_ACTIONABLE_HITS）。
    """
    out = []
    for hit in described.get("hits") or []:
        name, w = split_hit(hit)
        if w <= 0:
            continue
        if name in NON_ACTIONABLE_HITS:
            continue
        out.append({
            "hit": hit,
            "name": name,
            "weight": w,
            "advice": _advice_for_hit(name, hit, described),
        })
    out.sort(key=lambda p: (-p["weight"], p["name"]))
    return out


def _problem_names(problems):
    return [p["name"] for p in (problems or [])]


def lexical_directives(described):
    """没有残留痕迹时，给出这一轮的改写方向。

    2026-11-17 大改。原版是「换词 + 加语气词」：

        把书面词换成口语词（但是→可、所以→就、非常→挺）
        加一点接话词和口气（「吧」「啊」「就」「真的」这类）

    实测这条指令**把模型带进了死胡同**：跑 20 条 AI 评论，出来的全是
    「呀→吧」「挺好笑→真挺好笑的」「是猛→够猛」这种只动虚词的改法，
    换词后仍是一股 AI 味。

    用户实测有效的写法是**结构性**的：补主语、调因果顺序、砍评价句。
    所以这里改成结构指令，并且**明确禁止只加语气词**——那句话必须写死，
    否则模型默认就往虚词上改。
    """
    out = [
        "不要只加语气词（吧/啊/呢）或替换同义词——那不算改，读者一眼能看出原句",
        "补主语：光写「能跑到94分」读者不知道是谁，要写「max能跑到94分」",
        "调顺序：好处 → 但代价 → 所以决定。决定必须紧跟它的理由，中间不许插别的信息",
        "砍掉评价句和元评论（「这分数我看看就好」「这话我来说最没利益冲突」这类）",
    ]
    if float(described.get("heldout_ai") or 0.0) >= 0.5:
        out.append("留下的信息必须具体：写「37%」不写「那点」，写「二十多刀」不写「不少钱」")
    if float(described.get("ai_prob") or 0.0) >= 0.5:
        out.append("句子可以断，但断的地方不能是信息；两个短动作连着说，别用逗号切开")
    out.append("从头到尾换个说法，不是在原句上把虚词抹平")
    return out


# ============================================================================
# 指令构造
# ============================================================================
def _failure_sentence(failure):
    """把上一轮的失败原因翻成一句人话，让生成器知道为什么被打回。"""
    if not failure:
        return "上一轮没有可用产出。"
    v = failure.get("verdict")
    dr = failure.get("d_reward")
    dh = failure.get("d_held_human")
    num = ""
    if dr is not None and dh is not None:
        num = f"（奖励 {dr:+.3f}，用词 {dh:+.3f}）"

    # 上一轮一条都没留下 → 用拒绝原因的分布来说，比笼统说「被过滤」有用得多
    if v is None:
        n = failure.get("n_rejected") or 0
        t = failure.get("reject_tally") or {}
        who = f"上一轮 {n} 个候选**全部**" if n != 1 else "上一轮唯一的候选"
        if n and t.get("gaming") == n:
            return (f"{who}被判「钻空子」：奖励涨了，但用词验收没改善"
                    " —— 只改标点糊弄不过去，这一轮必须换词。")
        if n and t.get("fidelity") == n:
            return (f"{who}因「不保真」被打回：改掉了原文的信息点。"
                    "这一轮只换说法，一个信息点都不许丢。")
        if n and (t.get("degenerate", 0) + t.get("other", 0)) == n:
            return (f"{who}全是无效产出（空 / 过短 / 打分失败 / 跟原文一样），"
                    "等于没改。")
        return f"{who}被过滤，没有可用产出。"

    if v == "gaming":
        return ("上一轮最好的一版被判成「钻空子」" + num +
                "：奖励涨了，但用词验收没改善 —— 只改标点糊弄不过去，这一轮必须换词。")
    if v == "no-gain":
        return ("上一轮没有取得任何有效改善" + num +
                "：奖励和用词都没往真人方向走，那套改法无效，换思路。")
    if v == "ok":
        return "上一轮有达标候选，但没被选中（保真未过）。"
    return "上一轮全部候选被过滤（空 / 重复 / 不保真），没有可用产出。"


def build_instruction(round_index, original, described=None, base_text=None,
                      problems=None, cleared=None, failure=None,
                      stalled_rounds=0):
    """拼中文改写指令。

    第 1 轮：任务 + 硬约束 + 原文里逐条要消除的痕迹。
    第 2 轮起：**只针对上一轮没改掉的问题** + 明说哪些已经改掉别重新引入 +
              明说上一轮为什么被打回；没有残留问题时切到「换词」方向。
    这样每轮指令都不重复：残留集在收缩，措辞和落点也在换。

    described: 当前 base 的 describe 结果（第 1 轮传原文的）。只在
               「没有残留痕迹、要改方向」时用来判断往哪儿改。
    base_text: 上一轮最好的候选记录（dict，含 text），仅第 2 轮起传。
    stalled_rounds: 已经连续几轮「一条问题都没消掉」。连续失败时必须加码，
               否则第 2 轮和第 3 轮会发出同一封指令。
    """
    problems = list(problems or [])
    cleared = list(cleared or [])
    lines = []

    if round_index <= 1:
        lines.append("【任务】把下面这段 AI 写的评论改写成真人评论口吻。"
                     "只输出改写后的评论本身，不要解释、不要加标题。")
    else:
        lines.append(f"【任务】第 {round_index} 轮改写。在上一轮最好的候选基础上继续改，"
                     "只输出改写后的评论本身。")
        lines.append("【上一轮结论】" + _failure_sentence(failure))
        if cleared:
            lines.append("【已经改掉、不要重新引入】" + "、".join(cleared))

    lines.append("【硬约束】" + HARD_CONSTRAINT)

    if problems:
        if round_index <= 1:
            lines.append(f"【本轮要消除的 AI 痕迹】共 {len(problems)} 项，按扣分从高到低：")
            for i, p in enumerate(problems, 1):
                lines.append(f"  {i}) {p['name']}（规则 +{p['weight']}）→ {p['advice']}")
        else:
            # 第 2 轮起：只列「上一轮还没消掉」的那几项，并且换一种呈现方式
            # （不再是第 1 轮的编号清单），避免把同一串建议原样重复。
            if base_text:
                head = f"【本轮只处理这几项】上一轮最好的候选上还剩 {len(problems)} 项没消掉："
            else:
                head = (f"【本轮只处理这几项】上一轮没有产出可用候选，"
                        f"原文里还剩 {len(problems)} 项没动：")
            lines.append(head)
            for p in problems:
                lines.append(f"  - {p['name']}：{p['advice']}（还没消掉）")
            if not cleared:
                # 一条都没消掉 = 上一轮那套改法无效，必须明说并要求落到字词上
                lines.append("  ↑ 上一轮这几项一条都没消掉，说明照着原来的说法改没用。"
                             "这一轮请落到**具体字词**上：把哪个词换成哪个词、删掉哪半句。")
                techs = [f"{p['name']} → {RETRY_TECHNIQUE[p['name']]}"
                         for p in problems if p["name"] in RETRY_TECHNIQUE][:3]
                if techs:
                    lines.append("  换手法（同一项用另一种改法）：")
                    for t in techs:
                        lines.append(f"    · {t}")
    else:
        if round_index <= 1:
            lines.append("【本轮要消除的 AI 痕迹】原文没命中硬性句式规则，"
                         "重点是把用词改口语一些：")
        else:
            lines.append("【本轮只处理这几项】标点句式已经清干净了，"
                         "这一轮的问题不在标点，在**用词**：")
        for d in lexical_directives(described or {}):
            lines.append(f"  - {d}")

    if round_index > 1:
        lines.append("【不要重复上一轮的改法】上一轮试过的做法既然被判无效，就换一种表达；"
                     "也不要靠删标点、加表情来凑分。")
        if stalled_rounds >= 1:
            lines.append(
                f"【加码】已经连续 {stalled_rounds} 轮没消掉上面这些问题了，"
                "上一轮交回来的东西和原文几乎没有区别。"
                "这一轮如果某一项实在改不动，就**换掉整句话的说法**，"
                "不要只在原句上做微调；实在做不到就明说做不到，别把原文再抄一遍。")

    if base_text:
        lines.append("【上一轮最好的候选】\n" + (base_text.get("text") or ""))
        lines.append("【原文（信息点以原文为准）】\n" + original)
    else:
        lines.append("【原文】\n" + original)
    return "\n".join(lines)


# ============================================================================
# 退化候选闸门（简版信息保真，fidelity 缺席时的兜底）
# ============================================================================
def _degenerate_reason(text, original, min_ratio=MIN_LEN_RATIO,
                       max_ratio=MAX_LEN_RATIO):
    """返回拒绝原因字符串，None 表示通过。

    为什么必须要有：scorer 把空串判成 reward 0.999 / verdict "ok"（见模块头
    注释）。没有这道闸门，fidelity 一缺席就会选出空串当最佳改写。
    """
    t = text.strip()
    if not t:
        return "empty"
    olen = len(original.strip()) or 1
    ratio = len(t) / olen
    if ratio < min_ratio:
        return f"too_short({ratio:.2f}<{min_ratio})"
    if ratio > max_ratio:
        return f"too_long({ratio:.2f}>{max_ratio})"
    return None


def _similarity(a, b):
    """字符级相似度，只做诊断字段用（不参与过滤，理由见模块头注释）。"""
    if max(len(a), len(b)) > SIMILARITY_MAX_CHARS:
        return None
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 3)


def _norm_key(text):
    """去重用的归一化：忽略空白差异。"""
    return re.sub(r"\s+", "", text)


# ============================================================================
# 生成器接口（可插拔，模块内不含任何大模型 API 调用）
# ============================================================================
class Generator(Protocol):
    def __call__(self, original: str, instruction: str, n: int) -> list[str]:
        """给定原文和改写指令，返回 n 个候选改写"""
        ...


class NoneGenerator:
    """什么都不生成，直接抛异常 —— 防止忘了注入生成器就悄悄跑空。"""

    def __call__(self, original, instruction, n):
        raise RuntimeError(
            "NoneGenerator 不能用来生成：rewrite() 需要注入真实生成器"
            "（比如包一层大模型调用的实现），或者用 EchoGenerator 冒烟。")


# EchoGenerator 的变换表：(触发词, 名称, 变换函数)
# 只有指令里出现触发词才会应用 —— 这样「指令变了 → 输出变了」在测试里可验证。
_ECHO_RULES = [
    ("破折号", "去破折号", lambda t: re.sub(r"[—–]{1,}", "，", t)),
    ("引号", "去「」", lambda t: re.sub(r"[「」『』]", "", t)),
    ("分号", "去分号", lambda t: t.replace("；", "，").replace(";", "，")),
    ("冒号", "去冒号", lambda t: re.sub(r"[：:]", "，", t)),
    ("总结", "去总结词",
     lambda t: re.sub(r"(关键是|说白了|说到底|最重要的是|归根结底|本质是|内核是)", "", t)),
    ("换词", "换口语词",
     lambda t: (t.replace("但是", "可").replace("所以", "就").replace("非常", "挺")
                 .replace("因此", "所以").replace("并且", "还").replace("如果", "要是"))),
]


class EchoGenerator:
    """测试用生成器：按指令里的关键字做**简单字符串变换**，不追求改写质量。

    用途只有一个 —— 验证 workflow 的流程能跑通、指令能影响输出。
    它故意做得很笨：
      · 只看指令里有没有触发词，不理解语义；
      · 每条候选 = 单个变换，外加一条「全部变换叠加」的激进版；
      · 可能返回少于 n 个（没有可做的变换时就只返回原文），
        调用方必须能处理短列表 —— 这也是对 workflow 的容错测试。
    """

    def __init__(self, rules=None):
        self.rules = list(rules if rules is not None else _ECHO_RULES)

    def __call__(self, original, instruction, n):
        ins = instruction or ""
        original = original or ""
        active = [fn for trig, _name, fn in self.rules if trig in ins]
        variants = []
        for fn in active:
            try:
                v = fn(original)
            except Exception:
                continue
            if v and v != original:
                variants.append(v)
        if active:
            # 激进版：所有变换叠加（从后往前叠，前面的失败不影响后面）
            agg = original
            for fn in active:
                try:
                    agg = fn(agg)
                except Exception:
                    continue
            if agg and agg != original:
                variants.append(agg)
        # 去重保序
        seen, uniq = set(), []
        for v in variants:
            k = _norm_key(v)
            if k in seen:
                continue
            seen.add(k)
            uniq.append(v)
        if not uniq:
            uniq = [original]          # 没有可做的变换 → 原样返回，故意让这轮拿不到改善
        return uniq[:max(1, int(n))]


# ============================================================================
# fidelity 适配层（模块可能不存在 / 可能是半成品）
# ============================================================================
def _as_str_list(v):
    if v is None:
        return []
    if isinstance(v, str):
        return [v] if v else []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [str(v)]


def _skipped_fidelity(reason=None):
    return {
        "available": False, "status": "skipped", "ok": True, "score": None,
        "missing": [], "altered": [], "added": [], "notes": [],
        "reason": reason or "fidelity 模块不可用，已跳过保真检查",
    }


def _normalize_fidelity(raw):
    """把 fidelity.check 的返回值归一成固定结构，永不抛异常。"""
    base = {
        "available": True, "status": "error", "ok": False, "score": None,
        "missing": [], "altered": [], "added": [], "notes": [], "reason": None,
    }
    if not isinstance(raw, dict):
        base["reason"] = f"check() 返回了 {type(raw).__name__}，不是 dict"
        return base

    score = raw.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        score = None
    else:
        score = float(score)

    ok = raw.get("ok")
    if not isinstance(ok, bool):
        ok = None

    base["score"] = score
    base["missing"] = _as_str_list(raw.get("missing"))
    base["altered"] = _as_str_list(raw.get("altered"))
    base["added"] = _as_str_list(raw.get("added"))
    base["notes"] = _as_str_list(raw.get("notes"))

    if ok is None and score is not None:
        ok = score >= FIDELITY_MIN_SCORE
    if ok is None:
        base["reason"] = "check() 既没给 ok 也没给 score，无法判断保真"
        return base

    base["ok"] = bool(ok)
    base["status"] = "ok" if ok else "fail"
    if not ok:
        base["reason"] = "保真检查未通过（有信息点被改动或丢失）"
    return base


def _resolve_fidelity(check=None):
    """返回 (check_callable | None, 不可用原因 | None)。显式注入的 check 优先。"""
    if check is not None:
        return check, None
    try:
        from pipeline import fidelity as mod
    except Exception as e:                      # 模块不存在 / 导入期报错
        return None, f"pipeline.fidelity 导入失败（{type(e).__name__}: {e}）"
    fn = getattr(mod, "check", None)
    if not callable(fn):
        return None, "pipeline.fidelity 里没有可调用的 check()"
    return fn, None


def _probe_fidelity(check, sample):
    """开跑前用 (原文, 原文) 做恒等自检：恒等改写必须判 ok。

    半成品模块经常在真实文本上炸或者一律返回 False，与其让整轮跑不出结果，
    不如一次性降级 —— 这个探针就是那个开关。
    """
    if check is None:
        return False, None
    try:
        raw = check(sample, sample)
    except Exception as e:
        return False, f"保真自检（恒等改写）抛异常：{type(e).__name__}: {e}"
    norm = _normalize_fidelity(raw)
    if norm["status"] == "error":
        return False, f"保真自检返回值无法解析：{norm.get('reason')}"
    if norm["status"] != "ok":
        return False, "保真自检失败：恒等改写竟然没通过（模块可能还没写好）"
    return True, None


def _check_fidelity(check, original, candidate, skipped_reason=None):
    """单个候选的保真检查，永不抛异常。"""
    if check is None:
        return _skipped_fidelity(skipped_reason)
    try:
        return _normalize_fidelity(check(original, candidate))
    except Exception as e:
        d = _skipped_fidelity(f"保真检查抛异常：{type(e).__name__}: {e}")
        d.update({"available": True, "status": "error", "ok": False})
        return d


# ============================================================================
# 主流程
# ============================================================================
def _picked_view(c):
    if not c:
        return None
    return {
        "text": c["text"],
        "reward": c["reward"],
        "verdict": c["verdict"],
        "fidelity": (c.get("fidelity") or {}).get("status"),
        "fidelity_warn": bool(c.get("fidelity_warn")),
        "meets_bar": c["verdict"] == "ok",
    }


def _empty_stats():
    return {
        "generated": 0,            # 生成器一共吐了多少条（原始）
        "dropped_invalid": 0,      # 非字符串 / 空串
        "dropped_duplicate": 0,    # 与已见过的重复（跨轮去重）
        "dropped_degenerate": 0,   # 长度比越界等退化候选
        "dropped_unchanged": 0,    # 跟原文逐字一样的（等于没改，永远收敛不了）
        "dropped_scoring_error": 0,
        "scored": 0,               # 真正打了分的
        "dropped_gaming": 0,       # 被判钻空子
        "dropped_fidelity": 0,     # 保真没过
        "fidelity_warned": 0,      # soft 档下「没真通过但放行」的
        "truncated_candidates": 0,  # 超过单轮上限被截掉的
        "passed_gaming": 0,        # 过了 gaming 闸
        "passed_fidelity": 0,      # 过了保真闸（= 最终 kept；保真不可用时按跳过算通过）
        "kept": 0,
        "picked_reward": None,
        "converged_round": None,
    }


def rewrite(original, generator, max_rounds=DEFAULT_MAX_ROUNDS,
            n_per_round=DEFAULT_N_PER_ROUND, verbose=False, *,
            scorer=None, fidelity_check=None, fidelity_policy="strict",
            soft_fidelity_min=0.5,
            min_len_ratio=MIN_LEN_RATIO, max_len_ratio=MAX_LEN_RATIO):
    """跑完整的「指导-生成-打分-修改」循环。

    参数
        original       str   原文（AI 写的评论）
        generator            Generator 协议实现；本模块不内置任何 API 调用
        max_rounds     int   最多几轮（>=1）
        n_per_round    int   每轮要几个候选（>=1）
        verbose        bool  打印每轮候选与分数
        scorer               复用已加载的 Scorer（省 2 秒建模）；None 则用单例
        fidelity_check       显式注入保真检查函数（测试用）；None 则尝试
                             import pipeline.fidelity
        fidelity_policy      "strict"（默认，不保真就过滤）| "soft"（见模块头注释）
        soft_fidelity_min    soft 档下允许放行的最低保真分
        min_len_ratio / max_len_ratio  退化闸门的长度比上下限

    返回
        {
          "original": str,
          "best": str | None,        # 选中的改写；converged=False 时是「尽力而为」
          "converged": bool,         # 是否达到「verdict==ok 且保真通过」
          "rounds": int,             # 实际跑了几轮
          "history": [ {round, instruction, base, cleared, candidates, picked,
                        converged} ],
          "final": dict | None,      # 选中候选的完整打分信息（含 after describe）
          "fidelity": dict | None,   # 选中候选的保真结果
          "original_score": dict,    # 原文的 describe 结果
          "stats": {...}, "stop_reason": str, "notes": [str],
          "fidelity_available": bool,
        }

    history[].candidates[] 每条：
        {text, status: kept|rejected|scored, reject_reason, reward, d_reward,
         d_held_human, verdict, gaming, fidelity, fidelity_warn, len_ratio,
         similarity, after(describe 结果), problems}

    注意：`best` 在 converged=False 时也可能非空（尽力而为的最高分者），
    调用方**必须**看 converged 再决定能不能用。
    """
    text = "" if original is None else str(original)
    max_rounds = max(1, int(max_rounds))
    n_per_round = max(1, int(n_per_round))
    if fidelity_policy not in ("strict", "soft"):
        raise ValueError(f"fidelity_policy 只能是 strict / soft，收到 {fidelity_policy!r}")

    stats = _empty_stats()
    rec = {
        "original": text, "best": None, "converged": False, "rounds": 0,
        "history": [], "final": None, "fidelity": None, "original_score": None,
        "stats": stats, "stop_reason": None, "notes": [],
        "fidelity_available": False,
    }

    # ---------------- 空输入：直接退出，不进循环 ----------------
    if not text.strip():
        rec["stop_reason"] = "empty_input"
        rec["notes"].append("原文为空（或只有空白），没有可改写的内容，未进入循环")
        _log(verbose, "[跳过] 原文为空，未进入改写循环")
        return rec

    sc = scorer or get_scorer()

    # ---------------- 指导：先给原文打分 ----------------
    orig_score = sc.describe(text)
    rec["original_score"] = orig_score
    problems0 = describe_problems(orig_score)
    _log(verbose, f"[原文] len={len(text)}  奖励={orig_score['reward']:.3f}  "
                  f"验收(用词)={orig_score['heldout_ai']:.3f}")
    if problems0:
        _log(verbose, "  要修的痕迹: " +
             " / ".join(f"{p['name']}(+{p['weight']})" for p in problems0))
    else:
        _log(verbose, "  要修的痕迹: 无（原文没命中硬性句式规则）")

    # ---------------- 保真模块：解析 + 一次性自检降级 ----------------
    check, why = _resolve_fidelity(fidelity_check)
    if check is not None:
        usable, probe_why = _probe_fidelity(check, text)
        if not usable:
            check = None
            why = probe_why
    rec["fidelity_available"] = check is not None
    if check is None:
        rec["notes"].append(f"保真检查已跳过：{why}")
        _log(verbose, f"[保真] 跳过 —— {why}")
    else:
        rec["notes"].append("保真检查可用，已启用")
        _log(verbose, "[保真] 可用（恒等自检通过）")

    seen = set()                      # 跨轮去重（归一化文本）
    base = None                       # 上一轮最好的候选记录（dict）
    base_problem_names = _problem_names(problems0)
    prev_failure = None
    stalled = 0                       # 连续几轮一条问题都没消掉

    # ---------------- 循环 ----------------
    for k in range(1, max_rounds + 1):
        # 本轮指令：第 1 轮看原文痕迹；之后只看「上一轮剩下的」
        if k == 1:
            instruction = build_instruction(1, text, described=orig_score,
                                            problems=problems0)
            cleared = []
        else:
            cur_described = base["after"] if base else orig_score
            current = describe_problems(cur_described)
            cur_names = _problem_names(current)
            cleared = [n for n in base_problem_names if n not in cur_names]
            instruction = build_instruction(
                k, text, described=cur_described, base_text=base,
                problems=current, cleared=cleared, failure=prev_failure,
                stalled_rounds=stalled)

        _log(verbose, f"\n[第 {k} 轮] 指令 {len(instruction)} 字，请求 {n_per_round} 个候选")
        if verbose and k > 1 and cleared:
            _log(verbose, "  已改掉: " + "、".join(cleared))

        # ---- 生成 ----
        try:
            raw = generator(text, instruction, n_per_round)
        except Exception as e:
            rec["history"].append({
                "round": k, "instruction": instruction,
                "base": _picked_view(base), "cleared": cleared,
                "candidates": [], "picked": None, "converged": False,
                "error": f"{type(e).__name__}: {e}",
            })
            rec["rounds"] = k
            rec["stop_reason"] = "generator_error"
            rec["notes"].append(f"第 {k} 轮生成器抛异常：{type(e).__name__}: {e}")
            _log(verbose, f"  [错误] 生成器抛异常：{type(e).__name__}: {e}")
            break

        if isinstance(raw, str):
            raw = [raw]
        elif raw is None:
            raw = []
        elif not isinstance(raw, (list, tuple)):
            try:
                raw = list(raw)                 # 允许生成器返回迭代器
            except TypeError:
                raw = []
        raw = list(raw)
        if len(raw) > MAX_CANDIDATES_PER_ROUND:
            stats["truncated_candidates"] += len(raw) - MAX_CANDIDATES_PER_ROUND
            rec["notes"].append(
                f"第 {k} 轮生成器返回 {len(raw)} 条，只保留前 "
                f"{MAX_CANDIDATES_PER_ROUND} 条（单轮上限）")
            raw = raw[:MAX_CANDIDATES_PER_ROUND]
        stats["generated"] += len(raw)

        # ---- 候选清洗（非字符串 / 空 / 重复 / 退化） ----
        # 所有候选记录都用**同一套键**，没打分的字段填 None：
        # 下游（报表 / 人工检查）直接 c["reward"] 不会 KeyError。
        def _new_cand(txt, status, reason=None):
            return {
                "text": txt, "status": status, "reject_reason": reason,
                "reward": None, "d_reward": None, "d_held_human": None,
                "verdict": None, "gaming": False, "fidelity": None,
                "fidelity_warn": False,
                "len_ratio": round(len(txt) / max(1, len(text)), 3),
                "similarity": None, "after": None, "problems": [],
            }

        cands = []
        for t in raw:
            if not isinstance(t, str):
                stats["dropped_invalid"] += 1
                continue
            nt = t.strip()
            if not nt:
                stats["dropped_invalid"] += 1
                continue
            key = _norm_key(nt)
            if key in seen:
                stats["dropped_duplicate"] += 1
                continue
            seen.add(key)
            if key == _norm_key(text):
                # 和原文一字不差（连空白差异都抹平后相同）→ 等于没改。
                # 它的 d_reward 必然是 0，永远收敛不了，也不该被当成 best。
                stats["dropped_unchanged"] += 1
                cands.append(_new_cand(nt, "rejected", "unchanged"))
                continue
            bad = _degenerate_reason(nt, text, min_len_ratio, max_len_ratio)
            if bad:
                stats["dropped_degenerate"] += 1
                cands.append(_new_cand(nt, "rejected", f"degenerate:{bad}"))
                continue
            cands.append(_new_cand(nt, "pending"))

        # ---- 打分 + 保真 + 两道闸 ----
        for c in cands:
            if c["status"] != "pending":
                continue
            try:
                cmp = sc.compare(text, c["text"])
            except Exception as e:
                stats["dropped_scoring_error"] += 1
                c.update({"status": "rejected",
                          "reject_reason": f"scoring_error:{type(e).__name__}"})
                continue
            after = cmp["after"]
            stats["scored"] += 1
            c.update({
                "status": "scored",
                "reward": after["reward"],
                "d_reward": cmp["d_reward"],
                "d_held_human": cmp["d_held_human"],
                "verdict": cmp["verdict"],
                "gaming": bool(cmp["gaming_suspect"]),
                "len_ratio": round(len(c["text"]) / max(1, len(text)), 3),
                "similarity": _similarity(text, c["text"]),
                "after": after,
                "problems": describe_problems(after),
                "fidelity": _check_fidelity(check, text, c["text"], why),
            })
            if c["verdict"] == "gaming":
                stats["dropped_gaming"] += 1
                c.update({"status": "rejected", "reject_reason": "gaming"})
                continue
            stats["passed_gaming"] += 1
            fstat = c["fidelity"]["status"]
            fscore = c["fidelity"].get("score")
            if fstat in ("fail", "error"):
                # soft 档：只有「真判了不通过（fail）且分数够高」才放行并打警示；
                # error 一律不放行 —— 没法验证的东西不能当成验证过。
                soft_pass = (fidelity_policy == "soft" and fstat == "fail"
                             and isinstance(fscore, (int, float))
                             and fscore >= soft_fidelity_min)
                if not soft_pass:
                    stats["dropped_fidelity"] += 1
                    c.update({"status": "rejected",
                              "reject_reason": f"fidelity:{fstat}"})
                    continue
                c["fidelity_warn"] = True
                stats["fidelity_warned"] += 1
            stats["passed_fidelity"] += 1
            c["status"] = "kept"

        # ---- 选优：kept 按 reward 降序（reward 是绝对分，跨轮可比） ----
        kept = [c for c in cands if c["status"] == "kept"]
        kept.sort(key=lambda c: (-c["reward"], -(c["fidelity"].get("score") or 0.0),
                                 -c["d_held_human"], len(c["text"])))
        stats["kept"] += len(kept)
        ok = [c for c in kept if c["verdict"] == "ok"]
        # 保真真正通过的优先（soft 档里被放行的 warned 候选只在没有干净候选时才上）
        strict_ok = [c for c in ok if not c.get("fidelity_warn")]
        picked = (strict_ok or ok or kept or [None])[0]
        converged = bool(picked) and picked["verdict"] == "ok"

        rec["history"].append({
            "round": k, "instruction": instruction, "base": _picked_view(base),
            "cleared": cleared, "candidates": cands,
            "picked": _picked_view(picked), "converged": converged,
            "stalled_rounds": stalled,
        })
        rec["rounds"] = k

        # 这一轮有没有消掉问题（供下一轮判断要不要加码）；第 1 轮不算失败
        if k > 1:
            stalled = 0 if cleared else stalled + 1

        _print_round(verbose, k, cands, picked, converged)

        # ---- 记录「目前最好」（即使没收敛也留着，最后作尽力而为结果） ----
        # 收敛必须压过「分更高但没达标」的候选：bar 是 ok+保真，不是纯 reward
        if picked and (converged or rec["final"] is None
                       or picked["reward"] > rec["final"]["reward"]):
            rec["final"] = picked

        if converged:
            stats["converged_round"] = k
            rec["stop_reason"] = "converged"
            break

        # ---- 下一轮的 base / 失败原因 ----
        if picked:
            prev_failure = {
                "verdict": picked["verdict"], "d_reward": picked["d_reward"],
                "d_held_human": picked["d_held_human"], "reward": picked["reward"],
            }
            base = picked
            base_problem_names = _problem_names(picked["problems"])
        else:
            # 一条都没留下：把拒绝原因的分布带上，下一轮指令才能说出真正的病根
            # （「全部被判钻空子」和「全部不保真」要给出完全不同的改法）
            tally = {"gaming": 0, "fidelity": 0, "degenerate": 0, "other": 0}
            for c in cands:
                rr = c.get("reject_reason") or ""
                if rr == "gaming":
                    tally["gaming"] += 1
                elif rr.startswith("fidelity"):
                    tally["fidelity"] += 1
                elif rr.startswith("degenerate"):
                    tally["degenerate"] += 1
                else:
                    tally["other"] += 1
            prev_failure = {"verdict": None, "d_reward": None, "d_held_human": None,
                            "n_rejected": len(cands), "reject_tally": tally}
            # base 保持不变（可能仍是 None，指令就用原文当依据）

        # 本轮一条新候选都没有 → 生成器根本没响应指令，再来也不会有新东西
        if not cands:
            rec["stop_reason"] = "stagnant"
            rec["notes"].append(f"第 {k} 轮没有任何新候选（全是重复/空），提前结束")
            _log(verbose, "  [提前结束] 本轮没有新候选，生成器没有响应指令")
            break
    else:
        if rec["stop_reason"] is None:
            rec["stop_reason"] = "max_rounds"

    # ---------------- 收尾 ----------------
    best = rec["final"]
    rec["best"] = best["text"] if best else None
    rec["converged"] = stats["converged_round"] is not None
    rec["fidelity"] = best.get("fidelity") if best else None
    stats["picked_reward"] = best["reward"] if best else None
    if rec["stop_reason"] is None:
        rec["stop_reason"] = "max_rounds"
    if best is None:
        rec["notes"].append("没有任何候选通过过滤，best 为 None")
    elif not rec["converged"]:
        rec["notes"].append(
            f"未收敛，best 是尽力而为的最高分候选（reward {best['reward']:.3f}，"
            f"{best['verdict']}）")
    if best is not None and best.get("fidelity_warn"):
        fs = (best.get("fidelity") or {}).get("score")
        rec["notes"].append(
            f"best 的保真是 soft 档放行的（score {fs}，未真正通过），不能当合格产物")
    return rec


def _print_round(verbose, k, cands, picked, converged):
    if not verbose:
        return
    for i, c in enumerate(cands, 1):
        if c["status"] == "kept":
            tag = "⚠" if c.get("fidelity_warn") else "✓"
            fs = (c.get("fidelity") or {}).get("status")
            detail = (f"reward={c['reward']:.3f}  Δ={c['d_reward']:+.3f}  "
                      f"用词={c['d_held_human']:+.3f}  {c['verdict']}  保真={fs}")
            if c.get("fidelity_warn"):
                detail += "（soft 放行）"
        elif c.get("reward") is not None:
            tag = "×"
            fr = (c.get("fidelity") or {}).get("status") or "-"
            detail = (f"reward={c['reward']:.3f}  Δ={c['d_reward']:+.3f}  "
                      f"用词={c['d_held_human']:+.3f}  {c['verdict']}  "
                      f"→ 过滤：{c['reject_reason']}（保真={fr}）")
        else:
            tag = "×"
            detail = f"过滤：{c.get('reject_reason') or 'unknown'}"
        preview = c["text"].replace("\n", " ")
        if len(preview) > 46:
            preview = preview[:46] + "…"
        print(f"  {tag} #{i} {detail}")
        print(f"       {preview}")
    if converged:
        idx = next((i for i, c in enumerate(cands, 1) if c is picked), "?")
        print(f"  → 收敛：#{idx} 达标，收工")
    elif picked:
        idx = next((i for i, c in enumerate(cands, 1) if c is picked), "?")
        print(f"  → 未达标，本轮最好 #{idx}（{picked['verdict']}）继续下一轮")
    else:
        print("  → 本轮没有可用候选")


# ============================================================================
# 命令行
# ============================================================================
def _load_state_texts(path, limit):
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return []
    items = data.get("items") if isinstance(data, dict) else data
    out = []
    for it in (items or [])[:limit]:
        if isinstance(it, dict) and it.get("src"):
            out.append(it["src"])
    return out


def main():
    ap = argparse.ArgumentParser(
        description="改写 workflow 主干（EchoGenerator 冒烟版，不调用任何网络 API）")
    ap.add_argument("--text", help="要改写的文本；不给则读 --state 里的 src")
    ap.add_argument("--state", default=str(ROOT / "state" / "rewrite_abc.json"),
                    help="测试数据（格式 {items:[{id,src,A,B,C}]}）")
    ap.add_argument("--limit", type=int, default=5, help="读测试数据时最多几条")
    ap.add_argument("--rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    ap.add_argument("--n", type=int, default=DEFAULT_N_PER_ROUND)
    ap.add_argument("--fidelity-policy", choices=["strict", "soft"], default="strict",
                    help="strict=不保真就过滤（默认）；soft=高分放行但打警示")
    ap.add_argument("--quiet", action="store_true", help="不打印每轮细节")
    a = ap.parse_args()

    texts = [a.text] if a.text else _load_state_texts(a.state, a.limit)
    if not texts:
        print("没有可跑的文本：给 --text，或确认 --state 文件存在")
        return 1

    sc = get_scorer()
    print(f"[模型] 训练 {sc.n_train} 条（AI {sc.n_pos}）  奖励 {sc.n_feat} 维手工特征")
    print(f"[生成器] EchoGenerator（只做字符串变换，用于验证流程，不代表改写质量）")
    print(f"[参数] max_rounds={a.rounds}  n_per_round={a.n}  "
          f"保真档={a.fidelity_policy}")

    gen = EchoGenerator()
    for i, t in enumerate(texts, 1):
        print("\n" + "=" * 72)
        print(f"【{i}/{len(texts)}】{t[:60]}{'…' if len(t) > 60 else ''}")
        print("=" * 72)
        r = rewrite(t, gen, max_rounds=a.rounds, n_per_round=a.n,
                    verbose=not a.quiet, scorer=sc,
                    fidelity_policy=a.fidelity_policy)
        st = r["stats"]
        print(f"\n[结果] 收敛={r['converged']}  轮数={r['rounds']}  "
              f"退出原因={r['stop_reason']}")
        print(f"       生成 {st['generated']}  打分 {st['scored']}  "
              f"过游戏闸 {st['passed_gaming']}  过保真闸 {st['passed_fidelity']}  "
              f"作弊过滤 {st['dropped_gaming']}  退化 {st['dropped_degenerate']}")
        if r["best"]:
            print(f"       最佳(reward {st['picked_reward']:.3f}): {r['best']}")
        else:
            print("       最佳: None（没有候选通过过滤）")
        print(f"       保真可用: {r['fidelity_available']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
