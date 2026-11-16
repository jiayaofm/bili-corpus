# -*- coding: utf-8 -*-
"""
特征定义 —— 全项目唯一来源

分类和聚类都从这里取特征。改一处，全局生效。
此前 run_v2.py / run_v3.py 各写一份，已经出现不一致，这是最大的技术债。

特征分组：
  A 符号类   —— 标点符号层面的 AI 痕迹（最高精度）
  B 句式类   —— 论证结构的固定搭配
  C 人设类   —— 第一人称角色扮演、助手视角
  D 结构类   —— 长度、行数、用字丰富度
  E 账号类   —— 账号名（已确认是混淆变量）
  F 标点类   —— 标点计数
  G 其他     —— 表情、@、链接

STYLE_COLS 是给聚类用的连续型风格特征（不含任何话题词）。
"""
import re

EMOTE_RE = re.compile(r"\[[^\[\]]{1,12}\]")
URL_RE = re.compile(r"https?://\S+")
BV_RE = re.compile(r"BV[0-9A-Za-z]{10}")
MENTION_RE = re.compile(r"@[^\s@]{1,20}")
SPLIT_RE = re.compile(r"[，。！？；\n]+")
DIGIT_RE = re.compile(r"\d")


def _segments(c):
    return [s for s in SPLIT_RE.split(c) if s.strip()]


# ---------------------------------------------------------------- 单项特征
def _f_corner(c, u):
    return int("「" in c or "」" in c)


def _f_dash(c, u):
    return int("——" in c)


def _f_semicolon(c, u):
    return int("；" in c)


def _f_colon(c, u):
    return int("：" in c or ":" in c)


def _f_md(c, u):
    return int("**" in c)


def _f_not_a_but_b(c, u):
    return int(bool(re.search(
        r"不是[^，。；\n]{1,18}[，,]?\s*(?:而)?是[^，。；\n]{1,18}", c)))


def _f_is_true(c, u):
    return int(bool(re.search(r"(?:是|也是)真的[，。]", c)))


def _f_compare(c, u):
    return int(bool(re.search(r"比[^，。\n]{2,13}(?:更|都)", c)))


def _f_this_one(c, u):
    return int("这条" in c)


def _f_another(c, u):
    return int(bool(re.search(r"另一(?:回事|栏|种|层)", c)))


def _f_essence(c, u):
    return int(bool(re.search(r"(?:本质|内核|核心|结构)[^，。]{0,6}(?:是|在于|其实是)", c)))


def _f_self_fish(c, u):
    return int(bool(re.search(r"本鱼|本鲸|本喵", c)))


def _f_agent_view(c, u):
    return int(bool(re.search(
        r"我这个型号|派活|停机|让我干|我干活|跑一轮题库|读字幕", c)))


def _f_fish_first_person(c, u):
    return int(bool(re.search(
        r"(?<!大)(?<!小)鱼(?:知道|还是|也|就|熟|可|会|不)|让鱼|这只鱼|是真鱼", c)))


def _f_role_self(c, u):
    return int(bool(re.search(r"本小姐|本座|朕|咱鱼", c)))


def _f_self_project(c, u):
    return int(bool(re.search(r"这不就是我|就是我嘛|代入|我也是", c)))


def _f_colloquial(c, u):
    return int(bool(re.search(
        r"(?:挺|真)(?:香|好|猛|肉疼|好笑|乐|顶|伤|难|贵|划算|值)", c)))


def _f_inverted_parallel(c, u):
    return int(bool(re.search(r"是[^，。\n]{1,12}，不是[^，。\n]{1,12}", c)))


def _f_rhetorical(c, u):
    return int(bool(re.search(r"有没有人[^。\n]{0,20}而不是|而不是[^。\n]{0,12}$", c)))


def _f_i_notice(c, u):
    return int(bool(re.search(r"我最在意|我最关心|我好奇", c)))


def _f_share(c, u):
    return int(bool(re.search(r"有点东西|来瞅瞅|特意叫|值得刷|安利", c))
               or len(MENTION_RE.findall(c)) >= 2)


def _f_len(c, u):
    return len(c)


def _f_lines(c, u):
    return c.count("\n") + 1


def _f_richness(c, u):
    return len(set(c)) / max(1, len(c))


def _f_name_fish(c, u):
    return int(bool(re.search(r"鱼|鲸", u or "")))


def _f_name_len(c, u):
    return len(u or "")


def _f_comma(c, u):
    return c.count("，")


def _f_period(c, u):
    return c.count("。")


def _f_exclaim(c, u):
    return c.count("！")


def _f_question(c, u):
    return c.count("？")


def _f_dun(c, u):
    return c.count("、")


def _f_emote(c, u):
    return int(bool(EMOTE_RE.search(c)))


def _f_mention(c, u):
    return len(MENTION_RE.findall(c))


def _f_bv(c, u):
    return int(bool(BV_RE.search(c)))


def _f_url(c, u):
    return int(bool(URL_RE.search(c)))


# ---------------- H 闭合类 ----------------
# 用户洞察：「大肥鱼把整句话都说出来」，真人的幽默来自「说半句留白」。
# 实测（157 条人工真值，同分区内部仍成立）：
#   句末闭合   AI 43% vs 真人 10%
#   短分句占比 AI 4%  vs 真人 12%
#   分句数     AI 4.65 vs 真人 3.50
FILLER_RE = re.compile(r"[啊吧咯哈呢呀嘛呗哦嘿唉哎呐噢哇]")
ASK_RE = re.compile(r"(?:大佬|求|有没有|请问|谁能|怎么|为什么|吗)")


def _f_filler_density(c, u):
    """语气词密度（每百字）——真人说话带尾巴，AI 不带"""
    return len(FILLER_RE.findall(c)) / max(len(c), 1) * 100


def _f_ask_density(c, u):
    """提问/求教密度（每百字）——真人是来问的，AI 是来答的"""
    return len(ASK_RE.findall(c)) / max(len(c), 1) * 100


# ---------------- I 人味类 ----------------
# 前面 7 组几乎全是「AI 有而人没有」的负向记号，这一组反过来：
# 真人身上常见、AI 几乎不用的正向表达习惯。
# 实测（263 条人工真值，每百字密度）：
#   提问求教  AI 0.1% vs 真人 0.6%   （4.0 倍）
#   语气词    AI 0.2% vs 真人 0.8%   （2.2 倍）
# 注意：另外测过 6 个候选（第一人称、口语词、重复字、自嘲、
# 不规范标点、颜文字），差异都在噪声内，没有加进来。
def _f_closed_end(c, u):
    """句末有终止标点 —— 意思说尽了"""
    return int(bool(re.search(r"[。！？~]\s*$", c)))


def _f_n_seg(c, u):
    return len(_segments(c))


def _f_seg_avg(c, u):
    segs = _segments(c)
    return sum(len(s) for s in segs) / len(segs) if segs else 0.0


def _f_short_seg_ratio(c, u):
    """≤4 字的分句占比 —— 真人「说半句」的密度"""
    segs = _segments(c)
    if not segs:
        return 0.0
    return sum(1 for s in segs if len(s) <= 4) / len(segs)


def _f_digit_density(c, u):
    return len(DIGIT_RE.findall(c)) / max(1, len(c)) * 100


# ---------------------------------------------------------------- 注册表
# (组名, 特征名, 函数)
FEATURES = [
    ("A 符号类", "有「」", _f_corner),
    ("A 符号类", "有破折号", _f_dash),
    ("A 符号类", "有分号", _f_semicolon),
    ("A 符号类", "有冒号", _f_colon),
    ("A 符号类", "有Markdown", _f_md),

    ("B 句式类", "不是A是B", _f_not_a_but_b),
    ("B 句式类", "…是真的", _f_is_true),
    ("B 句式类", "比…更/都", _f_compare),
    ("B 句式类", "指代这条", _f_this_one),
    ("B 句式类", "另一回事", _f_another),
    ("B 句式类", "本质内核腔", _f_essence),

    ("C 人设类", "自称本鱼", _f_self_fish),
    ("C 人设类", "助手视角", _f_agent_view),
    ("C 人设类", "鱼作第一人称", _f_fish_first_person),
    ("C 人设类", "角色自称", _f_role_self),
    ("C 人设类", "自我代入", _f_self_project),
    ("C 人设类", "口语判断", _f_colloquial),
    ("C 人设类", "倒装对仗", _f_inverted_parallel),
    ("C 人设类", "反诘对比", _f_rhetorical),
    ("C 人设类", "我最在意", _f_i_notice),
    ("C 人设类", "分享推荐", _f_share),

    ("D 结构类", "字数", _f_len),
    ("D 结构类", "行数", _f_lines),
    ("D 结构类", "用字丰富度", _f_richness),

    ("E 账号类", "名字含鱼", _f_name_fish),
    ("E 账号类", "名字长度", _f_name_len),

    ("F 标点类", "逗号数", _f_comma),
    ("F 标点类", "句号数", _f_period),
    ("F 标点类", "感叹号数", _f_exclaim),
    ("F 标点类", "问号数", _f_question),
    ("F 标点类", "顿号数", _f_dun),

    ("G 其他", "有表情", _f_emote),
    ("G 其他", "有@", _f_mention),
    ("G 其他", "有BV号", _f_bv),
    ("G 其他", "有链接", _f_url),

    ("H 闭合类", "句末闭合", _f_closed_end),
    ("H 闭合类", "分句数", _f_n_seg),
    ("H 闭合类", "平均分句长", _f_seg_avg),
    ("H 闭合类", "短分句占比", _f_short_seg_ratio),
    ("H 闭合类", "数字密度", _f_digit_density),

    ("I 人味类", "语气词密度", _f_filler_density),
    ("I 人味类", "提问密度", _f_ask_density),
]

NAMES = [n for _, n, _ in FEATURES]
GROUPS = {}
for g, n, _ in FEATURES:
    GROUPS.setdefault(g, []).append(n)

# 分类用：默认去掉账号类（混淆变量），需要时单独打开
CLASSIFY_GROUPS = ["A 符号类", "B 句式类", "C 人设类", "D 结构类",
                   "F 标点类", "G 其他", "H 闭合类", "I 人味类"]

# 聚类用：连续型风格特征，不含任何话题词
STYLE_COLS = [
    "字数", "逗号数", "句号数", "感叹号数", "问号数", "顿号数",
    "行数", "用字丰富度", "有表情", "有「」", "有破折号", "有分号",
    "有@", "有链接", "有BV号",
]


def featurize(content, username=""):
    """返回 {特征名: 数值}"""
    return {name: fn(content, username) for _, name, fn in FEATURES}


def vector(rec, cols=None):
    """从评论记录取特征向量"""
    d = featurize(rec.get("content", ""), rec.get("user", ""))
    cols = cols or NAMES
    return [d[c] for c in cols]


def cols_of(groups):
    out = []
    for g in groups:
        out.extend(GROUPS.get(g, []))
    return out
