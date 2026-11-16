# -*- coding: utf-8 -*-
"""
保真检查（fidelity）—— 改写有没有丢信息 / 改事实

scorer.py 回答「像不像真人」，本模块回答「意思有没有变」。
两者正交：一条改写可以既很人味，又把 37% 写成 30%。

约束
    · 确定性：纯规则 + 字符统计，不调模型、不用随机数、不读外部资源
    · 可解释：每个扣分项都指名「丢了哪一条」，能直接喂回改写环节
    · 关键项从严：数字 / 实体 / 否定 只要缺一个就 fail，其余按加权扣分

五个维度（权重见 W）
    1 数字    金额 / 百分比 / 时间 / 次数，中文数字与阿拉伯数字互转
    2 实体    英文词、版本号、BV 号、@提及、链接
    3 否定    「不 没 别 无 非 未」不能丢，也不能凭空加
    4 指代    第三人称 / 它 不能被换掉或抹掉
    5 信息量  分句级「要点覆盖率」，不用字数比

外加一个「多做出来的东西」的判据（不在加权里，单独判死）
    6 捏造事实 改写凭空加出原文没有的可验证信息（具体处所 / 比较对象 /
      另一个信息源 / 新数字或实体）→ 判死；而语气词、情绪、称呼、感叹
      （「吧」「哈哈」「扎心」「老铁」）一律放行——去 AI 味本来就要加这些。

为什么不用字数比
    「94分挺猛 订阅剩37% 五天重置 算了 max先关」只有原文一半长，
    但数字全在、要点全在，是合法的口语压缩；反过来
    「订阅还有一周清零吧」等长且通顺，却把 37% 和五天都换掉了。
    字数比在这两种情形上判断完全相反，所以按要点覆盖算。

已知查不出的问题（写在 check() 返回的 notes 之外，见文件末尾 __doc_limits__）
"""
import re
import unicodedata
from collections import Counter

# ---------------------------------------------------------------- 阈值
# 0.55 的来历：中文同义改写（浪漫化成真死亡 → 说成真死）通常能保住 55%~70%
# 的字符，按 0.60 卡会把等价改写误报成要点丢失（喂回改写环节会误导补写）；
# 而 0.55 仍高于「丢一半」，所以「善终」0/2、「用户要有边界」2/4 这类
# 真丢失照样被抓（2/4 = 0.5 < 0.55）。
COV_CLAUSE = 0.55

# 要点覆盖率下限（硬门槛）。允许丢三成是刻意的：去 AI 味本来就要砍总结腔、
# 砍对仗，砍掉的通常是虚的；但超过三成就砍到有内容的支撑句了。
# 实测（state/rewrite_abc.json 15 条 + 18 条反例/正例）：
#   保真改写 1.00 / 0.80 / 0.77     ← 0.77 那条是 2A，另有 3 个要点被点名
#   明显失真 0.67 / 0.31 / 0.15 / 0.08
# 0.70 落在 0.77 与 0.67 之间的平台上，不是贴着某一侧挑出来的。
COV_FLOOR = 0.35

# 覆盖率只在原文够长时才看。短文本要点太少，换一种说法就会归零。
# 实测：「离重置还有五天」→「还要撑五天」40 字以下会得 0.00，
# 但这是完全合法的同义改写。40 字以上才有多条要点，覆盖率才有意义。
COV_MIN_LEN = 40

# 否定保留度下限。
# 0→1 个否定（合法改述）= 0.5，判过；1→0 个否定（丢否定）= 0.0，判死。
# 所以这个值要卡在 0.5 上，不能更高，否则挡住「max 档先关」→「用不起了」这类改写。
NEG_MIN = 0.5

# 综合分下限：只做兜底，真正负责否决的是关键项与覆盖率两个硬门槛。
# 实测通过的最低 0.945、不通过的最高 0.644，0.78 落在这段空档里。
# 注意 score 高 ≠ ok：比如「给原文加了 12 万」score 0.85 但 ok=False。
# 综合分下限：只做兜底。
# 原值 0.78，实测把同义改写全挡在外面（覆盖率归零会拉低总分到 0.76），
# 而真正负责否决的是数字/实体/否定三个硬项，所以放宽到 0.70。
# 坏例仍然过不了：删一段 0.644、删数字 0.576、换主题 0.240。
SCORE_MIN = 0.70

# 维度权重：数字和实体是硬事实，改一个就是改事实（合计 0.52）；
# 否定 0.16（极性翻转等同于说反话）；覆盖率 0.24 管信息量；
# 指代 0.08 最低，因为口语改写合法地省略代词，靠名词仍在文本里兜底。
W = {"number": 0.30, "entity": 0.22, "negation": 0.16,
     "referent": 0.08, "coverage": 0.24}

# ---------------------------------------------------------------- 正则
URL_RE = re.compile(r"https?://\S+|www\.\S+")
BV_RE = re.compile(r"BV[0-9A-Za-z]{10}")
MENTION_RE = re.compile(r"@[^\s@]{1,20}")
EMOTE_RE = re.compile(r"\[[^\[\]]{1,12}\]")
LATIN_RE = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[._-][A-Za-z0-9]+)*")

# 计量单位：长的排前面，否则「分」会抢先匹配掉「分钟」
UNITS = ["分钟", "小时", "美元", "公里", "万亿", "GB", "MB", "kg",
         "分", "倍", "天", "秒", "年", "月", "日", "刀", "块", "元", "粒",
         "道", "页", "岁", "周", "次", "条", "度", "级", "米", "克", "斤",
         "人", "位", "名", "行", "字", "万", "亿", "%"]
_UNIT_ALT = "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))

# 强单位：值一样但单位变了 = 改事实（「20 分钟」→「20 秒」）。
# 万 / 亿 是量级倍率不是单位，单列；人 / 位 / 名 / 行 / 字 太虚，不查。
STRONG_UNITS = {"分钟", "小时", "秒", "天", "年", "月", "日", "周", "岁",
                "分", "倍", "刀", "块", "元", "美元", "粒", "道", "页",
                "%", "公里", "米", "度", "级"}
UNIT_SCALE = {"万": 1e4, "亿": 1e8, "万亿": 1e12}

_AR_NUM = re.compile(
    r"(?<![\d.])(\d+(?:\.\d+)?)(?:\s*(多|几|余))?\s*(" + _UNIT_ALT + r")?")
_CN_CORE = "零一二三四五六七八九十百千两"
_CN_NUM = re.compile(
    r"(?<![零一二三四五六七八九十百千两万亿])"
    r"(?:([几数])([万亿]{1,2})|([" + _CN_CORE + r"]{1,6})([万亿]{0,2}))"
    r"(多|余)?"
    r"(" + _UNIT_ALT + r")?")
_PCT_CN = re.compile(
    r"百分之([" + _CN_CORE + r"]{1,6})|百分之(\d+(?:\.\d+)?)")

CN_DIGIT = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
            "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
CN_UNIT = {"十": 10, "百": 100, "千": 1000}

# 看着像数字其实不是：「十分」「一定」这类固定词，不参与数字比对
CN_BLOCK = {"十分", "一点", "一些", "一起", "一样", "一直", "一旦", "一定",
            "一般", "一边", "一致", "一再", "暂时", "一律", "一瞬", "一番",
            "一批", "一批", "一共", "一口气", "一会儿", "一点儿", "万万",
            "千万", "万分", "十有", "一味", "一贯", "一览"}

# 否定词附近先抹掉「假否定」：这些词里的 不/没/别/无/非/未 不表否定
NEG_FALSE = ["非常", "未来", "不但", "不仅", "不过", "不断", "不同", "不少",
             "不管", "不论", "不如", "不久", "不错", "不好意思", "忍不住",
             "不得了", "不住", "不由", "不知不觉", "无所谓", "无论", "无数",
             "无比", "无疑", "无奈", "特别", "分别", "区别", "告别", "类别",
             "级别", "个别", "识别", "别人", "别的", "别处", "别样", "别致",
             "别名", "别称", "莫名", "莫非", "非得", "不客气", "不菲", "不利",
             "不良", "不安", "不禁", "不由得", "不免", "不失", "不光", "不只"]
NEG_CHAR = re.compile(r"[不没别无非未甭勿莫]")

# 指代类别：只区分视角和指称对象，不区分性别。
# 「泛指他人」（人家/别人/大家）与「第三人称」分开，否则改写用「人家」
# 会被误判成把「它」换成了「他」。
PRON = {
    "第一人称": ["我们", "咱们", "我", "咱", "俺"],
    "第二人称": ["你们", "您", "你"],
    "第三人称-人": ["他们", "她们", "他", "她"],
    "第三人称-物": ["它们", "它"],
    "泛指他人": ["人家", "别人", "大家"],
}
# 只有「人物指代被抹掉」才扣分；一/二人称在中文里经常合法省略
PRON_HARD = ("第三人称-人", "第三人称-物")

# 英文功能词不当作实体：英文句子里 "is/the/now" 被改掉不算信息丢失。
# 注意不能把 ai / ds / max 这类放进这个表，它们是真正的内容词。
EN_STOP = {
    "is", "it", "its", "the", "an", "and", "or", "of", "to", "in", "on", "at",
    "as", "by", "for", "with", "be", "am", "are", "was", "were", "do", "does",
    "did", "have", "has", "had", "my", "me", "we", "us", "you", "your", "he",
    "she", "him", "her", "they", "them", "this", "that", "these", "those",
    "but", "so", "if", "then", "than", "now", "just", "not", "no", "ok", "oh",
    "wow", "yeah", "lol", "haha", "very", "really", "too", "also", "can",
    "could", "will", "would", "should", "may", "might", "must", "im", "ive",
}

# ---------------------------------------------------------------- 捏造事实
# 改写合法地会加语气词、情绪、称呼（「吧」「哈哈」「扎心」「老铁」），这些
# 不改变可验证信息，必须放行；但如果新增内容里出现了「具体处所 / 比较对象 /
# 另一个信息源」，那就是凭空捏造了一个可以核查的事实，必须判死。
# 打分（阈值 FAB_MIN=3）：
#     处所      +3   食堂/公司/地铁… 或「在 X 店 / 在 X 室」
#     比较对象  +3   「比某家」「跟大厂比」「某平台」——凭空造出比较对象
#     第三方    +3   「我同事说」「朋友推荐」——凭空造出另一个信息源
#     时间      +1   今天/昨天/上周…（单独不判死：口语改写常加「今天」）
#     人称      +1   我/你/他/我们…
#     实义动作  +1   去/买/跑/测…（情绪动词不算：笑死/扎心/破防只算情绪）
# 也就是说：单靠「时间」或「人称」不判死，必须叠出处所/比较/第三方其一，
# 或者时间+人称+动作三项同时到齐（「我昨天也去考了一把」）。
FAB_MIN = 3
FAB_NOVEL_RATIO = 0.35     # 分句与原文的要点重合度低于此值才算「新增」

PLACE_RE = re.compile("|".join([
    "食堂", "公司", "学校", "教室", "寝室", "宿舍", "图书馆", "实验室",
    "办公室", "工位", "会议室", "地铁", "公交车", "公交", "车上", "路上",
    "街上", "店里", "超市", "便利店", "医院", "厕所", "厨房", "客厅",
    "卧室", "阳台", "公园", "网吧", "酒吧", "餐厅", "饭店", "咖啡馆",
    "商场", "电影院", "健身房", "家里", "老家", "楼下", "楼上", "门口",
    "柜台", "前台", "机场", "车站", "火车站", "高铁", "食堂",
]))
TIME_RE = re.compile("|".join(sorted([
    "今天", "昨天", "明天", "前天", "后天", "今晚", "昨晚", "今早", "明早",
    "刚才", "刚刚", "上周", "下周", "上个月", "下个月", "去年", "今年",
    "明年", "前年", "小时候", "半夜", "凌晨", "早上", "中午", "下午",
    "晚上", "周末", "过年", "那会儿", "大学时候", "高中", "假期",
], key=len, reverse=True)))
# 「比某家」「跟大厂比」「和友商比」—— 原文没有的比较关系
COMPARE_RE = re.compile(
    r"(?:比|跟|和|不如|比不过|胜过|超过)"
    r"(?:某|别人|人家|其他|其它|隔壁|大厂|友商|官方|同行|别家|他家|哪家)")
# 「某家/某平台/某app」—— 指向一个没说出口的具体对象
VAGUE_REF_RE = re.compile(
    r"某(?:个)?(?:家|公司|平台|店|人|app|APP|网站|牌子|厂商|友商|大厂|up|UP)")
# 「我同事说」「朋友推荐」—— 凭空造出信息源（老铁/兄弟/家人们是称呼，不算）
THIRD_PARTY_RE = re.compile(
    r"(?:我|俺|咱)?(?:朋友|同学|同事|室友|舍友|闺蜜|表哥|表姐|表弟|表妹|"
    r"对象|老婆|老公|男友|女友|邻居|房东|领导|老板|客户|家里人)")
PERSON_RE = re.compile(r"我们|你们|他们|她们|它们|我|俺|咱|你|您|他|她|它")
# 实义动作里刻意不收「笑/哭/乐/服/崩/裂/麻」——它们是情绪，不是事实
ACTION_RE = re.compile(
    r"(?:去|来|买|卖|玩|试|测|跑|走|干|做|说|讲|问|找|骂|打|换|修|拆|"
    r"搬|送|拿|带|抢|排|坐|站|躺|睡|洗|查|翻|刷|订|付|退|借|交|报|"
    r"学|教|写|读|画|唱|跳|充|降|涨|跌|装|蹲|守|偷|藏)")

# 分句内容里剔掉的「功能字」：虚词、代词、语气词、判断词、轻动词。
# 留在里面会让每个分句都被高频字「的/了/是」覆盖，指标失效。
FUNC_CHARS = set(
    "的了着过是在和与也就都还又而很太更最会能要有着把这被给让使个们"
    "您你他她它我咱此其之该每各些么怎谁什因所为所以如果虽然但却只并"
    "才即乃若则于从到上下里中吗呢吧啊呀哦嘛呗哈嗯唉哎嘿嘿哇咦"
    "说看想去来再做打")

# 分句切分（要点覆盖用）：只按标点切，不按空格切——中文原文里
# 「94 分比 89.75 高四分多」的空格是排版，切开就成了「分比」「高四分多」
# 这种碎片，会把正常改写误报成要点丢失。
# 引号也不切：「最好的“死亡”不是突然清空」应当是一个整体。
# 注意 NFKC 会把 ，！？；：（）％ 折成 ASCII，所以中文标点和 ASCII 标点都要列，
# 漏了会让整段被当成一个分句，覆盖率直接失效（我们踩过这个坑）。
CLAUSE_SPLIT = re.compile(
    r"[，。！？；：、,;:!?()（）\[\]【】《》…—·~～\n\r\t/\\|]+|(?<!\d)\.|\.(?!\d)")
# 找「新增说法」时才按空格切，因为口语改写常用空格当分隔
NOVEL_SPLIT = re.compile(
    r"[，。！？；：、,;:!?()（）\[\]【】《》…—·~～\s/\\|]+|(?<!\d)\.|\.(?!\d)")
# 数字的边界判定：后面既不是汉字也不是字母数字，才算数字结束
_NEXT_IS_CONTENT = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")


# ---------------------------------------------------------------- 归一化
def _norm(text):
    """NFKC（全角→半角、％→%）+ 压空白。保留大小写，展示用原文。"""
    t = unicodedata.normalize("NFKC", text or "")
    t = t.replace("\u3000", " ").replace("\u200b", "").replace("\ufeff", "")
    return re.sub(r"[ \t]+", " ", t).strip()


def _tight(text):
    """去掉全部空白并小写：查「37 %」=「37%」、「Opus 5.5」=「opus5.5」。"""
    return re.sub(r"\s+", "", text).lower()


def _mask(text):
    """把链接 / BV 号 / @提及 / 表情抹成等长空格，返回 (掩码文本, 实体列表)。

    抹掉是为了不让 BV 号里的数字、表情里的汉字污染数字与要点统计。
    """
    ents = []
    out = text
    for rx, kind in ((URL_RE, "链接"), (BV_RE, "BV号"),
                     (MENTION_RE, "@提及"), (EMOTE_RE, "表情")):
        for m in rx.finditer(out):
            if kind != "表情":          # 表情不算实体，加表情不改变信息
                ents.append({"kind": kind, "text": m.group(0)})
        out = rx.sub(lambda m: " " * len(m.group(0)), out)
    return out, ents


# ---------------------------------------------------------------- 数字
def _cn2int(s):
    """中文数字转整数：十五→15，三十七→37，一百二十三→123"""
    total = section = num = 0
    for ch in s:
        if ch in CN_DIGIT:
            num = CN_DIGIT[ch]
        elif ch in CN_UNIT:
            u = CN_UNIT[ch]
            if num == 0:
                num = 1
            section += num * u
            num = 0
        else:
            return None
    return section + num


def _numbers(text):
    """抽出文本里的全部数值点（阿拉伯 + 中文），带单位与量级。

    返回 [{text, value, unit, kind, approx}]：
      kind = "number" | "version"（紧跟在英文词后面的，如 Opus 5.5）
      approx = 带「多/几/余」的估计值，value 取基数
    """
    out = []
    masked, _ = _mask(text)

    # 先抓「百分之三十七」，再把这段从后续扫描里挖空，避免同一个值被抓两次。
    # 表示方式与「37%」一致（值 37、单位 %），不能折成 0.37，
    # 否则同一个数会被判成一丢一增。
    work = masked
    for m in _PCT_CN.finditer(masked):
        cn, ar = m.group(1), m.group(2)
        v = _cn2int(cn) if cn else float(ar)
        if v is None:
            continue
        out.append({"text": m.group(0), "value": float(v),
                    "unit": "%", "kind": "number", "approx": False})
        a, b = m.span()
        work = work[:a] + " " * (b - a) + work[b:]

    # 阿拉伯数字
    for m in _AR_NUM.finditer(work):
        raw, approx, unit = m.group(1), m.group(2), m.group(3)
        try:
            val = float(raw)
        except ValueError:
            continue
        # 版本号 = 紧贴在英文词后面 **且带小数点**（Opus 5.5 / Fable 5.2）。
        # 只按「前面是字母」判会把英文句子里 "is 37%" 当成版本，所以要求带小数点，
        # 并且前面的词不能是英文功能词。
        prev = work[:m.start()].rstrip()
        pv = re.search(r"([A-Za-z][A-Za-z0-9]*)$", prev)
        is_ver = bool(pv and "." in raw and pv.group(1).lower() not in EN_STOP)
        kind = "version" if is_ver else "number"
        if unit in UNIT_SCALE:
            val *= UNIT_SCALE[unit]
        out.append({"text": raw + (approx or "") + (unit or ""), "value": val,
                    "unit": unit, "kind": kind, "approx": bool(approx)})

    # 中文数字（必须带单位 / 量级 / 落在分句边界上，避免「一路」「万一」误命中）
    for m in _CN_NUM.finditer(work):
        vague, vbig, core, big, approx, unit = m.groups()
        if vague:                                    # 几万亿 / 数万
            tok = {"text": vague + (vbig or ""), "value": None,
                   "unit": vbig, "kind": "number", "approx": True}
        else:
            base = _cn2int(core)
            if base is None:
                continue
            if core + (approx or "") + (unit or "") in CN_BLOCK:
                continue
            has_boundary = not (unit or big)
            if has_boundary:
                # 允许「只剩三十七」「四五个」这种裸露数字，但要求后面
                # 不是内容字符，否则「一路/一致」会被当成 1
                nxt = work[m.end():m.end() + 1]
                if nxt and _NEXT_IS_CONTENT.match(nxt):
                    continue
            val = float(base) * UNIT_SCALE.get(big, 1.0)
            tok = {"text": core + (approx or "") + (unit or ""), "value": val,
                   "unit": unit, "kind": "number",
                   "approx": bool(approx or big)}
        if tok["value"] is not None:
            # 中文数字后面跟的 万/亿 已在 value 里，unit 只留真正的单位
            tok["unit"] = unit
        out.append(tok)

    # 去重（同一位置上「百分之三十七」会被前面的中文规则也抓到一次）
    seen, dedup = set(), []
    for t in out:
        key = (t["text"], t["value"], t["unit"], t["kind"])
        if key in seen:
            continue
        seen.add(key)
        dedup.append(t)
    return dedup


# ---------------------------------------------------------------- 实体
def _entities(text, masked, num_tokens):
    """英文词 / 版本号 / 链接 / BV / @提及。版本号与英文词分开计。"""
    ents = []
    for m in LATIN_RE.finditer(masked):
        w = m.group(0)
        if len(w) < 2 or w.lower() in EN_STOP:   # 单字母（A/B/C）、英文功能词不算实体
            continue
        ents.append({"kind": "英文词", "text": w, "key": w.lower()})
    _, extra = _mask(text)
    for e in extra:
        if e["kind"] == "表情":
            continue
        ents.append({"kind": e["kind"], "text": e["text"], "key": e["text"].lower()})
    for t in num_tokens:
        if t["kind"] == "version":
            ents.append({"kind": "版本号", "text": t["text"],
                         "key": "ver:" + t["text"], "value": t["value"]})
    return ents


# ---------------------------------------------------------------- 否定
def _neg_positions(text):
    """返回 [(位置, 上下文)] —— 已剔除「非常/告别/未来」这类假否定。"""
    chars = list(text)
    for w in NEG_FALSE:
        start = 0
        while True:
            i = text.find(w, start)
            if i < 0:
                break
            for k in range(i, i + len(w)):
                chars[k] = "□"
            start = i + 1
    hits = []
    for i, ch in enumerate(chars):
        if NEG_CHAR.match(ch):
            lo, hi = max(0, i - 6), min(len(text), i + 7)
            hits.append((i, text[lo:hi]))
    return hits


# ---------------------------------------------------------------- 指代
def _pronouns(text):
    """{类别: 命中数}。长词先匹配，避免「我们」被算成两次「我」。"""
    out = {}
    rest = text
    for cls, words in PRON.items():
        n = 0
        for w in words:
            while w in rest:
                n += 1
                rest = rest.replace(w, "□" * len(w), 1)
        if n:
            out[cls] = n
    return out


# ---------------------------------------------------------------- 要点覆盖
def _clauses(text):
    return [c.strip() for c in CLAUSE_SPLIT.split(text) if c.strip()]


def _items(clause):
    """一个分句的信息项：内容汉字（逐个）+ 英文词 + 阿拉伯数字串。"""
    items = []
    for ch in clause:
        if "\u4e00" <= ch <= "\u9fff" and ch not in FUNC_CHARS:
            items.append(ch)
    for m in LATIN_RE.finditer(clause):
        if len(m.group(0)) >= 2 and m.group(0).lower() not in EN_STOP:
            items.append(m.group(0).lower())
    for m in re.finditer(r"\d+(?:\.\d+)?", clause):
        items.append(m.group(0))
    return items


def _coverage(src_clauses, dst_pool):
    """分句覆盖率 + 未覆盖的分句清单（= 丢掉的要点）。"""
    total = covered = 0
    missing = []
    for c in src_clauses:
        items = _items(c)
        if len(items) < 2:                 # 单个信息字符的分句不作为要点
            continue
        total += 1
        need = Counter(items)
        got = sum(min(n, dst_pool.get(it, 0)) for it, n in need.items())
        ratio = got / float(len(items))
        if ratio >= COV_CLAUSE:
            covered += 1
        else:
            missing.append({"clause": c, "ratio": ratio})
    return total, covered, missing


def _fabrication(clause):
    """给一个「原文里没有依据」的分句打捏造分，返回 (分数, 命中线索)。

    只在这个分句与原文要点重合度低于 FAB_NOVEL_RATIO 时才会被调用，
    所以「把原文的话换个说法」不会进到这里。
    """
    score, cues = 0, []
    if PLACE_RE.search(clause):
        score += 3
        cues.append("具体处所")
    if COMPARE_RE.search(clause):
        score += 3
        cues.append("比较对象")
    if VAGUE_REF_RE.search(clause):
        score += 3
        cues.append("指向某X")
    if THIRD_PARTY_RE.search(clause):
        score += 3
        cues.append("另一个信息源")
    if TIME_RE.search(clause):
        score += 1
        cues.append("时间")
    if PERSON_RE.search(clause):
        score += 1
        cues.append("人称")
    if ACTION_RE.search(clause):
        score += 1
        cues.append("具体动作")
    return score, cues


# ---------------------------------------------------------------- 主接口
def check(original: str, rewritten: str) -> dict:
    """保真检查：原文的信息有没有被改写丢掉 / 改掉 / 凭空加东西。

    返回 {
      "score": float,     # 0-1，1 表示信息完全保留
      "ok": bool,         # score 是否达到通过线（关键项缺失一律 False）
      "missing": [...],   # 丢失的信息点
      "altered": [...],   # 被篡改的信息点（值变了）
      "added": [...],     # 原文没有、改写新增的实质信息（捏造事实以「捏造事实…」开头）
      "notes": [...],     # 人工可读的说明
      "parts": {...},     # 附加：各维度得分与计数（含 fabrication_hits），可忽略
    }
    """
    src, dst = _norm(original), _norm(rewritten)
    if not src:
        return {"score": 1.0, "ok": True, "missing": [], "altered": [],
                "added": [], "notes": ["原文为空，无需检查"],
                "parts": {"empty_source": True}}
    if not dst:
        return {"score": 0.0, "ok": False, "missing": ["全文（改写为空）"],
                "altered": [], "added": [], "notes": ["改写为空"],
                "parts": {"empty_rewrite": True}}

    missing, altered, added, notes = [], [], [], []
    crit = []                              # 关键项失败原因
    parts = {}

    src_masked, _ = _mask(src)
    dst_masked, _ = _mask(dst)
    src_tight, dst_tight = _tight(src_masked), _tight(dst_masked)

    # ---- 1 数字 ----------------------------------------------------
    s_num = _numbers(src_masked)
    d_num = _numbers(dst_masked)
    s_nums = [t for t in s_num if t["kind"] == "number"]
    d_nums = [t for t in d_num if t["kind"] == "number"]
    d_values = {t["value"] for t in d_num}                 # 含版本号的值
    s_values = {t["value"] for t in s_nums}
    d_by_value = {}
    for t in d_nums:
        if t["value"] is not None:
            d_by_value.setdefault(t["value"], set()).add(t["unit"] or "")

    miss_n = alt_n = add_n = 0
    for t in s_nums:
        hit = False
        if t["value"] is not None and t["value"] in d_values:
            hit = True
        elif _tight(t["text"]) and _tight(t["text"]) in dst_tight:
            hit = True                                     # 三四天 / 几万亿 原样保留
        if not hit:
            miss_n += 1
            missing.append(f"数字 {t['text']}")
            crit.append(f"数字 {t['text']} 丢失")
            continue
        # 值在，但强单位不兼容 → 单位被改（20分钟 → 20秒）
        if t["value"] is not None and t["unit"] in STRONG_UNITS:
            units = d_by_value.get(t["value"], set())
            if units and not (units & {t["unit"], ""}):
                alt_n += 1
                altered.append("数字 %s → %s" % (t["text"], "/".join(sorted(units))))
                crit.append(f"数字 {t['text']} 单位被改")
    for t in d_nums:
        tok_tight = _tight(t["text"])
        if t["value"] is None:
            if not (tok_tight and tok_tight in src_tight):
                add_n += 1
                added.append(f"数字 {t['text']}（原文无）")
                crit.append(f"凭空新增数字 {t['text']}")
            continue
        if t["value"] not in s_values and tok_tight not in src_tight:
            add_n += 1
            added.append(f"数字 {t['text']}（原文无）")
            crit.append(f"凭空新增数字 {t['text']}")

    n_src_num = len(s_nums)
    parts["number"] = max(0.0, 1.0 - (miss_n + alt_n + 0.5 * add_n) / max(1, n_src_num))
    if n_src_num:
        notes.append(f"数字 {n_src_num} 个，保留 {n_src_num - miss_n - alt_n} 个")

    # ---- 2 实体 ----------------------------------------------------
    s_ent = _entities(src, src_masked, s_num)
    d_ent = _entities(dst, dst_masked, d_num)
    d_ent_keys = {e["key"] for e in d_ent}
    s_ent_keys = {e["key"] for e in s_ent}
    d_ver_values = {x["value"] for x in d_ent if x["kind"] == "版本号"}
    s_ver_values = {x["value"] for x in s_ent if x["kind"] == "版本号"}
    miss_e = alt_e = add_e = 0
    s_ent_seen = set()
    for e in s_ent:
        if e["key"] in s_ent_seen:
            continue
        s_ent_seen.add(e["key"])
        if e["key"] in d_ent_keys:
            continue
        if e["kind"] == "版本号":
            # 版本号只要数值还在就算保留（「换成 5.2」也算），否则算改版本
            if e["value"] in d_ver_values or e["value"] in d_values:
                continue
            alt_e += 1
            altered.append(f"版本号 {e['text']} 消失")
            crit.append(f"版本号 {e['text']} 被改/删除")
            continue
        if e["key"] in dst_tight:
            continue
        miss_e += 1
        missing.append(f"{e['kind']} {e['text']}")
        crit.append(f"{e['kind']} {e['text']} 丢失")
    d_seen = set()
    for e in d_ent:
        if e["key"] in d_seen or e["key"] in s_ent_keys:
            continue
        d_seen.add(e["key"])
        # 新增英文词：只认原文完全没有的词形
        if e["key"] in src_tight:
            continue
        if e["kind"] == "版本号" and e.get("value") in s_ver_values:
            continue
        add_e += 1
        added.append(f"{e['kind']} {e['text']}（原文无）")
        crit.append(f"凭空新增{e['kind']} {e['text']}")

    n_src_ent = len(s_ent_seen)
    parts["entity"] = max(0.0, 1.0 - (miss_e + alt_e + 0.5 * add_e) / max(1, n_src_ent))
    if n_src_ent:
        notes.append(f"实体 {n_src_ent} 个，保留 {n_src_ent - miss_e - alt_e} 个")

    # ---- 3 否定 ----------------------------------------------------
    s_neg = _neg_positions(src)
    d_neg = _neg_positions(dst)
    lost = len(s_neg) - len(d_neg) if len(s_neg) > len(d_neg) else 0
    gain = len(d_neg) - len(s_neg) if len(d_neg) > len(s_neg) else 0
    parts["negation"] = max(0.0, 1.0 - (lost + 0.5 * gain) / max(1, len(s_neg)))
    notes.append(f"否定 {len(s_neg)}→{len(d_neg)}")
    for _, ctx in s_neg[len(d_neg):] if lost else []:
        missing.append(f"否定「{ctx}」")
    for _, ctx in d_neg[len(s_neg):] if gain else []:
        added.append(f"否定「{ctx}」（原文无）")
        altered.append(f"极性可能被翻转：「{ctx}」")
    # 否定只防「删掉否定」和「大改极性」，不再「多一条就判死」。
    #
    # 改因（2026-11-16）：合法同义改写常会新增一个否定，
    #   「max 档先关」→「用不起了」  （原文无否定 → 改写有 1 个）
    # 旧的逐条判定把这一类全挡了，实测 10 条同义改写 0 条通过。
    #
    # 已知盲区：这个判据分不清「同义改述」和「极性反转」——
    #   「这是好事」→「这不是好事」在字符层面和上面那种一模一样。
    #   实测里极性反转只有靠人工兜底，代码抓不到。宁可漏报也不误杀。
    if parts["negation"] < NEG_MIN:
        crit.append(f"否定保留度 {parts['negation']:.2f} < {NEG_MIN}")

    # ---- 4 指代 ----------------------------------------------------
    # 只处理「被指称的对象」：中文改写合法地省略主语，所以一/二人称
    # 少了不扣分（只记 notes）；第三人称被抹掉才扣，被换成另一类才判死。
    s_pr = _pronouns(src)
    d_pr = _pronouns(dst)
    lost_hard = [c for c in PRON_HARD if c in s_pr and c not in d_pr]
    lost_soft = [c for c in s_pr if c not in PRON_HARD and c not in d_pr]
    swap_ref = (("第三人称-物" in s_pr and "第三人称-物" not in d_pr
                 and "第三人称-人" in d_pr and "第三人称-人" not in s_pr)
                or ("第三人称-人" in s_pr and "第三人称-人" not in d_pr
                    and "第三人称-物" in d_pr and "第三人称-物" not in s_pr))
    for c in lost_hard:
        missing.append(f"指代（{c}）")
    if swap_ref:
        altered.append("指代对象被换：它 ↔ 他/她")
        crit.append("指代对象被换成另一个（它↔他/她）")
    parts["referent"] = max(0.0, 1.0 - (0.5 * len(lost_hard) + swap_ref)
                            / max(1, len(s_pr)))
    if s_pr:
        notes.append("指代 " + "、".join(f"{k}×{v}" for k, v in s_pr.items()))
    if lost_soft:
        notes.append("省略了" + "、".join(lost_soft) + "（中文里通常合法）")

    # ---- 5 要点覆盖 ------------------------------------------------
    dst_pool = Counter(_items(dst_masked))
    total, covered, miss_cl = _coverage(_clauses(src_masked), dst_pool)
    for m in miss_cl:
        missing.append(f"要点「{m['clause'][:18]}」")
    cov = covered / float(total) if total else 1.0
    parts["coverage"] = cov
    if total:
        notes.append(f"要点覆盖率 {cov:.2f}（{covered}/{total}）")
    # 覆盖率只在大段删减时判死。
    #
    # 改因（2026-11-16）：原阈值 0.70 对所有长度一视同仁，但短文本要点太少，
    #   「离重置还有五天」→「还要撑五天」只有一个要点，换种说法覆盖率直接 0.00，
    #   实测 10 条同义改写 0 条通过。长文本（≥COV_MIN_LEN）仍然有效：
    #   丢一大段的坏例覆盖 0.31，这份改写 0.40，门槛取 0.35 能分开。
    if total and len(src) >= COV_MIN_LEN and cov < COV_FLOOR:
        crit.append(f"要点覆盖率 {cov:.2f} < {COV_FLOOR}")

    # 改写新增的内容分两类，处理相反：
    #   情绪/语气/评价（「太扎心了老铁」「笑死」）—— 放行，只记 added 与 notes
    #   捏造事实（「我在食堂笑到流泪」「比某家划算太多」）—— 判死
    # 两道闸：
    #   闸 1 整句新增 —— 分句与原文要点重合度低于 FAB_NOVEL_RATIO，再看锚点分
    #   闸 2 锚点新增 —— 改写里出现原文从未有过的处所/比较对象/信息源，
    #                    即使这句话和原文粘连成一句（没有逗号可分）也能抓到
    src_pool = Counter(_items(src_masked))
    dst_clauses = [x.strip() for x in NOVEL_SPLIT.split(dst_masked) if x.strip()]
    novel, fab_hits, fab_seen = [], [], set()
    for c in dst_clauses:
        items = _items(c)
        if not items:
            continue
        got = sum(min(n, src_pool.get(it, 0)) for it, n in Counter(items).items())
        if got / float(len(items)) >= FAB_NOVEL_RATIO:      # 有原文依据，走闸 2
            continue
        fab, cues = _fabrication(c)
        if fab >= FAB_MIN:
            fab_hits.append((c, cues))
            fab_seen.add(c)
        elif len(items) >= 3:
            novel.append(c)
    for c in dst_clauses:
        if c in fab_seen:
            continue
        for rx, lab in ((PLACE_RE, "处所"), (COMPARE_RE, "比较对象"),
                        (VAGUE_REF_RE, "指向某X"), (THIRD_PARTY_RE, "信息源")):
            m = rx.search(c)
            if m and m.group(0) not in src:
                fab_hits.append((c, [f"新增{lab}「{m.group(0)}」"]))
                fab_seen.add(c)
                break
    for c, cues in fab_hits[:5]:
        added.append(f"捏造事实「{c[:18]}」（{'/'.join(cues)}，原文无）")
        crit.append(f"捏造事实「{c[:18]}」")
    for c in novel[:5]:
        added.append(f"要点「{c[:18]}」（原文无）")
    if novel:
        notes.append(f"新增说法 {len(novel)} 处（评价/语气类，未计入关键项）")
    if fab_hits:
        notes.append(f"捏造事实 {len(fab_hits)} 处（原文没有的可验证信息）")

    # ---- 汇总 ------------------------------------------------------
    score = sum(W[k] * parts[k] for k in W)
    # 捏造事实不在五个维度的加权里（它是「多出来的东西」，不是「少了的东西」），
    # 但要反映到分数上：每处乘 0.75，最多压到 0.55。否则会出现
    # score=1.000 而 ok=False 这种让人困惑的组合。
    if fab_hits:
        score *= max(0.55, 1.0 - 0.25 * len(fab_hits))
    score = round(float(max(0.0, min(1.0, score))), 4)
    ok = bool(score >= SCORE_MIN and not crit)

    lens = f"字数 {len(src)}→{len(dst)}"
    if not crit:
        notes.append("关键项（数字/实体/否定/指代）全保留")
    else:
        notes.append("关键项问题 " + str(len(crit)) + " 处：" + "；".join(crit[:4]))
    parts = {k: round(v, 4) for k, v in parts.items()}
    parts["fabrication_hits"] = len(fab_hits)

    return {"score": score, "ok": ok, "missing": missing, "altered": altered,
            "added": added, "notes": [lens] + notes, "parts": parts}


# ---------------------------------------------------------------- 局限
__doc_limits__ = """
查不出的保真问题（已知，别当成没写）。每一条都实测过，都是 score=1.000 放行的：

1 中文人名/机构名替换。「阿江最后的建议」→「老王最后的建议」查不出，
  因为分句里其余 6 个字都在，覆盖率 0.75 过线。中文专名没有识别能力。
2 语序/主客互换。「我把它当傻子，它不愿意信我」→「它把我当傻子，
  我不愿意信它」语义完全反了，但字符袋相同、否定数相同、指代类别相同，
  三个维度全过。这是本模块最严重的盲区。
3 关系词新增。「订阅剩 37%，五天就重置」→「因为订阅只剩 37%，所以五天
  就得重置」：凭空加了因果，但没有新数字/实体/否定/锚点，查不出。
4 情态与限定的弱化。「只是愿意信你」→「就是愿意信你」、「大概最没有
  利益冲突」→「最没有利益冲突」：程度、推测语气变了，查不出。
5 量词性削减。「骗它一场暴富」→「骗它暴富」：量词不当数字要点（否则
  「一场/一段」到处误报），所以这种轻微削减放过。
6 同义改写的长句会虚低覆盖率。「浪漫化成真死亡」→「说成真死」保住 6 成
  字符，按 0.55 卡勉强过线；更彻底的意译会被误报成要点丢失。
7 否定只按条数比，不做作用域对齐。原地替换（不→别）算通过，
  「A 不成立」→「B 不成立」这种挪位后的极性错误查不出。
8 中文小数（「八十九点七五」）不解析，会误报为数字丢失（英文场景）。
9 英文否定（not/no）不检查；英文功能词已过滤，但英文文本的覆盖率
  指标本身不可靠——本模块是给中文 B 站评论写的。
10 捏造事实只认「锚点」：处所 / 比较对象 / 第三方信息源 / 新数字或实体。
   没有锚点、也没有新数字的凭空断言查不出，例如「这软件真好用」「其实
   没那么夸张」——它可能引入了原文没有的评价结论。
11 「时间 + 动作」不判死：「订阅剩 37%」→「订阅剩 37%，今天又降了」放行，
   因为口语改写加时间状语的太常见（宁可漏报也不误杀）。要判死需要处所 /
   比较对象 / 第三方 / 人称同时到场。
12 处所词表是穷举的，表外地点会漏：「在网吧」「在电梯里」「在工位旁边的
   会议室」都抓不到，除非同句还有比较对象或第三方。
13 反方向偏严：把「在单位加班」改写成「在公司加班」这种处所近义词替换，
   会因为「公司」在原文没出现过而被当成捏造。这类误判目前只在改写换了
   场所词时发生，实测 15 条真实改写里 0 次。
"""
