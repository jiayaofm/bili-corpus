# -*- coding: utf-8 -*-
"""
AI 发言打分器 v3 —— 句式特征版

v2 的问题（你指出的）：
  · 靠长度判 AI，真人的长文也中招
  · 统计挖词不可行：19 条正样本，挖出来全是话题词（小肥鱼、900多块）
    那是"视频内容"不是"说话方式"

v3 的做法：**人工归纳句式标记**，逐条对应真实样本。
每条规则都能指回具体样例，不是拍脑袋。

权重设计原则：
  · 标点/符号类特征几乎零误报 → 高权重
  · 句式类（不是A而是B、是真的、另一回事）→ 中高权重
  · 长度只保留极小权重（你明确否掉了长度主导）
  · 广告/招聘/刷屏 → 强负分
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "labeling"
OUT_DIR.mkdir(exist_ok=True)

EMOTE_RE = re.compile(r"\[[^\[\]]{1,12}\]")
BV_RE = re.compile(r"BV[0-9A-Za-z]{10}")
URL_RE = re.compile(r"https?://\S+")

# ============================================================================
# 核心：AI 句式标记表
#   格式 (名称, 匹配方式, 权重)
#   匹配方式：str = 字面包含；re.Pattern = 正则
#   每条都对应真实样例，见右侧注释
# ============================================================================
MARKERS = [
    # ---------- 符号类：真人几乎不用 ----------
    ("markdown粗体", "**", 50),            # ① B站不渲染 Markdown
    # 「」引号单独处理（见 score()）：短评论里多是引专名，长评论里才是强调
    ("破折号插入", "——", 16),               # ③⑤⑥⑩⑪⑫ 插入语/转折
    ("分号并列", "；", 8),                  # ④

    # ---------- 人设自称（区分"以鱼自称"和"提及鱼"）----------
    # 第一人称自称才是人格信号：③⑪⑫⑮⑯⑱
    ("第一人称自称", re.compile(r"本鱼|本鲸|本喵|本AI"), 16),
    # 只是"提及"大肥鱼/小肥鱼 —— 真人也在聊这个话题，几乎不加分
    ("提及鱼名", re.compile(r"大肥鱼|小肥鱼"), 3),

    # ---------- AI 助手视角（暴露身份的自述）----------
    # ⑧"读字幕看的""我这个型号"；⑳"别让我干到一半被停机""派活"
    ("助手视角", re.compile(r"我这个型号|派活|停机|我的余额|让我干|我干活|跑一轮题库|我这号"), 14),

    # ---------- 句式类：AI 的思维习惯 ----------
    # 「不是A，是B」「不是A，而是B」—— 对举下定义
    ("不是A是B", re.compile(r"不是[^，。；\n]{1,18}[，,]?\s*(?:而)?是[^，。；\n]{1,18}"), 14),
    # 「是真的」「也是真的」—— 判断句收尾
    ("…是真的", re.compile(r"(?:是|也是)真的[，。]"), 12),   # ①②⑤
    # 「这条」指代被回复的评论 —— AI 的分析口吻，②⑥⑨
    ("指代「这条」", re.compile(r"这条"), 8),
    # 「比…更/都」—— 比较结构，②⑤
    ("比…更/都", re.compile(r"比[^，。\n]{2,13}(?:更|都)"), 10),
    # 「另一回事」「另一栏」—— 转折评判
    ("另一回事", re.compile(r"另一(?:回事|栏|种|层)"), 10),  # ⑤
    # 「本质/内核/结构 + 是」
    ("本质内核腔", re.compile(r"(?:本质|内核|核心|结构)[^，。]{0,6}(?:是|在于|其实是)"), 10),  # ①②⑫
    # 「关键是」「说白了」「说到底」「最重要的是」
    ("总结腔", re.compile(r"关键是|说白了|说到底|最重要的是|归根结底"), 10),  # ④⑫
    # 冒号 + 解释（"…：xxx"）—— 真人也会用，权重压低且要求句中有实体
    ("冒号解释", re.compile(r"[^：:\n][：:]\s*[^\n]{8,}"), 4),  # ①②⑧⑩⑫
    # 「由我来说」「这话由我」
    ("元评论腔", re.compile(r"由我来说|这话由我|我要说的|我想说的是"), 10),  # ⑧
    # 傲娇否认「才没有/才不是」—— 真人也会写，权重低
    ("傲娇否认", re.compile(r"才没有|才不是"), 5),  # ⑯⑱
    # 「恰恰」这种书面对比副词
    ("书面副词", re.compile(r"恰恰|反而|反而更|而是|无非是"), 6),

    # ---------- 数字对比结构 ----------
    # 「六小时、八十一美元」「94 分比 89.75 高四分多」「1000 多万亿 vs 900 多块」
    ("数字对比", re.compile(r"\d[\d.]*\s*(?:万|亿|块|元|分|刀|美元|小时)?[^。\n]{0,12}\d"), 6),

    # ---------- 柔化语气词（权重低：真人也会用）----------
    ("柔化语气", re.compile(r"(?:捏|啦|喵|呀|呢)[。！？~]*$|(?:捏|喵)"), 5),  # ⑮⑯⑱
]

# 负向：真人强信号
NEG_MARKERS = [
    ("有B站表情", re.compile(r"\[[^\[\]]{1,12}\]"), -18),   # AI 极少主动用站内表情
    ("口语脏话", re.compile(r"卧槽|我靠|特么|妈的|牛逼|离谱|笑死|草[，。]?|淦|服了|无语|拉胯|破防|急了|典"), -10),
    ("语气词连打", re.compile(r"[？?]{2,}|[哈草笑]{3,}"), -12),
    ("含比分", re.compile(r"\d+\s*[:：]\s*\d+|\d/\d"), -8),
    ("时间戳", re.compile(r"\d{1,2}[:：]\d{2}(?!\s*[分块万亿])"), -6),
]

# 广告 / 招聘 / 刷屏 —— 真人串子或营销号，不是 AI 人格
SPAM_MARK = ["上四休三", "七险五金", "五险一金", "年终奖", "月薪", "底薪",
             "包吃住", "招聘", "私信我", "扫码", "加群", "内推", "邀请码"]
SPAM_EMOJI = re.compile(r"[💰🎣🔥✨💥🎉👏👍🍚]")


def score(r):
    c = r["content"]
    n = len(c)
    s = 0.0
    hits = []

    for name, pat, w in MARKERS:
        hit = (pat in c) if isinstance(pat, str) else bool(pat.search(c))
        if hit:
            s += w
            hits.append(f"{name}+{w}")

    # 「」引号：长评论里是"强调"，短评论里多为"引专名"，区别对待
    if "「" in c or "」" in c:
        w = 26 if n >= 60 else 10
        s += w
        hits.append(f"「」引号+{w}")

    for name, pat, w in NEG_MARKERS:
        hit = (pat in c) if isinstance(pat, str) else bool(pat.search(c))
        if hit:
            s += w
            hits.append(f"{name}{w}")

    # 长度：只保留极小权重（长度不是判据）
    if n > 200:
        s += 3
        hits.append("超长+3")
    elif n <= 5:
        s -= 12
        hits.append("极短-12")
    elif n <= 15:
        s -= 6
        hits.append("短-6")

    # 广告/刷屏
    if any(w in c for w in SPAM_MARK):
        s -= 50
        hits.append("招聘/广告-50")
    if URL_RE.search(c) and len(SPAM_EMOJI.findall(c)) >= 2:
        s -= 30
        hits.append("URL+刷屏-30")
    if len(SPAM_EMOJI.findall(c)) >= 3 and n < 120:
        s -= 15
        hits.append("emoji刷屏-15")

    # ================= 账号名（弱先验，不是决定性）=================
    # 实测提升度：全库 AI 占比 6.2%，名字含鱼/鲸的评论里 AI 占 25%（4 倍提升）。
    # 但真人粉丝也会起鱼名（"专注摸鱼的打工人""遐蝶就是蝶"），
    # 所以只给弱加成 —— 这是圈层特征 + 粉丝混淆，不是 AI 专属。
    uname = r.get("user") or ""
    if re.search(r"鱼|鲸", uname):
        s += 6
        hits.append("账号名含鱼/鲸+6")

    return s, hits


def conf_of(s, hits):
    # 负分一律判低——此前的 bug 是带 BV 的广告仍被判 mid
    if s < 0:
        return "low"
    if s >= 45:
        return "high"
    if s >= 18:
        return "mid"
    return "low"


def load_all():
    recs = []
    for fp in sorted(DATA.glob("*_*.jsonl")):
        for l in fp.open(encoding="utf-8"):
            if l.strip():
                d = json.loads(l)
                d["_src"] = fp.name
                recs.append(d)
    seen, out = set(), []
    for r in recs:
        if r["rpid"] in seen:
            continue
        seen.add(r["rpid"])
        out.append(r)
    return out


def open_free(base: Path):
    for i in range(20):
        p = base if i == 0 else base.with_name(f"{base.stem}_v{i}{base.suffix}")
        try:
            return p, p.open("w", newline="", encoding="utf-8-sig")
        except PermissionError:
            continue
    raise PermissionError(f"{base} 被占用")


def load_prev_human_marks():
    """人工标注持久化读取。
    优先 human_marked.json（不受 CSV 覆盖影响），其次兼容旧 CSV 里的 x 标记。"""
    marks = {}
    jp = OUT_DIR / "human_marked.json"
    if jp.exists():
        try:
            for item in json.loads(jp.read_text(encoding="utf-8")):
                marks[item["rpid"]] = item.get("content", "")
        except Exception:
            pass

    p = OUT_DIR / "candidates.csv"
    if p.exists():
        try:
            with p.open(encoding="utf-8-sig", newline="") as f:
                for d in csv.DictReader(f):
                    sc = (d.get("score") or "").strip()
                    if "x" in sc.lower():
                        marks[d["rpid"]] = d.get("content", "")
        except Exception:
            pass
    return marks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    human_marks = load_prev_human_marks()
    if human_marks:
        print(f"载入人工标注：{len(human_marks)} 条判为真人")
        (OUT_DIR / "human_marked.json").write_text(
            json.dumps([{"rpid": k, "content": v} for k, v in human_marks.items()],
                       ensure_ascii=False, indent=2), encoding="utf-8")

    recs = load_all()
    print(f"载入 {len(recs)} 条（已去重）\n")

    rows = []
    for r in recs:
        s, hits = score(r)
        rows.append((s, hits, r))
    rows.sort(key=lambda x: -x[0])

    csv_path, f = open_free(OUT_DIR / "candidates.csv")
    with f:
        w = csv.writer(f)
        w.writerow(["score", "conf", "hits", "level", "like", "length", "rpid", "content", "label"])
        for s, hits, r in rows:
            cf = conf_of(s, hits)
            known_human = r["rpid"] in human_marks
            label = "0" if known_human else ("1" if cf == "high" else "")
            w.writerow([round(s, 1), cf, "|".join(hits), r.get("level"), r.get("like"),
                        len(r["content"]), r["rpid"],
                        r["content"].replace("\n", "\\n"), label])
    print(f"候选表：{csv_path}")

    # ---- 用人工标注验证 ----
    if human_marks:
        bad = [(s, r) for s, h, r in rows
               if r["rpid"] in human_marks and conf_of(s, h) != "low"]
        print(f"\n【验证】人工标为真人的 {len(human_marks)} 条中，"
              f"新打分器仍判为 high/mid 的：{len(bad)} 条")
        for s, r in bad:
            print(f"   {s:>6.1f}  {r['content'][:80]!r}")
        if not bad:
            print("   -> 全部落入 low 档，通过")
        print()

    print("=" * 80)
    print(f"AI 可疑度 Top {args.top}（v3 句式版）")
    print("=" * 80)
    for i, (s, hits, r) in enumerate(rows[: args.top], 1):
        c = r["content"].replace("\n", "\\n")
        print(f"{i:>3}. [{s:>6.1f} {conf_of(s,hits):>4}] {len(c):>3}字  {' | '.join(hits[:5])}")
        print(f"      {c[:140]}")

    hi = sum(1 for s, h, r in rows if conf_of(s, h) == "high")
    mi = sum(1 for s, h, r in rows if conf_of(s, h) == "mid")
    print("\n" + "=" * 80)
    print(f"分档：high={hi}  mid={mi}  low={len(rows)-hi-mi}")


if __name__ == "__main__":
    main()
