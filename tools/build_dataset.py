#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""构造 QLoRA SFT 数据集：只用真人评论，学「B站评论区人话」的语域。

数据来源（全部项目内已有）：
  1. labeling/ground_truth.csv  —— 人工标注，label=="0" 为真人
  2. data/BV*_like.jsonl        —— 抓取的评论，去掉已标为 AI 的

处理原则：
  · 只用真人评论，绝不润色 —— 润色就失去「真人」的意义
  · 话题只用数据里真实存在的 video_title，不编造
  · 清洗：纯表情/纯@/纯数字/纯链接剔除，4 <= 字数 <= 300，去重

产出：D:\\dsh-ai\\lora\\data\\sft_train.jsonl / sft_eval_holdout.jsonl

用法：
  python tools/build_dataset.py --out D:\\dsh-ai\\lora\\data
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GT_CSV = ROOT / "labeling" / "ground_truth.csv"

# 指令模板。system 固定，user 三选一。
SYSTEM = "你在 B 站评论区发一条真实的短评。"
USER_TOPIC = "关于「{topic}」这个话题，说一句你的看法。"
USER_GENERIC = "说一句你此刻想说的话。"
# 有摘要时用这个：把「这个视频讲了什么」当输入，而不是干巴巴一个标题。
# 这是本项目的核心假设——模型需要看得见视频内容，才写得出不空心的评论。
USER_SUMMARY = (
    "视频《{title}》讲的是：{summary}\n"
    "\n"
    "就这条视频，说一句你的看法。"
)

# ---------------------------------------------------------------- 清洗规则

EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF"
    "\U0001F000-\U0001F2FF" "\U0000FE00-\U0000FE0F" "\U0000200D"
    "]+"
)
AT_RE = re.compile(r"@[^\s@]{1,30}")
# 行首的 @某人（可能连着好几个）。扩采来的评论里有 11.4% 是这个形状 ——
# 那是评论区里手动 @ 别人的回复，不是独立顶层评论。
# 本任务生成的是顶层评论，所以训练目标里要剥掉。
AT_PREFIX_RE = re.compile(r"^(?:\s*@[^\s@]{1,30}\s*)+")
URL_RE = re.compile(r"(?:https?://|www\.)\S+")
# B站表情名：[doge] 这种
BILI_EMO_RE = re.compile(r"\[[^\[\]]{1,10}\]")


# 已知 AI 号：二级回复里它们可能自己回自己。
# GrokBot 有独立标注文件（labeling/grokbot_ai.csv，28 条确认是 AI），
# 鲸府大肥鱼 是本项目追踪的人格号本体。
# 其余名字里带「鱼/鲸」的大多是仿号的人类（比如「DeepSeek大肥鱼-真人版」），
# 不做模糊匹配，只按精确用户名剔除。
BOT_USERS = {
    "GrokBot打工日记",
    "鲸府大肥鱼",
}


# 二级回复常以「回复 @某人 :」或「@某人 」开头，这是**回复格式**，
# 而本任务生成的是**顶层评论**——真人顶层评论里这个比例是 0%。
# 2026-11-17 实测：不处理的话模型会输出 21% 的「回复 @xxx」。
# 两种处理：整条丢掉（--no-replies），或只剥掉前缀保留正文（--strip-reply-prefix）。
REPLY_PREFIX_RE = re.compile(r"^\s*(?:回复\s*)?@[^\s:：]{1,25}\s*[:：]?\s*")


def _strip_noise(text: str) -> str:
    """剥掉表情/at/链接/标点，用于判断「是不是纯噪声」。"""
    s = BILI_EMO_RE.sub("", text)
    s = EMOJI_RE.sub("", s)
    s = AT_RE.sub("", s)
    s = URL_RE.sub("", s)
    s = re.sub(r"[\s\W_]+", "", s, flags=re.UNICODE)
    return s


def is_noise(text: str) -> bool:
    """纯表情 / 纯@ / 纯数字 / 纯链接 / 纯符号。"""
    core = _strip_noise(text)
    if not core:
        return True
    # 纯数字
    if re.fullmatch(r"\d+", core):
        return True
    return False


def char_len(text: str) -> int:
    """按可见字符计长度（不计换行和首尾空白）。"""
    return len(text.strip())


def norm_key(text: str) -> str:
    """去重键：「保留语义、忽略空白/全半角噪声」。"""
    s = _strip_noise(text)
    return s.lower()


def accepted(text: str) -> bool:
    if text is None:
        return False
    t = text.strip()
    if not t:
        return False
    if is_noise(t):
        return False
    n = char_len(t)
    if n < 4 or n > 300:
        return False
    return True


# ---------------------------------------------------------------- 读取

def load_gt() -> list[dict]:
    if not GT_CSV.exists():
        return []
    with GT_CSV.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_jsonl_comments() -> list[dict]:
    """顶层评论 + 二级回复都摊平。

    二级回复有 1374 条，比顶层还多，是把语料翻倍最直接的来源。
    但它们没有自己的 rpid，而且可能是 AI 号回的，所以打上 `_is_reply` 标记，
    由 build() 按用户名把已知机器人剔掉。
    """
    rows = []
    # 两路来源：
    #   data/BV*_like.jsonl  早期采集（160 条/视频，按热度取，带 sub_replies）
    #   data/deep/*.jsonl    深采（tools/deep_fetch.py，最多上千条/视频，只有顶层）
    # 重复的 rpid 会在 build() 里靠 seen_rpid 去重。
    sources = (sorted(glob.glob(str(ROOT / "data" / "BV*_like.jsonl")))
               + sorted(glob.glob(str(ROOT / "data" / "deep" / "*.jsonl"))))
    for p in sources:
        # 文件名去掉扩展名后取 bvid：
        #   data/BV1xxx_like.jsonl  → BV1xxx_like  → split('_') → BV1xxx
        #   data/deep/BV1xxx.jsonl  → BV1xxx
        bvid = Path(p).stem.split("_")[0].split("-")[0]
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                d["_is_reply"] = False
                d["bvid"] = bvid
                rows.append(d)
                for i, sr in enumerate(d.get("sub_replies") or []):
                    c = (sr.get("content") or "").strip()
                    if not c:
                        continue
                    rows.append({
                        "rpid": "sub:%s:%d" % (d.get("rpid", ""), i),
                        "content": c,
                        "user": sr.get("user") or "",
                        "video_title": d.get("video_title") or "",
                        "bvid": bvid,
                        "_is_reply": True,
                    })
    return rows


# ---------------------------------------------------------------- 主逻辑

def load_summaries() -> dict[str, dict]:
    """读 data/summaries/<bvid>.json，返回 {bvid: {summary, title, confidence}}。"""
    out: dict[str, dict] = {}
    for p in sorted(glob.glob(str(ROOT / "data" / "summaries" / "*.json"))):
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if d.get("bvid") and (d.get("summary") or "").strip():
            out[d["bvid"]] = d
    return out


def build(out_dir: Path, seed: int = 42, generic_ratio: float = 0.35,
          no_replies: bool = False, strip_reply_prefix: bool = False,
          holdout: int = 20, with_summary: bool = False,
          drop_at: bool = False) -> dict:
    rng = random.Random(seed)
    gt = load_gt()
    raw = load_jsonl_comments()

    # 已标为 AI 的 rpid（label == "1"）
    ai_rpids = {str(r["rpid"]).strip() for r in gt if str(r["label"]).strip() == "1"}
    human_rpids = {str(r["rpid"]).strip() for r in gt if str(r["label"]).strip() == "0"}

    # 只按 rpid 过滤不够：同一段 AI 文本可能以**另一个 rpid** 再次出现
    # （重复投稿/多视频共用文案）。所以再按内容做一层过滤。
    ai_content_keys = {
        norm_key(r["content"]) for r in gt
        if str(r["label"]).strip() == "1" and (r.get("content") or "").strip()
    }

    stats = {
        "gt_rows": len(gt),
        "gt_ai": len(ai_rpids),
        "gt_human": len(human_rpids),
        "jsonl_raw": len(raw),
    }

    # rpid -> video_title（来自抓取数据）
    title_by_rpid: dict[str, str] = {}
    bvid_by_rpid: dict[str, str] = {}
    for d in raw:
        rp = str(d.get("rpid", "")).strip()
        t = (d.get("video_title") or "").strip()
        if rp and t:
            title_by_rpid[rp] = t
        if rp and d.get("bvid"):
            bvid_by_rpid[rp] = d["bvid"]

    # 候选池：jsonl 里未被标为 AI 的 + gt 里标为人人的
    pool: list[tuple[str, str, str, str]] = []  # (source, content, topic, bvid)
    seen_rpid: set[str] = set()
    bot_skipped = 0
    reply_dropped = 0
    prefix_stripped = 0

    for d in raw:
        rp = str(d.get("rpid", "")).strip()
        content = d.get("content") or ""
        if not rp or rp in seen_rpid:
            continue
        if rp in ai_rpids:          # 去掉已确认的 AI 评论
            continue
        is_reply = bool(d.get("_is_reply"))
        if is_reply and (d.get("user") or "") in BOT_USERS:
            bot_skipped += 1        # 二级回复里 AI 号自己回的
            continue
        if is_reply and no_replies:
            reply_dropped += 1
            continue
        if is_reply and strip_reply_prefix:
            content2 = REPLY_PREFIX_RE.sub("", content or "")
            if content2 != (content or ""):
                prefix_stripped += 1
            content = content2
        seen_rpid.add(rp)
        pool.append(("jsonl" if not is_reply else "reply",
                     content, title_by_rpid.get(rp, ""), bvid_by_rpid.get(rp, "")))

    stats["bot_user_skipped"] = bot_skipped
    stats["reply_dropped"] = reply_dropped
    stats["reply_prefix_stripped"] = prefix_stripped

    for r in gt:
        rp = str(r["rpid"]).strip()
        if rp in seen_rpid:
            continue
        if str(r["label"]).strip() != "0":
            continue
        seen_rpid.add(rp)
        pool.append(("gt", r.get("content") or "",
                     title_by_rpid.get(rp, ""), bvid_by_rpid.get(rp, "")))

    stats["pool_after_ai_filter"] = len(pool)

    # 清洗 + 去重
    clean: list[dict] = []
    seen_keys: set[str] = set()
    drop = {"noise": 0, "length": 0, "dup": 0, "empty": 0, "ai_content": 0}
    for source, content, topic, bvid in pool:
        t = (content or "").strip()
        if not t:
            drop["empty"] += 1
            continue
        if is_noise(t):
            drop["noise"] += 1
            continue
        if drop_at:
            t2 = AT_PREFIX_RE.sub("", t).strip()
            if t2 != t:
                drop["at_stripped"] = drop.get("at_stripped", 0) + 1
                if char_len(t2) < 6:      # 剥完只剩几个字，是纯 @ 或寒暄，丢掉
                    drop["at_too_short"] = drop.get("at_too_short", 0) + 1
                    continue
                t = t2
        n = char_len(t)
        if n < 4 or n > 300:
            drop["length"] += 1
            continue
        k = norm_key(t)
        if not k or k in seen_keys:
            drop["dup"] += 1
            continue
        if k in ai_content_keys:      # 换个 rpid 重复出现的同一段 AI 文本
            drop["ai_content"] += 1
            continue
        seen_keys.add(k)
        clean.append({"source": source, "content": t, "topic": topic,
                      "bvid": bvid, "chars": n})

    stats["cleaned"] = len(clean)
    stats["dropped"] = drop
    stats["with_real_topic"] = sum(1 for c in clean if c["topic"])
    stats["topics_distinct"] = len({c["topic"] for c in clean if c["topic"]})

    # 切一个小的 holdout（仅用于观察过拟合，不参与训练）
    rng.shuffle(clean)
    holdout_rows = clean[:holdout]
    train_rows = clean[holdout:]

    # 构造对话样本
    summaries = load_summaries() if with_summary else {}

    def to_sample(rec: dict) -> dict:
        topic = rec["topic"]
        sm = summaries.get(rec.get("bvid") or "")
        if sm:
            # 有摘要：把「视频讲了什么」当输入。这是本项目的核心假设——
            # 模型得看见内容，才写得出不空心的评论。
            user = USER_SUMMARY.format(title=topic or sm.get("title") or "",
                                       summary=sm["summary"].strip())
            style = "summary"
        elif topic and rng.random() > generic_ratio:
            user = USER_TOPIC.format(topic=topic)
            style = "topic"
        else:
            user = USER_GENERIC
            style = "generic"
        return {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": rec["content"]},
            ],
            "_meta": {"source": rec["source"], "chars": rec["chars"],
                      "bvid": rec.get("bvid") or "",
                      "topic": topic if style != "generic" else "",
                      "style": style},
        }

    train_samples = [to_sample(r) for r in train_rows]
    eval_samples = [to_sample(r) for r in holdout_rows]

    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "sft_train.jsonl"
    eval_path = out_dir / "sft_eval_holdout.jsonl"
    for path, rows in ((train_path, train_samples), (eval_path, eval_samples)):
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 统计快照
    lens = sorted(c["chars"] for c in clean)
    stats["len_min"] = lens[0] if lens else 0
    stats["len_median"] = lens[len(lens) // 2] if lens else 0
    stats["len_max"] = lens[-1] if lens else 0
    stats["train"] = len(train_samples)
    stats["holdout"] = len(eval_samples)
    stats["style_counts"] = {
        k: sum(1 for s in train_samples if s["_meta"]["style"] == k)
        for k in ("summary", "topic", "generic")
    }
    stats["summaries_loaded"] = len(summaries)
    stats["train_sha256"] = hashlib.sha256(train_path.read_bytes()).hexdigest()
    stats["train_path"] = str(train_path)
    stats["eval_path"] = str(eval_path)

    (out_dir / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"D:\dsh-ai\lora\data", help="输出目录")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--generic-ratio", type=float, default=0.35,
                    help="用泛化指令（无话题）的比例")
    ap.add_argument("--holdout", type=int, default=20)
    ap.add_argument("--no-replies", action="store_true", help="丢掉全部二级回复")
    ap.add_argument("--strip-reply-prefix", action="store_true", help="剥掉二级回复的「回复 @某人 :」前缀")
    ap.add_argument("--drop-at", action="store_true",
                    help="剥掉行首的 @某人（扩采数据里 11% 是这种回复，不是顶层评论）")
    ap.add_argument("--with-summary", action="store_true",
                    help="用 data/summaries/ 里的视频摘要当输入（没有摘要的评论回落到标题式/泛化式）")
    ap.add_argument("--tag", default="", help="写进 stats 方便区分")

    a = ap.parse_args()
    stats = build(Path(a.out), seed=a.seed,
                  generic_ratio=a.generic_ratio, holdout=a.holdout,
        no_replies=a.no_replies, strip_reply_prefix=a.strip_reply_prefix,
        with_summary=a.with_summary, drop_at=a.drop_at)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
