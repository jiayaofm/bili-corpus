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

# 指令模板。system 固定，user 二选一。
SYSTEM = "你在 B 站评论区发一条真实的短评。"
USER_TOPIC = "关于「{topic}」这个话题，说一句你的看法。"
USER_GENERIC = "说一句你此刻想说的话。"

# ---------------------------------------------------------------- 清洗规则

EMOJI_RE = re.compile(
    "[" "\U0001F300-\U0001FAFF" "\U00002600-\U000027BF"
    "\U0001F000-\U0001F2FF" "\U0000FE00-\U0000FE0F" "\U0000200D"
    "]+"
)
AT_RE = re.compile(r"@[^\s@]{1,30}")
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
    for p in sorted(glob.glob(str(ROOT / "data" / "BV*_like.jsonl"))):
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
                        "_is_reply": True,
                    })
    return rows


# ---------------------------------------------------------------- 主逻辑

def build(out_dir: Path, seed: int = 42, generic_ratio: float = 0.35,
          holdout: int = 20) -> dict:
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
    for d in raw:
        rp = str(d.get("rpid", "")).strip()
        t = (d.get("video_title") or "").strip()
        if rp and t:
            title_by_rpid[rp] = t

    # 候选池：jsonl 里未被标为 AI 的 + gt 里标为人人的
    pool: list[tuple[str, str, str]] = []  # (source, content, topic)
    seen_rpid: set[str] = set()
    bot_skipped = 0

    for d in raw:
        rp = str(d.get("rpid", "")).strip()
        content = d.get("content") or ""
        if not rp or rp in seen_rpid:
            continue
        if rp in ai_rpids:          # 去掉已确认的 AI 评论
            continue
        if d.get("_is_reply") and (d.get("user") or "") in BOT_USERS:
            bot_skipped += 1        # 二级回复里 AI 号自己回的
            continue
        seen_rpid.add(rp)
        pool.append(("jsonl" if not d.get("_is_reply") else "reply",
                     content, title_by_rpid.get(rp, "")))

    stats["bot_user_skipped"] = bot_skipped

    for r in gt:
        rp = str(r["rpid"]).strip()
        if rp in seen_rpid:
            continue
        if str(r["label"]).strip() != "0":
            continue
        seen_rpid.add(rp)
        pool.append(("gt", r.get("content") or "", title_by_rpid.get(rp, "")))

    stats["pool_after_ai_filter"] = len(pool)

    # 清洗 + 去重
    clean: list[dict] = []
    seen_keys: set[str] = set()
    drop = {"noise": 0, "length": 0, "dup": 0, "empty": 0, "ai_content": 0}
    for source, content, topic in pool:
        t = (content or "").strip()
        if not t:
            drop["empty"] += 1
            continue
        if is_noise(t):
            drop["noise"] += 1
            continue
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
                      "chars": n})

    stats["cleaned"] = len(clean)
    stats["dropped"] = drop
    stats["with_real_topic"] = sum(1 for c in clean if c["topic"])
    stats["topics_distinct"] = len({c["topic"] for c in clean if c["topic"]})

    # 切一个小的 holdout（仅用于观察过拟合，不参与训练）
    rng.shuffle(clean)
    holdout_rows = clean[:holdout]
    train_rows = clean[holdout:]

    # 构造对话样本
    def to_sample(rec: dict) -> dict:
        topic = rec["topic"]
        # 有真实话题时，多数用话题式；否则/少数用泛化式
        use_topic = bool(topic) and rng.random() > generic_ratio
        user = USER_TOPIC.format(topic=topic) if use_topic else USER_GENERIC
        return {
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user},
                {"role": "assistant", "content": rec["content"]},
            ],
            "_meta": {"source": rec["source"], "chars": rec["chars"],
                      "topic": topic if use_topic else "",
                      "style": "topic" if use_topic else "generic"},
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
        "topic": sum(1 for s in train_samples if s["_meta"]["style"] == "topic"),
        "generic": sum(1 for s in train_samples if s["_meta"]["style"] == "generic"),
    }
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
    a = ap.parse_args()
    stats = build(Path(a.out), seed=a.seed,
                  generic_ratio=a.generic_ratio, holdout=a.holdout)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.exit(main())
