# -*- coding: utf-8 -*-
"""
改写效果评测：训出来的模型到底有没有把「鱼腥味」去掉，内容有没有丢。

## 四个指标，缺一不可

| 指标 | 测什么 | 为什么要 |
|---|---|---|
| **鱼腥味** | 输出还像不像大肥鱼 | 本项目的核心目标 |
| **内容保留** | 输入里说的东西还在不在 | 早期 pipeline 就是死在这（5.7%） |
| **P(AI)** | 有没有掉回「通用 AI 腔」 | 防止修好一个坏掉另一个 |
| **形态** | 长度/表情/骨架重复 | 输出得像正常人写的 |

只报一个指标一定会骗自己——尤其「鱼腥味降了」通常靠「写得更短更空」就能做到。

## 两种输入方式

```bash
# ① 已有预测结果（不占 GPU，先验指标逻辑用）
python -X utf8 tools/eval_rewrite.py --predictions pred.jsonl

# ② 直接加载 adapter 生成（占 GPU）
python -X utf8 tools/eval_rewrite.py --adapter D:\\dsh-ai\\lora\\out\\qwen3-1.7b-fish \\
    --data data/rewrite/b_dev.jsonl
```

`--predictions` 每行：`{"bvid": "...", "x": "鱼腔输入", "y": "真人目标", "pred": "模型输出"}`

## 鱼腥味检测器怎么来的

用**同一批训练数据**里未参与本评测的样本训一个二分类器：

```
正类  x（鱼腔改写）     负类  y（真人原话）
特征  字符 TF-IDF 1-3gram
```

然后给模型输出打分。**分数越低 = 越不像鱼。**

**注意**：如果模型输出和训练时的 x 高度相似（比如模型学会了「抄输入」），
鱼腥味分数会显得很高——这反而是对的，说明它没改。

## 内容保留怎么算

从输入 x 里抽「可核查的具体信息」：

- 阿拉伯数字（含百分比、金额、单位前缀）
- 拉丁字母/英文词
- 汉字 bigram（去掉纯虚词组合）

然后看这些在输出里还剩多少。同时也报「相对 y 的保留率」当参照——
因为 y 本身就是「正确改写后该有的样子」，理想情况下输出应该跟 y 一样全。

## 拒绝出数

样本数不够时直接报错，不打印虚指标（和 persona_detector.py 一个规矩）。
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EMOJI_RE = re.compile(r"\[[^\[\]]{1,12}\]")
NUM_RE = re.compile(r"\d+(?:\.\d+)?%?")
LATIN_RE = re.compile(r"[A-Za-z]{2,}")

# 纯虚词组合的 bigram 不算「内容」
STOP_CHARS = set("的了是在和有不这那也就都还你我他她它大小上下很太就都没个么什怎因为所以但是如果"
                 "而且然后可以可能应该已经现在时候什么怎么这样那样一些一下这个那个")

MIN_SAMPLES = 30

# 标点归一化：评测时剥掉标点和表情，只剩选词和句法。
# 理由：检测器在原始文本上 AUC 0.956，但「只换标点加表情」的朴素基线
# 就能拿到比真人还低的分数——说明标点贡献了可被作弊的分辨力。
# 剥掉之后 AUC 仍有 0.707，剩下的才是真信号。
_EMOJI_MARK_RE = re.compile(r"\[[^\[\]]{1,12}\]")          # [笑哭] [doge] 这类表情标记
_KEEP_RE = re.compile(r"[^\u4e00-\u9fff0-9A-Za-z]")        # 只留汉字/数字/字母


def normalize_punct(text: str) -> str:
    """剥掉标点、空白和表情标记，只留内容字。"""
    return _KEEP_RE.sub("", _EMOJI_MARK_RE.sub("", text))


# ------------------------------------------------------------------ 特征
def key_terms(text: str) -> set[str]:
    """抽「可核查的具体信息」：数字、英文词、内容 bigram。"""
    terms: set[str] = set()
    for m in NUM_RE.findall(text):
        if len(m) >= 1:
            terms.add(m)
    for m in LATIN_RE.findall(text):
        terms.add(m.lower())
    cjk = re.sub(r"[^\u4e00-\u9fff]", "", text)
    for i in range(len(cjk) - 1):
        bg = cjk[i:i + 2]
        if all(ch in STOP_CHARS for ch in bg):
            continue
        terms.add(bg)
    return terms


def recall_of(src_terms: set[str], out_text: str) -> float:
    if not src_terms:
        return 1.0
    out_terms = key_terms(out_text)
    hit = sum(1 for t in src_terms if t in out_terms)
    return hit / len(src_terms)


def dup_ratio(texts: list[str], prefix: int = 6) -> float:
    """骨架重复：前 N 字相同的比例。"""
    if len(texts) < 2:
        return 0.0
    keys = [t[:prefix] for t in texts]
    uniq = len(set(keys))
    return 1 - uniq / len(keys)


# ------------------------------------------------------------------ 朴素基线
NAIVE_EMO = ["[笑哭]", "[doge]", "[星星眼]", "[吃瓜]"]


def naive_de_fish(text: str, idx: int = 0, emoji_rate: float = 1.0) -> str:
    """朴素基线：只做「换标点 + 加表情」，一个字都不改。

    为什么要这个基线：鱼腥味检测器主要靠句号认人（`。` 权重 +3.66）。
    如果模型只比这条基线好一点点，说明它学的不是「像人」，是「像标点」。

    这是**合格线**：跑不过它，等于没训。
    """
    t = text.replace("。", "，").replace("、", "，")
    t = re.sub(r"\s+", "", t)                 # 鱼腔特有的数字后空格
    t = t.rstrip("，, ")
    if (idx % 100) >= int(emoji_rate * 100):
        return t
    return t + NAIVE_EMO[idx % len(NAIVE_EMO)]


# ------------------------------------------------------------------ 鱼腥味检测器
def fit_fish_detector(xs: list[str], ys: list[str]):
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    texts = xs + ys
    lab = np.array([1] * len(xs) + [0] * len(ys))
    clf = make_pipeline(
        TfidfVectorizer(analyzer="char", ngram_range=(1, 3), min_df=1, sublinear_tf=True),
        LogisticRegression(C=0.3, max_iter=3000, class_weight="balanced"),
    )
    clf.fit(texts, lab)
    return clf


# ------------------------------------------------------------------ 加载
def load_predictions(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not all(k in r for k in ("x", "y", "pred")):
                continue
            rows.append(r)
    return rows


def generate_with_adapter(adapter: Path, data: Path, max_new: int, temp: float) -> list[dict]:
    """加载 adapter 逐条生成。占 GPU，需外部确认。"""
    import os
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # 和 webdemo 一样：不依赖调用者的 shell 环境，自己找模型缓存
    for cand in (Path(r"D:\dsh-ai\lora\hf"), Path.home() / ".cache" / "huggingface"):
        hub = cand / "hub"
        try:
            if hub.is_dir() and any(hub.glob("models--Qwen*")):
                os.environ.setdefault("HF_HOME", str(cand))
                break
        except OSError:
            continue
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    MODEL = "Qwen/Qwen3-1.7B"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                                device_map="auto")
    model = PeftModel.from_pretrained(model, str(adapter)).eval()

    rows = []
    with data.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            msgs = s["messages"]
            prompt_msgs = msgs[:-1]
            target = msgs[-1]["content"]
            # 干净的原评论：内容保留指标只能拿它算。
            # 不能拿 prompt_msgs[-1]，B 变体里混了 200 字视频摘要，
            # 会把「输出没复述摘要」误判成「内容丢了」。
            meta = s.get("_meta") or {}
            clean_x = meta.get("x") or prompt_msgs[-1]["content"]
            text = tok.apply_chat_template(prompt_msgs, tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
            ids = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=max_new, do_sample=True,
                                     temperature=temp, top_p=0.95,
                                     pad_token_id=tok.eos_token_id)
            pred = tok.decode(out[0][ids["input_ids"].shape[1]:],
                              skip_special_tokens=True).strip().replace("\n", " ")
            rows.append({
                "bvid": meta.get("bvid", ""),
                "x": clean_x,
                "y": target,
                "pred": pred,
                "prompt_chars": sum(len(m["content"]) for m in prompt_msgs),
            })
    return rows


# ------------------------------------------------------------------ 报告
def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", type=Path, help="预测结果 jsonl")
    ap.add_argument("--adapter", type=Path, help="LoRA adapter 目录（需 GPU）")
    ap.add_argument("--data", type=Path, help="dev 数据 jsonl（配 --adapter）")
    ap.add_argument("--triggers", type=Path,
                    default=ROOT / "data" / "rewrite" / "b_train.jsonl",
                    help="用来训鱼腥味检测器的训练集")
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--show", type=int, default=6, help="打印几条对照")
    ap.add_argument("--normalize", action="store_true",
                    help="剥掉标点和表情再判断，防止「换标点」作弊")
    ap.add_argument("--save-preds", type=Path, default=None,
                    help="把预测结果存成 jsonl，下次可直接 --predictions 复用")
    a = ap.parse_args()

    print("=" * 70)
    print("  改写效果评测")
    print("=" * 70)

    # 鱼腥味检测器：优先用训练集里的 (x, y) 对
    det = None
    if a.triggers.exists():
        xs, ys = [], []
        with a.triggers.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                s = json.loads(line)
                m = s["messages"]
                # 从 user 里抠出「这条评论：xxx」不方便，直接用内容字段
                xs.append(m[-2]["content"])
                ys.append(m[-1]["content"])
        if a.normalize:
            xs = [normalize_punct(t) for t in xs]
            ys = [normalize_punct(t) for t in ys]
        if len(xs) >= 20:
            det = fit_fish_detector(xs, ys)
            print(f"  鱼腥味检测器：用 {len(xs)} 对训练样本拟合"
                  + ("（已剥标点和表情）" if a.normalize else ""))
    if det is None:
        print("  ⚠️ 没有训练集可拟合鱼腥味检测器，跳过该指标")

    if a.predictions:
        rows = load_predictions(a.predictions)
    elif a.adapter and a.data:
        rows = generate_with_adapter(a.adapter, a.data, a.max_new, a.temp)
    else:
        print("  ❌ 需要 --predictions，或 --adapter + --data")
        return 2

    if a.save_preds:
        a.save_preds.parent.mkdir(parents=True, exist_ok=True)
        with a.save_preds.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({k: r[k] for k in ("bvid", "x", "y", "pred")},
                                   ensure_ascii=False) + "\n")
        print(f"  预测已存：{a.save_preds}")

    if len(rows) < MIN_SAMPLES:
        print()
        print(f"  ❌ 样本只有 {len(rows)} 条，少于 {MIN_SAMPLES}，拒绝出指标")
        return 2

    preds = [r["pred"] for r in rows]
    xs = [r["x"] for r in rows]
    ys = [r["y"] for r in rows]

    # ---- 鱼腥味
    print()
    print("【一】鱼腥味（越低越不像大肥鱼）")
    if det is not None:
        import numpy as np
        # x 是 user 段（含模板文字），用它当「改造前的上界」参考
        if a.normalize:
            xs = [normalize_punct(t) for t in xs]
            preds = [normalize_punct(t) for t in preds]
            ys = [normalize_punct(t) for t in ys]
        fish_x = det.predict_proba(xs)[:, 1]
        fish_p = det.predict_proba(preds)[:, 1]
        fish_y = det.predict_proba(ys)[:, 1]
        # 朴素基线两档：强档（每条都加表情）和匹配档（只有 26% 加，和真人一致）
        base_strong = [naive_de_fish(r["x"], i, 1.0) for i, r in enumerate(rows)]
        base_match = [naive_de_fish(r["x"], i, 0.26) for i, r in enumerate(rows)]
        fish_bs = det.predict_proba(base_strong)[:, 1]
        fish_bm = det.predict_proba(base_match)[:, 1]
        base_preds = base_strong
        fish_b = fish_bm
        print(f"    改造前（输入）   均值 {fish_x.mean():.3f}")
        print(f"    模型输出         均值 {fish_p.mean():.3f}   "
              f"中位 {statistics.median(fish_p):.3f}")
        print(f"    真人目标（下限） 均值 {fish_y.mean():.3f}")
        print(f"    基线·强（每条加表情）  均值 {fish_bs.mean():.3f}")
        print(f"    基线·匹配（26% 加表情） 均值 {fish_bm.mean():.3f}")
        drop = fish_x.mean() - fish_p.mean()
        span = max(1e-6, fish_x.mean() - fish_y.mean())
        bdrop_s = fish_x.mean() - fish_bs.mean()
        bdrop_m = fish_x.mean() - fish_bm.mean()
        print(f"    → 模型降幅 {drop:+.3f}，走完理想距离的 {drop / span * 100:.0f}%")
        print(f"    → 强基线降幅 {bdrop_s:+.3f}（{bdrop_s / span * 100:.0f}%）"
              f"  匹配基线 {bdrop_m:+.3f}（{bdrop_m / span * 100:.0f}%）")
        if fish_bs.mean() < fish_y.mean():
            print("    ⚠️ 只换标点加表情就能低于真人目标——**这个检测器可以被标点欺骗**，")
            print("       所以它不能单独作为「成功」的依据。必须同时看内容保留和形态。")
        if drop < bdrop_m * 0.8:
            print("    ❌ 模型明显弱于匹配基线，改善可能只来自表层特征")
        elif drop < bdrop_m:
            print("    ⚠️ 模型略低于匹配基线，改善的可信度不足")
        else:
            print("    ✅ 模型不低于匹配基线")
    else:
        print("    （跳过）")
        base_preds = []

    # ---- 内容保留
    print()
    print("【二】内容保留（输入里的具体信息还剩多少）")
    rec_x, rec_y = [], []
    for r in rows:
        terms = key_terms(r["x"])
        rec_x.append(recall_of(terms, r["pred"]))
        rec_y.append(recall_of(terms, r["y"]))
    print(f"    模型输出  均值 {statistics.mean(rec_x):.3f}  "
          f"中位 {statistics.median(rec_x):.3f}")
    print(f"    真人目标  均值 {statistics.mean(rec_y):.3f}  "
          f"（这是「改对了」该有的水平）")
    lost = sum(1 for v in rec_x if v < 0.5)
    lost_y = sum(1 for v in rec_y if v < 0.5)
    # 用真人的分布当参照，而不是拍一个 0.5 的绝对阈值。
    # v4_dev 上真人原文自己有 66% 落在 0.5 以下（这批改写幅度本来就大），
    # 用绝对阈值会把「正确的改写」也算成失败。
    print(f"    真人目标里同样低于 0.5 的   {lost_y}/{len(rows)}"
          f"（{lost_y/len(rows)*100:.0f}%）← 这是参照线")
    print(f"    模型低于 0.5 的            {lost}/{len(rows)}"
          f"（{lost/len(rows)*100:.0f}%）")
    if lost <= lost_y * 1.3:
        print("    → 与真人水平相当，不算丢信息")
    else:
        print("    → 明显多于真人，改写时丢了内容")

    # ---- P(AI)
    print()
    print("【三】P(AI)（现有分类器，防掉回通用 AI 腔）")
    try:
        from pipeline.scorer import Scorer
        sc = Scorer()
        pa = [sc.describe(t)["ai_prob"] for t in preds]
        py = [sc.describe(t)["ai_prob"] for t in ys]
        print(f"    模型输出  均值 {statistics.mean(pa):.3f}  "
              f"中位 {statistics.median(pa):.3f}")
        print(f"    真人目标  均值 {statistics.mean(py):.3f}")
        hi = sum(1 for v in pa if v >= 0.5)
        print(f"    判成 AI 的   {hi}/{len(rows)}（{hi/len(rows)*100:.0f}%）")
    except Exception as e:
        print(f"    （分类器不可用：{type(e).__name__}）")

    # ---- 形态
    print()
    print("【四】形态")
    for name, arr in (("模型输出", preds), ("真人目标", ys)):
        lens = [len(t) for t in arr]
        emo = sum(1 for t in arr if EMOJI_RE.search(t)) / len(arr) * 100
        print(f"    {name}  中位 {statistics.median(lens):.0f} 字  "
              f"<10字 {sum(1 for l in lens if l < 10)/len(lens)*100:.0f}%  "
              f"表情 {emo:.0f}%  骨架重复 {dup_ratio(arr)*100:.1f}%")

    # ---- 抽样对照
    if a.show:
        print()
        print("【五】抽样对照")
        for r in rows[:a.show]:
            print(f"    输入  {r['x'][:60]}")
            print(f"    输出  {r['pred'][:60]}")
            print(f"    真人  {r['y'][:60]}")
            print()

    print("  ⚠️ 提醒：只看「鱼腥味下降」会被「写得更短更空」骗到，")
    print("          必须同时看内容保留。两者要一起改善才算真的成功。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
