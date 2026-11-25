# -*- coding: utf-8 -*-
"""测一测那套「去 AI 味 / 去鱼腥味」提示词到底有没有用。

做法：同一个基座模型，同一个话题，生成两组评论：
  A 组：裸提示（"写一条B站评论"）—— 模拟没做人设优化的默认输出
  B 组：套上 docs/人设提示词-v1.md —— 模拟优化后的人设

然后比较两组的表层特征，以及 scorer v4 给的 AI 分位。
判断标准不是"读起来像不像"，而是提示词里那些硬指标有没有真的变。
"""
import json
import re
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.stdout.reconfigure(encoding="utf-8")

import pipeline.features as F  # noqa: E402
from comment_server import ClassifierV3, ensure_base, BASE_TOK, BASE_MODEL  # noqa: E402

TOPICS = [
    "四大网红探店仿膳，皇上要是吃到这样的宫廷菜，估计要全给咔嚓了！",
    "为什么说「下饭」这个词特别穷酸？",
    "一个人吃饭的时候看什么视频最解压",
    "盘点那些年把我们看笑的老电影桥段",
]

BARE = "你在 B站 看了一个视频，标题是《{t}》。写一条评论发在评论区。"

FULL_PROMPT = """你是 B站 的一个普通用户，喜欢在这个平台上刷视频、发评论。

看完视频后，随手在评论区打一条话。就是普通网友那种随手一打，不是写文章，不是做总结。

死规矩：
1. 不要用句号。句子之间用逗号，或者直接空开。
2. 结尾不要加标点。
3. 不要用「」直角引号。
4. 不要用破折号 ——。
5. 不要用分号 ；。
6. 不要用 Markdown 符号（**、#、- 列表）。
7. 不要用「不是 A，是 B」这种对仗。
8. 不要用「…是真的」「比…更/都」这种强调结构。
9. 不要用「关键是」「说白了」「说到底」「恰恰」「反而」「而是」。
10. 不要用「有点 + 形容词」。
11. 不要自称「本鱼」「本鲸」「本喵」这类。
12. 不要暴露自己是被派来干活的（我这个型号、别让我停机、读字幕看的）。
13. 不要用「捏」「呀」「啦」「喵」收尾装可爱。

要写的具体样子：
- 长度看内容，别硬凑。一句话能说完就一句话，真有话说就多说两句。
- 分句要长，不要切成碎块。人说话是一口气说完一段。
- 多重复字、多带语气词（哈哈哈哈、卧槽、啊、吧、呗）。
- 约五分之一的评论带 B站 表情，写 [doge] [笑哭] [打call] 这种。
- 数字和英文照抄，不要转汉字。
- 写你看视频时真的想到的东西，具体到哪一句、哪个画面。不要总结视频，不要升华。

对照：
✗ 语言是把词和语法拼一块的沟通系统。人靠它表达自己的想法。
✓ 语言就是把词和语法一块拼起来的交流系统，人靠它表达自己的意思

✗ 这期视频讲了三种盒饭，第一份是真材实料，后面两份纯粹凑数。
✓ 这期讲了三种盒饭。头一份是真材实料，后头两份就图个好看

写完检查：有句号吗？结尾有标点吗？有「」——；** 吗？有「不是A是B」吗？
是不是切成三字一句了？像不像一个人随口打的？

现在看这个视频：《{t}》
在评论区打一条。只输出评论本身，不要任何解释。"""


def main():
    z = np.load(ROOT / "state" / "scorer_v4.npz", allow_pickle=True)
    clf = ClassifierV3(ROOT / "state" / "scorer_v4.npz")
    ref = json.loads((ROOT / "state" / "ai_ref.json").read_text(encoding="utf-8"))["values"]
    ref = np.array(sorted(ref))

    # 复用 comment_server 里那一份底座，别再加载第二个实例——
    # Windows 页面文件会扛不住（实测 OSError 1455）。
    if not ensure_base():
        raise SystemExit("底座加载失败")
    import comment_server as CS
    tok, mdl = CS.BASE_TOK, CS.BASE_MODEL
    tok.padding_side = "left"

    gen = {}
    for name, tpl in (("A", BARE), ("B", FULL_PROMPT)):
        prompts = []
        for t in TOPICS:
            msgs = [{"role": "user", "content": tpl.format(t=t)}]
            prompts.append(tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False))
        enc = tok(prompts, return_tensors="pt", padding=True).to(mdl.device)
        with torch.inference_mode():
            o = mdl.generate(**enc, max_new_tokens=160, do_sample=True,
                             temperature=0.95, top_p=0.95,
                             pad_token_id=tok.eos_token_id)
        g = o[:, enc["input_ids"].shape[1]:]
        gen[name] = [s.strip().replace("\n", " ") for s in
                     tok.batch_decode(g, skip_special_tokens=True)]

    allt = gen["A"] + gen["B"]
    probs = clf.score(allt)
    pct = [float((ref < p).mean() * 100) for p in probs]

    print("=" * 78)
    for name, label in (("A", "裸提示（默认人设）"), ("B", "套用提示词 v1")):
        print()
        print(f"  【{label}】")
        for i, t in enumerate(gen[name]):
            print(f"    {i + 1}. {t[:78]}")

    print()
    print("=" * 78)
    print("  表层特征对比（越接近「真人」列越好）")
    print()
    print("  %-16s %10s %10s %10s" % ("指标", "A 裸提示", "B 提示词", "真人目标"))
    print("  " + "-" * 52)

    def stat_group(texts):
        per = [F.featurize(t, "") for t in texts]
        return {k: st.median([p[k] for p in per]) for k in per[0]}

    sa, sb = stat_group(gen["A"]), stat_group(gen["B"])
    rows = [
        ("句号每百字", "句号数", 1.19, "低"),
        ("分号数", "分号数", 0.0, "低"),
        ("平均分句长", "平均分句长", 14.76, "高"),
        ("叠字重复", "用字丰富度", None, ""),
    ]
    for label, key, target, _ in rows:
        if key not in sa or target is None:
            continue
        print("  %-16s %10.2f %10.2f %10.2f" % (label, sa[key], sb[key], target))

    for idx, (name, label) in enumerate((("A", "裸提示"), ("B", "提示词"))):
        pass
    print()
    print("  %-16s %10.1f %10.1f" % (
        "AI 分位（中位）",
        st.median(pct[:len(gen["A"])]), st.median(pct[len(gen["A"]):])))
    print("  %-16s %10d %10d" % (
        "判红（≥90）",
        sum(1 for p in pct[:len(gen["A"])] if p >= 90),
        sum(1 for p in pct[len(gen["A"]):] if p >= 90)))
    print()
    print("  参考：真人评论的 AI 分位中位约 62，误判率阈值 90 分位时约 5%")


if __name__ == "__main__":
    main()
