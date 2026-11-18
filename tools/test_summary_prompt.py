# -*- coding: utf-8 -*-
"""
测一件事：**已经训过的 LoRA，喂摘要能不能直接用**，还是必须重训。

用户的问题：现在这版喂了摘要不是能写人话吗，为什么要重训？

对。所以要测两个条件：
    A  只给标题（训练时的格式，分布内）
    B  标题 + 摘要（训练时没见过这种输入，分布外）

如果 B 明显更好，说明不用重训，改输入就行。
如果 B 反而更差或没变化，说明 LoRA 忽略了摘要，需要重训。

用法：
    python tools/test_summary_prompt.py --n 6
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.stdout.reconfigure(encoding="utf-8")

# 测试用的临时摘要。等用户的插件产出真摘要后替换。
SUMMARIES = {
    "BV1ArZnYUE9a": (
        "《晓艳天气预报》",
        "游戏",
        "一个恶搞天气预报的整活栏目。主播用正经的新闻播报腔调念完全离谱的天气内容，"
        "配夸张音效和表情，包袱全在「一本正经说瞎话」这个反差上。",
    ),
    "BV1edWM63Ewd": (
        "Claude Fable 5.2实测：94分刷爆了我们的私有题库，但我真用不起了！",
        "科技",
        "作者用自建的私有编程题库测 Claude Fable 5.2，跑出 94 分，比 Opus 5.5 高四分多。"
        "但跑一遍题库要六小时、八十一美元，日常用不起，作者最后建议日常还是用 Opus 5.5。",
    ),
    "BV1w6ZEYrEsb": (
        '为什么中国人对一道菜的最高评价是"下饭"?',
        "美食",
        "从饮食结构和主食习惯聊为什么中国人评价一道菜的最高标准是「下饭」，"
        "说到米饭在日常饮食里的地位、汤汁和咸鲜口味对下饭的作用。",
    ),
}

SYS = "你在 B 站评论区发一条真实的短评。"

USER_TITLE = "关于「{title}」这个话题，说一句你的看法。"
USER_SUMMARY = "关于「{title}」这个话题，说一句你的看法。\n视频内容：{summary}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=r"D:\dsh-ai\lora\out\qwen3-1.7b-gen2")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    a = ap.parse_args()

    import torch
    from eval_generation import load_model, generate_many

    print("加载模型…")
    tok, model = load_model("Qwen/Qwen3-1.7B", a.adapter)

    for bv, (title, dom, summary) in SUMMARIES.items():
        print("=" * 74)
        print("《%s》 [%s]" % (title[:44], dom))
        print("-" * 74)

        for label, user in (
            ("A 只给标题（训练分布内）", USER_TITLE.format(title=title)),
            ("B 标题+摘要（分布外）", USER_SUMMARY.format(title=title, summary=summary)),
        ):
            msgs = [{"role": "system", "content": SYS},
                    {"role": "user", "content": user}]
            outs = generate_many(tok, model, msgs, a.n, a.temperature, a.max_new_tokens)
            print("【%s】" % label)
            for t in outs:
                t = (t or "").strip()
                if t:
                    print("   [%2d字] %s" % (len(t), t.replace("\n", " ")[:62]))
            print()


if __name__ == "__main__":
    main()
