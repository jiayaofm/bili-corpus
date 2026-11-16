# -*- coding: utf-8 -*-
"""
统一打分器 —— 给「指导-生成-打分-修改」workflow 用

和 audit.py 的区别：

  audit.py  只回答「这条是不是 AI」，输出一个融合分。当检测器够用，
            但当**奖励函数**有三个毛病：
              1. 离散、有尖峰，没有梯度
              2. 容易被作弊
              3. 分不出「没有 AI 标记」和「像人」

  本模块  输出两个互相独立的数：

              reward   优化目标（越高越好）
                       规则分 + 手工特征随机森林，看标点和句式统计

              heldout  验收指标，不参与优化
                       TF-IDF 字 n-gram + 逻辑回归，看**用词**
                       两种表示完全不同：改标点骗得到前者，骗不到后者

  判定：奖励涨了、验收没涨 → 钻空子，打回重写。

为什么验收要用 TF-IDF（试错记录）：
  第一版把「逗号数/分句数/短分句占比/用字丰富度」当作留出特征，
  结果全被标点驱动——把「——」换成「，」就能让它们同时"变人"，
  作弊案例 4/4 被放行，机制完全失效。
  第二版只留「语气词密度/提问密度」，作弊是抓住了，但太窄：
  真实改写只要没加语气词就被误判。
  最终改用 TF-IDF 判别器，实测分离度：
      作弊（只改标点） 验收改善 +0.04
      真实改写（换词） 验收改善 +0.15 ~ 0.19

用法：
    python pipeline/scorer.py "一条评论"
    python pipeline/scorer.py --compare 原文本 改写后
"""
import csv
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT))

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from label_ai import score as rule_score  # noqa: E402
from pipeline import features as F  # noqa: E402

GT = ROOT / "labeling" / "ground_truth.csv"

# 判定「钻空子」的两个门槛（实测校准：作弊 +0.04 / 真改写 +0.15）
GAMING_REWARD_MIN = 0.05     # 奖励至少涨这么多才值得怀疑
GAMING_HELDOUT_MIN = 0.08    # 验收至少涨这么多才算真改善

# 退化输入的下限。
# 不设这个门槛会有灾难性后果：空串没有任何 AI 标记，规则分 0、
# 奖励模型判真人、TF-IDF 也判真人，于是 describe("") 给 0.999 分，
# compare(原文, "") 直接 verdict=ok —— 空字符串成了「最优改写」。
# 极端短的文本同理（单个字）。
MIN_LEN = 2

# 规则命中项 → 修改建议（供 workflow 的「修改」环节定向执行）
#
# 注意：只能写规则真的会产出的 hit 名。
# 早先这里有过「句末闭合」「反诘对比|倒装对仗」两条，但这两项是
# features.py 的特征名，规则并不产出对应 hit，所以它们永远不触发。
# 现在改由下面的 FEATURE_ADVICE 按特征值直接判断。
ADVICE = [
    (r"破折号", "去掉破折号：改成逗号断开，或干脆删掉那半句"),
    (r"「」", "去掉「」引号：口语里没人这么强调"),
    (r"分号", "去掉分号：拆成两个短句"),
    (r"数字对比|数字密度", "删掉多余的精确数字，最多留一个"),
    (r"不是A是B", "别用「不是A，是B」这种对仗，像在总结"),
    (r"总结腔|本质|内核", "删掉「本质是」「总而言之」这类总结句"),
    (r"自称本鱼|助手视角|角色自称", "别自称角色，用普通第一人称或第三人称"),
    (r"超长", "拆成更短的句子，别一口气说完"),
    (r"比…更|我是真的", "别用比较级强化，直接说事"),
]

# 特征值 → 修改建议（规则不产 hit、但特征能直接看出来）
FEATURE_ADVICE = [
    ("句末闭合", lambda v: v >= 1, "句末别收死：句号改逗号或省略号，或删掉最后的结论句"),
    ("反诘对比", lambda v: v >= 1, "别构造对仗反问句，直接说事"),
    ("倒装对仗", lambda v: v >= 1, "别构造对仗句，打散成平常语序"),
    ("分句数", lambda v: v >= 6, "分句太多了，删掉一半，别一口气说完"),
    ("短分句占比", lambda v: v == 0, "全是没有短句的长句，插几个四五字的短句"),
]


class Scorer:
    """双表示打分器：手工特征算奖励，TF-IDF 算验收。"""

    def __init__(self, seed=42):
        self.cols = F.cols_of(F.CLASSIFY_GROUPS)
        texts, y = self._load()

        # 奖励模型：手工特征的随机森林
        X = np.array([[F.featurize(t, "")[c] for c in self.cols] for t in texts], float)
        self.reward_model = RandomForestClassifier(
            n_estimators=500, class_weight="balanced", random_state=seed)
        self.reward_model.fit(X, y)

        # 验收模型：字 n-gram TF-IDF + 逻辑回归（看用词，不看标点统计）
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                                   min_df=2, max_features=6000)
        Xt = self.vec.fit_transform(texts)
        self.heldout_model = LogisticRegression(
            max_iter=2000, class_weight="balanced").fit(Xt, y)

        self.n_train = len(y)
        self.n_pos = int(y.sum())
        self.n_feat = len(self.cols)

    def _load(self):
        texts, labels = [], []
        for r in csv.DictReader(GT.open(encoding="utf-8-sig", newline="")):
            texts.append(r["content"])
            labels.append(int(r["label"]))
        return texts, np.array(labels)

    @staticmethod
    def _norm_rule(s):
        """规则分归一到 0-1，60 分以上视为天花板"""
        return float(min(max(s / 60.0, 0.0), 1.0))

    def _ai_reward(self, text):
        x = np.array([[F.featurize(text, "")[c] for c in self.cols]], float)
        return float(self.reward_model.predict_proba(x)[0, 1])

    def _ai_heldout(self, text):
        return float(self.heldout_model.predict_proba(
            self.vec.transform([text]))[0, 1])

    def describe(self, text, username=""):
        rule, hits = rule_score({"content": text, "user": username})
        ai_reward = self._ai_reward(text)
        ai_held = self._ai_heldout(text)
        feats = F.featurize(text, "")

        # 奖励 = 规则越轻越好 + 奖励模型越判真人越好
        reward = 0.5 * (1.0 - self._norm_rule(rule)) + 0.5 * (1.0 - ai_reward)

        degenerate = len(text.strip()) < MIN_LEN
        if degenerate:
            # 空文本没有任何 AI 标记，按公式会得到接近满分。
            # 直接压到 0：没写过东西不可能是好改写。
            reward = 0.0

        advice = []
        for pat, tip in ADVICE:
            if any(re.search(pat, h) for h in hits):
                advice.append(tip)
        for feat, cond, tip in FEATURE_ADVICE:
            if feat in feats and cond(feats[feat]) and tip not in advice:
                advice.append(tip)

        return {
            "text": text,
            "ai_rule": round(float(rule), 1),
            "ai_prob": round(ai_reward, 4),
            "human_prob": round(1.0 - ai_reward, 4),
            "heldout_ai": round(ai_held, 4),
            "reward": round(reward, 4),
            "hits": hits,
            "advice": advice,
            "degenerate": degenerate,
        }

    def compare(self, before, after):
        b = self.describe(before)
        a = self.describe(after)
        d_reward = a["reward"] - b["reward"]
        d_held = b["heldout_ai"] - a["heldout_ai"]   # 正值 = 用词更像人
        gaming = d_reward >= GAMING_REWARD_MIN and d_held < GAMING_HELDOUT_MIN
        if a["degenerate"] or b["degenerate"]:
            # 退化文本不可能是有效改写；即使算出来的奖励是涨的也不作数
            verdict = "no-gain"
        elif gaming:
            verdict = "gaming"
        elif d_reward > 0 and d_held >= GAMING_HELDOUT_MIN:
            verdict = "ok"
        else:
            verdict = "no-gain"
        return {
            "before": b,
            "after": a,
            "d_reward": round(d_reward, 4),
            "d_held_human": round(d_held, 4),
            "gaming_suspect": gaming,
            "verdict": verdict,
        }


def _fmt(d):
    return (f"规则 {d['ai_rule']:>6.1f}  P(AI) {d['ai_prob']:.3f}  "
            f"验收 {d['heldout_ai']:.3f}  奖励 {d['reward']:.3f}\n"
            f"  {d['text'][:80]}")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="AI 味打分器（双表示 + 防作弊验收）")
    ap.add_argument("text", nargs="?", help="待评文本")
    ap.add_argument("--compare", nargs=2, metavar=("原文", "改写"), help="对比改写前后")
    a = ap.parse_args()

    s = Scorer()
    print(f"[模型] 训练 {s.n_train} 条（AI {s.n_pos}）")
    print(f"       奖励表示：{s.n_feat} 个手工特征 + 规则分")
    print(f"       验收表示：TF-IDF 字 2-4 gram + 逻辑回归\n")

    if a.compare:
        r = s.compare(*a.compare)
        print("改写前:", _fmt(r["before"]))
        if r["before"]["hits"]:
            print("   命中:", " / ".join(r["before"]["hits"]))
        print("\n改写后:", _fmt(r["after"]))
        if r["after"]["hits"]:
            print("   命中:", " / ".join(r["after"]["hits"]))
        print(f"\n奖励 {r['d_reward']:+.4f}　验收(用词) {r['d_held_human']:+.4f}")
        if r["verdict"] == "gaming":
            print("  ⚠ 钻空子：奖励涨了，但用词没变人 → 打回重写")
        elif r["verdict"] == "ok":
            print("  ✓ 两项同向改善")
        else:
            print("  · 未取得有效改善")
        if r["after"]["advice"]:
            print("\n还需改：")
            for t in r["after"]["advice"]:
                print("  -", t)
    elif a.text:
        d = s.describe(a.text)
        print(_fmt(d))
        if d["hits"]:
            print("  命中:", " / ".join(d["hits"]))
        if d["advice"]:
            print("  建议:")
            for t in d["advice"]:
                print("   -", t)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
