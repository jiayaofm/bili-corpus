# -*- coding: utf-8 -*-
"""把预标注 CSV 转成方便阅读的 Markdown 复核表"""
import csv
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
_arg = sys.argv[1] if len(sys.argv) > 1 else "labeling/ai_prelabels.csv"
_out = sys.argv[2] if len(sys.argv) > 2 else "labeling/ai_prelabels_round1.md"
_title = sys.argv[3] if len(sys.argv) > 3 else "第 1 轮"
SRC = ROOT / _arg
OUT = ROOT / _out

VERDICT = {"1": "AI", "0": "真人", "": "不确定"}
CONF = {"high": "高", "mid": "中", "low": "低"}


def main():
    rows = list(csv.DictReader(SRC.open(encoding="utf-8-sig", newline="")))
    ai = [r for r in rows if r["label"] == "1"]
    un = [r for r in rows if r["label"] == ""]
    hu = [r for r in rows if r["label"] == "0"]

    L = []
    L.append(f"# AI 预标注结果 · {_title}")
    L.append("")
    L.append(f"> {len(rows)} 条未标注评论的盲判草稿　|　生成于 "
             f"{datetime.now():%Y-%m-%d %H:%M}")
    L.append(">")
    L.append("> **修正方式**：在我判定那行后面加 `→ 0` 或 `→ 1` 即可，例如")
    L.append("> `判定：**AI** → 0` 表示你认为这条其实是真人。")
    L.append("")
    L.append("## 概要")
    L.append("")
    L.append("| 判定 | 条数 | 建议动作 |")
    L.append("|---|---|---|")
    L.append(f"| **AI** | {len(ai)} | 重点复核，判错会污染训练集 |")
    L.append(f"| **不确定** | {len(un)} | 请逐条定夺 |")
    L.append(f"| 真人 | {len(hu)} | 抽 10% 扫一眼即可 |")
    L.append("")
    L.append("---")
    L.append("")

    # ---- AI ----
    L.append(f"## 一、判为 AI（{len(ai)} 条）")
    L.append("")
    for r in ai:
        L.append(f"### #{int(r['id']):03d} · 置信 {CONF.get(r['conf'], r['conf'])} · `{r['domain']}`")
        L.append("")
        L.append(f"> {r['content']}")
        L.append("")
        L.append(f"判定：**AI**　判据：{r['reason']}")
        L.append("")
        L.append("---")
        L.append("")

    # ---- 不确定 ----
    L.append(f"## 二、不确定（{len(un)} 条）")
    L.append("")
    for r in un:
        L.append(f"### #{int(r['id']):03d} · `{r['domain']}`")
        L.append("")
        L.append(f"> {r['content']}")
        L.append("")
        L.append(f"判定：**?**　疑点：{r['reason']}")
        L.append("")
        L.append("---")
        L.append("")

    # ---- 真人 ----
    L.append(f"## 三、判为真人（{len(hu)} 条）")
    L.append("")
    L.append("> 抽查即可。若发现明显是 AI 的，在「修正」列写 `→ 1`。")
    L.append("")
    L.append("| # | 内容 | 修正 |")
    L.append("|---|---|---|")
    for r in hu:
        c = r["content"].replace("\n", " ").replace("|", "\\|")
        if len(c) > 68:
            c = c[:68] + "…"
        L.append(f"| {int(r['id']):03d} | {c} |  |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 判据说明")
    L.append("")
    L.append("我用的规律（来自你之前的标注和纠正）：")
    L.append("")
    L.append("| AI 特征 | 真人特征 |")
    L.append("|---|---|")
    L.append("| 句末闭合（以句号结尾） | 说半句、留白、不结尾 |")
    L.append("| 精确复述数字后给判断 | 提数字但提问或感叹 |")
    L.append("| @ 出现在句末（装饰性收尾） | @ 在开头（指名） |")
    L.append("| 对仗、排比、结构对称 | 口语碎片、语气词（啊/吧/咯） |")
    L.append("| 总结型判断「说明…」「这就是…」 | 玩梗、跑题、个人经历 |")
    L.append("| 无 B站表情、无错别字 | 表情、错别字、颜文字、刷屏 |")
    L.append("| 措辞绝对、干练 | 限定词（也就/估计/可能/那么一些） |")
    L.append("")
    L.append("**已知弱点**：表情能伪装；短评 AI 容易漏判。")

    OUT.write_text("\n".join(L), encoding="utf-8", newline="\n")
    print(f"写入 {OUT}")
    print(f"  AI {len(ai)} / 不确定 {len(un)} / 真人 {len(hu)}")
    print(f"  共 {len(L)} 行，{OUT.stat().st_size} 字节")


if __name__ == "__main__":
    main()
