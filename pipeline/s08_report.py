# -*- coding: utf-8 -*-
"""
s08 汇总报告

把 s02/s04/s05/s06/s07 的产物汇总成 analysis/REPORT.md。
不重复计算，只读取已有结果与 manifest。
"""
import csv
import json
from collections import Counter
from datetime import datetime

import common
from common import P, banner, info, read_jsonl, read_manifest, record

STAGE = "s08_report"
OUT_REL = "analysis/REPORT.md"


def _read_md(rel):
    p = common.ROOT / rel
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")


def _table_section(md, heading):
    """从 md 里截取某个二级标题下的内容"""
    if not md:
        return None
    i = md.find(heading)
    if i < 0:
        return None
    j = md.find("\n## ", i + 1)
    return md[i:j if j > 0 else len(md)].strip()


def main(force=False):
    banner("s08 汇总报告")

    corpus_path = common.ROOT / "data/corpus.jsonl"
    gt_path = P["labeling"] / "ground_truth.csv"
    if not corpus_path.exists():
        info("[失败] 缺少 data/corpus.jsonl")
        return 1

    corpus = read_jsonl(corpus_path)
    doms = Counter(r["domain"] for r in corpus)
    bvids = Counter(r["bvid"] for r in corpus)
    lens = [len(r["content"]) for r in corpus]

    gt = []
    if gt_path.exists():
        with gt_path.open(encoding="utf-8-sig", newline="") as f:
            gt = list(csv.DictReader(f))
    n_pos = sum(1 for r in gt if r["label"] == "1")
    n_neg = sum(1 for r in gt if r["label"] == "0")
    n_x = sum(1 for r in gt if r["strength"] == "x")
    n_m = sum(1 for r in gt if r["strength"] == "m")

    m = read_manifest()
    cls = m.get("s06_classify", {})
    clu = m.get("s07_cluster", {})

    L = []
    L.append("# B站评论区「AI 发言」识别与分析")
    L.append("")
    L.append(f"> 生成时间：{datetime.now():%Y-%m-%d %H:%M}　|　"
             f"流水线：run_all.py　|　阶段：s08")
    L.append("")
    L.append("## 目录")
    L.append("")
    L.append("1. [研究问题](#一研究问题)")
    L.append("2. [数据](#二数据)")
    L.append("3. [标注](#三标注)")
    L.append("4. [分类结果](#四分类结果)")
    L.append("5. [聚类结果](#五聚类结果)")
    L.append("6. [局限性](#六局限性)")
    L.append("7. [复现方式](#七复现方式)")
    L.append("")

    L.append("## 一、研究问题")
    L.append("")
    L.append("目标：识别 B站 评论区中由 AI（大模型人格号）生成的发言，"
             "并分析其语言特征。")
    L.append("")
    L.append("对应课程要求的两个方向：**分类**（AI vs 真人）与"
             "**聚类**（评论形态/体裁的无监督发现）。")
    L.append("")

    L.append("## 二、数据")
    L.append("")
    L.append(f"- 评论总数：**{len(corpus)}** 条")
    L.append(f"- 视频数：{len(bvids)} 个，覆盖 {len(doms)} 个分区")
    L.append(f"- 平均长度 {sum(lens)/len(lens):.1f} 字，"
             f"中位 {sorted(lens)[len(lens)//2]} 字")
    L.append("")
    L.append("| 分区 | 条数 |")
    L.append("|---|---|")
    for d, n in doms.most_common():
        L.append(f"| {d} | {n} |")
    L.append("")
    L.append("采集方式：B站公开评论接口，按点赞排序，每个视频上限固定条数，"
             "含断点续传与风控冷却。")
    L.append("")

    L.append("## 三、标注")
    L.append("")
    L.append(f"- 人工盲标样本：**{len(gt)}** 条（分层抽样：规则候选层全取 + 随机层抽样）")
    L.append(f"- 明确判为 AI（x）：{n_x} 条；疑似（m）：{n_m} 条；判为真人：{n_neg} 条")
    L.append("")
    L.append("**关键设计：盲标。** 标注时隐藏规则分数、命中特征、账号名与点赞数，"
             "避免标注者被既有判断带偏。")
    L.append("")

    # 分类
    cls_md = _read_md("analysis/v5_classify.md")
    L.append("## 四、分类结果")
    L.append("")

    sec = _table_section(cls_md, "## 一、抽样偏差与加权还原")
    if sec:
        L.append(sec.replace("## 一、抽样偏差与加权还原", "### 抽样偏差与加权还原"))
        L.append("")

    L.append("> 下表为**加权口径（主结论）**。样本中 AI 密度 23.6%，"
             "全库真实约 7.9%；未加权指标会明显高估召回率。")
    L.append("")

    sec = _table_section(cls_md, "## 二、模型对比")
    if sec:
        L.append(sec.replace("## 二、模型对比（主结论：加权口径）", "### 模型对比"))
        L.append("")
    sec = _table_section(cls_md, "## 三、与规则打分器对比")
    if sec:
        L.append(sec.replace("## 三、与规则打分器对比（加权口径）",
                             "### 与规则基线对比"))
        L.append("")
    sec = _table_section(cls_md, "## 六、特征组消融")
    if sec:
        L.append(sec.replace("## 六、特征组消融", "### 特征组消融"))
        L.append("")
    if cls.get("extra"):
        e = cls["extra"]
        L.append(f"最佳模型：**{e.get('best')}**，"
                 f"加权 F1 = {e.get('f1_weighted')}"
                 f"（样本口径 {e.get('f1_sample')}）。")
        L.append("")
    L.append("完整报告见 `analysis/v5_classify.md`。")
    L.append("")

    # 聚类
    clu_md = _read_md("analysis/v5_cluster.md")
    L.append("## 五、聚类结果")
    L.append("")
    if clu.get("extra"):
        e = clu["extra"]
        L.append("| 表示方法 | silhouette | AMI（与分区吻合） |")
        L.append("|---|---|---|")
        L.append(f"| 文字 TF-IDF | — | {e.get('text_ami')} |")
        L.append(f"| 风格特征 | {e.get('style_sil')} | {e.get('style_ami')} |")
        L.append("")
    sec = _table_section(clu_md, "## 三、结论")
    if sec:
        L.append(sec.replace("## 三、结论", "### 结论"))
        L.append("")
    L.append("完整报告见 `analysis/v5_cluster.md`。")
    L.append("")

    L.append("## 六、局限性")
    L.append("")
    L.append("1. **话题与分区共线**：每个分区只采 1 个视频，"
             "无法区分「分区效应」与「话题效应」。要严格论证体裁聚类，"
             "需每分区 2–3 个不同话题的视频。")
    L.append(f"2. **正样本规模有限**：分类任务正类仅 {n_pos} 条，"
             "指标置信区间较宽。")
    L.append("3. **标注者单一**：全部标注由一人完成，存在个人判断偏差，"
             "未计算标注者间一致性（Kappa）。")
    L.append("4. **采集受限**：B站 接口存在 IP 级频次限制，"
             "语料规模受采集节奏约束。")
    L.append("5. **账号名是混淆变量**：真人粉丝与 AI 账号共用命名风格，"
             "已从分类特征中默认剔除。")
    L.append("6. **标注标准随批次变化**：第一批 151 条判 AI 占 36.4%，"
             "第二批 157 条降至 23.6%，说明标注门槛在收紧；"
             "跨批次指标不可直接比较。")
    L.append("")

    L.append("## 七、复现方式")
    L.append("")
    L.append("```bash")
    L.append("python run_all.py              # 全流程，已完成的阶段自动跳过")
    L.append("python run_all.py --from s05   # 从特征阶段开始重跑")
    L.append("python run_all.py --only s06   # 只跑分类")
    L.append("python run_all.py --force      # 全部强制重跑")
    L.append("```")
    L.append("")
    L.append("所有随机种子、视频列表、采样比例集中在 `config.json`。")
    L.append("")

    out = common.ROOT / OUT_REL
    out.write_text("\n".join(L), encoding="utf-8")
    info(f"报告 {OUT_REL}（{len(L)} 行）")

    record(STAGE, {OUT_REL: len(L)},
           extra={"corpus": len(corpus), "labeled": len(gt)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
