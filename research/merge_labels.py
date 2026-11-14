# -*- coding: utf-8 -*-
"""
回收盲标结果 → ground truth，并算出规则打分器的真实精确率

用户标注格式（写在「编号」列前缀）：
   x17  → 明确认为第 17 条是 AI
   m23  → 认为第 23 条可能是 AI
   45   → 无前缀，视为真人

两种统计口径：
   严格：x=AI，m=不确定（不计入）
   宽松：x=AI，m=也计入 AI
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).parent
LAB = ROOT / "labeling"
OUT = ROOT / "analysis"
OUT.mkdir(exist_ok=True)

PREFIX_RE = re.compile(r"^\s*([A-Za-z]*)\s*(\d+)\s*$")


def main():
    rows = list(csv.DictReader(
        (LAB / "blind_151.csv").open(encoding="utf-8-sig", newline="")))
    amap = json.loads((LAB / "blind_151_map.json").read_text(encoding="utf-8"))

    # 编号 → rpid / 是否候选层（统一成 int 再比较，兼容 Excel 去掉前导零）
    id2rpid = {int(it["编号"]): it["rpid"] for it in amap["items"]}
    id2cand = {int(it["编号"]): it["was_candidate"] for it in amap["items"]}
    rpid2id = {v: k for k, v in id2rpid.items()}

    certain, maybe, human, unparsed = set(), set(), set(), []
    content = {}
    for r in rows:
        k = (r["编号"] or "").strip()
        m = PREFIX_RE.match(k)
        if not m:
            unparsed.append(k)
            continue
        pre, num = m.group(1).lower(), int(m.group(2))
        content[num] = r["评论内容"]
        if pre == "x":
            certain.add(num)
        elif pre == "m":
            maybe.add(num)
        elif pre == "":
            human.add(num)
        else:
            unparsed.append(k)

    total = len(certain) + len(maybe) + len(human)
    print(f"解析成功 {total} 条：明确AI(x) {len(certain)}  "
          f"疑似AI(m) {len(maybe)}  真人 {len(human)}")
    if unparsed:
        print(f"注意：{len(unparsed)} 条编号无法解析：{unparsed[:10]}")

    # ---------- 分层统计 ----------
    cand_ids = [i for i in id2cand if id2cand[i] == 1]
    rand_ids = [i for i in id2cand if id2cand[i] == 0]
    n_cand_total = amap["candidate_count"]
    n_rand_total = amap["random_count"]
    pool_rest = 368 - n_cand_total - amap["excluded_known_human"]

    def tally(ids, pos):
        return sum(1 for i in ids if i in pos)

    lines = ["# 盲标结果（人工 ground truth）", ""]
    lines.append(f"- 盲标样本 {total} 条（规则候选层 {len(cand_ids)} + 随机层 {len(rand_ids)}）")
    lines.append(f"- 你的标注：明确是AI **{len(certain)}** 条，疑似 **{len(maybe)}** 条，"
                 f"真人 {len(human)} 条")
    lines.append("")
    lines.append("> 标注方式：`x` = 明确是 AI，`m` = 可能是 AI，无前缀 = 真人")

    lines.append("\n## 一、规则打分器的真实精确率\n")
    lines.append("这是本项目第一个**不带循环性**的数字。\n")
    lines.append("| 口径 | 候选层样本 | 你认同是AI | 精确率 | 误报数 |")
    lines.append("|---|---|---|---|---|")
    strict_hit = tally(cand_ids, certain)
    loose_hit = tally(cand_ids, certain | maybe)
    n_c = len(cand_ids)
    lines.append(f"| 严格（只算 x） | {n_c} | {strict_hit} | "
                 f"{strict_hit/n_c*100:.1f}% | {n_c-strict_hit} |")
    lines.append(f"| 宽松（x+m） | {n_c} | {loose_hit} | "
                 f"{loose_hit/n_c*100:.1f}% | {n_c-loose_hit} |")
    print(f"\n候选层 {n_c} 条 → 严格认同 {strict_hit} 条 ({strict_hit/n_c*100:.1f}%)，"
          f"宽松 {loose_hit} 条 ({loose_hit/n_c*100:.1f}%)")

    # ---------- 随机层：真实分布 ----------
    lines.append("\n## 二、随机层的 AI 密度（反映真实分布）\n")
    lines.append("| 口径 | 随机层样本 | 判为AI | AI占比 |")
    lines.append("|---|---|---|---|")
    rs = tally(rand_ids, certain)
    rl = tally(rand_ids, certain | maybe)
    n_r = len(rand_ids)
    lines.append(f"| 严格 | {n_r} | {rs} | {rs/n_r*100:.1f}% |")
    lines.append(f"| 宽松 | {n_r} | {rl} | {rl/n_r*100:.1f}% |")
    print(f"随机层 {n_r} 条 → 严格 {rs} 条 ({rs/n_r*100:.1f}%)，宽松 {rl} 条 ({rl/n_r*100:.1f}%)")

    # 全库加权估计
    scale = pool_rest / n_r
    est_s = len(certain) + rs * scale - rs   # 候选层已知 + 随机层外推的其余
    lines.append("")
    lines.append(f"**全库 368 条中 AI 占比加权估计**："
                 f"严格约 {(strict_hit + rs*scale)/368*100:.1f}%，"
                 f"宽松约 {(loose_hit + rl*scale)/368*100:.1f}%")

    # ---------- 分歧样例 ----------
    disagree = [i for i in cand_ids if i in human]
    lines.append(f"\n## 三、规则误报样例（{len(disagree)} 条我判 AI、你判真人）\n")
    if disagree:
        lines.append("| 编号 | 内容 |")
        lines.append("|---|---|")
        for i in sorted(disagree)[:25]:
            txt = content.get(i, "").replace("\n", " ").replace("|", "\\|")[:110]
            lines.append(f"| {i} | {txt} |")

    # 反向：我漏掉的
    missed = [i for i in rand_ids if i in certain or i in maybe]
    lines.append(f"\n## 四、规则漏报样例（{len(missed)} 条你判 AI、我判真人）\n")
    if missed:
        lines.append("| 编号 | 标注 | 内容 |")
        lines.append("|---|---|---|")
        for i in sorted(missed)[:25]:
            tag = "x 明确" if i in certain else "m 疑似"
            txt = content.get(i, "").replace("\n", " ").replace("|", "\\|")[:110]
            lines.append(f"| {i} | {tag} | {txt} |")

    # ---------- 保存 ----------
    gt = {
        "certain_ai": [id2rpid[i] for i in sorted(certain)],
        "maybe_ai": [id2rpid[i] for i in sorted(maybe)],
        "human": [id2rpid[i] for i in sorted(human)],
        "note": "x=明确AI, m=疑似AI, 无前缀=真人",
    }
    (LAB / "ground_truth.json").write_text(
        json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")

    # CSV 版，方便后续直接读
    with (LAB / "ground_truth.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rpid", "label", "strength", "was_candidate", "content"])
        for i in sorted(certain | maybe | human):
            lbl = 1 if (i in certain or i in maybe) else 0
            st = "x" if i in certain else ("m" if i in maybe else "")
            w.writerow([id2rpid[i], lbl, st, id2cand[i], content.get(i, "")])

    (OUT / "blind_result.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n真值：{LAB / 'ground_truth.csv'}")
    print(f"报告：{OUT / 'blind_result.md'}")


if __name__ == "__main__":
    main()
