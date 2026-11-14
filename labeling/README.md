# 标注目录（不入库）

本目录被 `.gitignore` 排除，**不会提交到仓库**。

原因：这里的 CSV 包含 B站 评论原文和 `@用户名` 提及，
属于其他用户的生成内容，不适合随仓库分发。

## 文件说明

| 文件 | 由谁生成 | 作用 |
|---|---|---|
| `blind_sample.csv` | `pipeline/s03_sample.py` | 待人工标注的盲标表 |
| `blind_map.json` | `pipeline/s03_sample.py` | 编号 ↔ rpid 对照，以及抽样参数 |
| `ground_truth.csv` | `pipeline/s04_labels.py` | 回收后的真值（rpid / label / strength / was_candidate / content） |
| `blind_151.csv`、`human_marked.json`、`candidates*.csv` | 早期探索脚本 | 历史产物，仅作参考 |

## 怎么自己生成

```bash
# 1. 生成盲标表（分层抽样）
python run_all.py --only s03

# 2. 打开 labeling/blind_sample.csv，在「判断」列填：
#       1 或 x  -> 是 AI
#       0 或空   -> 真人
#       ? 或 m   -> 拿不准
#    也可以直接在「编号」前加 x / m 前缀，两种写法都支持。
#    注意：不要看规则分数、账号名、点赞数——盲标才有意义。

# 3. 回收标注
python run_all.py --only s04
```

`s04` 会在缺少标注时以退出码 2 中止流水线，这是设计上的「人工闸门」。
