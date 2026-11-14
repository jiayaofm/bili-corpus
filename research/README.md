# research —— 探索期脚本归档

这里的脚本是正式流水线（`pipeline/`）形成之前的**过程产物**，
保留下来是为了记录方法是怎么一步步试出来的，**不保证可运行**。

正式流程请看根目录的 `run_all.py` 和 `pipeline/`。

## 为什么不能直接跑

这些脚本用 `Path(__file__).parent` 定位项目根目录。
搬到 `research/` 之后，它们会把 `research/` 当成根，找不到 `data/`。

如果要复现某个脚本，把它拷回项目根目录再运行：

```bash
copy research\run_v3.py .
python run_v3.py
```

## 各脚本做了什么

### 特征探索
| 脚本 | 作用 |
|---|---|
| `label_ai.py` | 规则打分器（v1–v4 迭代），最终特征权重写在注释里 |
| `find_ai.py` | 按账号等级切片，验证「二级号 = AI」这个假设（结论：不成立） |
| `mine_keywords.py` | 用 n-gram + log-odds 统计挖 AI 特征词（结论：挖出来全是话题词，失败） |
| `test_features.py` | 第一轮候选特征测试（26 个候选） |
| `test_features2.py` | 第二轮：账号名规则 + 句式候选 |
| `test_closure.py` | 「句子闭合度」特征测试，即 H 组的来源 |

### 语料与标注
| 脚本 | 作用 |
|---|---|
| `analyze.py` | 早期指标分析（长度分布、标点密度、表情率） |
| `preview.py` | 快速预览原始评论 |
| `probe_level.py` | 检查 `level` 字段缺失的原因（v2 采集器的 bug） |
| `make_blind.py` | 手工版盲标表生成（s03 的前身） |
| `merge_labels.py` | 手工版标注回收（s04 的前身） |
| `recover_labels.py` | 一次数据事故的补救：CSV 被覆盖后按内容前缀恢复人工标注 |

### 实验
| 脚本 | 作用 |
|---|---|
| `run_v1.py` | 第一版：TF-IDF 聚类 + 手工特征分类（结论：silhouette 0.037，聚类失败） |
| `run_v2.py` | 特征组累积消融 + PR/ROC + 阈值扫描 |
| `run_v3.py` | 首次用人工真值训练（无循环性） |
| `run_v3_ablation.py` | 人工真值下的特征组消融 |
| `run_v4_cluster.py` | 跨领域文字聚类（AMI 0.234，但靠话题词） |
| `run_v4b_style_cluster.py` | 风格特征聚类对照（silhouette 0.29，但与分区无关） |
| `weighted_eval.py` | 加权评估原型，已抽成 `pipeline/evaluate.py` |

### 采集与登录
| 脚本 | 作用 |
|---|---|
| `bili_login.py` | 第一版扫码登录，**已废弃**——第三方库解析 cookie 有 bug，产出的凭证为空 |

## 三个值得记住的失败

1. **统计挖词不可行**：只有 19 条正样本时，n-gram 挖出来的「特征词」全是
   视频本身的话题词（`小肥鱼`、`900多块`），不是说话方式。
2. **聚类不能按分区**：每个分区只有 1 个视频，话题与分区完全共线，
   TF-IDF 聚出来的是话题，不是体裁。
3. **标签循环性**：早期用规则生成的标签训练模型，得到 F1=1.000 的假结果——
   模型在复现规则，不是在识别 AI。换人工盲标后才拿到真实数字。
