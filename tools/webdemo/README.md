# 课程展示网站

一个零依赖的本地网站，五页展示这个项目的四个方向。

## 怎么启动

```powershell
cd E:\DSH\bili-corpus
D:\dsh-ai\lora\venv\Scripts\python.exe -X utf8 -u tools\webdemo\server.py
```

**不用设任何环境变量。** 脚本自己会找模型缓存（依次试 `D:\dsh-ai\lora\hf`、
`~\.cache\huggingface`、工作区内的 `.bili-home`），并自动开启离线模式。

启动时会打印找到的缓存路径：

```
  模型缓存: D:\dsh-ai\lora\hf
```

如果你确实要用别的位置，显式设 `HF_HOME` 即可，脚本会尊重你的设置（用 `setdefault`）。

**必须用 `D:\dsh-ai\lora\venv` 那个 Python**，不能用项目里的 `.venv`：

| venv | sklearn | torch | 能不能跑 |
|---|:---:|:---:|---|
| `E:\DSH\bili-corpus\.venv` | ✓ | ✗ | 分类能用，**生成不能用** |
| `D:\dsh-ai\lora\venv` | ✓ | ✓ | **全都能用** ← 用这个 |

启动后模型在后台预热（约 6~60 秒）。前四页立即可用，生成页会显示「预热中」，
好了变成「模型已就绪」。

> **曾经踩过的坑**：如果启动时 shell 里没 `HF_HOME`，而脚本也不自己设，
> 就会报 `OSError: We couldn't connect to 'https://huggingface.co'` ——
> 模型缓存在 D 盘，默认路径找不到，transformers 就去联网下载，而这台机器连不上。
> 现在脚本自己处理了，不会再出现。

## 五个页面

| 页 | 讲什么 | 能不能交互 |
|---|---|---|
| **概览** | 语料 9169 条、标注 263 条、管线八阶段 | — |
| **分类器** | 输入任意评论 → 实时打分 + 判断依据 + 改写建议 | ✓ 15~150 毫秒一条 |
| **聚类** | 八种发言类型 + AI 占比 + 两张 PCA 图 | 静态 |
| **关联规则** | 前提 → 是 AI 的规则表 + 支持度/置信度图 | 静态 |
| **生成** | 选视频 → 看摘要 → 生成 3 条 → 和真人对照 | ✓ 约 20 秒 3 条 |

## 演示脚本（约 5 分钟）

```
1  概览页       「我们从 B站 真实评论区取了 9169 条」
2  分类器页     点「示例·像 AI」→ 99%，看判断依据和改写建议
                再点「示例·像真人」→ 2%，对比一下就出来了
3  聚类页       「AI 挤在长评簇 81%，短句簇只有 3.2%，p < 1e-5」
4  关联规则页   「字数≥36 + 有@ ≥1 → 95% 是 AI，提升度 3.97 倍」
5  生成页       选一个视频，点生成，20 秒后看 3 条候选 + 真人对照
```

## 演示前的检查清单

```
□ 服务启动了（控制台看到 → http://127.0.0.1:8848）
□ 生成页显示「模型已就绪」而不是「预热中」
□ 分类器页点一次「判断」，确认有数字（首次可能 500 毫秒）
□ 浏览器缩放 100%，窗口宽度 ≥ 1200（页面在窄屏会变单栏）
□ 五个 tab 都点一遍，确认图都加载了
```

## 排练用的自动模式

```
http://127.0.0.1:8848/?auto=1#generate
```

自动点一次生成，不用手点。**正式演示时别用**，手动点更有参与感。

## 直接用网址切页

```
http://127.0.0.1:8848/#overview
http://127.0.0.1:8848/#classify
http://127.0.0.1:8848/#cluster
http://127.0.0.1:8848/#assoc
http://127.0.0.1:8848/#generate
```

投屏时可以直接开某一页，不用现场点。

## 出问题了怎么办

| 症状 | 原因 | 处理 |
|---|---|---|
| 页面空、没数字 | JS 报错 | F12 看 Console；多半是我改代码时手滑了 |
| 生成页一直「预热中」 | 显存被占 / 模型加载失败 | 看 `state\webdemo.err.log` |
| 端口被占 | 上次没关干净 | `Get-CimInstance Win32_Process -Filter "Name='python.exe'"` 后杀掉 |
| 关联规则页空 | `state/rules.json` 没了 | `python tools\assoc_rules.py --json state/rules.json` |

**杀进程的正确姿势**（Windows 允许重复绑定同一端口，光关窗口不一定干净）：

```powershell
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
  Where-Object { $_.CommandLine -like '*webdemo*server.py*' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

## 文件

```
tools/webdemo/server.py    标准库 http.server，约 400 行
tools/webdemo/index.html   单页，五个 tab，无外部依赖
```

不联网、不装包、不用 Node 构建。改完 `index.html` 刷新即可生效，
改 `server.py` 要重启。
