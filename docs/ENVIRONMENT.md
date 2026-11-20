# 运行环境

> 本文档说明这个项目的 Python 环境怎么建、怎么用、有哪些坑。
> 最后核实：2026-11-18。

---

## 一句话

**所有 Python 命令都用项目自己的 `.venv`，不要用系统的 `python`，也不要用 DSH 运行时的 python。**

```powershell
# 项目根目录下
.\.venv\Scripts\python.exe -X utf8 <脚本>
```

加 `-X utf8` 是因为 Windows 默认编码会让中文输出变乱码。

---

## 环境概况

| 项 | 值 |
|---|---|
| 位置 | `E:\DSH\bili-corpus\.venv` |
| Python | **3.12.14** |
| 建法 | `python -m venv .venv`（从 DSH 运行时的 python 派生，但**互不依赖**） |
| 包数量 | 42 个（见 `requirements.lock.txt`） |

### 关键包版本

```
numpy        2.5.5
scipy        1.18.1
scikit-learn 1.9.1
pandas       3.1.1
matplotlib   3.11.2
bilibili-api-python  17.4.2
```

---

## 这些包是怎么来的（换机器时必看）

**不是一条 `pip install -r` 装出来的。** 当时 PyPI 网络不可用，实测：

```
pypi.org                    单次响应  57 秒
pypi.tuna.tsinghua.edu.cn   单次响应  30 秒
mirrors.aliyun.com/pypi     单次响应 245 秒
```

在这种速度下装十几个包要几个小时。所以分成三路：

| 包 | 来源 |
|---|---|
| numpy / scipy / pandas / sklearn（+ cloudpickle / narwhals 等） | **从 `D:\dsh-ai\lora\venv` 直接复制**（同为 Python 3.12.14，二进制兼容） |
| matplotlib（+ contourpy / cycler / fonttools / kiwisolver / pyparsing） | **从 DSH 运行时的 site-packages 复制** |
| bilibili-api-python / curl_cffi / qrcode / lxml 等 | pip 装（极慢但最终成功） |

**换机器时不能照抄这个流程**（那边不一定有可借的环境）。正常做法是：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果网络还是慢，就分多次装，每次装两三个包。

---

## 常用命令

```powershell
$py = '.\.venv\Scripts\python.exe'

# 走完整流水线
& $py -X utf8 run_all.py

# 只跑某一步
& $py -X utf8 pipeline\s06_classify.py

# 改写流程自检（不调 API，85 项断言）
& $py -X utf8 tools\run_rewrite_demo.py

# 显存门槛纯函数测试
& $py -X utf8 tools\test_preflight.py

# 显存预检
& $py -X utf8 tools\preflight.py
```

---

## 采集：两套并存

| 方式 | 实现 | 状态 |
|---|---|---|
| **DSH 插件**（推荐） | `E:\DSH\dsh-bili-whale`（Node），匿名免登录 | ✅ 已验证可爬评论 |
| 项目自带 | `bili_harvest.py`（Python，需 `bilibili_api` + 凭证） | 可用但已不首选 |

插件爬评论：

```
node E:\DSH\dsh-bili-whale\tools\bili.mjs comments <BV> --ps 20 --pn 1 --sort hot --json
```

**为什么改用它**：匿名可读、免登录、自带 412 风控处理、输出直接是 JSON。

`bilibili_api` 仍然装着（`research/bili_login.py` 等少数脚本要用），但**不是分析流水线的依赖**——`pipeline/` 下没有任何文件 import 它。

---

## 坑（都踩过）

### 1. 系统 `python` 是 Windows 应用商店的占位程序

```
C:\Users\<用户>\AppData\Local\Microsoft\WindowsApps\python.exe
```

跑起来会弹商店。**永远用 `.venv\Scripts\python.exe`。**

### 2. DSH 运行时的 python 会丢包

它在 DSH 重新配置时被重建，**装过的包会消失**（本项目实测丢过 sklearn / scipy / matplotlib / bilibili_api）。

**所以不要往那里装东西。** 项目依赖一律进 `.venv`。

### 3. `-X utf8` 不能省

Windows 默认 cp936，中文写文件/打印可能乱码或抛 `UnicodeEncodeError`。

### 4. `.venv` 不入库

已在 `.gitignore` 里。各自本地建，版本对齐看 `requirements.lock.txt`。

### 5. Python 进程只能写工作区（**最容易踩**）

DSH 沙箱下，**Python 子进程只能往工作区 `E:\DSH\bili-corpus` 里写**。实测：

```
E:\DSH\bili-corpus\...          ✓ 可写
C:\Users\<你>\.bili-ai\...      ✗ PermissionError
C:\Users\<你>\.dsh\...          ✗ PermissionError
D:\dsh-ai\...                   ✗ PermissionError
```

而 **pwsh 和它启动的 node 不受这个限制**：

```
pwsh 写 ~/.bili-ai              ✓
node（pwsh 启动）写 ~/.bili-ai   ✓
Python 写 ~/.bili-ai            ✗
```

**后果**：不能直接 `subprocess.run(["node", "bili.mjs", ...])` ——
插件会把运行时状态写进 `~/.bili-ai/`，第一步 `open traffic-state.json.lock` 就 EPERM。

**解法**（`tools/fetch_materials.py` 用的就是这招）：
把 `USERPROFILE` / `DSH_HOME` / `TEMP` 全部重定向到工作区内的 `.bili-home/`，
再把凭据复制一份进去。插件就把状态写在那儿，Python 写得到。

```python
env.update({
    "USERPROFILE": str(ROOT / ".bili-home"),
    "DSH_HOME":    str(ROOT / ".bili-home" / ".dsh"),
    "TEMP":        str(ROOT / ".tmp"),
    "TMP":         str(ROOT / ".tmp"),
    "DSH_NODE":    r"C:\Program Files\nodejs\node.exe",
})
```

⚠️ **`.bili-home/.bili-ai/credentials.json` 是 B站 凭据副本**，已在 `.gitignore` 里。
换机器时**必须自己重新扫码登录**，不要从任何地方拷贝这个文件。

### 6. Windows 执行策略禁止跑 `.ps1`

```
无法加载文件 ...ps1，因为在此系统上禁止运行脚本
```

所以采集脚本都写成 `.py`，不要写 `.ps1`。

---

## 采集：两套并存

| 方式 | 实现 | 状态 |
|---|---|---|
| **DSH 插件**（推荐） | `E:\DSH\dsh-bili-whale`（Node），扫码登录 | ✅ 已验证 |
| 项目自带 | `bili_harvest.py`（Python，需 `bilibili_api` + 凭证） | 可用但已不首选 |

插件命令（在 pwsh 里直接跑，**不要在 Python 里跑**，见坑 5）：

```powershell
$node = 'C:\Program Files\nodejs\node.exe'
$bili = 'E:\DSH\dsh-bili-whale\tools\bili.mjs'

& $node $bili comments  <BV> --ps 20 --sort hot --json   # 热评（匿名可读）
& $node $bili video     <BV>                             # 简介+标签+字幕+弹幕
& $node $bili subtitles <BV> --out <目录>                 # 字幕落盘
& $node $bili debrief   <BV> --no-frames                 # 一条龙素材（含弹幕热区）
```

抓素材的正式入口是 `python tools\fetch_materials.py`（内部已处理坑 5）。

---

## 自检清单

新环境建完，跑这三条，全过才算好：

```powershell
& $py -X utf8 -c "import numpy,scipy,sklearn,pandas,matplotlib,bilibili_api;print('core ok')"

& $py -X utf8 tools\run_rewrite_demo.py      # 应输出「结论：全部通过」

& $py -X utf8 tools\test_preflight.py        # 应输出「ALL PASS」
```
