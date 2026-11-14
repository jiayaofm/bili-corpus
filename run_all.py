# -*- coding: utf-8 -*-
"""
流水线总入口

用法：
    python run_all.py                 # 全流程，已完成的阶段自动跳过
    python run_all.py --from s05      # 从某阶段开始
    python run_all.py --only s06      # 只跑某阶段
    python run_all.py --force         # 忽略 manifest，全部重跑
    python run_all.py --list          # 只显示状态

退出码：0 全部成功；1 某阶段失败；2 需要人工标注（gate）
"""
import importlib
import sys
import traceback
from datetime import datetime
from pathlib import Path

# 阶段脚本内部使用 `import common` / `import features` 同级导入，
# 把 pipeline/ 加入 sys.path（直接运行单阶段时 Python 会自动加，这里补上）
_PKG_DIR = Path(__file__).resolve().parent / "pipeline"
if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))

from pipeline.common import is_done, read_manifest

# (短码, manifest 键, 模块名, 中文名)
STAGES = [
    ("s01", "s01_fetch", "s01_fetch", "采集"),
    ("s02", "s02_corpus", "s02_corpus", "语料构建"),
    ("s03", "s03_sample", "s03_sample", "盲标抽样"),
    ("s04", "s04_labels", "s04_labels", "回收标注（人工 gate）"),
    ("s05", "s05_features", "s05_features", "特征矩阵"),
    ("s06", "s06_classify", "s06_classify", "分类"),
    ("s07", "s07_cluster", "s07_cluster", "聚类"),
    ("s08", "s08_report", "s08_report", "汇总报告"),
]
CODES = [s[0] for s in STAGES]


def parse_args(argv):
    opts = {"from": None, "only": None, "force": False, "list": False}
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--force":
            opts["force"] = True
        elif a == "--list":
            opts["list"] = True
        elif a in ("--from", "--only") and i + 1 < len(argv):
            opts[a.lstrip("-")] = argv[i + 1]
            i += 1
        elif a in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        i += 1
    return opts


def resolve(code_or_key):
    """接受 s05 或 s05_features 两种写法，返回短码"""
    if code_or_key in CODES:
        return code_or_key
    for code, key, _, _ in STAGES:
        if code_or_key == key:
            return code
    return None


def status_table():
    m = read_manifest()
    print("\n" + "=" * 74)
    print(f"{'阶段':<6}{'名称':<24}{'状态':<8}产物")
    print("-" * 74)
    n_done = 0
    for code, key, _, label in STAGES:
        done = is_done(key)
        n_done += 1 if done else 0
        rec = m.get(key)
        if rec:
            outs = rec.get("outputs", {})
            extra = f"{len(outs)} 个文件 @ {rec.get('at','')[:19]}"
        else:
            extra = "-"
        print(f"{code:<6}{label:<24}{'完成' if done else '未完成':<8}{extra}")
    print("=" * 74)
    print(f"进度 {n_done}/{len(STAGES)}    manifest: {read_manifest and ''}"
          f"state/manifest.json")
    return n_done


def main(argv):
    opts = parse_args(argv)

    if opts["list"]:
        status_table()
        return 0

    force = opts["force"]
    sel_only = resolve(opts["only"]) if opts["only"] else None
    sel_from = resolve(opts["from"]) if opts["from"] else None

    if opts["only"] and not sel_only:
        print(f"未知阶段: {opts['only']}，可选: {', '.join(CODES)}")
        return 1
    if opts["from"] and not sel_from:
        print(f"未知阶段: {opts['from']}，可选: {', '.join(CODES)}")
        return 1

    selected = []
    for code, key, modname, label in STAGES:
        if sel_only:
            if code != sel_only:
                continue
        elif sel_from:
            if CODES.index(code) < CODES.index(sel_from):
                continue
        selected.append((code, key, modname, label))

    print(f"\n开始 {datetime.now():%Y-%m-%d %H:%M:%S}　"
          f"计划: {', '.join(s[0] for s in selected)}"
          + ("（强制重跑）" if force else ""))

    completed, skipped = [], []

    for code, key, modname, label in selected:
        if not force and is_done(key):
            print(f"\n[{code}] {label} —— 已完成，跳过")
            skipped.append(code)
            continue

        print(f"\n[{code}] {label} —— 开始（{datetime.now():%H:%M:%S}）")
        try:
            mod = importlib.import_module(f"pipeline.{modname}")
            rc = mod.main(force=force) if _accepts_force(mod) else mod.main()
        except Exception:
            print("\n[异常] 阶段执行出错：")
            traceback.print_exc()
            print(f"\n流水线在 {code} 中止。")
            status_table()
            return 1

        if rc == 2:
            print(f"\n[{code}] 需要人工介入，流水线暂停。")
            status_table()
            return 2
        if rc not in (0, None):
            print(f"\n[{code}] 返回码 {rc}，流水线中止。")
            status_table()
            return 1

        completed.append(code)

    print(f"\n结束 {datetime.now():%Y-%m-%d %H:%M:%S}　"
          f"本次执行 {len(completed)} 个，跳过 {len(skipped)} 个")
    status_table()
    return 0


def _accepts_force(mod):
    import inspect
    try:
        return "force" in inspect.signature(mod.main).parameters
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
