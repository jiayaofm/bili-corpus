# -*- coding: utf-8 -*-
"""
发布前校验：按 .gitignore 语义推演「哪些文件会被提交」

为什么需要这个脚本：
    有些环境（比如这台机器）没有安装 git，无法直接跑 `git status`。
    本脚本实现 gitignore 的核心匹配语义，用来在不依赖 git 的前提下
    确认敏感文件确实不会被提交。

    注意：**git 本身仍是权威**。装了 git 之后请再跑一次：
        git add -A --dry-run
        git status --porcelain -uall

用法：
    python tools/verify_publish.py          # 列出会提交 / 被忽略的文件
    python tools/verify_publish.py --brief  # 只输出统计与违规项

匹配规则实现范围（覆盖本仓库用到的写法）：
    · 注释 #、空行
    · 取反 !pattern（被取反的规则优先级更高，后匹配覆盖前匹配）
    · 结尾 / 表示只匹配目录
    · 含 / 的模式锚定到仓库根；不含 / 的模式匹配任意层级的同名文件
    · * 匹配除 / 外任意字符；? 匹配单个字符；** 跨目录
    · [abc] 字符类
    · 父目录被忽略时，子文件无法再被取反包含（git 的真实行为）
"""
import fnmatch
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
GITIGNORE = ROOT / ".gitignore"

# 绝对不允许出现在提交列表里的路径（红线）
FORBIDDEN = [
    "bili_credential.json",
    "qrcode_login.png",
]
FORBIDDEN_PREFIX = [
    "state/",
    "labeling/",
]
# 允许出现在被禁目录里的例外文件
PREFIX_ALLOW = {
    "labeling/README.md",
}
FORBIDDEN_SUFFIX = [
    ".state.json",
]
FORBIDDEN_CONTAINS = [
    "__pycache__",
]

# 必须出现在提交列表里的文件
REQUIRED = [
    "README.md",
    "LICENSE",
    ".gitignore",
    "requirements.txt",
    "config.json",
    "run_all.py",
    "pipeline/features.py",
    "pipeline/evaluate.py",
    "docs/METHOD.md",
    "data/sample/sample.jsonl",
]


def parse_gitignore(path):
    rules = []
    if not path.exists():
        return rules
    for raw in path.read_text(encoding="utf-8").split("\n"):
        line = raw.rstrip()
        if not line or line.startswith("#"):
            continue
        negate = line.startswith("!")
        if negate:
            line = line[1:]
        dir_only = line.endswith("/")
        if dir_only:
            line = line[:-1]
        if not line:
            continue
        anchored = "/" in line
        line = line.lstrip("/") if line.startswith("/") else line
        rules.append({
            "pattern": line,
            "negate": negate,
            "dir_only": dir_only,
            "anchored": anchored,
        })
    return rules


def _to_regex(pat):
    """把 gitignore 通配模式转成正则"""
    out = []
    i = 0
    n = len(pat)
    while i < n:
        ch = pat[i]
        if ch == "*":
            if pat[i:i + 3] == "**/":
                out.append("(?:.*/)?")
                i += 3
                continue
            if pat[i:i + 2] == "**":
                out.append(".*")
                i += 2
                continue
            out.append("[^/]*")
            i += 1
        elif ch == "?":
            out.append("[^/]")
            i += 1
        elif ch == "[":
            j = pat.find("]", i + 1)
            if j < 0:
                out.append(re.escape(ch))
                i += 1
            else:
                out.append(pat[i:j + 1])
                i = j + 1
        else:
            out.append(re.escape(ch))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def _match_rule(rule, rel, is_dir):
    if rule["dir_only"] and not is_dir:
        return False
    pat = rule["pattern"]
    rx = _to_regex(pat)
    if rule["anchored"]:
        return bool(rx.match(rel))
    # 不含 /：匹配任意层级的基名
    parts = rel.split("/")
    return any(rx.match(p) for p in parts)


def _status(rules, rel, is_dir):
    """返回 (ignored, 命中规则)"""
    ignored = False
    hit = None
    for r in rules:
        if _match_rule(r, rel, is_dir):
            ignored = not r["negate"]
            hit = r
    return ignored, hit


def is_ignored(rules, rel, is_dir):
    """父目录被忽略时，子文件一律忽略（git 的真实语义）"""
    parts = rel.split("/")
    for i in range(1, len(parts)):
        anc = "/".join(parts[:i])
        ig, _ = _status(rules, anc, True)
        if ig:
            return True
    ig, _ = _status(rules, rel, is_dir)
    return ig


def collect_files():
    out = []
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT).as_posix()
        if rel.startswith(".git/"):
            continue
        out.append((rel, p.is_dir()))
    return out


def main():
    brief = "--brief" in sys.argv
    rules = parse_gitignore(GITIGNORE)

    kept, dropped = [], []
    for rel, is_dir in collect_files():
        if is_dir:
            continue
        if is_ignored(rules, rel, False):
            dropped.append(rel)
        else:
            kept.append(rel)

    print(f"gitignore 规则 {len(rules)} 条")
    print(f"会被提交 {len(kept)} 个文件，被忽略 {len(dropped)} 个文件\n")

    # ---- 红线检查 ----
    violations = []
    for rel in kept:
        name = rel.split("/")[-1]
        if name in FORBIDDEN:
            violations.append(f"凭证文件会被提交: {rel}")
        for pre in FORBIDDEN_PREFIX:
            if rel.startswith(pre) and rel not in PREFIX_ALLOW:
                violations.append(f"敏感目录会被提交: {rel}")
        for suf in FORBIDDEN_SUFFIX:
            if rel.endswith(suf):
                violations.append(f"状态文件会被提交: {rel}")
        for sub in FORBIDDEN_CONTAINS:
            if sub in rel:
                violations.append(f"缓存文件会被提交: {rel}")

    print("=" * 60)
    if violations:
        print("违规：有敏感文件会被提交")
        for v in violations:
            print("  [X]", v)
    else:
        print("红线检查通过：凭证 / 状态 / 缓存 / 原始语料 / 标注文件均不会被提交")

    missing = [r for r in REQUIRED if r not in kept]
    if missing:
        print("\n警告：以下必需文件被误忽略")
        for m in missing:
            print("  [!]", m)
    else:
        print("必需文件齐全")

    print("=" * 60)

    if not brief:
        print("\n--- 会被提交的文件 ---")
        for rel in kept:
            print(f"  {rel}")
        print("\n--- 被忽略的敏感文件（抽样） ---")
        show = [d for d in dropped if any(
            k in d for k in ("credential", "qrcode", "state.json",
                             "labeling/", "corpus", "manifest", "__pycache__"))]
        for rel in show[:40]:
            print(f"  {rel}")
        if len(show) > 40:
            print(f"  ...（共 {len(show)} 个）")

    return 1 if (violations or missing) else 0


if __name__ == "__main__":
    raise SystemExit(main())
