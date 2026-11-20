# -*- coding: utf-8 -*-
"""
抓视频素材，供模型写摘要用。

## 素材只用这三路

    ✓ 简介        up 自己写的
    ✓ 官方字幕    它到底说了什么（几百行台词，最准）
    ✓ 标签        up 自己打的分类

## 刻意不取

    ✗ 热评    那是我们要预测的标签，取了就是泄题
    ✗ 弹幕    v1 先不做（解析成本高，价值次之）

`bili.mjs video` 这条命令本身不含热评，所以泄题风险在源头被结构性排除。

## 为什么要「假 home」

DSH 沙箱下 **Python 进程只能写工作区**。而插件把运行时状态写在
`~/.bili-ai/`（traffic-state.json 的锁文件等），Python 直接调它会：

    EPERM: operation not permitted, open '...\\.bili-ai\\traffic-state.json.lock'

实测结论：

    pwsh 写 ~/.bili-ai            ✓
    node（pwsh 启动）写 ~/.bili-ai  ✓
    Python 写 ~/.bili-ai          ✗ PermissionError

所以这里把 `USERPROFILE` / `DSH_HOME` / `TEMP` 全部重定向到工作区内的
`.bili-home/`，插件就把状态写在那里 —— Python 写得到。

代价：凭据要复制一份进 `.bili-home/.bili-ai/`。
**该目录已在 .gitignore 里，不会入库。**

## 用法

    python tools/fetch_materials.py                 # config.json 里全部视频
    python tools/fetch_materials.py BV1w6ZEYrEsb    # 指定视频
    python tools/fetch_materials.py --force         # 重抓

产物：

    data/materials/<BV>.json          结构化素材
    data/subs/<BV>-ai-zh-plain.txt    字幕纯文本（模型读的就是它）
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BILI = Path(r"E:\DSH\dsh-bili-whale\tools\bili.mjs")
NODE = Path(r"C:\Program Files\nodejs\node.exe")

MAT_DIR = ROOT / "data" / "materials"
SUB_DIR = ROOT / "data" / "subs"
FAKE_HOME = ROOT / ".bili-home"


def ensure_fake_home() -> dict:
    """建工作区内的假 home，把凭据和配置复制过去，返回子进程要用的 env。"""
    (FAKE_HOME / ".bili-ai").mkdir(parents=True, exist_ok=True)
    (FAKE_HOME / ".dsh").mkdir(parents=True, exist_ok=True)
    (ROOT / ".tmp").mkdir(parents=True, exist_ok=True)

    real_home = Path(os.environ.get("USERPROFILE") or os.path.expanduser("~"))
    for src, dst in [
        (real_home / ".bili-ai" / "credentials.json", FAKE_HOME / ".bili-ai" / "credentials.json"),
        (real_home / ".dsh" / "bili-whale.config.json", FAKE_HOME / ".dsh" / "bili-whale.config.json"),
    ]:
        if src.exists():
            shutil.copy2(src, dst)

    env = dict(os.environ)
    env.update({
        "USERPROFILE": str(FAKE_HOME),
        "DSH_HOME": str(FAKE_HOME / ".dsh"),
        "TEMP": str(ROOT / ".tmp"),
        "TMP": str(ROOT / ".tmp"),
        "DSH_NODE": str(NODE),
    })
    return env


def run_bili(*args: str, env: dict, timeout: int = 180) -> tuple[int, str]:
    cmd = [str(NODE), str(BILI), *args]
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=timeout, env=env)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


# --------------------------------------------------------------------------- #
def parse_video(out: str) -> dict:
    """解析 `bili.mjs video <BV>` 的分段文本输出。"""
    m: dict = {"title": "", "up": "", "tname": "", "duration": 0,
               "aid": 0, "cid": 0, "desc": "", "tags": [], "cover": ""}
    lines = out.splitlines()

    for i, ln in enumerate(lines):
        if ln.strip().startswith("=== 元信息"):
            for nxt in lines[i + 1:]:
                if nxt.strip():
                    m["title"] = nxt.strip()
                    break
            break

    for ln in lines:
        s = ln.strip()
        if s.startswith("UP "):
            parts = [p.strip() for p in s[3:].split("|")]
            if parts:
                m["up"] = parts[0]
            if len(parts) > 1:
                m["tname"] = parts[1]
            if len(parts) > 2:
                mm = re.search(r"(\d+)s", parts[2])
                if mm:
                    m["duration"] = int(mm.group(1))
        elif s.startswith("aid="):
            mm = re.search(r"aid=(\d+)\s+cid=(\d+)", s)
            if mm:
                m["aid"], m["cid"] = int(mm.group(1)), int(mm.group(2))
        elif s.startswith("封面:"):
            m["cover"] = s.split(":", 1)[1].strip()
        elif s.startswith("标签:"):
            m["tags"] = [t.strip() for t in s.split(":", 1)[1].split("/") if t.strip()]

    in_desc = False
    desc_lines: list[str] = []
    for ln in lines:
        s = ln.strip()
        if s.startswith("简介原文"):
            in_desc = True
            continue
        if in_desc:
            if s.startswith("封面:"):
                break
            desc_lines.append(s)
    m["desc"] = "\n".join(x for x in desc_lines if x).strip()
    return m


def fetch_one(bv: str, env: dict, force: bool = False) -> dict:
    out_json = MAT_DIR / f"{bv}.json"
    # 只缓存成功的结果：失败记录留着会让人误以为不用重抓
    if out_json.exists() and not force:
        cached = json.loads(out_json.read_text(encoding="utf-8"))
        if cached.get("ok"):
            return cached

    MAT_DIR.mkdir(parents=True, exist_ok=True)
    SUB_DIR.mkdir(parents=True, exist_ok=True)

    rec: dict = {"bvid": bv, "ok": False, "materials": {}, "errors": []}

    code, out = run_bili("video", bv, env=env)
    meta = parse_video(out)
    rec["materials"].update({
        "title": meta["title"], "up": meta["up"], "tname": meta["tname"],
        "duration": meta["duration"], "aid": meta["aid"], "cid": meta["cid"],
        "desc": meta["desc"], "tags": meta["tags"], "cover": meta["cover"],
    })
    if not meta["title"]:
        rec["errors"].append(f"video 解析失败 exit={code}")

    code, out = run_bili("subtitles", bv, "--out", str(SUB_DIR), env=env)
    flat = SUB_DIR / f"{bv}-ai-zh-plain.txt"
    if flat.exists() and flat.stat().st_size > 0:
        text = flat.read_text(encoding="utf-8").strip()
        rec["materials"]["subtitle_chars"] = len(text)
        rec["materials"]["subtitle_lines"] = text.count("\n") + 1
        rec["materials"]["subtitle_path"] = str(flat.relative_to(ROOT))
    else:
        rec["materials"].update({"subtitle_chars": 0, "subtitle_lines": 0,
                                 "subtitle_path": None})
        rec["errors"].append("无官方字幕（这一路缺失）")

    rec["ok"] = bool(rec["materials"]["title"]) and rec["materials"]["subtitle_chars"] > 0
    out_json.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    return rec


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    force = "--force" in argv
    bvs = [a for a in argv if a.startswith("BV")]

    if not bvs:
        cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
        bvs = [v["bvid"] for v in cfg["videos"]]
        extra = "BV1Sbnz6EEix"
        if (ROOT / "data" / f"{extra}_like.jsonl").exists() and extra not in bvs:
            bvs.append(extra)

    env = ensure_fake_home()
    print(f"要抓 {len(bvs)} 个视频的素材\n")
    print(f"{'BV':<14} {'标题':<30} {'字幕行':>6} {'摘要源':>7}")
    print("-" * 64)

    ok = 0
    for bv in bvs:
        try:
            rec = fetch_one(bv, env, force=force)
        except Exception as e:
            print(f"{bv:<14} ✗ {type(e).__name__}: {str(e)[:40]}")
            continue
        m = rec["materials"]
        print(f"{bv:<14} {m['title'][:28]:<30} {m['subtitle_lines']:>6} "
              f"{'✓' if rec['ok'] else '✗':>7}")
        if rec["ok"]:
            ok += 1

    print("-" * 64)
    print(f"完成 {ok}/{len(bvs)}，素材在 data/materials/")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
