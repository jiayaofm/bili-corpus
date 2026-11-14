# -*- coding: utf-8 -*-
"""
流水线公共设施：路径、配置、状态清单、IO 工具

设计原则：
  · 所有路径从 ROOT 派生，不硬编码
  · 每次写产物都登记到 state/manifest.json（条数 + 时间 + sha256 前缀）
  · run_all.py 靠 manifest 判断某阶段是否已完成
"""
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"


# ---------------- 配置 ----------------
def load_config():
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


CFG = load_config()
P = {k: ROOT / v for k, v in CFG["paths"].items()}
for d in P.values():
    d.mkdir(parents=True, exist_ok=True)

MANIFEST = P["state"] / "manifest.json"


# ---------------- 清单 ----------------
def read_manifest():
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def write_manifest(m):
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")


def record(stage, outputs, extra=None):
    """登记某阶段的产物。outputs: {相对路径: 记录数 或 说明}"""
    m = read_manifest()
    entry = {
        "at": datetime.now().isoformat(timespec="seconds"),
        "outputs": {},
    }
    for rel, meta in outputs.items():
        full = ROOT / rel
        rec = {"exists": full.exists()}
        if full.exists():
            rec["bytes"] = full.stat().st_size
            rec["sha256_8"] = sha8(full)
        rec.update(meta if isinstance(meta, dict) else {"count": meta})
        entry["outputs"][rel] = rec
    if extra:
        entry.update(extra)
    m[stage] = entry
    write_manifest(m)
    return entry


def sha8(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def is_done(stage, required=None):
    """该阶段是否已完成且产物齐全"""
    m = read_manifest()
    if stage not in m:
        return False
    outs = m[stage].get("outputs", {})
    for rel, rec in outs.items():
        if not (ROOT / rel).exists():
            return False
    if required:
        for rel in required:
            if rel not in outs:
                return False
            if not (ROOT / rel).exists():
                return False
    return True


# ---------------- IO ----------------
def read_jsonl(path):
    out = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def write_jsonl(path, rows):
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def banner(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def info(msg):
    print(f"  {msg}")
