# -*- coding: utf-8 -*-
"""
课程展示用的本地网站（零依赖，只用标准库 http.server）。

## 为什么不用 Flask / FastAPI

两个 venv 里都没装 web 框架，而装它要走那条慢得离谱的 PyPI。
`http.server` 是标准库，一台笔记本上跑演示完全够。

## 跑在哪个 venv

必须用 `D:\dsh-ai\lora\venv`：

    E:\\DSH\\bili-corpus\\.venv        有 sklearn，但**没有 torch** → 生成功能跑不了
    D:\\dsh-ai\\lora\\venv              torch + transformers + peft + sklearn + scipy
                                       → 打分和生成都能跑

## 启动

    D:\\dsh-ai\\lora\\venv\\Scripts\\python.exe tools\\webdemo\\server.py

然后浏览器开 http://127.0.0.1:8848

## 设计取舍

- **分类器实时打分**（15 毫秒一条，无风险）
- **聚类/关联规则读预计算好的 JSON**，页面秒开，不在演示时算
- **生成功能懒加载**：服务启动后台预热，页面显示「预热中」，好了就能用。
  这样即使 GPU 出问题，前四页照样能演示。
"""
from __future__ import annotations

import csv
import json
import mimetypes
import os
import re
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

PORT = 8848
ADAPTER = Path(r"D:\dsh-ai\lora\out\qwen3-1.7b-gen4")
MODEL_ID = "Qwen/Qwen3-1.7B"

SYSTEM = "你在 B 站评论区发一条真实的短评。"
USER_TMPL = "视频《{title}》讲的是：{summary}\n\n就这条视频，说一句你的看法。"


def setup_hf_env() -> str:
    """让服务不依赖「调用者 shell 里 export 过什么」。

    ## 踩过的坑

    在 shell 里先 `$env:HF_HOME='D:\\dsh-ai\\lora\\hf'` 再启动，一切正常。
    换一个窗口直接 `python server.py`，生成页就报：

        OSError: We couldn't connect to 'https://huggingface.co' to load the files,
        and couldn't find them in the cached files.

    因为模型缓存在 D 盘，默认路径 `C:\\Users\\<用户>\\.cache\\huggingface` 找不到，
    transformers 就转去联网下载，而这台机器连不上。

    **所以脚本必须自己把环境变量钉死，不能靠外部 shell。**
    """
    candidates = [
        Path(r"D:\dsh-ai\lora\hf"),
        Path.home() / ".cache" / "huggingface",
        ROOT / ".bili-home" / ".cache" / "huggingface",
    ]
    chosen = None
    for c in candidates:
        hub = c / "hub"
        try:
            if hub.is_dir() and any(hub.glob("models--Qwen*")):
                chosen = c
                break
        except OSError:
            continue
    if chosen:
        # 用 setdefault：外部显式指定的优先，没指定才用我们找到的
        os.environ.setdefault("HF_HOME", str(chosen))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("DEAI_ALLOW_GPU", "1")
    return str(chosen) if chosen else "(没找到含 Qwen 的缓存目录)"


HF_HOME_USED = setup_hf_env()

# ---------------------------------------------------------------- 全局状态
_state: dict = {
    "scorer": None,
    "gen": None,          # (tok, model)
    "gen_ready": False,
    "gen_error": None,
    "clusters": [],
    "rules": [],
    "videos": [],
    "overview": {},
}
_lock = threading.Lock()


# ---------------------------------------------------------------- 数据准备
def load_scorer():
    from pipeline.scorer import Scorer
    _state["scorer"] = Scorer()
    print("[ok] 分类器就绪", flush=True)


def build_clusters() -> None:
    """用 s07 的同一套预处理还原风格簇，并交叉 AI 标注。"""
    import numpy as np
    from scipy import stats
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    X = np.load(ROOT / "state" / "X_style.npy")
    meta = json.loads((ROOT / "state" / "meta_style.json").read_text(encoding="utf-8"))
    rpids = [str(x) for x in meta["rpids"]]
    scols = meta["cols"]

    Xr = X.astype(float).copy()
    li = scols.index("字数")
    base_len = np.maximum(Xr[:, li], 1.0)
    for name in ("逗号数", "句号数", "感叹号数", "问号数", "顿号数"):
        if name in scols:
            j = scols.index(name)
            Xr[:, j] = Xr[:, j] / base_len * 100.0
    Xr[:, li] = np.log1p(Xr[:, li])
    Xs = StandardScaler().fit_transform(Xr)
    labels = KMeans(n_clusters=8, random_state=42, n_init=10).fit_predict(Xs)

    gt: dict[str, int] = {}
    with (ROOT / "labeling" / "ground_truth.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rp = str(row.get("rpid", "")).strip()
            lab = str(row.get("label", "")).strip()
            if rp and lab in ("0", "1"):
                gt[rp] = int(lab)
    n_gt, n_ai = len(gt), sum(gt.values())
    base_rate = n_ai / n_gt if n_gt else 0.0

    cid = {rp: int(labels[i]) for i, rp in enumerate(rpids)}
    idx_of: dict[int, list[int]] = {k: [] for k in range(8)}
    for i, rp in enumerate(rpids):
        idx_of[int(labels[i])].append(i)

    NAME = {1: "极短·互动党", 7: "极短·独白", 2: "表情党", 0: "中长句·无表情",
            5: "长评·独白", 3: "长评·带@", 4: "长评·高@", 6: "超长评·带@"}
    DESC = {
        1: "两三个字就完事，一半在 @ 人，路过打个招呼",
        7: "同样极短，但不 @ 人，纯自言自语",
        2: "短句 + 必带表情，情绪靠 emoji 表达",
        0: "中等长度、不用表情，最像正经留一句",
        5: "七八十字的独白，不 @ 人，少数派",
        3: "六七十字、一半 @ 人，AI 最密集的一类",
        4: "中等长度但六成 @ 人，量很小",
        6: "最长的一类，多行 + 高频 @ 人，量极小",
    }

    out = []
    for k in range(8):
        in_c = [rp for rp in gt if cid.get(rp) == k]
        idxs = idx_of[k]
        if not idxs:
            continue
        n = len(in_c)
        a = sum(gt[rp] for rp in in_c)
        p = a / n if n else 0.0
        fp = None
        if n >= 10 and n_ai > a and (n_gt - n_ai) > (n - a):
            fp = stats.fisher_exact(
                [[a, n - a], [n_ai - a, (n_gt - n_ai) - (n - a)]])[1]
            fp = min(1.0, fp * 8)
        out.append({
            "k": k,
            "name": NAME.get(k, f"簇{k}"),
            "desc": DESC.get(k, ""),
            "size": len(idxs),
            "avg_len": round(float(np.expm1(np.mean([Xr[i, li] for i in idxs]))), 1),
            "emoji": round(float(np.mean([X[i, scols.index("有表情")] for i in idxs])) * 100),
            "at": round(float(np.mean([X[i, scols.index("有@")] for i in idxs])) * 100),
            "n_labeled": n,
            "n_ai": a,
            "ai_rate": round(p * 100, 1),
            "lift": round(p / base_rate, 2) if base_rate else 0,
            "p": fp,
        })
    out.sort(key=lambda r: -r["ai_rate"])
    _state["clusters"] = out
    _state["base_rate"] = round(base_rate * 100, 1)
    _state["n_labeled"] = n_gt
    print(f"[ok] 聚类就绪 K=8，标注 {n_gt} 条", flush=True)


def build_videos() -> None:
    # 先按 bvid 收集真人评论，演示时拿来和模型生成并排对照。
    # 注意：摘要里**不含**热评（那是要预测的标签），这里只是生成完成后的对照展示。
    by_bvid: dict[str, list[str]] = {}
    try:
        with (ROOT / "data" / "corpus.jsonl").open(encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                c = (r.get("content") or "").strip()
                b = r.get("bvid") or ""
                # 只留「像样的短评」当例子：4~60 字，不带 @
                if b and 4 <= len(c) <= 60 and not c.startswith("@"):
                    by_bvid.setdefault(b, []).append(c)
    except OSError:
        pass

    vs = []
    for p in sorted((ROOT / "data" / "summaries").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if not (d.get("summary") or "").strip():
            continue
        # 标题本身可能已经带书名号，去掉一层，否则提示词里会套成《《…》》
        title = (d.get("title") or "").strip().strip("《》「」")
        pool = by_bvid.get(d["bvid"], [])
        # 固定步长取样，保证每次打开页面看到的例子一样（演示要可复现）
        step = max(1, len(pool) // 3) if pool else 1
        samples = [pool[i] for i in range(0, len(pool), step)][:3] if pool else []
        vs.append({
            "bvid": d["bvid"],
            "title": title,
            "summary": d["summary"].strip(),
            "confidence": d.get("confidence", ""),
            "human_samples": samples,
            "human_total": len(pool),
        })
    _state["videos"] = vs
    print(f"[ok] 视频摘要 {len(vs)} 条（含真人对照片段）", flush=True)


def load_rules() -> None:
    p = ROOT / "state" / "rules.json"
    if not p.exists():
        print("[--] state/rules.json 不存在，关联规则页会空着", flush=True)
        return
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        print("[--] rules.json 解析失败", flush=True)
        return

    # 结构（由 tools/assoc_rules.py 产出）：
    #   rules_to_ai  : [{ant, n_ant, n_both, support, confidence, lift, p, p_bonf, ...}]
    #                  ant 是特征列表，结论固定是「这条是 AI」
    #   general_rules: [{ant, cons, ...}]  特征 → 特征
    rules = []
    for r in (d.get("rules_to_ai") or [])[:40]:
        rules.append({
            "kind": "to_ai",
            "lhs": r.get("ant") or [],
            "rhs": ["是 AI 评论"],
            "n": r.get("n_ant"),
            "hit": r.get("n_both"),
            "support": r.get("support"),
            "confidence": r.get("confidence"),
            "lift": r.get("lift"),
            "p": r.get("p_bonf", r.get("p")),
        })
    for r in (d.get("general_rules") or [])[:20]:
        rules.append({
            "kind": "general",
            "lhs": r.get("ant") or [],
            "rhs": r.get("cons") or [],
            "n": r.get("n_ant"),
            "hit": r.get("n_both"),
            "support": r.get("support"),
            "confidence": r.get("confidence"),
            "lift": r.get("lift"),
            "p": r.get("p_bonf", r.get("p")),
        })
    _state["rules"] = rules
    _state["assoc_meta"] = {
        "n": d.get("n"),
        "k_ai": d.get("k_ai"),
        "base": d.get("base"),
        "n_items": d.get("n_items"),
        "n_freq_itemsets": d.get("n_freq_itemsets"),
        "n_rules": d.get("n_rules"),
        "n_rules_to_ai": d.get("n_rules_to_ai"),
        "n_candidate_rules": d.get("n_candidate_rules"),
        "permutation": d.get("permutation"),
        "cv": d.get("cv"),
    }
    print(f"[ok] 关联规则 {len(rules)} 条（含规则→AI {min(40, len(d.get('rules_to_ai') or []))} 条）",
          flush=True)


def build_overview() -> None:
    def count_jsonl(p: Path) -> int:
        try:
            with p.open(encoding="utf-8") as f:
                return sum(1 for _ in f)
        except OSError:
            return 0

    deep = 0
    for p in (ROOT / "data" / "deep").glob("*.jsonl"):
        if not p.name.endswith("-time.jsonl"):
            deep += count_jsonl(p)

    _state["overview"] = {
        "corpus": count_jsonl(ROOT / "data" / "corpus.jsonl"),
        "deep": deep,
        "labeled": _state.get("n_labeled", 0),
        "videos": len(_state.get("videos", [])),
        "summaries": len(_state.get("videos", [])),
        "base_rate": _state.get("base_rate", 0),
        "rules": len(_state.get("rules", [])),
    }


# ---------------------------------------------------------------- 生成（懒加载）
def warm_generator() -> None:
    """后台预热。失败不影响前四页。"""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16, device_map="auto")
        model = PeftModel.from_pretrained(model, str(ADAPTER))
        model.eval()
        with _lock:
            _state["gen"] = (tok, model)
            _state["gen_ready"] = True
        print(f"[ok] 生成模型就绪，{time.time()-t0:.0f} 秒（缓存 {HF_HOME_USED}）", flush=True)
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        low = msg.lower()
        # 把最常见的两种失败翻译成人话，否则演示现场看到一屏英文没法处理
        if "huggingface.co" in low or "offline" in low or "cached" in low:
            hint = (f"找不到模型缓存（HF_HOME={HF_HOME_USED}）。"
                    f"确认 D:\\dsh-ai\\lora\\hf\\hub\\models--Qwen--Qwen3-1.7B 存在，"
                    f"或手动设 HF_HOME 后重启。")
        elif "out of memory" in low:
            hint = "显存不够。先把别的占显存的程序关掉，再重启服务。"
        else:
            hint = "生成功能不可用，前四页不受影响。详见 state\\webdemo.err.log"
        with _lock:
            _state["gen_error"] = hint
        print(f"[!!] 生成模型预热失败 —— {hint}", flush=True)
        traceback.print_exc()


def generate(text_summary: str, title: str, temperature: float = 1.1) -> str:
    import torch
    tok, model = _state["gen"]
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TMPL.format(title=title, summary=text_summary)}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=90, do_sample=True,
                             temperature=temperature, top_p=0.95,
                             pad_token_id=tok.eos_token_id)
    g = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return g.replace("\n", " ").strip()


# ---------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "bili-corpus-demo"

    def log_message(self, fmt, *args):
        pass  # 演示时别刷屏

    # -- helpers
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _read_json(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    # -- routes
    def do_GET(self) -> None:
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                html = (Path(__file__).parent / "index.html").read_bytes()
                self._send(200, html, "text/html; charset=utf-8")
            elif path == "/api/overview":
                self._json(_state["overview"])
            elif path == "/api/clusters":
                self._json({"clusters": _state["clusters"],
                            "base_rate": _state.get("base_rate", 0),
                            "n_labeled": _state.get("n_labeled", 0)})
            elif path == "/api/rules":
                self._json({"rules": _state["rules"],
                            "meta": _state.get("assoc_meta", {})})
            elif path == "/api/videos":
                self._json({"videos": _state["videos"]})
            elif path == "/api/genstatus":
                self._json({"ready": _state["gen_ready"],
                            "error": _state["gen_error"]})
            elif path.startswith("/fig/"):
                name = os.path.basename(path[5:])
                fp = ROOT / "analysis" / name
                if fp.is_file():
                    ctype = mimetypes.guess_type(fp.name)[0] or "image/png"
                    self._send(200, fp.read_bytes(), ctype)
                else:
                    self._json({"error": "not found"}, 404)
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        try:
            if path == "/api/classify":
                body = self._read_json()
                text = (body.get("text") or "").strip()
                if not text:
                    self._json({"error": "空文本"}, 400)
                    return
                if len(text) > 2000:
                    text = text[:2000]
                t0 = time.time()
                d = _state["scorer"].describe(text)
                dt = (time.time() - t0) * 1000
                self._json({
                    "text": text,
                    "ai_prob": round(float(d["ai_prob"]), 4),
                    "human_prob": round(float(d["human_prob"]), 4),
                    "rule_score": d.get("ai_rule"),
                    "hits": d.get("hits"),
                    "advice": d.get("advice"),
                    "ms": round(dt, 1),
                })
            elif path == "/api/generate":
                body = self._read_json()
                bvid = body.get("bvid") or ""
                v = next((x for x in _state["videos"] if x["bvid"] == bvid), None)
                if not v:
                    self._json({"error": "没有这个视频"}, 404)
                    return
                if not _state["gen_ready"]:
                    self._json({"error": _state["gen_error"] or "模型正在预热，稍等",
                                "warming": not _state["gen_error"]}, 503)
                    return
                t0 = time.time()
                with _lock:
                    # 一次给 3 条：模型在温度 1.1 下方差很大，
                    # 只展示一条等于用一次坏运气代表它的水平。
                    outs = [generate(v["summary"], v["title"]) for _ in range(3)]
                self._json({"comments": [x for x in outs if x],
                            "seconds": round(time.time() - t0, 1),
                            "title": v["title"],
                            "summary": v["summary"]})
            else:
                self._json({"error": "not found"}, 404)
        except Exception as e:
            self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if not (Path(__file__).parent / "index.html").exists():
        print("[error] 缺 index.html")
        return 1

    print("=" * 56)
    print("  B站 评论语料 —— 课程展示")
    print("=" * 56)
    print(f"  模型缓存: {HF_HOME_USED}")
    load_scorer()
    build_clusters()
    build_videos()
    load_rules()
    build_overview()

    threading.Thread(target=warm_generator, daemon=True).start()

    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"\n  → http://127.0.0.1:{PORT}\n  （Ctrl+C 停止）\n", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
