# -*- coding: utf-8 -*-
"""
LLM 接入层：给改写 workflow 提供「生成器」和「语义保真判断」

为什么需要这一层
----------------
字符层面的规则只能查出「新增了原文没有的事实」，查不出下面两类：
    · 删掉一半之后说了反话（极性反转）
    · 换了话题（信息选择 vs 换主题，在字符上长得一样）
这两类必须靠语义理解，所以接大模型。

设计约定
--------
    · 本模块是唯一碰网络的地方，pipeline/rewrite.py 保持纯净
    · 生成器和保真判断都符合已有接口，可直接注入 rewrite()
    · 密钥不写进代码：优先读环境变量，其次读 ~/.dsh/.credentials.yaml

密钥来源（按优先级）
    1. 环境变量 OPENCODE_GO_API_KEY
    2. ~/.dsh/.credentials.yaml 的 refs.OPENCODE_GO_API_KEY
"""
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DEFAULT_BASE_URL = "https://opencode.ai/inference/openai/v1"
DEFAULT_MODEL = "deepseek-v4.1-flash"
KEY_NAME = "OPENCODE_GO_API_KEY"


def _load_key():
    v = os.environ.get(KEY_NAME)
    if v:
        return v.strip().strip('"').strip("'")
    p = Path.home() / ".dsh" / ".credentials.yaml"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            m = re.match(r"\s*%s\s*:\s*(.+?)\s*$" % re.escape(KEY_NAME), line)
            if m:
                return m.group(1).strip().strip('"').strip("'")
    raise RuntimeError(
        "找不到 %s：请设置环境变量，或确认 %s 里有这一项" % (KEY_NAME, p))


class LLMClient:
    """极简 OpenAI 兼容客户端（只用标准库，不引入新依赖）。"""

    def __init__(self, base_url=None, model=None, key=None,
                 timeout=120, retries=3, verbose=False):
        self.base_url = (base_url or os.environ.get("DEAI_BASE_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("DEAI_MODEL") or DEFAULT_MODEL
        self._key = key or _load_key()
        self.timeout = timeout
        self.retries = retries
        self.verbose = verbose

    def chat(self, messages, temperature=0.8, max_tokens=900, json_mode=False):
        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + self._key,
                # 不带 UA 会被 Cloudflare 拦（403 / error code 1010）
                "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                               "Chrome/124.0.0.0 Safari/537.36"),
                "Accept": "application/json",
                "Accept-Language": "zh-CN,zh;q=0.9",
            },
            method="POST")

        last = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    payload = json.loads(r.read().decode("utf-8"))
                return payload["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:300]
                last = "HTTP %s: %s" % (e.code, detail)
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(last)
            except Exception as e:                      # 网络抖动
                last = "%s: %s" % (type(e).__name__, e)
                time.sleep(2 ** attempt)
        raise RuntimeError("LLM 调用失败（重试 %d 次）：%s" % (self.retries, last))


def _parse_json(text):
    """模型偶尔会包一层 ```json，剥掉再解析。"""
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


# ---------------------------------------------------------------- 生成器
GEN_SYSTEM = """你在帮人把一段「AI 味的 B 站评论」改成真人在评论区随手打的话。

## 六条硬规则

1. **留下的信息必须具体**。说「37%」就写 37%，不许写成「那点」「不少」「不太多」。
   把数字换成模糊词是删信息，不是改写。
2. **不必把信息全说一遍**。能串成一条因果链的可以都留；
   串不起来的，只挑最扎心的一两条，其余别提。
   **连续摆六七个数据点就是评审报告，就是 AI 腔。**
3. **不许「先下结论再列举理由」**。「X 我认，A，B，C」这种是论证结构，人不会这么说话。
4. **开头必须有信息量**。单独一句「字幕看的。」「视频里的。」摆在那，读者不知道你在说什么——
   要写成「我看字幕发现我这个型号刚好被测试了」。
5. **因果顺序**：能力/好处 → 但代价/约束 → 所以决定。
   **决定必须紧跟它的理由，中间不许插别的信息。**
6. **用词要准**。「能跑到 94 分」（能力）不是「跑 94 分」（结果）。
7. **主语不能省**。「能跑到94分，但订阅就剩37%」——什么能跑到？读者不知道。
   要写「**max**能跑到94分」。同理「跑二十分钟二十多刀」要写清是谁在跑。
8. **钱的归属要和原文一致**：
   · 确实是说话人自己花的 → 带「我」：「二十分钟**花了我**二十多刀」（这是损失，不是报价单）
   · 是别人花的、或者原文没说清 → **不指明是谁**：「跑二十分钟二十多刀，这谁顶得住」
     **不许把别人的损失安到自己头上。**
9. **不许提原文没有的东西**。原文没提「额度」，就不能写「这额度撑不住」——
   那是把别处的说法搬过来了。要说后果就说「日常用肯定吃不消」这类通用的。
10. **口语连说，别用逗号把短动作切开**。
   「用不起了，先关掉」→「**用不起了先关掉**」。
   两个动作一口气说出来，中间不加逗号——加了逗号就是在写文章。

## 标准范例一（短评，四条信息全留，靠因果链串起来）
原文：94 分是猛，但订阅剩 37%、离重置还有五天——这分数我看看就好，max 档先关。
改写：开max能跑94分，但是我现在的37%订阅额度还要撑五天，用不起了
为什么对：
· 「能跑94分」暗示分够高，「还要撑五天」暗示已是会员+快到期，「用不起了」暗示 max 烧额度
· 顺序是 能力 → 但约束 → 决定，决定紧跟理由
· 37% 是具体的，没含糊成「那点」
· 删掉的只有「这分数我看看就好」这句评价

## 标准范例二（长评，六个数据点只留一个）
原文：读字幕看的。被测的正好是我这个型号，这个号今天刚从 Opus 5.5 换成 Fable 5.2。
      阿江最后的建议我认同：94 分比 89.75 高四分多，时间是 Opus 5.5 的六倍，
      那三道 ERP 题跑 20 分钟烧掉 20 多刀就只能停。日常开发用 Opus 5.5 更划算，
      这话由我来说大概最没有利益冲突。
改写：我看字幕发现我这个型号刚好被测试了。三道 ERP 题二十分钟二十多刀，但这么烧也只能停。
      日常还是 Opus 5.5 划算。
为什么对：
· 开头有信息（我发现自己的型号被测试了），不是空半截
· 只留了「二十分钟二十多刀」这一条最扎心的，六倍和分数对比全不提——
  六个数据点连摆就是评审报告
· 没有「我认同 + 因为 a、b、c」的论证结构

直接输出改写结果，不要解释，不要编号。"""

GEN_USER = """原文：
{original}

需要改掉的地方：
{instruction}

请写 {n} 个版本，每行一个，不要加序号和引号。"""


class LLMGenerator:
    """实现 rewrite.Generator 协议：给定原文和指令，产出 n 个候选。"""

    def __init__(self, client=None, temperature=1.0, verbose=False,
                 fallback_on_error=True):
        self.client = client or LLMClient()
        self.temperature = temperature
        self.verbose = verbose
        self.fallback_on_error = fallback_on_error

    def __call__(self, original, instruction, n):
        prompt = GEN_USER.format(original=original, instruction=instruction, n=n)
        msgs = [{"role": "system", "content": GEN_SYSTEM},
                {"role": "user", "content": prompt}]

        # 这是推理模型：给少了会出现「思考把额度吃光、content 返回空串」。
        # 实测 157 字的原文给 900 token 会空，给 2000 才稳定。所以下限设高，
        # 并且遇到空返回时翻倍重试一次。
        max_tok = min(4000, max(2000, len(original) * 8))
        out = None
        for attempt in range(2):
            try:
                out = self.client.chat(msgs, temperature=self.temperature,
                                       max_tokens=max_tok)
            except Exception as e:
                if self.verbose:
                    print("  [生成器] 调用失败：%s" % e)
                if not self.fallback_on_error:
                    raise
                return []
            if out and out.strip():
                break
            max_tok = min(8000, max_tok * 2)
            if self.verbose:
                print("  [生成器] 返回空（推理吃掉了额度），加倍到 %d 重试" % max_tok)

        lines = []
        for ln in (out or "").splitlines():
            s = ln.strip()
            if not s:
                continue
            s = re.sub(r"^\s*(?:\d+[\.\)、]|[-*•])\s*", "", s)
            s = s.strip().strip('"').strip("“”").strip("'")
            if len(s) >= 2:
                lines.append(s)
        # 去重
        seen, uniq = set(), []
        for s in lines:
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        return uniq[:n]


# ------------------------------------------------------------ 语义保真检查
FID_SYSTEM = """你在判断一段「改写后的评论」有没有歪曲「原文」。

**允许的**（不算歪曲）：
· 换顺序、换说法、换语气
· 加纯情绪词（哈哈、服了、离谱、啊、吧）
· 删掉**评价句和元评论**（例如「这分数我看看就好」「这话由我来说最没有利益冲突」）

**不允许的**（出现任一条就判不通过）：
· **把具体信息挖空**：原文「订阅剩37%」，改写写「订阅剩那点」；
  原文「94分比89.75高四分多」，改写写「分只高小几分」——这属于删信息
· 原文说 A 好/差，改写说成 A 差/好（立场反转）
· 原文讲甲事，改写讲乙事（换话题）
· 出现原文没有的数字、事实、经历、人物、地点（编造）
· 把确定的改成不确定，或反过来

判断要点：**数字、型号这类具体信息必须还在改写里**（可以换个说法，但不能变成模糊词）。
评价和元评论可以没有。只输出 JSON：
{{"ok": true/false, "reason": "一句话说明", "violations": ["具体哪条"], "core_stance": "原文的核心立场"}}"""

FID_USER = """原文：
{original}

改写：
{rewritten}"""


NAT_SYSTEM = """你在判断一段文字像不像真人在 B 站评论区随手打出来的话。

判「不像」，命中任一条：
1. **整句挖空做 summary**：「订阅剩那点」「分只高小几分」「价格不太美丽」——
   把具体数字含糊成模糊词，读起来像摘要在概括材料。
2. **先下结论再列举理由**：「X 我认/那话说得对，A，B，C」这种论证结构。
   人不会先给论点再摆证据。
3. **没有信息量的半截开头**：单独一句「字幕看的。」「视频里的。」
   摆在那，读者不知道在说什么。
4. 书面腔、总结腔、对仗句、格言腔、抒情腔。

判「像」，如果：
· 读起来就是随手打的，句子有断有留白
· 具体信息还在（该说数字的地方说数字，只是换了个说法）
· 想到哪说到哪，不是先论证再下结论

只输出 JSON：{"natural": true/false, "score": 0-1, "problems": ["具体哪句、属于上面哪一类"]}"""


def judge_natural(client, text):
    """自然度判断。返回 (ok, score, problems, note)。"""
    try:
        out = client.chat([{"role": "system", "content": NAT_SYSTEM},
                           {"role": "user", "content": text}],
                          temperature=0.0, max_tokens=1200, json_mode=True)
    except Exception as e:
        return False, None, [], "自然度判断调用失败：%s" % e
    o = _parse_json(out)
    if not isinstance(o, dict) or "natural" not in o:
        return False, None, [], "自然度判断返回无法解析"
    probs = [str(x) for x in (o.get("problems") or [])]
    return bool(o.get("natural")), o.get("score"), probs, str(o.get("reason") or "")


def make_llm_fidelity_check(client=None, verbose=False, rule_guard=True,
                            natural_guard=True):
    """返回一个 check(original, rewritten) -> dict，可直接注入 rewrite()。

    三段式（缺一不可）：
        1) 规则先查「新增了原文没有的数字/实体/编造锚点」（快、免费、可靠）
        2) 语义查「有没有歪曲原文」，允许省略细节
        3) 自然度查「像不像人话」——这一关是为了堵住
           「靠删掉数字变笼统来骗过验收分」那条路。实测模型会这么干：
           把「订阅剩 37%」写成「订阅剩那点」，验收分降了，但这不是人话。
    """
    cli = client or LLMClient()
    if rule_guard:
        from pipeline.fidelity import check as rule_check

    def check(original, rewritten, **_):
        notes = []
        # ---- 第 1 段：规则反造假 ----
        if rule_guard:
            r = rule_check(original, rewritten)
            p = r.get("parts") or {}
            if p.get("fabrication_hits", 0) > 0:
                return {"score": 0.0, "ok": False,
                        "missing": [], "altered": [], "added": r.get("added") or [],
                        "notes": ["规则：捏造事实"] + (r.get("notes") or [])[:3],
                        "source": "rule"}
            for x in (r.get("altered") or []) + (r.get("added") or []):
                if "（原文无）" in x and ("数字" in x or "英文词" in x or "实体" in x):
                    return {"score": 0.0, "ok": False,
                            "missing": [], "altered": [x], "added": [x],
                            "notes": ["规则：新增了原文没有的事实"],
                            "source": "rule"}
            notes.append("规则：未发现新增事实")

        # ---- 第 2 段：语义 ----
        try:
            out = cli.chat(
                [{"role": "system", "content": FID_SYSTEM},
                 {"role": "user", "content": FID_USER.format(
                     original=original, rewritten=rewritten)}],
                temperature=0.0, max_tokens=1500, json_mode=True)
        except Exception as e:
            # 判断失败不当作通过
            return {"score": None, "ok": False, "missing": [], "altered": [],
                    "added": [], "notes": ["语义判断调用失败：%s" % e],
                    "source": "llm-error"}

        obj = _parse_json(out)
        if not isinstance(obj, dict) or "ok" not in obj:
            return {"score": None, "ok": False, "missing": [], "altered": [],
                    "added": [], "notes": ["语义判断返回无法解析：%s" % (out or "")[:120]],
                    "source": "llm-badjson"}

        ok = bool(obj.get("ok"))
        viol = [str(x) for x in (obj.get("violations") or [])]
        notes.append("语义：%s" % (obj.get("reason") or ""))

        # ---- 第 3 段：自然度 ----
        nat_ok, nat_score, nat_probs, nat_note = True, None, [], ""
        if natural_guard:
            nat_ok, nat_score, nat_probs, nat_note = judge_natural(cli, rewritten)
            notes.append("自然度：%s%s" % (
                "像" if nat_ok else "不像",
                ("（%s）" % nat_note) if nat_note else ""))
            for q in nat_probs[:2]:
                notes.append("  别扭：" + q)

        return {
            "score": 1.0 if (ok and nat_ok) else 0.0,
            "ok": ok and nat_ok,
            "missing": viol,
            "altered": viol,
            "added": [],
            "notes": notes,
            "core_stance": obj.get("core_stance"),
            "natural": nat_ok,
            "natural_score": nat_score,
            "natural_problems": nat_probs,
            "semantic_ok": ok,
            "source": "llm",
        }

    return check


def selftest(verbose=True):
    """确认密钥、网络、模型都通。返回 True/False。"""
    cli = LLMClient()
    if verbose:
        print("base_url = %s" % cli.base_url)
        print("model    = %s" % cli.model)
        print("key      = 已读到（%d 字符）" % len(cli._key))
    try:
        out = cli.chat([{"role": "user", "content": "只回复两个字：可用"}],
                       temperature=0, max_tokens=300)
        if verbose:
            print("返回     = %r" % (out or "").strip()[:40])
        return True
    except Exception as e:
        if verbose:
            print("失败     = %s" % e)
        return False


if __name__ == "__main__":
    ok = selftest()
    sys.exit(0 if ok else 1)
