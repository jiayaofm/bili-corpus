#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B站扫码登录 —— 手工实现版

为什么不用 bilibili_api 的 QrCodeLogin：
    它的 WEB 分支从 events["url"] 的查询串里抠 cookie：
        cookies_list = cred_url.split("?")[1].split("&")
    而 B站 现在通过 Set-Cookie 响应头下发凭证，URL 里不再带 SESSDATA，
    结果是"登录成功但凭证全为空"。本脚本同时读 Set-Cookie 和 URL 兜底。

输出：bili_credential.json（sessdata / bili_jct / buvid3 / dedeuserid）

依赖：pip install aiohttp qrcode pillow
"""

import asyncio
import json
import sys
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import aiohttp

try:
    import qrcode
except ImportError:
    sys.exit("缺少依赖：pip install qrcode pillow")

OUT = Path(__file__).parent / "bili_credential.json"
QR_PNG = Path(__file__).parent / "qrcode_login.png"

BASE = "https://passport.bilibili.com/x/passport-login/web/qrcode"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.bilibili.com/",
}

CODE_NOT_SCANNED = 86101
CODE_NOT_CONFIRMED = 86090
CODE_EXPIRED = 86038


def make_qr(url: str) -> None:
    qr = qrcode.QRCode(box_size=8, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    qr.make_image(fill_color="black", back_color="white").save(QR_PNG)


def extract_from_url(u: str) -> dict:
    """B站 成功后会返回 crossDomain URL，查询串里带凭证（兜底来源）。"""
    out = {}
    try:
        q = parse_qs(urlparse(u).query)
        for key, target in (("SESSDATA", "sessdata"), ("bili_jct", "bili_jct"),
                            ("DedeUserID", "dedeuserid")):
            if key in q and q[key] and q[key][0]:
                out[target] = q[key][0]
    except Exception:
        pass
    return out


async def main() -> int:
    print("=" * 50)
    print("B站扫码登录（手工实现）")
    print("=" * 50)

    async with aiohttp.ClientSession(headers=HEADERS) as sess:
        # 1) 申请二维码
        async with sess.get(f"{BASE}/generate", timeout=aiohttp.ClientTimeout(total=20)) as r:
            gen = await r.json(content_type=None)
        if gen.get("code") != 0:
            print(f"申请二维码失败: {gen}")
            return 1

        data = gen["data"]
        qr_url = data["url"]
        qr_key = data["qrcode_key"]

        make_qr(qr_url)
        print(f"\n二维码图片：{QR_PNG}")
        print("\n请用 B站 App 扫码。")
        print("确认时务必选「记住我 / 长期」，否则 24 小时就失效。\n")

        # 2) 轮询
        last_msg = None
        for i in range(180):          # 最长约 6 分钟
            async with sess.get(
                f"{BASE}/poll", params={"qrcode_key": qr_key},
                timeout=aiohttp.ClientTimeout(total=20)
            ) as r:
                poll = await r.json(content_type=None)
                # 成功后 Set-Cookie 就在这里
                jar = {c.key: c.value for c in r.cookies.values()}

            body = poll.get("data") or {}
            code = body.get("code")

            if code == CODE_NOT_SCANNED:
                msg = "等待扫码"
            elif code == CODE_NOT_CONFIRMED:
                msg = "已扫码，请在手机上确认"
            elif code == CODE_EXPIRED:
                print("\n二维码已过期，请重新运行本脚本")
                return 1
            else:
                msg = "处理中"

            if msg != last_msg:
                print(f"  → {msg}")
                last_msg = msg

            # 成功判定：poll 的 data.code == 0
            if code == 0:
                cred = {}
                # 来源 A：Set-Cookie 响应头（B站现在的实际下发方式）
                for k in ("SESSDATA", "bili_jct", "DedeUserID", "buvid3"):
                    if k in jar and jar[k]:
                        cred[k] = unquote(jar[k])
                # 来源 B：crossDomain URL 查询串（兜底）
                for k, v in extract_from_url(body.get("url") or "").items():
                    if v and not cred.get(k):
                        cred[k] = unquote(v)

                sessdata = cred.get("SESSDATA") or cred.get("sessdata") or ""
                bili_jct = cred.get("bili_jct") or ""
                dedeuserid = cred.get("DedeUserID") or cred.get("dedeuserid") or ""
                buvid3 = cred.get("buvid3") or ""

                if not sessdata:
                    print("\n登录返回到达，但没拿到 SESSDATA。")
                    print(f"  Set-Cookie 键: {sorted(jar.keys())}")
                    print(f"  url 存在: {bool(body.get('url'))}")
                    return 1

                payload = {
                    "sessdata": sessdata,
                    "bili_jct": bili_jct,
                    "buvid3": buvid3,
                    "dedeuserid": dedeuserid,
                }
                OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                print("\n[OK] 登录成功，凭证已保存")
                print(f"     {OUT}")
                print(f"     sessdata 长度={len(sessdata)}  bili_jct={'有' if bili_jct else '无'}  "
                      f"buvid3={'有' if buvid3 else '无'}")
                return 0

            await asyncio.sleep(2)

    print("\n等待超时")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except KeyboardInterrupt:
        print("\n已取消")
        sys.exit(130)
