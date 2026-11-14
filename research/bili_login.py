#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B站扫码登录助手 —— 生成 bili_credential.json

为什么需要它：采集脚本要读评论，而 B站 的评论接口在未登录时返回受限。
这个脚本只做一件事：让你扫码，把 sessdata / bili_jct / buvid3 / dedeuserid 存到本地。

注意：凭证等同于你的登录态，不要提交到 git、不要发给任何人。

依赖（两个都要）：
    pip install bilibili-api-python aiohttp

用法（在**有网络的普通 PowerShell 窗口**里跑）：
    <python> bili_login.py

扫码时务必选「记住我 / 长期」，否则 24 小时就过期。
"""

import asyncio
import json
import sys
from pathlib import Path

def _preflight():
    """依赖自检：bilibili-api 需要一个 HTTP 客户端，否则运行期才报错。"""
    have_api = True
    try:
        import bilibili_api  # noqa: F401
    except ImportError:
        have_api = False

    http_lib = None
    for m in ("curl_cffi", "httpx", "aiohttp"):
        try:
            __import__(m)
            http_lib = m
            break
        except ImportError:
            continue

    if have_api and http_lib:
        return

    pkgs = []
    if not have_api:
        pkgs.append("bilibili-api-python")
    if not http_lib:
        pkgs.append("aiohttp")
    sys.exit(
        "缺少依赖：\n"
        f"    pip install {' '.join(pkgs)}\n"
        "说明：bilibili-api 需要一个 HTTP 客户端（aiohttp / httpx / curl_cffi 任一），\n"
        "      单独装主库不会自动带上。"
    )


_preflight()

try:
    from bilibili_api.login_v2 import QrCodeLogin, QrCodeLoginEvents
except ImportError:
    sys.exit("导入 login_v2 失败，请确认 bilibili-api-python 版本较新。")

OUT = Path(__file__).parent / "bili_credential.json"


async def main():
    print("=" * 50)
    print("B站扫码登录")
    print("=" * 50)

    qr = QrCodeLogin()
    await qr.generate_qrcode()

    # 终端 ASCII 二维码
    print(qr.get_qrcode_terminal())

    # 同时存一张 PNG，方便找不到终端二维码时用
    try:
        pic = qr.get_qrcode_picture()
        png = Path(__file__).parent / "qrcode_login.png"
        png.write_bytes(pic.content)
        print(f"二维码图片已存到：{png}")
    except Exception:
        pass

    print("\n请用 B站 App 扫码，**确认时选「记住我 / 长期」**")
    print("等待中...\n")

    shown = None
    for _ in range(180):
        state = await qr.check_state()

        if state != shown:
            if state == QrCodeLoginEvents.SCAN:
                print("  → 已扫码，请在手机上确认")
            elif state == QrCodeLoginEvents.CONF:
                print("  → 已确认，正在处理")
            shown = state

        if state == QrCodeLoginEvents.DONE:
            cred = qr.get_credential()
            data = {
                "sessdata": cred.sessdata,
                "bili_jct": cred.bili_jct,
                "buvid3": cred.buvid3,
                "dedeuserid": cred.dedeuserid,
            }
            OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\n[OK] 登录成功，凭证已保存：{OUT}")
            print("   下一步：python bili_harvest.py fetch --bvid BV1YxWx69Ens --order both")
            return

        if state == QrCodeLoginEvents.TIMEOUT:
            print("\n[失败] 二维码已过期，请重新运行本脚本")
            return

        await asyncio.sleep(2)

    print("\n[失败] 等待超时，请重新运行")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已取消")
