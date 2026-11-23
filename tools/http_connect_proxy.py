# -*- coding: utf-8 -*-
"""让 ssh 走 HTTP 代理（CONNECT 隧道）。

为什么需要：机器上 Clash 监听 127.0.0.1:7897，但只对 HTTP/HTTPS 生效；
ssh 不走它，直连 github 会被重置。Windows OpenSSH 缺 connect.exe，
所以用这个脚本充当 ProxyCommand，把 stdio 和代理隧道对接起来。

要点：全程单线程 + 裸文件描述符（os.read/os.write + select）。
用 sys.stdin.buffer 这类缓冲对象 + 守护线程会在解释器退出时炸：
  Fatal Python error: _enter_buffered_busy: could not acquire lock

用法（给 git 用，不碰全局 ssh 配置）：

    git -c core.sshCommand='ssh -o ProxyCommand="python tools/http_connect_proxy.py %h %p"' push

参数：<host> [port]，端口默认 443。
"""
import os
import select
import socket
import sys

PROXY = os.environ.get("DSH_HTTP_PROXY", "127.0.0.1:7897")


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: http_connect_proxy.py <host> [port]", file=sys.stderr)
        return 2
    host = sys.argv[1]
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 443
    phost, pport = PROXY.rsplit(":", 1)

    try:
        sock = socket.create_connection((phost, int(pport)), timeout=20)
    except OSError as e:
        print(f"连不上代理 {PROXY}: {e}", file=sys.stderr)
        return 1

    sock.sendall(
        f"CONNECT {host}:{port} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Proxy-Connection: keep-alive\r\n\r\n".encode()
    )

    # 读响应头，确认 200；顺便把已经带回的数据留着
    buf = b""
    while b"\r\n\r\n" not in buf:
        chunk = sock.recv(4096)
        if not chunk:
            print("代理提前关闭连接", file=sys.stderr)
            return 1
        buf += chunk
    head, _, rest = buf.partition(b"\r\n\r\n")
    status = head.split(b"\r\n", 1)[0]
    if b" 200" not in status:
        print(f"代理拒绝：{status.decode(errors='replace')}", file=sys.stderr)
        return 1

    in_fd = sys.stdin.fileno()
    out_fd = sys.stdout.fileno()
    if rest:
        os.write(out_fd, rest)

    sock.settimeout(None)
    while True:
        try:
            readable, _, _ = select.select([in_fd, sock], [], [])
        except (OSError, ValueError):
            break
        if in_fd in readable:
            try:
                data = os.read(in_fd, 65536)
            except OSError:
                break
            if not data:
                break
            try:
                sock.sendall(data)
            except OSError:
                break
        if sock in readable:
            try:
                data = sock.recv(65536)
            except OSError:
                break
            if not data:
                break
            try:
                os.write(out_fd, data)
            except OSError:
                break

    try:
        sock.close()
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
