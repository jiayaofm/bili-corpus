@echo off
chcp 65001 >nul
cd /d "%~dp0"
set HF_HOME=D:\dsh-ai\lora\hf
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1
set PYTHONIOENCODING=utf-8

echo ============================================================
echo   B站评论打分 / 改写 本地服务
echo   浏览器扩展连的是 http://127.0.0.1:8849
echo   启动要加载两个模型，大约等 60~90 秒
echo ============================================================
echo.

"D:\dsh-ai\lora\venv\Scripts\python.exe" -u -X utf8 tools\comment_server.py ^
  --port 8849 --lora --adapter D:\dsh-ai\lora\out\fish-v4

echo.
echo 服务已退出。按任意键关闭窗口。
pause >nul
