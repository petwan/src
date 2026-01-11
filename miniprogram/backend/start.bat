@echo off
REM Astrapi 项目快速启动脚本
REM 双击此文件启动开发服务器

echo ===================================
echo   Astrapi 开发服务器
echo ===================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未找到 Python，请先安装 Python 3.11+
    pause
    exit /b 1
)

REM 检查 .env 文件
if not exist .env (
    echo [提示] .env 文件不存在，正在从 .env.example 复制...
    copy .env.example .env >nul
    echo [完成] .env 文件已创建，请根据需要修改配置
    echo.
)

REM 检查依赖
if not exist venv (
    echo [提示] 虚拟环境不存在，正在创建...
    python -m venv venv
    echo [完成] 虚拟环境已创建
    echo.
)

REM 激活虚拟环境
call venv\Scripts\activate.bat

REM 检查依赖是否已安装
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo [提示] 正在安装依赖...
    pip install -r requirements.txt
    echo [完成] 依赖已安装
    echo.
)

echo ===================================
echo   服务启动信息
echo ===================================
echo   访问地址: http://localhost:8000
echo   API 文档: http://localhost:8000/docs
echo   ReDoc:    http://localhost:8000/redoc
echo.
echo   按 Ctrl+C 停止服务
echo ===================================
echo.

REM 启动开发服务器
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

pause
