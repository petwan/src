# Astrapi 项目 PowerShell 脚本
# 用于 Windows 环境下简化常用开发操作

param(
    [Parameter(Position=0)]
    [ValidateSet('help', 'install', 'update', 'clean', 'run', 'dev', 'test', 'lint', 'format', 'type-check',
                'db-upgrade', 'db-create', 'docker-up', 'docker-down', 'docker-logs', 'info')]
    [string]$Command = 'help',

    [string]$Message,
    [string]$Name
)

# 颜色输出函数
function Write-ColorOutput {
    param([string]$Message, [string]$Color = "White")
    Write-Host $Message -ForegroundColor $Color
}

# 显示帮助信息
function Show-Help {
    Write-ColorOutput "=== Astrapi 项目常用命令 ===" "Cyan"
    Write-Host ""
    Write-ColorOutput "环境管理:" "Green"
    Write-Host "  .\Make.ps1 install          # 安装项目依赖"
    Write-Host "  .\Make.ps1 update           # 更新依赖"
    Write-Host "  .\Make.ps1 clean            # 清理缓存和临时文件"
    Write-Host ""
    Write-ColorOutput "运行应用:" "Green"
    Write-Host "  .\Make.ps1 run              # 启动生产服务器"
    Write-Host "  .\Make.ps1 dev              # 启动开发服务器（热重载）"
    Write-Host ""
    Write-ColorOutput "代码质量:" "Green"
    Write-Host "  .\Make.ps1 lint             # 运行代码检查"
    Write-Host "  .\Make.ps1 format           # 格式化代码"
    Write-Host "  .\Make.ps1 type-check       # 运行类型检查"
    Write-Host ""
    Write-ColorOutput "测试:" "Green"
    Write-Host "  .\Make.ps1 test             # 运行所有测试"
    Write-Host ""
    Write-ColorOutput "数据库:" "Green"
    Write-Host "  .\Make.ps1 db-upgrade       # 升级数据库到最新版本"
    Write-Host "  .\Make.ps1 db-create -Message '描述' # 创建新的数据库迁移"
    Write-Host ""
    Write-ColorOutput "Docker:" "Green"
    Write-Host "  .\Make.ps1 docker-up        # 启动 Docker 容器"
    Write-Host "  .\Make.ps1 docker-down      # 停止 Docker 容器"
    Write-Host "  .\Make.ps1 docker-logs      # 查看 Docker 日志"
    Write-Host ""
    Write-ColorOutput "其他:" "Green"
    Write-Host "  .\Make.ps1 info             # 显示项目信息"
    Write-Host "  .\Make.ps1 help             # 显示帮助信息"
    Write-Host ""
}

# 安装依赖
function Install-Dependencies {
    Write-ColorOutput "安装项目依赖..." "Blue"
    if (Test-Path poetry.lock) {
        poetry install
    } elseif (Test-Path requirements.txt) {
        pip install -r requirements.txt
    } else {
        Write-ColorOutput "未找到依赖文件" "Red"
        return
    }
    Write-ColorOutput "依赖安装完成" "Green"
}

# 更新依赖
function Update-Dependencies {
    Write-ColorOutput "更新依赖..." "Blue"
    poetry update
    Write-ColorOutput "依赖更新完成" "Green"
}

# 清理缓存
function Clean-Cache {
    Write-ColorOutput "清理缓存..." "Blue"

    # 删除 Python 缓存目录
    Get-ChildItem -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Recurse -File -Filter "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Recurse -File -Filter "*.pyo" | Remove-Item -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Recurse -File -Filter "*.pyd" | Remove-Item -Force -ErrorAction SilentlyContinue

    # 删除数据库文件
    Get-ChildItem -Recurse -File -Filter "*.db" | Remove-Item -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Recurse -File -Filter "*.sqlite" | Remove-Item -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Recurse -File -Filter "*.sqlite3" | Remove-Item -Force -ErrorAction SilentlyContinue

    # 删除测试和工具缓存
    Remove-Item -Recurse -Force -Path .pytest_cache -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force -Path .ruff_cache -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force -Path .mypy_cache -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force -Path htmlcov -ErrorAction SilentlyContinue
    Remove-Item -Force -Path .coverage -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force -Path dist -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force -Path build -ErrorAction SilentlyContinue

    Write-ColorOutput "清理完成" "Green"
}

# 启动生产服务器
function Start-ProductionServer {
    Write-ColorOutput "启动生产服务器..." "Blue"
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
}

# 启动开发服务器
function Start-DevelopmentServer {
    Write-ColorOutput "启动开发服务器..." "Blue"
    Write-ColorOutput "访问地址: http://localhost:8000" "Yellow"
    Write-ColorOutput "API 文档: http://localhost:8000/docs" "Yellow"
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
}

# 运行测试
function Run-Tests {
    Write-ColorOutput "运行测试..." "Blue"
    pytest -v
}

# 代码检查
function Run-Lint {
    Write-ColorOutput "运行代码检查..." "Blue"
    ruff check app/
    Write-ColorOutput "代码检查完成" "Green"
}

# 格式化代码
function Format-Code {
    Write-ColorOutput "格式化代码..." "Blue"
    ruff format app/
    Write-ColorOutput "代码格式化完成" "Green"
}

# 类型检查
function Run-TypeCheck {
    Write-ColorOutput "运行类型检查..." "Blue"
    mypy app/
    Write-ColorOutput "类型检查完成" "Green"
}

# 升级数据库
function Upgrade-Database {
    Write-ColorOutput "升级数据库..." "Blue"
    alembic upgrade head
    Write-ColorOutput "数据库升级完成" "Green"
}

# 创建数据库迁移
function Create-DatabaseMigration {
    if ([string]::IsNullOrEmpty($Message)) {
        Write-ColorOutput "错误: 请使用 -Message 参数指定迁移描述" "Red"
        Write-Host "示例: .\Make.ps1 db-create -Message '添加用户表'"
        return
    }
    Write-ColorOutput "创建数据库迁移: $Message" "Blue"
    alembic revision --autogenerate -m $Message
    Write-ColorOutput "迁移文件已创建" "Green"
}

# Docker 操作
function Start-DockerContainers {
    Write-ColorOutput "启动 Docker 容器..." "Blue"
    docker-compose up -d
    Write-ColorOutput "容器已启动" "Green"
    Write-ColorOutput "查看日志: .\Make.ps1 docker-logs" "Yellow"
}

function Stop-DockerContainers {
    Write-ColorOutput "停止 Docker 容器..." "Blue"
    docker-compose down
    Write-ColorOutput "容器已停止" "Green"
}

function Show-DockerLogs {
    docker-compose logs -f
}

# 显示项目信息
function Show-ProjectInfo {
    Write-ColorOutput "=== 项目信息 ===" "Cyan"
    Write-Host "  Python 版本: $(python --version)"
    if (Get-Command poetry -ErrorAction SilentlyContinue) {
        Write-Host "  Poetry 版本: $(poetry --version)"
    } else {
        Write-Host "  Poetry 版本: 未安装"
    }

    if (Test-Path .env) {
        Write-Host "  .env 文件: 已存在"
    } else {
        Write-Host "  .env 文件: 不存在"
    }
}

# 主命令路由
switch ($Command) {
    'help' { Show-Help }
    'install' { Install-Dependencies }
    'update' { Update-Dependencies }
    'clean' { Clean-Cache }
    'run' { Start-ProductionServer }
    'dev' { Start-DevelopmentServer }
    'test' { Run-Tests }
    'lint' { Run-Lint }
    'format' { Format-Code }
    'type-check' { Run-TypeCheck }
    'db-upgrade' { Upgrade-Database }
    'db-create' { Create-DatabaseMigration }
    'docker-up' { Start-DockerContainers }
    'docker-down' { Stop-DockerContainers }
    'docker-logs' { Show-DockerLogs }
    'info' { Show-ProjectInfo }
    default {
        Write-ColorOutput "未知命令: $Command" "Red"
        Write-ColorOutput "使用 '.\Make.ps1 help' 查看帮助" "Yellow"
    }
}
