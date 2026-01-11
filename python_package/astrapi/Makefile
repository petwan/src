# Astrapi 项目 Makefile
# 用于简化常用开发操作

.PHONY: help install update clean run dev test lint format type-check docker-build docker-up docker-down db-upgrade db-create db-migration

# 颜色输出
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# 默认目标
.DEFAULT_GOAL := help

# ============================================
# 帮助信息
# ============================================

help: ## 显示帮助信息
	@echo "$(BLUE)Astrapi 项目常用命令$(NC)"
	@echo ""
	@echo "$(GREEN)开发环境:$(NC)"
	@sed -n 's/^##//p' ${MAKEFILE_LIST} | column -t -s ':' | sed -e 's/^/ /'
	@echo ""
	@echo "$(GREEN)示例:$(NC)"
	@echo "  make install      # 安装依赖"
	@echo "  make dev         # 启动开发服务器"
	@echo "  make test        # 运行测试"

# ============================================
# 环境管理
# ============================================

install: ## 安装项目依赖
	@echo "$(BLUE)安装项目依赖...$(NC)"
	@if [ -f poetry.lock ]; then \
		poetry install; \
	else \
		pip install -r requirements.txt; \
	fi
	@echo "$(GREEN)依赖安装完成$(NC)"

install-dev: ## 安装开发依赖
	@echo "$(BLUE)安装开发依赖...$(NC)"
	poetry install --with dev
	@echo "$(GREEN)开发依赖安装完成$(NC)"

update: ## 更新依赖
	@echo "$(BLUE)更新依赖...$(NC)"
	poetry update
	@echo "$(GREEN)依赖更新完成$(NC)"

clean: ## 清理缓存和临时文件
	@echo "$(BLUE)清理缓存...$(NC)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name "*.db" -delete
	find . -type f -name "*.sqlite" -delete
	find . -type f -name "*.sqlite3" -delete
	rm -rf .pytest_cache
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf dist
	rm -rf build
	rm -rf *.egg-info
	@echo "$(GREEN)清理完成$(NC)"

# ============================================
# 运行应用
# ============================================

run: ## 启动生产服务器
	@echo "$(BLUE)启动生产服务器...$(NC)"
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

dev: ## 启动开发服务器（热重载）
	@echo "$(BLUE)启动开发服务器...$(NC)"
	@echo "$(YELLOW)访问地址: http://localhost:8000$(NC)"
	@echo "$(YELLOW)API 文档: http://localhost:8000/docs$(NC)"
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

dev-with-env: ## 使用 .env 文件启动开发服务器
	@echo "$(BLUE)启动开发服务器（使用 .env）...$(NC)"
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)警告: .env 文件不存在，从 .env.example 复制$(NC)"; \
		cp .env.example .env; \
	fi
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# ============================================
# 代码质量
# ============================================

lint: ## 运行代码检查
	@echo "$(BLUE)运行代码检查...$(NC)"
	ruff check app/
	@echo "$(GREEN)代码检查完成$(NC)"

lint-fix: ## 自动修复代码问题
	@echo "$(BLUE)自动修复代码问题...$(NC)"
	ruff check --fix app/
	@echo "$(GREEN)代码修复完成$(NC)"

format: ## 格式化代码
	@echo "$(BLUE)格式化代码...$(NC)"
	ruff format app/
	@echo "$(GREEN)代码格式化完成$(NC)"

format-check: ## 检查代码格式
	@echo "$(BLUE)检查代码格式...$(NC)"
	ruff format --check app/

type-check: ## 运行类型检查
	@echo "$(BLUE)运行类型检查...$(NC)"
	mypy app/
	@echo "$(GREEN)类型检查完成$(NC)"

check-all: lint type-check ## 运行所有检查（代码检查 + 类型检查）
	@echo "$(GREEN)所有检查完成$(NC)"

# ============================================
# 测试
# ============================================

test: ## 运行所有测试
	@echo "$(BLUE)运行测试...$(NC)"
	pytest -v

test-cov: ## 运行测试并生成覆盖率报告
	@echo "$(BLUE)运行测试（覆盖率）...$(NC)"
	pytest --cov=app --cov-report=html --cov-report=term
	@echo "$(GREEN)覆盖率报告: htmlcov/index.html$(NC)"

test-unit: ## 运行单元测试
	@echo "$(BLUE)运行单元测试...$(NC)"
	pytest tests/unit -v

test-integration: ## 运行集成测试
	@echo "$(BLUE)运行集成测试...$(NC)"
	pytest tests/integration -v

# ============================================
# 数据库
# ============================================

db-upgrade: ## 升级数据库到最新版本
	@echo "$(BLUE)升级数据库...$(NC)"
	alembic upgrade head
	@echo "$(GREEN)数据库升级完成$(NC)"

db-downgrade: ## 回退数据库一个版本
	@echo "$(BLUE)回退数据库...$(NC)"
	alembic downgrade -1
	@echo "$(GREEN)数据库回退完成$(NC)"

db-create: ## 创建新的数据库迁移
	@echo "$(BLUE)创建数据库迁移...$(NC)"
	@if [ -z "$(message)" ]; then \
		echo "$(RED)错误: 请使用 'make db-create message=\"描述信息' ' 指定迁移描述$(NC)"; \
		exit 1; \
	fi
	alembic revision --autogenerate -m "$(message)"
	@echo "$(GREEN)迁移文件已创建$(NC)"

db-reset: ## 重置数据库（删除所有表）
	@echo "$(YELLOW)警告: 这将删除所有数据库表！$(NC)"
	@read -p "确定要继续吗？[y/N] " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		alembic downgrade base; \
		alembic upgrade head; \
		echo "$(GREEN)数据库已重置$(NC)"; \
	else \
		echo "$(YELLOW)操作已取消$(NC)"; \
	fi

db-check: ## 检查数据库连接
	@echo "$(BLUE)检查数据库连接...$(NC)"
	@python -c "import asyncio; from app.database.session import check_db_connection; asyncio.run(check_db_connection())"

# ============================================
# Docker
# ============================================

docker-build: ## 构建 Docker 镜像
	@echo "$(BLUE)构建 Docker 镜像...$(NC)"
	docker-compose build
	@echo "$(GREEN)镜像构建完成$(NC)"

docker-up: ## 启动 Docker 容器
	@echo "$(BLUE)启动 Docker 容器...$(NC)"
	docker-compose up -d
	@echo "$(GREEN)容器已启动$(NC)"
	@echo "$(YELLOW)查看日志: make docker-logs$(NC)"

docker-down: ## 停止 Docker 容器
	@echo "$(BLUE)停止 Docker 容器...$(NC)"
	docker-compose down
	@echo "$(GREEN)容器已停止$(NC)"

docker-logs: ## 查看 Docker 日志
	@docker-compose logs -f

docker-restart: ## 重启 Docker 容器
	@echo "$(BLUE)重启 Docker 容器...$(NC)"
	docker-compose restart
	@echo "$(GREEN)容器已重启$(NC)"

docker-clean: ## 清理 Docker 容器和卷
	@echo "$(BLUE)清理 Docker 资源...$(NC)"
	docker-compose down -v
	docker system prune -f
	@echo "$(GREEN)清理完成$(NC)"

# ============================================
# Celery
# ============================================

celery-worker: ## 启动 Celery Worker
	@echo "$(BLUE)启动 Celery Worker...$(NC)"
	celery -A app.tasks.celery_app worker --loglevel=info

celery-beat: ## 启动 Celery Beat（定时任务）
	@echo "$(BLUE)启动 Celery Beat...$(NC)"
	celery -A app.tasks.celery_app beat --loglevel=info

celery-flower: ## 启动 Celery Flower（监控）
	@echo "$(BLUE)启动 Celery Flower...$(NC)"
	@echo "$(YELLOW)访问地址: http://localhost:5555$(NC)"
	celery -A app.tasks.celery_app flower

# ============================================
# 文档
# ============================================

docs: ## 生成 API 文档
	@echo "$(BLUE)生成 API 文档...$(NC)"
	@echo "$(GREEN)Swagger UI: http://localhost:8000/docs$(NC)"
	@echo "$(GREEN)ReDoc: http://localhost:8000/redoc$(NC)"
	@echo "$(GREEN)OpenAPI JSON: http://localhost:8000/api/v1/openapi.json$(NC)"

docs-export: ## 导出 OpenAPI JSON
	@echo "$(BLUE)导出 OpenAPI JSON...$(NC)"
	@curl http://localhost:8000/api/v1/openapi.json -o openapi.json
	@echo "$(GREEN)OpenAPI JSON 已导出到 openapi.json$(NC)"

# ============================================
# 项目管理
# ============================================

init: ## 初始化项目（首次使用）
	@echo "$(BLUE)初始化项目...$(NC)"
	@echo "$(YELLOW)1. 复制 .env 文件$(NC)"
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN)✓ .env 文件已创建$(NC)"; \
	fi
	@echo "$(YELLOW)2. 安装依赖$(NC)"
	@poetry install
	@echo "$(GREEN)✓ 依赖已安装$(NC)"
	@echo "$(YELLOW)3. 创建数据库迁移$(NC)"
	@alembic revision --autogenerate -m "初始化数据库"
	@echo "$(GREEN)✓ 迁移文件已创建$(NC)"
	@echo "$(YELLOW)4. 升级数据库$(NC)"
	@alembic upgrade head
	@echo "$(GREEN)✓ 数据库已升级$(NC)"
	@echo ""
	@echo "$(GREEN)项目初始化完成！$(NC)"
	@echo "$(YELLOW)运行 'make dev' 启动开发服务器$(NC)"

new-module: ## 创建新模块（使用 name=模块名）
	@echo "$(BLUE)创建新模块: $(name)$(NC)"
	@if [ -z "$(name)" ]; then \
		echo "$(RED)错误: 请使用 'make new-module name=模块名' 指定模块名称$(NC)"; \
		exit 1; \
	fi
	@mkdir -p app/api/v1/$(name)
	@touch app/api/v1/$(name)/__init__.py
	@touch app/api/v1/$(name)/models.py
	@touch app/api/v1/$(name)/schemas.py
	@touch app/api/v1/$(name)/crud.py
	@touch app/api/v1/$(name)/router.py
	@touch app/api/v1/$(name)/service.py
	@touch app/api/v1/$(name)/dependencies.py
	@echo "$(GREEN)模块 $(name) 已创建$(NC)"

# ============================================
# 系统信息
# ============================================

info: ## 显示项目信息
	@echo "$(BLUE)项目信息:$(NC)"
	@echo "  项目名称: $(shell grep APP_NAME .env.example 2>/dev/null | cut -d'=' -f2 || echo 'Astrapi')"
	@echo "  Python 版本: $(shell python --version)"
	@echo "  Poetry 版本: $(shell poetry --version 2>/dev/null || echo '未安装')"
	@echo "  数据库驱动: $(shell grep DATABASE_URL .env.example 2>/dev/null | cut -d':' -f1 || echo 'unknown')"

env-show: ## 显示环境变量（不包含敏感信息）
	@echo "$(BLUE)环境变量:$(NC)"
	@echo "  APP_NAME: $(APP_NAME)"
	@echo "  APP_VERSION: $(APP_VERSION)"
	@echo "  DEBUG: $(DEBUG)"
	@echo "  LOG_LEVEL: $(LOG_LEVEL)"

ps: ## 显示运行中的进程
	@echo "$(BLUE)运行中的进程:$(NC)"
	@ps aux | grep -E "uvicorn|celery" | grep -v grep || echo "没有运行中的进程"

# ============================================
# 快捷命令
# ============================================

all: clean install lint test ## 完整构建（清理 + 安装 + 检查 + 测试）

ci: lint type-check test ## CI 流程（检查 + 类型检查 + 测试）

rebuild: clean install db-upgrade ## 重新构建项目

# ============================================
# 帮助命令分类显示
# ============================================
##开发环境: make dev, make run, make install, make update
##代码质量: make lint, make format, make type-check, make check-all
##测试: make test, make test-cov, make test-unit, make test-integration
##数据库: make db-upgrade, make db-create message='描述', make db-reset
##Docker: make docker-up, make docker-down, make docker-logs
##Celery: make celery-worker, make celery-beat, make celery-flower
##项目管理: make init, make new-module name='模块名', make info
