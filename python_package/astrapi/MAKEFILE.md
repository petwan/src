# Makefile 使用指南

本项目提供了 Makefile (Linux/Mac) 和 PowerShell 脚本 (Windows) 来简化常用开发操作。

## 快速开始

### Linux / macOS

```bash
# 查看所有可用命令
make help

# 启动开发服务器
make dev

# 运行测试
make test
```

### Windows

```powershell
# 查看所有可用命令
.\Make.ps1 help

# 启动开发服务器
.\Make.ps1 dev

# 运行测试
.\Make.ps1 test
```

或者直接双击 `start.bat` 启动开发服务器。

## 命令分类

### 环境管理

| 命令 | 说明 |
|------|------|
| `make install` / `.\Make.ps1 install` | 安装项目依赖 |
| `make install-dev` | 安装开发依赖 |
| `make update` / `.\Make.ps1 update` | 更新依赖 |
| `make clean` / `.\Make.ps1 clean` | 清理缓存和临时文件 |

### 运行应用

| 命令 | 说明 |
|------|------|
| `make run` / `.\Make.ps1 run` | 启动生产服务器 |
| `make dev` / `.\Make.ps1 dev` | 启动开发服务器（热重载） |
| `make dev-with-env` | 使用 .env 文件启动开发服务器 |

### 代码质量

| 命令 | 说明 |
|------|------|
| `make lint` / `.\Make.ps1 lint` | 运行代码检查 |
| `make lint-fix` | 自动修复代码问题 |
| `make format` / `.\Make.ps1 format` | 格式化代码 |
| `make format-check` | 检查代码格式 |
| `make type-check` / `.\Make.ps1 type-check` | 运行类型检查 |
| `make check-all` | 运行所有检查（代码检查 + 类型检查）|

### 测试

| 命令 | 说明 |
|------|------|
| `make test` / `.\Make.ps1 test` | 运行所有测试 |
| `make test-cov` | 运行测试并生成覆盖率报告 |
| `make test-unit` | 运行单元测试 |
| `make test-integration` | 运行集成测试 |

### 数据库

| 命令 | 说明 |
|------|------|
| `make db-upgrade` / `.\Make.ps1 db-upgrade` | 升级数据库到最新版本 |
| `make db-downgrade` | 回退数据库一个版本 |
| `make db-create message="描述"` / `.\Make.ps1 db-create -Message "描述"` | 创建新的数据库迁移 |
| `make db-reset` | 重置数据库（删除所有表）|
| `make db-check` | 检查数据库连接 |

### Docker

| 命令 | 说明 |
|------|------|
| `make docker-build` | 构建 Docker 镜像 |
| `make docker-up` / `.\Make.ps1 docker-up` | 启动 Docker 容器 |
| `make docker-down` / `.\Make.ps1 docker-down` | 停止 Docker 容器 |
| `make docker-logs` / `.\Make.ps1 docker-logs` | 查看 Docker 日志 |
| `make docker-restart` | 重启 Docker 容器 |
| `make docker-clean` | 清理 Docker 容器和卷 |

### Celery

| 命令 | 说明 |
|------|------|
| `make celery-worker` | 启动 Celery Worker |
| `make celery-beat` | 启动 Celery Beat（定时任务）|
| `make celery-flower` | 启动 Celery Flower（监控）|

### 文档

| 命令 | 说明 |
|------|------|
| `make docs` | 生成 API 文档 |
| `make docs-export` | 导出 OpenAPI JSON |

### 项目管理

| 命令 | 说明 |
|------|------|
| `make init` | 初始化项目（首次使用）|
| `make new-module name=模块名` | 创建新模块 |
| `make info` / `.\Make.ps1 info` | 显示项目信息 |
| `make ps` | 显示运行中的进程 |

### 系统信息

| 命令 | 说明 |
|------|------|
| `make help` | 显示帮助信息 |
| `make env-show` | 显示环境变量 |

### 快捷命令

| 命令 | 说明 |
|------|------|
| `make all` | 完整构建（清理 + 安装 + 检查 + 测试）|
| `make ci` | CI 流程（检查 + 类型检查 + 测试）|
| `make rebuild` | 重新构建项目 |

## 使用示例

### 首次使用项目

```bash
# 1. 初始化项目
make init

# 2. 启动开发服务器
make dev
```

### 日常开发流程

```bash
# 1. 启动开发服务器
make dev

# 2. 在另一个终端运行代码检查
make lint

# 3. 运行测试
make test

# 4. 创建新的数据库迁移
make db-create message="添加用户表"

# 5. 升级数据库
make db-upgrade
```

### 部署前检查

```bash
# 运行所有检查
make check-all

# 运行测试并查看覆盖率
make test-cov

# 类型检查
make type-check
```

### Docker 部署

```bash
# 构建并启动
make docker-build
make docker-up

# 查看日志
make docker-logs

# 停止并清理
make docker-down
make docker-clean
```

## Windows 用户说明

由于 Windows 默认不支持 Makefile，我们提供了三种方式：

### 1. 使用 PowerShell 脚本

```powershell
# 查看帮助
.\Make.ps1 help

# 运行命令
.\Make.ps1 dev
```

### 2. 使用批处理文件

直接双击 `start.bat` 文件即可启动开发服务器。

### 3. 安装 Make for Windows

如果你习惯使用 Make 命令，可以安装 [Make for Windows](https://www.cygwin.com/) 或使用 WSL。

## 常见问题

### Q: 如何在 Windows 上使用 Make？
A: 安装 WSL (Windows Subsystem for Linux) 或使用提供的 PowerShell 脚本。

### Q: 如何自定义命令？
A: 编辑 Makefile 文件添加新的目标，或修改现有命令。

### Q: 如何查看某个命令的详细说明？
A: 运行 `make help` 或 `.\Make.ps1 help` 查看所有命令。

### Q: make dev 和 make run 有什么区别？
A: `make dev` 启动开发服务器（支持热重载），`make run` 启动生产服务器（多进程）。

## 注意事项

1. **首次使用前**请运行 `make init` 初始化项目
2. **开发时**推荐使用 `make dev` 启动热重载服务器
3. **提交代码前**运行 `make check-all` 确保代码质量
4. **修改数据库结构**后记得运行 `make db-create` 和 `make db-upgrade`
5. **Docker 环境**使用 `make docker-up` 而不是 `make dev`

## 扩展 Makefile

如需添加自定义命令，在 Makefile 中添加新的目标即可：

```makefile
custom-command: ## 自定义命令描述
	@echo "执行自定义操作"
	# 你的命令
```

然后运行 `make help` 查看新添加的命令。
