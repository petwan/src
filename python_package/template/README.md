# Python 包 Cookiecutter 模板

这是一个用于创建现代 Python 包的 [Cookiecutter](https://github.com/cookiecutter/cookiecutter) 模板，使用 [Poetry](https://python-poetry.org/) 作为包管理器。

## 特性

- 使用 Poetry 进行现代化 Python 打包
- 使用 Typer 构建命令行界面（现代化的Click替代品，支持自动生成帮助和类型提示）
- 使用 Loguru 进行日志记录（更好地支持多线程和多进程）
- 预配置的常用路径常量（ROOT_PATH, DEFAULT_CONFIG_FILE等）
- 使用 pytest 进行预配置测试
- 使用 Ruff 进行快速代码格式化和检查（替代Black和Flake8）
- 使用 MyPy 进行类型检查
- 预配置 pre-commit hooks（使用本地配置，更快更可靠）
- GitHub Actions CI/CD
- Docker 和 Docker Compose 支持
- 标准项目结构

## 要求

- Python 3.6+
- [Cookiecutter](https://github.com/cookiecutter/cookiecutter): `pip install cookiecutter`
- [Poetry](https://python-poetry.org/): `pip install poetry`

## 使用方法

创建新的 Python 项目使用此模板：

```bash
cookiecutter .
```

这将提示您输入几个变量：

- `package_name`: Python 包名称
- `package_version`: 初始版本（默认：0.1.0）
- `package_description`: 包的简短描述
- `author_name`: 您的姓名
- `author_email`: 您的邮箱
- `github_user`: 您的 GitHub 用户名
- `license`: 许可证类型（MIT、BSD-3-Clause、Apache-2.0 或 GNU GPL v3.0）
- `python_version`: 最低 Python 版本（默认：3.8）

## 模板变量

该模板接受以下变量：

- `package_name`: Python 包名称（用于目录名和导入）
- `package_version`: 包的初始版本（默认：0.1.0）
- `package_description`: 包的描述
- `author_name`: 作者姓名
- `author_email`: 作者邮箱
- `github_user`: GitHub 用户名
- `license`: 包的许可证（选项：MIT、BSD-3-Clause、Apache-2.0、GNU GPL v3.0）
- `python_version`: 最低 Python 版本（默认：3.8）

## 生成的项目内容

运行 cookiecutter 命令后，您将得到一个包含以下内容的新项目：

- 标准 Python 包结构
- 用于依赖管理的 Poetry 配置
- 使用 Typer 的 CLI 入口点（现代化的Click替代品）
- 使用 Loguru 的日志配置（更好地支持多线程和多进程）
- 预配置的常用路径常量
- 使用 pytest 的测试配置
- 代码质量工具（Ruff、MyPy）
- 本地预配置的 pre-commit hooks（更快更可靠）
- GitHub Actions 工作流程
- Docker 和 Docker Compose 文件
- 包含常用命令的 Makefile
- 文档模板

### 生成的项目结构

```
your-package-name/
├── your_package_name/           # 主要包目录
│   ├── __init__.py              # 包初始化
│   ├── cli/                     # 命令行接口模块
│   │   ├── __init__.py
│   │   └── main.py              # 使用 Typer 的 CLI 入口点
│   └── utils/                   # 实用程序模块
│       ├── __init__.py
│       ├── constants.py          # 常用路径常量
│       ├── exceptions.py         # 自定义异常
│       ├── logging_config.py     # 使用 Loguru 的日志配置
│       └── type_utils.py         # 类型定义
├── tests/                       # 测试目录
│   └── test_example.py          # 示例测试文件
├── .github/                     # GitHub 配置
│   └── workflows/
│       └── test.yml             # GitHub Actions 工作流程
├── .gitignore                   # Git 忽略规则
├── .pre-commit-config.yaml      # 使用本地配置的 pre-commit hooks
├── Dockerfile                   # Docker 配置
├── docker-compose.yml           # Docker Compose 配置
├── Makefile                     # 常用开发命令
├── README.md                    # 项目 README
├── example.py                   # 示例用法
└── pyproject.toml               # Poetry 配置
```

### 开始使用您的新项目

1. 导航到新创建的项目：
   ```bash
   cd your-package-name
   ```

2. 使用 Poetry 安装依赖项：
   ```bash
   poetry install
   ```

3. 运行示例以确保一切正常工作：
   ```bash
   poetry run python example.py
   ```

4. 运行 CLI：
   ```bash
   poetry run your_package_name
   ```

5. 检查版本：
   ```bash
   poetry run your_package_name version
   ```

6. 运行测试：
   ```bash
   make test
   ```

7. 安装 pre-commit hooks：
   ```bash
   poetry run pre-commit install
   ```

### 开发命令

生成的项目包含一个 Makefile，其中包含有用的命令：

- `make install`: 安装依赖项
- `make test`: 运行测试
- `make lint`: 使用 Ruff 检查代码
- `make check`: 运行类型检查
- `make format`: 使用 Ruff 格式化代码
- `make clean`: 清理缓存文件
- `make ci`: 运行所有检查（lint、type check、tests）

## 许可证

此 Cookiecutter 模板根据 MIT 许可证授权。