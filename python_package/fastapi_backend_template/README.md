1. 使用 cookiecutter 创建项目
bash
pip install cookiecutter
cd fastapi_backend_template
cookiecutter .
1. 开发环境设置
bash
cd <your-project-name>
poetry install
cp .env.example .env
poetry run dev
1. 使用 Docker
bash
docker-compose up -d
1. 运行测试
bash
make test
# 或
poetry run pytest