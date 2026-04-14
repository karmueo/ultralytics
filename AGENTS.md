# Repository Guidelines

## 项目结构与模块组织
- `ultralytics/`：核心库代码（模型、训练引擎、配置、工具、跟踪等）。
- `tests/`：pytest 测试用例（如 `test_cli.py`、`test_engine.py`）。
- `docs/`：文档与指南（MkDocs 体系）。
- `examples/`：示例脚本与用法演示。
- `docker/`：容器与部署相关文件。
- `DeepStream-Yolo/`：DeepStream 集成示例。
- `runs/`：训练/推理输出目录（通常为本地生成产物）。

## 构建、测试与开发命令
- `pip install -e .[dev]`：以可编辑模式安装并包含开发依赖。
- `yolo predict model=yolo26n.pt source='path/to/image.jpg'`：CLI 快速预测。
- `pytest`：运行默认测试（含 doctest）。
- `pytest --slow`：运行标记为 slow 的测试。
- `ruff format .` / `ruff check .`：格式化与静态检查。

## 编码风格与命名规范
- 主要语言为 Python；行宽 120，遵循 `ruff`/`yapf`/`isort` 配置。
- 新增函数或类请写 Google 风格 docstring，建议用中文说明；返回类型若写明请使用括号格式。
- 命名遵循 Python 习惯：模块/函数 `snake_case`，类 `CamelCase`，常量 `UPPER_SNAKE_CASE`。

## 测试指南
- 测试框架为 `pytest`，覆盖率由 `pytest-cov`/`coverage` 管理。
- 测试文件以 `test_*.py` 命名；优先在现有测试文件中补充用例。
- 新功能或修复必须包含可复现的测试，避免引入回归。

## 提交与 PR 指南
- 提交信息多为动词开头的简短描述（如 `Fix ...`、`Update ...`），可在末尾附 `(#12345)`。
- PR 需描述变更动机与范围，关联问题/需求；通过 CI 测试后再提交。
- 首次贡献需遵循小范围变更原则并完成 CLA 签署。

## 许可与合规提示
- 本仓库采用 AGPL-3.0；涉及模型或代码复用时注意开源合规要求。

## 规则说明
- 所有回答使用中文回复。
- 如果有不明确的信息先提问或查阅网上最新信息，不要臆断后直接修改代码。
