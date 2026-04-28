# Kimodo fork 安装说明

[English version](INSTALL.en.md)

本文档记录本 fork 推荐的 uv 安装流程。上游 README 仍然保留原始说明；本文件主要用于本地开发、调试和复现当前 fork 的 Python 环境。

## 基本原则

- 使用 uv 管理虚拟环境、依赖同步、锁文件和命令运行。
- 保留 `setup.py` 负责 `MotionCorrection` 的 CMake/C++ extension 构建。
- Docker 依赖锁定流程暂不改动，继续使用 `docker_requirements.in`、`docker_requirements.txt` 和 `kimodo/scripts/lock_requirements.py`。
- PyTorch 建议按本机 CUDA/CPU 环境单独安装，避免 uv/pip 自动选择到不合适的 wheel。

## 前置要求

- Python 3.10 或更新版本
- uv
- CMake 3.15 或更新版本
- C++17 编译器
  - Windows 推荐 Visual Studio Build Tools，或可用的 MinGW toolchain。
  - Linux/macOS 使用系统 C++ 编译器即可。

`MotionCorrection` 的 C++ extension 本身不是 CUDA extension；编译时主要需要 CMake、Python development headers、pybind11、Eigen 和 C++ 编译器。运行 Kimodo 仍然需要 PyTorch，生成任务推荐使用 GPU 版 PyTorch。

## 创建虚拟环境

Windows PowerShell:

```powershell
uv venv --python 3.10
.\.venv\Scripts\activate
```

Linux/macOS:

```bash
uv venv --python 3.10
source .venv/bin/activate
```

## 安装 PyTorch

CUDA 12.4 示例:

```powershell
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

CPU 版或其他 CUDA 版本请按 PyTorch 官方安装命令替换上面的命令。这里不把 PyTorch 额外写入 uv dependency group，是为了避免在不同机器上解析到不合适的 CUDA/CPU wheel。

## 同步项目依赖

推荐主入口:

```powershell
uv sync --extra all --group dev --group docs
```

如果你已经按本机 CUDA 版本手动安装了 PyTorch，并希望避免 uv 覆盖它，推荐使用:

```powershell
uv sync --extra all --group dev --group docs --no-install-package torch
```

`torch` 虽然没有在本项目中作为直接依赖特别配置，但会通过 `peft`/`transformers` 等依赖进入解析结果。使用 `--no-install-package torch` 可以保留你手动安装的 CUDA/CPU 版本。

以上命令会安装当前项目，并通过以下链路触发 `MotionCorrection` 的 CMake 构建:

```text
uv sync
-> setuptools.build_meta
-> setup.py
-> build_ext
-> cmake
```

如果只想同步依赖，不想安装当前项目或触发 CMake 构建，可以使用:

```powershell
uv sync --no-install-project --group dev --group docs
```

## 可选安装包

本 fork 把推理服务相关依赖放在 `server` extra 中，避免基础安装强制引入 FastAPI/uvicorn 等 HTTP server 依赖。

只安装 server 依赖:

```powershell
uv sync --extra server
```

开发时通常可以继续使用完整入口:

```powershell
uv sync --extra all --group dev --group docs
```

`all` extra 已包含 `server`、demo 和 SOMA 相关依赖。后续 Kimodo server/client 架构会优先放在 `kimodo/server` 和 `kimodo/houdini` 目录中。

## 常用命令

```powershell
uv run kimodo_gen --help
uv run kimodo_demo
uv run python -m kimodo.scripts.generate --help
uv run python kimodo/scripts/lock_requirements.py
```

重新生成 uv 锁文件:

```powershell
uv lock
```

`uv lock` 会根据 `pyproject.toml` 更新 `uv.lock`；安装时再通过 `uv sync --extra all --group dev --group docs` 选择需要同步的 extras 和 dependency groups。

## 验证安装

```powershell
uv run python -c "import kimodo; import motion_correction"
uv run kimodo_gen --help
```

## Troubleshooting

如果 `uv lock` 或 `uv sync` 提示 Python 版本不兼容，请确认当前环境使用的是 Python 3.10 或更新版本。

如果 `uv sync` 在构建 `MotionCorrection` 时失败，优先检查:

- `cmake --version` 是否可用。
- 是否安装了 C++17 编译器。
- Windows 上是否已安装 Visual Studio Build Tools，或者 PATH 中是否有可用的 `g++`。
- 如果 CMake 尝试下载 pybind11/Eigen 失败，说明本机无法访问 GitHub/GitLab；可先通过系统包管理器或其他方式预装 pybind11/Eigen。

如果 PyTorch 相关导入或 CUDA 检测失败，重新按本机 CUDA 版本安装对应 PyTorch wheel。
