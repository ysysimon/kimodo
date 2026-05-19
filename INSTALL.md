# Kimodo fork 安装说明

[English version](INSTALL.en.md)

本文档记录本 fork 推荐的 uv 安装流程。上游 README 仍然保留原始说明；本文件主要用于本地开发、调试和复现当前 fork 的 Python 环境。

## 基本原则

- 使用 uv 管理虚拟环境、依赖同步、锁文件和命令运行。
- 保留 `setup.py` 负责 `MotionCorrection` 的 CMake/C++ extension 构建。
- Docker 依赖锁定流程暂不改动，继续使用 `docker_requirements.in`、`docker_requirements.txt` 和 `kimodo/scripts/lock_requirements.py`。
- PyTorch 已纳入 uv 管理。Windows/Linux 默认使用 PyTorch CUDA 13.0 wheel；macOS 回退到 PyPI wheel。

## 前置要求

- Python 3.10 到 3.13。当前项目不使用 Python 3.14+，因为 PyTorch 的部分 JIT 路径在 3.14+ 上仍会产生兼容性警告。
- uv
- CMake 3.15 或更新版本
- C++17 编译器
  - Windows 推荐 Visual Studio Build Tools，或可用的 MinGW toolchain。
  - Linux/macOS 使用系统 C++ 编译器即可。

`MotionCorrection` 的 C++ extension 本身不是 CUDA extension；编译时主要需要 CMake、Python development headers、pybind11、Eigen 和 C++ 编译器。运行 Kimodo 仍然需要 PyTorch，生成任务推荐使用 GPU 版 PyTorch。

## 创建虚拟环境

Windows PowerShell:

```powershell
uv venv --python 3.13
.\.venv\Scripts\activate
```

Linux/macOS:

```bash
uv venv --python 3.13
source .venv/bin/activate
```

## PyTorch 版本

本 fork 在 `pyproject.toml` 中直接声明 `torch`，并通过 uv source 配置控制 PyTorch 来源:

- Windows/Linux: `https://download.pytorch.org/whl/cu130`
- macOS: PyPI 默认 wheel

因此通常不需要手动运行 `uv pip install torch ...`。`uv sync` 会按当前平台安装 `uv.lock` 中锁定的 PyTorch 版本。同步后可用下面的命令确认:

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

如果你确实需要 CPU-only 或其他 CUDA 版本，可以临时跳过项目锁定的 torch，然后手动安装目标 wheel:

```powershell
uv sync --extra all --group dev --group docs --no-install-package torch
uv pip install torch --index-url <your-pytorch-index-url>
```

## 同步项目依赖

推荐主入口:

```powershell
uv sync --extra all --group dev --group docs
```

如果你已经按本机 CUDA 版本手动安装了 PyTorch，并希望避免 uv 覆盖它，推荐使用:

```powershell
uv sync --extra all --group dev --group docs --no-install-package torch
```

`torch` 现在是项目直接依赖。使用 `--no-install-package torch` 可以保留你手动安装的 CUDA/CPU 版本，但正常开发推荐直接让 uv 按锁文件管理 PyTorch。

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
uv run poe
uv run poe torch-info
uv run poe demo
uv run poe text-encoder
uv run poe server
uv run poe server-fake
uv run poe check-import
uv run python kimodo/scripts/lock_requirements.py
```

推荐使用项目锁定的 `poe` 版本，即 `uv run poe <task>`。如果你已经用 `pipx` 或其他方式全局安装了 `poethepoet`，也可以在完成 `uv sync` 后直接运行 `poe <task>`。

## 运行推理服务

真实 `ModelRuntime` 服务会加载模型，并按 `.env.local` 或当前环境变量中的服务端和 text encoder 策略启动。默认监听 `127.0.0.1:8000`：

```powershell
uv run poe server
```

服务端监听地址可以写在 `.env.local` 中：

```dotenv
KIMODO_SERVER_HOST=127.0.0.1
KIMODO_SERVER_PORT=8000
```

Houdini 客户端默认也会使用这两个变量拼出 `http://<host>:<port>`。如果 Houdini 需要连接另一台机器，推荐在 shelf 的 `Configure Server` 中保存完整 URL，或设置：

```dotenv
KIMODO_REMOTE_MOTION_SERVER_URL=http://192.168.1.10:8000
```

如果需要临时覆盖 host、port 或其他参数，可以直接运行 server entrypoint；CLI 参数优先级高于环境变量：

```powershell
uv run python -m kimodo.scripts.run_server --host 127.0.0.1 --port 8000
```

只想检查 HTTP server/client wiring，不加载真实模型、CUDA 或 text encoder 时，使用 fake runtime：

```powershell
uv run poe server-fake
```

如果显式使用 `TEXT_ENCODER_MODE=external`，需要先在另一个终端启动 text encoder：

```powershell
uv run poe text-encoder
```

更多 route、请求格式和 text encoder 策略见 `kimodo/server/README_cn.md`。

## 运行测试

推荐通过 uv 运行 pytest，这样测试使用的 Python、依赖和 `uv.lock` 保持一致:

```powershell
uv run poe test
uv run poe test-server
```

## 本地 text encoder 环境变量

如果无法直接访问 Hugging Face 上的 gated `meta-llama/Meta-Llama-3-8B-Instruct`，可以先从其他来源下载兼容的 Llama base model 到本地目录，然后通过 `LLM2VEC_BASE_MODEL_PATH` 指定给 Kimodo 的 LLM2Vec text encoder 使用。

复制 `.env.example` 为本机配置文件：

```powershell
Copy-Item .env.example .env.local
```

然后在 `.env.local` 中填写本机路径，例如：

```dotenv
HF_HOME='D:\AI_cache\hf_cache'
LLM2VEC_BASE_MODEL_PATH='D:\AI_cache\Meta-Llama-3-8B-Instruct'
TEXT_ENCODER_MODE=local
TEXT_ENCODER_DEVICE=cuda:0
```

启动 text encoder：

```powershell
uv run poe text-encoder
```

`LLM2VEC_BASE_MODEL_PATH` 只覆盖 LLM2Vec 的 base model 路径；PEFT adapter 仍按 Hugging Face cache 或 `TEXT_ENCODERS_DIR` 查找。可以用 `TEXT_ENCODER_DEVICE=cuda:0`、`cuda:1` 等指定 CUDA 设备；如果希望降低显存占用，可以在 `.env.local` 中设置 `TEXT_ENCODER_DEVICE=cpu`。

重新生成 uv 锁文件:

```powershell
uv lock
```

`uv lock` 会根据 `pyproject.toml` 更新 `uv.lock`；安装时再通过 `uv sync --extra all --group dev --group docs` 选择需要同步的 extras 和 dependency groups。

## 验证安装

```powershell
uv run poe check-import
uv run poe torch-info
```

## Troubleshooting

如果 `uv lock` 或 `uv sync` 提示 Python 版本不兼容，请确认当前环境使用的是 Python 3.10 到 3.13。

如果 `uv sync` 在构建 `MotionCorrection` 时失败，优先检查:

- `cmake --version` 是否可用。
- 是否安装了 C++17 编译器。
- Windows 上是否已安装 Visual Studio Build Tools，或者 PATH 中是否有可用的 `g++`。
- 如果 CMake 尝试下载 pybind11/Eigen 失败，说明本机无法访问 GitHub/GitLab；可先通过系统包管理器或其他方式预装 pybind11/Eigen。

如果 PyTorch 相关导入或 CUDA 检测失败，先用上面的确认命令检查 `torch.__version__`、`torch.version.cuda` 和 `torch.cuda.is_available()`。默认 Windows/Linux 环境应解析到 `+cu130` wheel；如果你需要其他 CUDA/CPU wheel，再使用 `--no-install-package torch` 加手动安装覆盖。
