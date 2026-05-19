# Kimodo fork installation guide

[中文版](INSTALL.md)

This document records the recommended uv-based installation flow for this fork. The upstream README remains unchanged; this file is intended for local development, debugging, and reproducing the Python environment for this fork.

## Principles

- Use uv to manage the virtual environment, dependency synchronization, lockfile, and command execution.
- Keep `setup.py` responsible for building the `MotionCorrection` CMake/C++ extension.
- Leave the Docker dependency locking flow unchanged: continue using `docker_requirements.in`, `docker_requirements.txt`, and `kimodo/scripts/lock_requirements.py`.
- PyTorch is managed by uv. Windows/Linux use the PyTorch CUDA 13.0 wheel by default; macOS falls back to the PyPI wheel.

## Requirements

- Python 3.10 through 3.13. This project does not use Python 3.14+ yet because some PyTorch JIT paths still emit compatibility warnings on 3.14+.
- uv
- CMake 3.15 or newer
- A C++17 compiler
  - On Windows, Visual Studio Build Tools is recommended. A working MinGW toolchain is also acceptable.
  - On Linux/macOS, the system C++ compiler is usually enough.

The `MotionCorrection` C++ extension itself is not a CUDA extension. Building it mainly requires CMake, Python development headers, pybind11, Eigen, and a C++ compiler. Running Kimodo still requires PyTorch, and GPU PyTorch is recommended for generation workloads.

## Create a virtual environment

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

## PyTorch version

This fork declares `torch` directly in `pyproject.toml` and uses uv sources to control where PyTorch is resolved from:

- Windows/Linux: `https://download.pytorch.org/whl/cu130`
- macOS: the default PyPI wheel

In normal installs you do not need to run `uv pip install torch ...` manually. `uv sync` installs the PyTorch version locked in `uv.lock` for the current platform. Verify it with:

```powershell
uv run python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

If you specifically need CPU-only PyTorch or a different CUDA version, skip the project-locked torch package and then install the desired wheel manually:

```powershell
uv sync --extra all --group dev --group docs --no-install-package torch
uv pip install torch --index-url <your-pytorch-index-url>
```

## Sync project dependencies

Recommended main entry point:

```powershell
uv sync --extra all --group dev --group docs
```

If you already installed PyTorch manually for your local CUDA version and want to avoid uv replacing it, use:

```powershell
uv sync --extra all --group dev --group docs --no-install-package torch
```

`torch` is now a direct project dependency. `--no-install-package torch` keeps your manually installed CUDA/CPU PyTorch version in place, but normal development should let uv manage PyTorch from the lockfile.

The commands above install the current project and trigger the `MotionCorrection` CMake build through this chain:

```text
uv sync
-> setuptools.build_meta
-> setup.py
-> build_ext
-> cmake
```

If you only want to sync dependencies without installing the current project or triggering the CMake build, use:

```powershell
uv sync --no-install-project --group dev --group docs
```

## Optional extras

This fork keeps inference-server dependencies in the `server` extra so the base install does not force FastAPI/uvicorn or other HTTP server packages into every environment.

Install only the server extra:

```powershell
uv sync --extra server
```

For development, the full entry point remains:

```powershell
uv sync --extra all --group dev --group docs
```

The `all` extra includes `server`, demo, and SOMA-related dependencies. The Kimodo server/client architecture will live primarily under `kimodo/server` and `kimodo/houdini`.

## Common commands

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

The recommended entry point is the project-locked Poe version: `uv run poe <task>`. If you installed `poethepoet` globally with `pipx` or another tool, you can also run `poe <task>` directly after completing `uv sync`.

## Run the inference server

The real `ModelRuntime` service loads the model and uses the server and text encoder strategy from `.env.local` or the current environment. It listens on `127.0.0.1:8000` by default:

```powershell
uv run poe server
```

The server bind address can be set in `.env.local`:

```dotenv
KIMODO_SERVER_HOST=127.0.0.1
KIMODO_SERVER_PORT=8000
```

The Houdini client uses the same variables by default to build `http://<host>:<port>`. If Houdini needs to connect to another machine, save the full URL with the shelf `Configure Server` tool, or set:

```dotenv
KIMODO_REMOTE_MOTION_SERVER_URL=http://192.168.1.10:8000
```

To temporarily override the host, port, or other startup options, run the server entry point directly. CLI arguments take precedence over environment variables:

```powershell
uv run python -m kimodo.scripts.run_server --host 127.0.0.1 --port 8000
```

For HTTP server/client wiring checks without loading the real model, CUDA, or text encoder, use the fake runtime:

```powershell
uv run poe server-fake
```

If you explicitly use `TEXT_ENCODER_MODE=external`, start the text encoder in another terminal first:

```powershell
uv run poe text-encoder
```

See `kimodo/server/README.md` for routes, request formats, and text encoder strategy details.

## Run tests

Run pytest through uv so the tests use the Python interpreter, dependencies, and lockfile resolved for this project:

```powershell
uv run poe test
uv run poe test-server
```

## Local text encoder environment variables

If you cannot access the gated `meta-llama/Meta-Llama-3-8B-Instruct` repository on Hugging Face directly, you can download a compatible Llama base model from another source and point Kimodo's LLM2Vec text encoder at the local directory with `LLM2VEC_BASE_MODEL_PATH`.

Copy `.env.example` to a local config file:

```powershell
Copy-Item .env.example .env.local
```

Then fill in machine-specific values in `.env.local`, for example:

```dotenv
HF_HOME='D:\AI_cache\hf_cache'
LLM2VEC_BASE_MODEL_PATH='D:\AI_cache\Meta-Llama-3-8B-Instruct'
TEXT_ENCODER_MODE=local
TEXT_ENCODER_DEVICE=cuda:0
```

Start the text encoder:

```powershell
uv run poe text-encoder
```

`LLM2VEC_BASE_MODEL_PATH` only overrides the LLM2Vec base model path. The PEFT adapter is still resolved from the Hugging Face cache or `TEXT_ENCODERS_DIR`. Use `TEXT_ENCODER_DEVICE=cuda:0`, `cuda:1`, and so on to choose a CUDA device; to reduce VRAM usage, set `TEXT_ENCODER_DEVICE=cpu` in `.env.local`.

Regenerate the uv lockfile:

```powershell
uv lock
```

`uv lock` updates `uv.lock` from `pyproject.toml`; choose extras and dependency groups at install time with `uv sync --extra all --group dev --group docs`.

## Verify the installation

```powershell
uv run poe check-import
uv run poe torch-info
```

## Troubleshooting

If `uv lock` or `uv sync` reports an incompatible Python version, confirm that the current environment is using Python 3.10 through 3.13.

If `uv sync` fails while building `MotionCorrection`, check:

- `cmake --version` is available.
- A C++17 compiler is installed.
- On Windows, Visual Studio Build Tools is installed, or `g++` is available on `PATH`.
- If CMake fails while downloading pybind11/Eigen, the machine likely cannot access GitHub/GitLab. Preinstall pybind11/Eigen through a system package manager or another available method.

If PyTorch imports or CUDA detection fail, first check `torch.__version__`, `torch.version.cuda`, and `torch.cuda.is_available()` with the verification command above. Windows/Linux should resolve to a `+cu130` wheel by default; use `--no-install-package torch` plus a manual install only if you need a different CUDA/CPU wheel.
