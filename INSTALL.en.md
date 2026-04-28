# Kimodo fork installation guide

[中文版](INSTALL.md)

This document records the recommended uv-based installation flow for this fork. The upstream README remains unchanged; this file is intended for local development, debugging, and reproducing the Python environment for this fork.

## Principles

- Use uv to manage the virtual environment, dependency synchronization, lockfile, and command execution.
- Keep `setup.py` responsible for building the `MotionCorrection` CMake/C++ extension.
- Leave the Docker dependency locking flow unchanged: continue using `docker_requirements.in`, `docker_requirements.txt`, and `kimodo/scripts/lock_requirements.py`.
- Install PyTorch separately for your CUDA/CPU environment to avoid uv/pip selecting an unsuitable wheel automatically.

## Requirements

- Python 3.10 or newer
- uv
- CMake 3.15 or newer
- A C++17 compiler
  - On Windows, Visual Studio Build Tools is recommended. A working MinGW toolchain is also acceptable.
  - On Linux/macOS, the system C++ compiler is usually enough.

The `MotionCorrection` C++ extension itself is not a CUDA extension. Building it mainly requires CMake, Python development headers, pybind11, Eigen, and a C++ compiler. Running Kimodo still requires PyTorch, and GPU PyTorch is recommended for generation workloads.

## Create a virtual environment

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

## Install PyTorch

CUDA 12.4 example:

```powershell
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

For CPU-only or other CUDA versions, replace the command above with the official PyTorch installation command for your machine. PyTorch is not added as a special uv dependency group here so different machines can keep the correct CUDA/CPU wheel.

## Sync project dependencies

Recommended main entry point:

```powershell
uv sync --extra all --group dev --group docs
```

If you already installed PyTorch manually for your local CUDA version and want to avoid uv replacing it, use:

```powershell
uv sync --extra all --group dev --group docs --no-install-package torch
```

Although `torch` is not configured as a special direct dependency in this project, it can enter the resolved dependency graph through packages such as `peft` and `transformers`. `--no-install-package torch` keeps your manually installed CUDA/CPU PyTorch version in place.

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
uv run kimodo_gen --help
uv run kimodo_demo
uv run python -m kimodo.scripts.generate --help
uv run python kimodo/scripts/lock_requirements.py
```

Regenerate the uv lockfile:

```powershell
uv lock
```

`uv lock` updates `uv.lock` from `pyproject.toml`; choose extras and dependency groups at install time with `uv sync --extra all --group dev --group docs`.

## Verify the installation

```powershell
uv run python -c "import kimodo; import motion_correction"
uv run kimodo_gen --help
```

## Troubleshooting

If `uv lock` or `uv sync` reports an incompatible Python version, confirm that the current environment is using Python 3.10 or newer.

If `uv sync` fails while building `MotionCorrection`, check:

- `cmake --version` is available.
- A C++17 compiler is installed.
- On Windows, Visual Studio Build Tools is installed, or `g++` is available on `PATH`.
- If CMake fails while downloading pybind11/Eigen, the machine likely cannot access GitHub/GitLab. Preinstall pybind11/Eigen through a system package manager or another available method.

If PyTorch imports or CUDA detection fail, reinstall the PyTorch wheel that matches your local CUDA version.
