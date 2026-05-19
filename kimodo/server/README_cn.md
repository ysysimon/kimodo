# Kimodo Server

Kimodo server 模块提供一组 framework-neutral 的服务端构件，用于把 Kimodo 模型作为持久化生成服务运行，并提供一个薄的 FastAPI adapter 用于 HTTP 访问。

English documentation: [README.md](README.md).

## 当前状态

目前 server 模块已经包含：

- `Runtime`：执行一次 motion generation 的抽象接口。
- `ModelRuntime`：加载并缓存 Kimodo 模型，执行真实生成。
- `FakeRuntime`：用于测试和 smoke check 的轻量 runtime。
- `JobManager`：提交、运行、查询和取消 generation job。
- `JobStorage`：管理 job 目录、request/status JSON 和 artifact 文件。
- `GenerationRequest`：一次生成请求的数据结构。
- `ArtifactRecord`：生成文件的可下载元数据。
- `TextEncoderServerConfig`：配置 text encoder 的启动策略。
- `create_fastapi_app`：用于 HTTP route 的 FastAPI adapter factory。

FastAPI adapter 只做薄封装：route、request/response 转换和文件下载属于 adapter；模型加载、job 生命周期和 artifact 路径解析仍然保留在 server core。

## 最小使用示例

下面的例子使用 `FakeRuntime`，不会加载模型或使用 CUDA，适合检查 server wiring：

```python
from kimodo.server.app import create_app
from kimodo.server.runtime import FakeRuntime
from kimodo.server.schemas import GenerationRequest

app = create_app(storage_root="outputs/server-smoke", runtime=FakeRuntime())
jobs = app["jobs"]

record = jobs.submit(
    GenerationRequest(
        texts=["A person walks forward."],
        durations=[2.0],
        formats=["npz"],
    )
)

print(record.job_id)
```

真实服务应使用默认的 `ModelRuntime`：

```python
from kimodo.server.app import create_app

app = create_app(storage_root="outputs/server-jobs")
jobs = app["jobs"]
```

如果需要显式指定 text encoder 策略，可以在创建 app 时传入 `TextEncoderServerConfig`：

```python
from kimodo.server.app import create_app
from kimodo.server.runtime import TextEncoderServerConfig

app = create_app(
    text_encoder_config=TextEncoderServerConfig(
        mode="external",
        url="http://127.0.0.1:9550/",
    )
)
```

## FastAPI 使用

启动真实 runtime 服务：

```bash
uv run poe server
```

默认监听 `127.0.0.1:8000`。可以在 `.env.local` 中配置服务端监听地址；CLI 参数会覆盖环境变量：

```dotenv
KIMODO_SERVER_HOST=127.0.0.1
KIMODO_SERVER_PORT=8000
```

Houdini 客户端默认也会使用这两个变量拼出 `http://<host>:<port>`；如果客户端要连接另一台机器，可以通过 shelf 的 `Configure Server` 保存完整 URL。

```bash
uv run python -m kimodo.scripts.run_server --host 127.0.0.1 --port 8000
```

如果只想做轻量 smoke check，不加载模型、CUDA 或 text encoder，可以使用 `FakeRuntime`：

```bash
uv run python -m kimodo.scripts.run_server --runtime fake --storage-root outputs/server-smoke
```

也可以使用 Poe task 做 smoke check：

```bash
uv run poe server-fake
```

安装后的环境也可以使用 `kimodo_server` console script。

HTTP adapter 当前暴露：

- `GET /health`
- `POST /jobs`
- `GET /jobs/{job_id}`
- `POST /jobs/{job_id}/cancel`
- `GET /jobs/{job_id}/artifacts/{artifact_key}`

`POST /jobs` 接收与 `GenerationRequest` 相同的 generation 字段，但不接收 `job_id`；`job_id` 由服务端生成。Job 响应会刻意省略服务端本地路径 `job_dir`。Artifact 下载通过 `JobStorage.resolve_artifact(job_id, artifact_key)` 解析真实文件；adapter 不会手动拼接本地文件路径。

## Text Encoder 策略

Text encoder 策略同时影响正式 `ModelRuntime` 服务和 `runtime and inference` 真实推理测试。两者都会通过 `TextEncoderServerConfig.from_env()` 读取环境变量，并在加载 Kimodo 模型前准备 text encoder。

常用模式：

- `TEXT_ENCODER_MODE=local`：在当前进程内加载 LLM2Vec text encoder。这是 server runtime 的默认模式。
- `TEXT_ENCODER_MODE=external`：连接已经运行的 text encoder service，并检查 `TEXT_ENCODER_URL`。
- `TEXT_ENCODER_MODE=managed`：由 server runtime 启动并管理一个 text encoder subprocess。
- `TEXT_ENCODER_MODE=auto`：先尝试 external；如果 external 不可用，回退到 local。

关键环境变量：

- `TEXT_ENCODER_URL`：external/auto/managed 使用的 service URL；未设置时默认使用 `GRADIO_SERVER_PORT` 组成本机 URL。
- `GRADIO_SERVER_NAME`、`GRADIO_SERVER_PORT`：managed subprocess 启动 text encoder service 时使用的 host/port。
- `TEXT_ENCODER_DEVICE`：text encoder 使用的设备，例如 `cpu`、`cuda:0` 或 `cuda:1`。
- `TEXT_ENCODER_FP32`：是否使用 fp32 text encoder。
- `TEXT_ENCODER`：text encoder preset，当前默认是 `llm2vec`。
- `TEXT_ENCODER_TMP_FOLDER`：managed text encoder subprocess 使用的临时目录。
- `LLM2VEC_BASE_MODEL_PATH`：只覆盖 LLM2Vec 的 base model 路径，不覆盖 PEFT model。
- `TEXT_ENCODERS_DIR`：给默认 base/PEFT model 名称加本地根目录前缀；如果同时设置了 `LLM2VEC_BASE_MODEL_PATH`，base model 优先使用 `LLM2VEC_BASE_MODEL_PATH`。
- `HF_HOME`、`HUGGINGFACE_CACHE_DIR`：影响 Hugging Face/transformers 缓存位置；`HUGGINGFACE_CACHE_DIR` 会作为 LLM2Vec 加载时的 cache dir。

`local` 和 `auto` fallback 到 local 时，需要当前环境能加载 LLM2Vec，以及 gated 的 `meta-llama/Meta-Llama-3-8B-Instruct` base model。通常需要 Hugging Face 账号有 gated repo 权限并完成 `hf auth login`，或者用 `LLM2VEC_BASE_MODEL_PATH` 指向本地已有的 base model。PEFT model 仍然来自配置中的 `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised`，除非通过 `TEXT_ENCODERS_DIR` 提供本地 PEFT 路径。

本地开发时可以复制 `.env.example` 为 `.env.local`，填写自己的缓存和模型目录，然后通过 Poe 启动 text encoder：

```powershell
Copy-Item .env.example .env.local
uv run poe text-encoder
```

更多细节见 [Text Encoder](docs/cn/text-encoder.md)。

## 测试

server 测试分为默认轻量测试和显式启用的真实推理测试。默认测试不会加载真实模型，不会使用 CUDA，适合本地开发和 CI：

```bash
uv run poe test
```

这会按照项目 pytest 配置收集 `tests/` 下的所有测试；server 的真实推理测试也会被收集，但不会在普通 `uv run poe test` 中执行。

如果只想运行 server 模块的轻量测试：

```bash
uv run poe test-server
```

也可以运行所有带 `server` marker 的测试；其中真实推理测试会在未设置环境变量时自动跳过：

```bash
uv run poe test-server-all
```

真实 `ModelRuntime` 推理测试需要在具备模型、text encoder 和设备环境的机器上显式开启：

```bash
uv run poe test-runtime-real
```

如果希望测试进程直接在本地加载 text encoder，而不是 probe `TEXT_ENCODER_URL`，可以使用：

```bash
uv run poe test-runtime-real-local
```

只设置 `KIMODO_RUN_REAL_RUNTIME=1` 后运行普通 `uv run poe test` 不会执行真实推理测试；必须同时用 `-m "runtime and inference"` 显式选择这一层。真实推理测试会使用上一节的 text encoder 策略；如果 text encoder、模型权限、模型缓存或设备条件不满足，测试会跳过并提示环境未准备好。

真实推理测试专用环境变量：

- `KIMODO_RUN_REAL_RUNTIME`：真实推理测试的显式开关。只有设置为 `1`，并且 pytest 命令同时使用 `-m "runtime and inference"` 选择这一层时，真实推理测试才会执行。
- `KIMODO_REAL_MODEL`：真实推理测试使用的模型名，默认 `kimodo-soma-rp`。
- `KIMODO_REAL_DEVICE`：真实推理测试使用的设备，默认 `cpu`，例如 `cuda:0`。

这些变量只影响测试本身。上一节的 text encoder 相关环境变量不是测试专用；它们会同时影响正式 `ModelRuntime` 服务和真实推理测试。

## 阅读路径

- [请求参数](docs/cn/requests.md)：如何构造 `GenerationRequest`，各字段的含义和默认值。
- [Job 与 Artifact](docs/cn/jobs-and-artifacts.md)：job 生命周期、状态记录和 artifact 下载模型。
- [Runtime](docs/cn/runtime.md)：`Runtime`、`ModelRuntime`、`FakeRuntime` 的职责和注入方式。
- [Text Encoder](docs/cn/text-encoder.md)：text encoder 的 `external/local/managed/auto` 启动策略。
- [模块架构](docs/cn/architecture.md)：server 模块内部边界，以及 FastAPI adapter 如何接入。
