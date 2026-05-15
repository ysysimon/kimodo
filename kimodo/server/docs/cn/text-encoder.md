# Text Encoder

server runtime 在加载 Kimodo 模型前会准备 text encoder。`TextEncoderServerConfig` 定义 server 侧 text encoder 的启动策略，`ModelRuntime.prepare_text_encoder()` 会委托内部 `TextEncoderService` 执行具体准备工作。

这套策略同时影响正式运行的 server runtime 和 `runtime and inference` 真实推理测试。真实推理测试不会定义另一套 text encoder 行为；它同样读取 `TextEncoderServerConfig.from_env()`，因此环境变量的含义和优先级与正式 `ModelRuntime` 一致。

## 启动模式

| 模式 | 含义 | 适用场景 |
| --- | --- | --- |
| `external` | 使用已经运行的 text encoder service，并在模型加载前 probe URL。 | 生产或手动管理 text encoder 进程。 |
| `local` | 不使用 API service，让模型在同一进程内创建并使用 text encoder 对象。 | 简单单进程运行，或不想额外启动 text encoder service。 |
| `managed` | server runtime 启动并管理一个 text encoder subprocess。 | 希望 server 自动拉起 text encoder。 |
| `auto` | 优先 probe external URL，失败时 fallback 到 local。 | 开发环境或兼容多种部署方式。 |

`TEXT_ENCODER_MODE=api` 会被兼容解析为 `external`。

## 配置来源

可以直接构造 config：

```python
from kimodo.server.runtime import TextEncoderServerConfig

config = TextEncoderServerConfig(
    mode="external",
    url="http://127.0.0.1:9550/",
    device="cpu",
)
```

也可以通过环境变量或 CLI args 构造。`text_encoder_config_from_args()` 会先读取 `TextEncoderServerConfig.from_env()`，再用显式 CLI args 覆盖对应字段。

## 本地 Poe 工作流

本地开发时，把本机专属配置放在 `.env.local`，再通过项目 Poe task 启动 text encoder：

```powershell
Copy-Item .env.example .env.local
uv run poe text-encoder
```

Linux/macOS 使用同一份 `.env.local`，启动命令也保持一致：

```bash
cp .env.example .env.local
uv run poe text-encoder
```

`text-encoder` task 会加载可选的 `.env.local`，然后运行 `kimodo_textencoder`。额外 CLI 参数会继续透传，例如 `uv run poe text-encoder --fp32`。

## 环境变量

不是所有 text encoder 环境变量都会在所有模式下生效。先按当前模式判断变量作用范围：

| 变量 | 主要生效模式 | 用途 |
| --- | --- | --- |
| `TEXT_ENCODER_MODE` | 全部 | server text encoder 模式：`external`、`local`、`managed`、`auto` 或兼容值 `api`。 |
| `TEXT_ENCODER_URL` | `external`、`auto`、`managed` probe | service URL。`local` 不使用它；`managed` 会用它等待 subprocess 可用。 |
| `TEXT_ENCODER_FP32` | `local`、`managed`、直接启动 `kimodo_textencoder` | 是否使用 fp32 text encoder。纯 `external` 时，当前进程不会用它加载 encoder。 |
| `TEXT_ENCODER_DEVICE` | `local`、`managed`、`auto` fallback | text encoder 使用的 device，例如 `cpu` 或 `cuda:0`。纯 `external` 时，应在实际运行 text encoder service 的进程中设置。 |
| `TEXT_ENCODER` | `local`、`managed`、直接启动 `kimodo_textencoder` | text encoder 名称，默认 `llm2vec`。纯 `external` 时，远端服务使用哪个 encoder 由远端进程决定。 |
| `TEXT_ENCODER_TMP_FOLDER` | `managed`、直接启动 `kimodo_textencoder` | text encoder server 写临时 embedding 文件的目录。 |
| `GRADIO_SERVER_NAME` | `managed`、直接启动 `kimodo_textencoder` | text encoder service host。 |
| `GRADIO_SERVER_PORT` | `managed`、直接启动 `kimodo_textencoder`、默认 URL | text encoder service port，也用于未设置 `TEXT_ENCODER_URL` 时生成默认 URL。 |

LLM2Vec 相关变量：

| 变量 | 主要生效模式 | 用途 |
| --- | --- | --- |
| `LLM2VEC_BASE_MODEL_PATH` | `local`、`managed`、`auto` fallback | 覆盖 LLM2Vec 的 base model 路径。只影响 base model，不覆盖 PEFT model。 |
| `TEXT_ENCODERS_DIR` | `local`、`managed`、`auto` fallback | 给默认 base/PEFT model 名称加本地根目录前缀。如果同时设置了 `LLM2VEC_BASE_MODEL_PATH`，base model 使用 `LLM2VEC_BASE_MODEL_PATH`，PEFT model 仍按 `TEXT_ENCODERS_DIR` 解析。 |
| `HF_HOME` | 本地加载 LLM2Vec 的进程 | Hugging Face 默认缓存根目录，由 Hugging Face/transformers 读取。 |
| `HUGGINGFACE_CACHE_DIR` | 本地加载 LLM2Vec 的进程 | 作为 LLM2Vec 加载时传给 transformers 的 cache dir。 |

纯 `external` 模式下，当前 server 进程只需要知道 `TEXT_ENCODER_URL`。如果外部 text encoder service 也在本机由你手动启动，那么 `HF_HOME`、`LLM2VEC_BASE_MODEL_PATH`、`TEXT_ENCODER_DEVICE` 等变量应该设置在启动那个 service 的进程里。

`LLM2VEC_BASE_MODEL_PATH` 的作用点在 `LLM2VecEncoder` 内部：它会替换配置中的 `base_model_name_or_path`，例如默认的 LLM2Vec base model 名称；但 `peft_model_name_or_path` 不会被它替换。若需要本地 PEFT model，请使用 `TEXT_ENCODERS_DIR` 提供对应目录布局，或调整实际加载配置。

## CLI 参数

`add_text_encoder_args(parser)` 会添加：

```text
--text-encoder-mode
--text-encoder-url
--text-encoder-host
--text-encoder-port
--text-encoder-fp32
--text-encoder-device
```

这些参数用于 server entrypoint 或未来 HTTP adapter 的启动命令。当前核心模块只提供解析和配置能力，不定义具体 CLI。

## Managed 模式

`managed` 模式会通过当前 Python 解释器启动：

```text
python -m kimodo.scripts.run_text_encoder_server
```

server 会设置 `TEXT_ENCODER`、`TEXT_ENCODER_TMP_FOLDER`、`GRADIO_SERVER_NAME`、`GRADIO_SERVER_PORT` 等环境变量，并等待 `TEXT_ENCODER_URL` 可用。若进程提前退出或在 `startup_timeout_seconds` 内没有准备好，会抛出 `RuntimeError` 并关闭 subprocess。

## Auto 模式

`auto` 模式会先 probe `url`：

- probe 成功：使用 external/API text encoder。
- probe 失败：打印 fallback 信息，并切换到同一进程内的 text encoder 对象。

这适合开发环境，但生产环境更建议显式使用 `external`、`local` 或 `managed`。

注意：`auto` fallback 到 `local` 后，需要当前进程能加载 LLM2Vec 和 gated 的 `meta-llama/Meta-Llama-3-8B-Instruct` base model。通常需要 Hugging Face 账号有 gated repo 访问权限并完成认证，或者设置 `LLM2VEC_BASE_MODEL_PATH` 指向本地已有的 base model。否则正式 runtime 会加载失败；真实推理测试会把这类环境未准备好的情况报告为 skip。

## 资源释放

如果 runtime 使用 `managed` 模式，服务关闭时应调用：

```python
runtime.close()
```

`ModelRuntime.close()` 会委托 `TextEncoderService.close()` 终止 managed subprocess。`TextEncoderService` 也会注册 `atexit` 清理作为兜底。
