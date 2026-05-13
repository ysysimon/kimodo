# Text Encoder

server runtime 在加载 Kimodo 模型前会准备 text encoder。`TextEncoderServerConfig` 定义 server 侧 text encoder 的启动策略，`ModelRuntime.prepare_text_encoder()` 会委托内部 `TextEncoderService` 执行具体准备工作。

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

## 环境变量

| 变量 | 用途 |
| --- | --- |
| `TEXT_ENCODER_MODE` | server text encoder 模式：`external`、`local`、`managed`、`auto` 或兼容值 `api`。 |
| `TEXT_ENCODER_URL` | external/auto/managed 模式使用的 service URL。 |
| `TEXT_ENCODER_FP32` | 是否使用 fp32 text encoder。 |
| `TEXT_ENCODER_DEVICE` | text encoder 使用的 device，例如 `cpu` 或 `cuda:0`。 |
| `TEXT_ENCODER` | text encoder 名称，默认 `llm2vec`。 |
| `TEXT_ENCODER_TMP_FOLDER` | managed subprocess 使用的临时目录。 |
| `GRADIO_SERVER_NAME` | managed text encoder service host。 |
| `GRADIO_SERVER_PORT` | managed text encoder service port，也用于默认 URL 端口。 |

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

## 资源释放

如果 runtime 使用 `managed` 模式，服务关闭时应调用：

```python
runtime.close()
```

`ModelRuntime.close()` 会委托 `TextEncoderService.close()` 终止 managed subprocess。`TextEncoderService` 也会注册 `atexit` 清理作为兜底。
