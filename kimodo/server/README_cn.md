# Kimodo Server

Kimodo server 模块提供一组 framework-neutral 的服务端构件，用于把 Kimodo 模型作为持久化生成服务运行。这里的 server 并非某个具体 HTTP 框架的应用，而是未来 HTTP/FastAPI adapter 可以复用的核心层。

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

HTTP adapter 尚未接入。未来的 FastAPI 或其他 HTTP adapter 应该保持薄封装，只负责 route、request/response 转换和文件下载，不应该把模型加载、job 生命周期或 artifact 路径解析写进 adapter 里。

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

## 阅读路径

- [请求参数](docs/cn/requests.md)：如何构造 `GenerationRequest`，各字段的含义和默认值。
- [Job 与 Artifact](docs/cn/jobs-and-artifacts.md)：job 生命周期、状态记录和 artifact 下载模型。
- [Runtime](docs/cn/runtime.md)：`Runtime`、`ModelRuntime`、`FakeRuntime` 的职责和注入方式。
- [Text Encoder](docs/cn/text-encoder.md)：text encoder 的 `external/local/managed/auto` 启动策略。
- [模块架构](docs/cn/architecture.md)：server 模块内部边界，以及未来 HTTP adapter 应如何接入。
