# Runtime

runtime 层负责真正执行一次 generation。`JobManager` 只依赖 `Runtime` protocol，因此测试、smoke check 和真实模型执行可以使用不同 runtime。

## Runtime Protocol

`Runtime` protocol 只有一个核心方法：

```python
def generate(self, request: GenerationRequest, *, job_dir: str | Path) -> GenerationResult:
    ...
```

实现者需要在 `job_dir` 下写入 artifact 文件，并返回 `GenerationResult`。`JobManager` 会把 `GenerationResult.artifacts` 写入 `JobRecord`。

## ModelRuntime

`ModelRuntime` 是真实模型 runtime。它负责：

- 根据设备可用性选择 `cuda:0` 或 `cpu`，也可以通过 `device` 显式指定。
- 在首次使用某个 model name 时调用 `kimodo.load_model()`。
- 缓存已加载模型，后续同名 model 复用同一个实例。
- 在加载模型前准备 text encoder。
- 调用模型生成 motion，并通过 `exports.py` 写出 NPZ、BVH 或 zip artifact。

`ModelRuntime.close()` 会释放 runtime 拥有的资源，例如 managed text encoder subprocess。服务关闭时应调用它。

## FakeRuntime

`FakeRuntime` 用于测试和 smoke check。它不会加载 Kimodo 模型，不会使用 CUDA，也不会运行 diffusion；它只写入很小的 placeholder artifact，并保持和真实 runtime 相同的 public metadata 形状。

示例：

```python
from kimodo.server.app import create_app
from kimodo.server.runtime import FakeRuntime

app = create_app(storage_root="outputs/server-tests", runtime=FakeRuntime())
```

这适合验证 `JobManager`、`JobStorage`、artifact metadata 和未来 HTTP adapter 的 wiring。

## Runtime 注入

`create_app()` 默认创建 `ModelRuntime`：

```python
from kimodo.server.app import create_app

app = create_app()
```

测试或 adapter smoke check 可以注入自定义 runtime：

```python
from kimodo.server.app import create_app
from kimodo.server.runtime import FakeRuntime

app = create_app(runtime=FakeRuntime())
```

如果传入了 `runtime`，就不能同时传 `text_encoder_config`。`text_encoder_config` 只在 `create_app()` 自己创建 `ModelRuntime` 时使用。
