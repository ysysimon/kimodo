# Runtime

The runtime layer executes generation. `JobManager` depends only on the
`Runtime` protocol, so tests, smoke checks, and real model execution can use
different runtime implementations.

## Runtime Protocol

The `Runtime` protocol has one core method:

```python
def generate(self, request: GenerationRequest, *, job_dir: str | Path) -> GenerationResult:
    ...
```

An implementation should write artifact files under `job_dir` and return a
`GenerationResult`. `JobManager` writes `GenerationResult.artifacts` into the
`JobRecord`.

## ModelRuntime

`ModelRuntime` is the real model runtime. It is responsible for:

- Choosing `cuda:0` or `cpu` based on device availability, unless `device` is
  provided explicitly.
- Calling `kimodo.load_model()` the first time a model name is used.
- Caching loaded models so later jobs with the same model name reuse the same
  instance.
- Preparing the text encoder before loading a model.
- Calling the model to generate motion, then using `exports.py` to write NPZ,
  BVH, or zip artifacts.

`ModelRuntime.close()` releases runtime-owned resources such as a managed text
encoder subprocess. Services should call it during shutdown.

## FakeRuntime

`FakeRuntime` is for tests and smoke checks. It does not load Kimodo models, use
CUDA, or run diffusion. It only writes tiny placeholder artifacts while
preserving the same public metadata shape as the real runtime.

Example:

```python
from kimodo.server.app import create_app
from kimodo.server.runtime import FakeRuntime

app = create_app(storage_root="outputs/server-tests", runtime=FakeRuntime())
```

This is useful for validating `JobManager`, `JobStorage`, artifact metadata, and
future HTTP adapter wiring.

## Runtime Injection

`create_app()` creates a `ModelRuntime` by default:

```python
from kimodo.server.app import create_app

app = create_app()
```

Tests or adapter smoke checks can inject a custom runtime:

```python
from kimodo.server.app import create_app
from kimodo.server.runtime import FakeRuntime

app = create_app(runtime=FakeRuntime())
```

If `runtime` is provided, `text_encoder_config` cannot be provided at the same
time. `text_encoder_config` is only used when `create_app()` creates its own
`ModelRuntime`.
