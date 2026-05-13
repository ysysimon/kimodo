# Kimodo Server

Kimodo server provides framework-neutral building blocks for running Kimodo
models as a persistent generation service. The server package is not tied to a
specific HTTP framework yet; it is the core layer that a future HTTP/FastAPI
adapter can wrap.

Chinese documentation is also available: [README_cn.md](README_cn.md).

## Current Status

The server package currently includes:

- `Runtime`: the protocol for running one motion generation request.
- `ModelRuntime`: loads and caches Kimodo models, then runs real generation.
- `FakeRuntime`: lightweight runtime for tests and smoke checks.
- `JobManager`: submits, runs, queries, and cancels generation jobs.
- `JobStorage`: manages job directories, request/status JSON, and artifact files.
- `GenerationRequest`: the data contract for one generation request.
- `ArtifactRecord`: downloadable metadata for generated files.
- `TextEncoderServerConfig`: configures how the text encoder is prepared.

A concrete HTTP adapter is not included yet. A future FastAPI or other HTTP
adapter should stay thin: routes, request/response conversion, and file
downloads belong there; model loading, job lifecycle management, and artifact
path resolution should remain in the server core.

## Minimal Usage

The following example uses `FakeRuntime`. It does not load a model, use CUDA, or
run diffusion, so it is useful for checking server wiring:

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

Real service usage should use the default `ModelRuntime`:

```python
from kimodo.server.app import create_app

app = create_app(storage_root="outputs/server-jobs")
jobs = app["jobs"]
```

To configure the text encoder explicitly, pass `TextEncoderServerConfig` when
creating the app:

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

## Reading Path

- [Request parameters](docs/en/requests.md): how to build a `GenerationRequest`
  and what each field means.
- [Jobs and artifacts](docs/en/jobs-and-artifacts.md): job lifecycle, status
  records, and artifact download model.
- [Runtime](docs/en/runtime.md): responsibilities and injection patterns for
  `Runtime`, `ModelRuntime`, and `FakeRuntime`.
- [Text Encoder](docs/en/text-encoder.md): the `external/local/managed/auto`
  text encoder startup strategies.
- [Architecture](docs/en/architecture.md): server module boundaries and how a
  future HTTP adapter should connect.
