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

## Text Encoder Strategy

The text encoder strategy applies to both real `ModelRuntime` services and the
`runtime and inference` real runtime test. Both paths read environment variables
through `TextEncoderServerConfig.from_env()` and prepare the text encoder before
loading a Kimodo model.

Common modes:

- `TEXT_ENCODER_MODE=external`: connect to an already-running text encoder
  service and probe `TEXT_ENCODER_URL`. This is the server runtime default.
- `TEXT_ENCODER_MODE=local`: load the LLM2Vec text encoder inside the current
  process.
- `TEXT_ENCODER_MODE=managed`: let the server runtime start and manage a text
  encoder subprocess.
- `TEXT_ENCODER_MODE=auto`: try `external` first; if unavailable, fall back to
  `local`.

Important environment variables:

- `TEXT_ENCODER_URL`: service URL used by `external`, `auto`, and `managed`;
  when unset, the default URL is derived from `GRADIO_SERVER_PORT`.
- `GRADIO_SERVER_NAME`, `GRADIO_SERVER_PORT`: host/port used by a managed text
  encoder subprocess.
- `TEXT_ENCODER_DEVICE`: text encoder device, for example `cpu` or `cuda:0`.
- `TEXT_ENCODER_FP32`: whether to use an fp32 text encoder.
- `TEXT_ENCODER`: text encoder preset, currently defaulting to `llm2vec`.
- `TEXT_ENCODER_TMP_FOLDER`: temporary directory used by the managed text
  encoder subprocess.
- `LLM2VEC_BASE_MODEL_PATH`: overrides only the LLM2Vec base model path; it does
  not override the PEFT model.
- `TEXT_ENCODERS_DIR`: prefixes the default base/PEFT model names with a local
  root directory. If `LLM2VEC_BASE_MODEL_PATH` is also set, the base model uses
  `LLM2VEC_BASE_MODEL_PATH` instead.
- `HF_HOME`, `HUGGINGFACE_CACHE_DIR`: affect Hugging Face/transformers cache
  locations; `HUGGINGFACE_CACHE_DIR` is passed as LLM2Vec's cache dir.

When `local` is used, or when `auto` falls back to `local`, the environment must
be able to load LLM2Vec and the gated `meta-llama/Meta-Llama-3-8B-Instruct` base
model. Usually this means the Hugging Face account has gated repo access and
`hf auth login` is configured, or `LLM2VEC_BASE_MODEL_PATH` points to an existing
local base model. The PEFT model still comes from
`McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised` unless
`TEXT_ENCODERS_DIR` provides a local PEFT path.

For local development, copy `.env.example` to `.env.local`, fill in your cache
and model directories, then start the text encoder through Poe:

```powershell
Copy-Item .env.example .env.local
uv run poe text-encoder
```

See [Text Encoder](docs/en/text-encoder.md) for more detail.

## Testing

Server tests are split into default lightweight tests and explicitly enabled
real inference tests. The default layer does not load real models or use CUDA,
so it is suitable for local development and CI:

```bash
uv run poe test
```

This follows the project pytest config and collects every test under `tests/`.
The server real inference test is collected too, but it does not run during a
plain `uv run poe test`.

To run only the lightweight server tests:

```bash
uv run poe test-server
```

You can also run every test with the `server` marker. The real inference test is
selected but skipped unless its opt-in environment variable is set:

```bash
uv run poe test-server-all
```

Run the real `ModelRuntime` inference smoke test only on a machine with the
required model, text encoder, and device setup:

```bash
uv run poe test-runtime-real
```

If you want the test process to load the text encoder locally instead of
probing `TEXT_ENCODER_URL`, use:

```bash
uv run poe test-runtime-real-local
```

Setting `KIMODO_RUN_REAL_RUNTIME=1` and then running plain `uv run poe test` is
not enough to run real inference; the
`-m "runtime and inference"` marker selection is required too. The real
inference test uses the text encoder strategy described above. If the text
encoder, model access, local cache, or device prerequisites are not ready, the
test is skipped with an environment setup message.

Real inference test-only environment variables:

- `KIMODO_RUN_REAL_RUNTIME`: explicit opt-in switch for the real inference test.
  The test only runs when this is set to `1` and pytest also selects
  `-m "runtime and inference"`.
- `KIMODO_REAL_MODEL`: model name for the real inference test. Defaults to
  `kimodo-soma-rp`.
- `KIMODO_REAL_DEVICE`: device for the real inference test. Defaults to `cpu`,
  for example `cuda:0`.

These variables only affect the test itself. The text encoder environment
variables in the previous section are not test-only; they affect both real
`ModelRuntime` services and the real inference test.

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
