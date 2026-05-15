# Text Encoder

The server runtime prepares the text encoder before loading a Kimodo model.
`TextEncoderServerConfig` defines the server-side text encoder startup strategy,
and `ModelRuntime.prepare_text_encoder()` delegates the actual work to the
internal `TextEncoderService`.

This strategy applies to both real server runtime usage and the
`runtime and inference` real runtime test. The test does not define separate
text encoder behavior; it reads `TextEncoderServerConfig.from_env()` just like a
real `ModelRuntime` service, so the same environment variables and precedence
rules apply.

## Startup Modes

| Mode | Meaning | Useful when |
| --- | --- | --- |
| `external` | Use an already-running text encoder service and probe its URL before model loading. | Production deployments or manually managed text encoder processes. |
| `local` | Do not use an API service; create and use the text encoder object inside the same process as the model. | Simple single-process runs, or when you do not want an extra text encoder service. |
| `managed` | Let the server runtime start and manage a text encoder subprocess. | You want the server to bring up the text encoder automatically. |
| `auto` | Probe the external URL first; if it fails, fall back to `local`. | Development environments or deployments that need to tolerate multiple setups. |

`TEXT_ENCODER_MODE=api` is accepted as a compatibility alias for `external`.

## Configuration Sources

You can construct a config directly:

```python
from kimodo.server.runtime import TextEncoderServerConfig

config = TextEncoderServerConfig(
    mode="external",
    url="http://127.0.0.1:9550/",
    device="cpu",
)
```

You can also build it from environment variables or CLI args.
`text_encoder_config_from_args()` first reads `TextEncoderServerConfig.from_env()`
and then applies explicit CLI args on top.

## Local Poe Workflow

For local development, keep machine-specific values in `.env.local` and start
the text encoder through the project Poe task:

```powershell
Copy-Item .env.example .env.local
uv run poe text-encoder
```

On Linux/macOS, use the same `.env.local` file and run the same Poe task:

```bash
cp .env.example .env.local
uv run poe text-encoder
```

The `text-encoder` task loads optional `.env.local` values and runs
`kimodo_textencoder`. Extra CLI args are passed through, for example
`uv run poe text-encoder --fp32`.

## Environment Variables

Not every text encoder environment variable is active in every mode. First use
the current mode to decide which variables matter:

| Variable | Main modes | Purpose |
| --- | --- | --- |
| `TEXT_ENCODER_MODE` | All | Server text encoder mode: `external`, `local`, `managed`, `auto`, or compatibility value `api`. |
| `TEXT_ENCODER_URL` | `external`, `auto`, `managed` probe | Service URL. `local` ignores it; `managed` uses it while waiting for the subprocess to become available. |
| `TEXT_ENCODER_FP32` | `local`, `managed`, direct `kimodo_textencoder` runs | Whether to use an fp32 text encoder. In pure `external` mode, the current process does not use it to load an encoder. |
| `TEXT_ENCODER_DEVICE` | `local`, `managed`, `auto` fallback | Device for the text encoder, for example `cpu` or `cuda:0`. In pure `external` mode, set it on the process that actually runs the text encoder service. |
| `TEXT_ENCODER` | `local`, `managed`, direct `kimodo_textencoder` runs | Text encoder name, defaulting to `llm2vec`. In pure `external` mode, the remote service decides which encoder it uses. |
| `TEXT_ENCODER_TMP_FOLDER` | `managed`, direct `kimodo_textencoder` runs | Directory where the text encoder server writes temporary embedding files. |
| `GRADIO_SERVER_NAME` | `managed`, direct `kimodo_textencoder` runs | Host for the text encoder service. |
| `GRADIO_SERVER_PORT` | `managed`, direct `kimodo_textencoder` runs, default URL | Port for the text encoder service; also used to derive the default URL when `TEXT_ENCODER_URL` is unset. |

LLM2Vec-specific variables:

| Variable | Main modes | Purpose |
| --- | --- | --- |
| `LLM2VEC_BASE_MODEL_PATH` | `local`, `managed`, `auto` fallback | Overrides the LLM2Vec base model path only. It does not override the PEFT model. |
| `TEXT_ENCODERS_DIR` | `local`, `managed`, `auto` fallback | Prefixes the default base/PEFT model names with a local root directory. If `LLM2VEC_BASE_MODEL_PATH` is also set, the base model uses `LLM2VEC_BASE_MODEL_PATH`, while the PEFT model still resolves through `TEXT_ENCODERS_DIR`. |
| `HF_HOME` | The process loading LLM2Vec locally | Default Hugging Face cache root read by Hugging Face/transformers. |
| `HUGGINGFACE_CACHE_DIR` | The process loading LLM2Vec locally | Passed as the transformers cache dir when LLM2Vec loads. |

In pure `external` mode, the current server process only needs
`TEXT_ENCODER_URL`. If you manually start the external text encoder service on
the same machine, set `HF_HOME`, `LLM2VEC_BASE_MODEL_PATH`,
`TEXT_ENCODER_DEVICE`, and related loading variables on that service process.

`LLM2VEC_BASE_MODEL_PATH` is consumed inside `LLM2VecEncoder`: it replaces the
configured `base_model_name_or_path`, such as the default LLM2Vec base model
name. It does not replace `peft_model_name_or_path`. To provide a local PEFT
model, use `TEXT_ENCODERS_DIR` with the expected directory layout, or adjust the
actual loading config.

## CLI Args

`add_text_encoder_args(parser)` adds:

```text
--text-encoder-mode
--text-encoder-url
--text-encoder-host
--text-encoder-port
--text-encoder-fp32
--text-encoder-device
```

These args are intended for a server entrypoint or future HTTP adapter startup
command. The core server package only provides parsing and configuration; it
does not define a concrete CLI here.

## Managed Mode

`managed` mode starts the text encoder through the current Python interpreter:

```text
python -m kimodo.scripts.run_text_encoder_server
```

The server sets environment variables such as `TEXT_ENCODER`,
`TEXT_ENCODER_TMP_FOLDER`, `GRADIO_SERVER_NAME`, and `GRADIO_SERVER_PORT`, then
waits for `TEXT_ENCODER_URL` to become available. If the process exits early or
does not become ready within `startup_timeout_seconds`, the server raises
`RuntimeError` and closes the subprocess.

## Auto Mode

`auto` mode probes `url` first:

- If the probe succeeds, it uses the external/API text encoder.
- If the probe fails, it prints a fallback message and switches to the text
  encoder object created inside the same process.

This is convenient for development. Production deployments should usually choose
`external`, `local`, or `managed` explicitly.

Note: after `auto` falls back to `local`, the current process must be able to
load LLM2Vec and the gated `meta-llama/Meta-Llama-3-8B-Instruct` base model.
Usually that means the Hugging Face account has gated repo access and is
authenticated, or `LLM2VEC_BASE_MODEL_PATH` points to an existing local base
model. Otherwise, a real runtime service will fail during loading; the real
runtime test reports this kind of missing environment as a skip.

## Resource Cleanup

If the runtime uses `managed` mode, call this during service shutdown:

```python
runtime.close()
```

`ModelRuntime.close()` delegates to `TextEncoderService.close()` to terminate the
managed subprocess. `TextEncoderService` also registers `atexit` cleanup as a
fallback.
