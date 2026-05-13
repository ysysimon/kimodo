# Text Encoder

The server runtime prepares the text encoder before loading a Kimodo model.
`TextEncoderServerConfig` defines the server-side text encoder startup strategy,
and `ModelRuntime.prepare_text_encoder()` delegates the actual work to the
internal `TextEncoderService`.

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

## Environment Variables

| Variable | Purpose |
| --- | --- |
| `TEXT_ENCODER_MODE` | Server text encoder mode: `external`, `local`, `managed`, `auto`, or compatibility value `api`. |
| `TEXT_ENCODER_URL` | Service URL used by `external`, `auto`, and `managed` modes. |
| `TEXT_ENCODER_FP32` | Whether to use an fp32 text encoder. |
| `TEXT_ENCODER_DEVICE` | Device for the text encoder, for example `cpu` or `cuda:0`. |
| `TEXT_ENCODER` | Text encoder name, defaulting to `llm2vec`. |
| `TEXT_ENCODER_TMP_FOLDER` | Temporary directory used by the managed subprocess. |
| `GRADIO_SERVER_NAME` | Host for the managed text encoder service. |
| `GRADIO_SERVER_PORT` | Port for the managed text encoder service; also used by the default URL. |

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

## Resource Cleanup

If the runtime uses `managed` mode, call this during service shutdown:

```python
runtime.close()
```

`ModelRuntime.close()` delegates to `TextEncoderService.close()` to terminate the
managed subprocess. `TextEncoderService` also registers `atexit` cleanup as a
fallback.
