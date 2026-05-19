# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command line entry point for the Kimodo FastAPI server."""

from __future__ import annotations

import argparse
import os

import uvicorn

from kimodo.server.asgi import create_fastapi_app
from kimodo.server.runtime import FakeRuntime, add_text_encoder_args, text_encoder_config_from_args

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
HOST_ENV = "KIMODO_SERVER_HOST"
PORT_ENV = "KIMODO_SERVER_PORT"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Kimodo FastAPI inference server.")
    parser.add_argument("--host", default=None, help=f"Host interface for uvicorn. Defaults to ${HOST_ENV}.")
    parser.add_argument("--port", type=int, default=None, help=f"Port for uvicorn. Defaults to ${PORT_ENV}.")
    parser.add_argument("--storage-root", default=None, help="Directory for server job state and artifacts.")
    parser.add_argument(
        "--runtime",
        choices=["real", "fake"],
        default="real",
        help="Runtime implementation. Use fake for smoke tests without model/CUDA dependencies.",
    )
    return add_text_encoder_args(parser)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    runtime = FakeRuntime() if args.runtime == "fake" else None
    text_encoder_config = None if runtime is not None else text_encoder_config_from_args(args)
    app = create_fastapi_app(
        storage_root=args.storage_root,
        runtime=runtime,
        text_encoder_config=text_encoder_config,
    )
    uvicorn.run(app, host=args.host or os.environ.get(HOST_ENV, DEFAULT_HOST), port=_resolve_port(args.port))


def _resolve_port(cli_port: int | None) -> int:
    if cli_port is not None:
        return cli_port
    return int(os.environ.get(PORT_ENV, DEFAULT_PORT))


if __name__ == "__main__":
    main()
