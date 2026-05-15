# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command line entry point for the Kimodo FastAPI server."""

from __future__ import annotations

import argparse

import uvicorn

from kimodo.server.asgi import create_fastapi_app
from kimodo.server.runtime import FakeRuntime, add_text_encoder_args, text_encoder_config_from_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Kimodo FastAPI inference server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface for uvicorn.")
    parser.add_argument("--port", type=int, default=8000, help="Port for uvicorn.")
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
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
