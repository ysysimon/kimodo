# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Text encoder startup strategies for the Kimodo server runtime."""

from __future__ import annotations

import argparse
import atexit
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Literal

TextEncoderMode = Literal["external", "local", "managed", "auto"]


@dataclass(frozen=True)
class TextEncoderServerConfig:
    """Server-level text encoder startup strategy."""

    mode: TextEncoderMode = "external"
    url: str = "http://127.0.0.1:9550/"
    host: str = "127.0.0.1"
    port: int = 9550
    fp32: bool = False
    device: str | None = None
    text_encoder: str = "llm2vec"
    tmp_folder: str = "/tmp/text_encoder/"
    startup_timeout_seconds: float = 120.0

    @classmethod
    def from_env(cls) -> "TextEncoderServerConfig":
        mode = _normalize_text_encoder_mode(os.environ.get("TEXT_ENCODER_MODE", "external"))
        port = int(os.environ.get("GRADIO_SERVER_PORT", "9550"))
        return cls(
            mode=mode,
            url=os.environ.get("TEXT_ENCODER_URL", f"http://127.0.0.1:{port}/"),
            host=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
            port=port,
            fp32=_env_flag("TEXT_ENCODER_FP32"),
            device=os.environ.get("TEXT_ENCODER_DEVICE") or None,
            text_encoder=os.environ.get("TEXT_ENCODER", "llm2vec"),
            tmp_folder=os.environ.get("TEXT_ENCODER_TMP_FOLDER", "/tmp/text_encoder/"),
        )


def add_text_encoder_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add server text encoder strategy options to a CLI parser."""
    parser.add_argument(
        "--text-encoder-mode",
        choices=["external", "local", "managed", "auto"],
        default=None,
        help="Server text encoder strategy. Defaults to TEXT_ENCODER_MODE or external.",
    )
    parser.add_argument(
        "--text-encoder-url",
        default=None,
        help="External text encoder URL used by external/auto modes.",
    )
    parser.add_argument(
        "--text-encoder-host",
        default=None,
        help="Host for managed text encoder service.",
    )
    parser.add_argument(
        "--text-encoder-port",
        type=int,
        default=None,
        help="Port for managed text encoder service.",
    )
    parser.add_argument(
        "--text-encoder-fp32",
        action="store_true",
        help="Use fp32 for the text encoder.",
    )
    parser.add_argument(
        "--text-encoder-device",
        default=None,
        help="Override TEXT_ENCODER_DEVICE, for example cpu or cuda:0.",
    )
    return parser


def text_encoder_config_from_args(args: argparse.Namespace) -> TextEncoderServerConfig:
    """Build a text encoder server config from argparse args and environment defaults."""
    base = TextEncoderServerConfig.from_env()
    port = args.text_encoder_port if args.text_encoder_port is not None else base.port
    if args.text_encoder_url:
        url = args.text_encoder_url
    elif args.text_encoder_port is not None:
        url = f"http://127.0.0.1:{port}/"
    else:
        url = base.url
    return TextEncoderServerConfig(
        mode=_normalize_text_encoder_mode(args.text_encoder_mode) if args.text_encoder_mode else base.mode,
        url=url,
        host=args.text_encoder_host or base.host,
        port=port,
        fp32=bool(args.text_encoder_fp32 or base.fp32),
        device=args.text_encoder_device if args.text_encoder_device is not None else base.device,
        text_encoder=base.text_encoder,
        tmp_folder=base.tmp_folder,
        startup_timeout_seconds=base.startup_timeout_seconds,
    )


class TextEncoderService:
    """Manage text encoder environment and optional managed subprocess."""

    def __init__(self, config: TextEncoderServerConfig) -> None:
        self.config = config
        self._ready = False
        self._managed_process: subprocess.Popen | None = None

    def prepare(self) -> None:
        """Apply the configured text encoder strategy before model loading."""
        if self._ready:
            return

        config = self.config
        if config.mode == "local":
            self._apply_local_text_encoder_env()
        elif config.mode == "external":
            self._apply_external_text_encoder_env(config.url)
            self._probe_text_encoder_url(config.url)
        elif config.mode == "managed":
            self._start_managed_text_encoder()
            self._apply_external_text_encoder_env(config.url)
        elif config.mode == "auto":
            self._prepare_auto_text_encoder()
        else:
            raise ValueError(f"Unsupported text encoder mode: {config.mode!r}")

        self._ready = True

    def close(self) -> None:
        """Release runtime-owned resources such as managed text encoder processes."""
        if self._managed_process is None:
            return
        process = self._managed_process
        self._managed_process = None
        if process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

    def _prepare_auto_text_encoder(self) -> None:
        config = self.config
        try:
            self._probe_text_encoder_url(config.url)
        except Exception as error:
            print(
                "Text encoder service is unreachable; server is falling back to "
                f"local text encoder. ({type(error).__name__}: {error})"
            )
            self._apply_local_text_encoder_env()
            return
        self._apply_external_text_encoder_env(config.url)

    def _apply_local_text_encoder_env(self) -> None:
        os.environ["TEXT_ENCODER_MODE"] = "local"
        self._apply_shared_text_encoder_env()

    def _apply_external_text_encoder_env(self, url: str) -> None:
        os.environ["TEXT_ENCODER_MODE"] = "api"
        os.environ["TEXT_ENCODER_URL"] = url
        self._apply_shared_text_encoder_env()

    def _apply_shared_text_encoder_env(self) -> None:
        config = self.config
        os.environ["TEXT_ENCODER"] = config.text_encoder
        if config.device:
            os.environ["TEXT_ENCODER_DEVICE"] = config.device

    def _start_managed_text_encoder(self) -> None:
        config = self.config
        if self._managed_process is not None and self._managed_process.poll() is None:
            return

        env = os.environ.copy()
        env["TEXT_ENCODER"] = config.text_encoder
        env["TEXT_ENCODER_TMP_FOLDER"] = config.tmp_folder
        env["GRADIO_SERVER_NAME"] = config.host
        env["GRADIO_SERVER_PORT"] = str(config.port)
        if config.device:
            env["TEXT_ENCODER_DEVICE"] = config.device

        command = [
            sys.executable,
            "-m",
            "kimodo.scripts.run_text_encoder_server",
            "--text-encoder",
            config.text_encoder,
            "--tmp-folder",
            config.tmp_folder,
        ]
        if config.fp32:
            command.append("--fp32")

        self._managed_process = subprocess.Popen(command, env=env)
        atexit.register(self.close)

        deadline = time.time() + config.startup_timeout_seconds
        last_error: Exception | None = None
        try:
            while time.time() < deadline:
                return_code = self._managed_process.poll()
                if return_code is not None:
                    self._managed_process = None
                    raise RuntimeError(f"Managed text encoder exited early with code {return_code}.")
                try:
                    self._probe_text_encoder_url(config.url)
                    return
                except Exception as error:
                    last_error = error
                    time.sleep(1.0)
        except Exception:
            self.close()
            raise

        self.close()
        raise RuntimeError(
            "Timed out waiting for managed text encoder at "
            f"{config.url}. Last error: {last_error}"
        )

    def _probe_text_encoder_url(self, url: str) -> None:
        from kimodo.model.text_encoder_api import TextEncoderAPI

        text_encoder = TextEncoderAPI(url)
        text_encoder(["healthcheck"])


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").lower() in {"1", "true", "yes", "on"}


def _normalize_text_encoder_mode(mode: str) -> TextEncoderMode:
    normalized = mode.lower()
    if normalized == "api":
        normalized = "external"
    if normalized not in {"external", "local", "managed", "auto"}:
        raise ValueError(
            "Invalid text encoder mode: "
            f"{mode!r}. Expected one of external, local, managed, auto."
        )
    return normalized  # type: ignore[return-value]
