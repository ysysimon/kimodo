# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persistent model runtime for Kimodo server jobs."""

from __future__ import annotations

import argparse
import atexit
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Literal, Protocol
from zipfile import ZIP_DEFLATED, ZipFile

from .schemas import ArtifactRecord, GenerationRequest, GenerationResult

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


class Runtime(Protocol):
    """Minimal generation runtime contract used by server jobs."""

    def generate(self, request: GenerationRequest, *, job_dir: str | Path) -> GenerationResult:
        """Generate artifacts for one request under ``job_dir``."""


def _download_url(job_id: str | None, key: str) -> str | None:
    return f"/jobs/{job_id}/artifacts/{key}" if job_id else None


class ModelRuntime:
    """Load Kimodo models once and reuse them across generation jobs."""

    def __init__(
        self,
        device: str | None = None,
        text_encoder_config: TextEncoderServerConfig | None = None,
    ) -> None:
        import torch

        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.text_encoder_config = text_encoder_config or TextEncoderServerConfig.from_env()
        self._models = {}
        self._lock = Lock()
        self._text_encoder_ready = False
        self._managed_process: subprocess.Popen | None = None

    def get_model(self, model_name: str):
        from kimodo import load_model

        with self._lock:
            if model_name not in self._models:
                self.prepare_text_encoder()
                self._models[model_name] = load_model(
                    model_name,
                    device=self.device,
                    default_family="Kimodo",
                    text_encoder_fp32=self.text_encoder_config.fp32,
                )
            return self._models[model_name]

    def prepare_text_encoder(self) -> None:
        """Apply the configured text encoder strategy before model loading."""
        if self._text_encoder_ready:
            return

        config = self.text_encoder_config
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

        self._text_encoder_ready = True

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
        config = self.text_encoder_config
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
        config = self.text_encoder_config
        os.environ["TEXT_ENCODER"] = config.text_encoder
        if config.device:
            os.environ["TEXT_ENCODER_DEVICE"] = config.device

    def _start_managed_text_encoder(self) -> None:
        config = self.text_encoder_config
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

    @staticmethod
    def _resolve_cfg_kwargs(request: GenerationRequest) -> dict:
        """Resolve server CFG fields into Kimodo model keyword arguments."""
        from kimodo.model.cfg import CFG_TYPES

        cfg_type = request.cfg_type
        cfg_weight = request.cfg_weight

        if cfg_type is not None and cfg_type not in CFG_TYPES:
            raise ValueError(f"Invalid cfg_type: {cfg_type!r}. Expected one of {CFG_TYPES}.")

        if cfg_type == "nocfg":
            if cfg_weight is not None:
                raise ValueError("cfg_weight is not used when cfg_type is 'nocfg'.")
            return {"cfg_type": "nocfg"}

        if cfg_type == "regular":
            if not isinstance(cfg_weight, (float, int)):
                raise ValueError("cfg_type 'regular' requires cfg_weight to be one float.")
            return {"cfg_type": "regular", "cfg_weight": float(cfg_weight)}

        if cfg_type == "separated":
            if not isinstance(cfg_weight, list) or len(cfg_weight) != 2:
                raise ValueError("cfg_type 'separated' requires cfg_weight to be [text_weight, constraint_weight].")
            return {"cfg_type": "separated", "cfg_weight": [float(cfg_weight[0]), float(cfg_weight[1])]}

        if cfg_weight is None:
            return {}
        if isinstance(cfg_weight, (float, int)):
            return {"cfg_type": "regular", "cfg_weight": float(cfg_weight)}
        if isinstance(cfg_weight, list) and len(cfg_weight) == 2:
            return {"cfg_type": "separated", "cfg_weight": [float(cfg_weight[0]), float(cfg_weight[1])]}
        raise ValueError("cfg_weight must be one float or a two-float list.")

    def generate(self, request: GenerationRequest, *, job_dir: str | Path) -> GenerationResult:
        from kimodo.constraints import load_constraints_lst
        from kimodo.tools import seed_everything

        from .exports import save_bvh_artifacts, save_npz_artifacts, save_zip_artifact

        model = self.get_model(request.model)
        job_dir = Path(job_dir)

        if len(request.texts) != len(request.durations):
            raise ValueError("texts and durations must have the same length.")
        if request.num_transition_frames < 1:
            raise ValueError("num_transition_frames must be at least 1.")
        if isinstance(request.first_heading_angle, list) and len(request.first_heading_angle) not in (
            1,
            request.num_samples,
        ):
            raise ValueError("first_heading_angle must be one value or one value per sample.")
        if request.seed is not None:
            seed_everything(request.seed)

        num_frames = [int(duration * model.fps) for duration in request.durations]
        constraint_lst = (
            load_constraints_lst(request.constraints, model.skeleton, device=self.device)
            if request.constraints
            else []
        )

        cfg_kwargs = self._resolve_cfg_kwargs(request)

        # G1 post-processing is intentionally disabled in the CLI/demo too.
        use_postprocess = False if "g1" in request.model.lower() else request.postprocess

        output = model(
            request.texts,
            num_frames,
            constraint_lst=constraint_lst,
            num_denoising_steps=request.diffusion_steps,
            num_samples=request.num_samples,
            multi_prompt=True,
            first_heading_angle=request.first_heading_angle,
            num_transition_frames=request.num_transition_frames,
            post_processing=use_postprocess,
            root_margin=request.root_margin,
            return_numpy=True,
            **cfg_kwargs,
        )

        artifacts_dir = job_dir / "artifacts"
        formats = {fmt.lower() for fmt in request.formats}
        artifacts = {}
        if "npz" in formats:
            artifacts.update(
                save_npz_artifacts(
                    artifacts_dir,
                    output,
                    job_id=request.job_id,
                )
            )
        if "bvh" in formats:
            artifacts.update(
                save_bvh_artifacts(
                    artifacts_dir,
                    output,
                    skeleton=model.skeleton,
                    fps=model.fps,
                    device=self.device,
                    job_id=request.job_id,
                )
            )

        if request.zip_output and artifacts:
            artifacts = {
                "zip": save_zip_artifact(
                    artifacts_dir,
                    artifacts,
                    job_id=request.job_id,
                )
            }

        return GenerationResult(job_id=request.job_id, artifacts=artifacts)


class FakeRuntime:
    """Fast deterministic runtime for server interface tests and smoke checks.

    This runtime intentionally does not load Kimodo models, touch CUDA, or run
    diffusion. It only writes tiny placeholder artifacts with the same public
    metadata shape as the real runtime.
    """

    def generate(self, request: GenerationRequest, *, job_dir: str | Path) -> GenerationResult:
        if len(request.texts) != len(request.durations):
            raise ValueError("texts and durations must have the same length.")

        artifacts_dir = Path(job_dir) / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        formats = {fmt.lower() for fmt in request.formats}

        artifacts: dict[str, ArtifactRecord] = {}
        for fmt in ("npz", "bvh"):
            if fmt not in formats:
                continue
            artifacts.update(self._write_format_artifacts(artifacts_dir, fmt, request=request))

        if request.zip_output and artifacts:
            zip_path = artifacts_dir / "artifacts.zip"
            with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as zip_file:
                for artifact in artifacts.values():
                    zip_file.write(artifacts_dir / artifact.filename, arcname=artifact.filename)
            artifacts = {
                "zip": self._artifact_record(
                    zip_path,
                    key="zip",
                    content_type="application/zip",
                    job_id=request.job_id,
                )
            }

        return GenerationResult(job_id=request.job_id, artifacts=artifacts)

    def _write_format_artifacts(
        self,
        artifacts_dir: Path,
        fmt: str,
        *,
        request: GenerationRequest,
    ) -> dict[str, ArtifactRecord]:
        extension = fmt
        content_type = "application/octet-stream"
        n_samples = request.num_samples

        if n_samples == 1:
            path = artifacts_dir / f"motion.{extension}"
            self._write_dummy_artifact(path, request=request, key=fmt)
            return {fmt: self._artifact_record(path, key=fmt, content_type=content_type, job_id=request.job_id)}

        artifacts = {}
        for sample_idx in range(n_samples):
            key = f"{fmt}_{sample_idx:02d}"
            path = artifacts_dir / f"motion_{sample_idx:02d}.{extension}"
            self._write_dummy_artifact(path, request=request, key=key)
            artifacts[key] = self._artifact_record(
                path,
                key=key,
                content_type=content_type,
                job_id=request.job_id,
            )
        return artifacts

    @staticmethod
    def _write_dummy_artifact(path: Path, *, request: GenerationRequest, key: str) -> None:
        path.write_text(
            "\n".join(
                [
                    "Kimodo fake server artifact",
                    f"key={key}",
                    f"job_id={request.job_id}",
                    f"model={request.model}",
                    f"num_samples={request.num_samples}",
                    f"texts={request.texts!r}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _artifact_record(path: Path, *, key: str, content_type: str, job_id: str | None) -> ArtifactRecord:
        return ArtifactRecord(
            key=key,
            filename=path.name,
            content_type=content_type,
            size_bytes=path.stat().st_size,
            download_url=_download_url(job_id, key),
        )
