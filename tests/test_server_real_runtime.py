# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in real ModelRuntime inference smoke test."""

from __future__ import annotations

import os
import sys

import pytest

from kimodo.server.runtime import ModelRuntime, TextEncoderServerConfig
from kimodo.server.runtime.text_encoder import TextEncoderService
from kimodo.server.schemas import GenerationRequest


def _real_runtime_test_enabled() -> bool:
    if os.environ.get("KIMODO_RUN_REAL_RUNTIME") != "1":
        return False
    return _marker_expression_selects_inference()


def _marker_expression_selects_inference() -> bool:
    args = sys.argv[1:]
    for index, arg in enumerate(args):
        if arg == "-m" and index + 1 < len(args):
            marker_expression = args[index + 1]
        elif arg.startswith("-m") and len(arg) > 2:
            marker_expression = arg[2:]
        else:
            continue
        return "runtime" in marker_expression and "inference" in marker_expression
    return False


def _text_encoder_config_or_skip() -> TextEncoderServerConfig:
    config = TextEncoderServerConfig.from_env()
    if config.mode != "external":
        return config
    try:
        TextEncoderService(config)._probe_text_encoder_url(config.url)
    except Exception as error:
        pytest.skip(
            "external text encoder is unavailable at "
            f"{config.url} ({type(error).__name__}: {error}); start the text encoder "
            "or set TEXT_ENCODER_MODE to local, managed, or auto for this test environment"
        )
    return config


pytestmark = [
    pytest.mark.server,
    pytest.mark.runtime,
    pytest.mark.inference,
    pytest.mark.slow,
    pytest.mark.skipif(
        not _real_runtime_test_enabled(),
        reason='set KIMODO_RUN_REAL_RUNTIME=1 and run pytest with -m "runtime and inference"',
    ),
]


def test_real_model_runtime_generates_npz_artifact(tmp_path):
    model_name = os.environ.get("KIMODO_REAL_MODEL", "kimodo-soma-rp")
    device = os.environ.get("KIMODO_REAL_DEVICE", "cpu")
    runtime = ModelRuntime(device=device, text_encoder_config=_text_encoder_config_or_skip())
    request = GenerationRequest(
        texts=["A person walks forward."],
        durations=[0.5],
        model=model_name,
        diffusion_steps=1,
        num_samples=1,
        formats=["npz"],
        job_id="real-runtime-job",
    )

    try:
        result = _generate_or_skip_for_environment(runtime, request, tmp_path)
    finally:
        runtime.close()

    assert set(result.artifacts) == {"npz"}
    artifact = result.artifacts["npz"]
    assert artifact.filename == "motion.npz"
    assert artifact.download_url == "/jobs/real-runtime-job/artifacts/npz"
    assert (tmp_path / "artifacts" / artifact.filename).is_file()


def _generate_or_skip_for_environment(runtime: ModelRuntime, request: GenerationRequest, tmp_path):
    try:
        return runtime.generate(request, job_dir=tmp_path)
    except Exception as error:
        if _is_environment_setup_error(error):
            pytest.skip(f"real runtime environment is not ready: {type(error).__name__}: {error}")
        raise


def _is_environment_setup_error(error: Exception) -> bool:
    message = str(error).lower()
    setup_error_fragments = [
        "gated repo",
        "cannot access gated repo",
        "has been rejected by the repo's authors",
        "make sure to have access",
        "not a local folder",
        "could not fetch config",
        "does not appear to have a file",
        "llm2vec_base_model_path",
        "full_key: text_encoder",
    ]
    return any(fragment in message for fragment in setup_error_fragments)
