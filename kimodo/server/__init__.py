# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Server-side building blocks for running Kimodo as a persistent inference service."""

from .jobs import JobManager
from .runtime import ModelRuntime, TextEncoderServerConfig, add_text_encoder_args, text_encoder_config_from_args
from .schemas import ArtifactRecord, GenerationRequest, GenerationResult, JobRecord, JobStatus
from .storage import JobStorage

__all__ = [
    "GenerationRequest",
    "GenerationResult",
    "ArtifactRecord",
    "JobManager",
    "JobRecord",
    "JobStatus",
    "JobStorage",
    "ModelRuntime",
    "TextEncoderServerConfig",
    "add_text_encoder_args",
    "text_encoder_config_from_args",
    "create_fastapi_app",
]


def __getattr__(name: str):
    if name == "create_fastapi_app":
        from .asgi import create_fastapi_app

        return create_fastapi_app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
