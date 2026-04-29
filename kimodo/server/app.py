# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP application entry point for the Kimodo inference server.

The concrete web framework is intentionally left out of the core package for
now. Keep request handling thin here and delegate generation to ``JobManager``
and ``ModelRuntime`` as the server grows.
"""

from .jobs import JobManager
from .runtime import ModelRuntime, Runtime
from .storage import JobStorage


def create_app(storage_root: str | None = None, runtime: Runtime | None = None):
    """Create a server application.

    This placeholder wires the core objects together. A future FastAPI/Flask
    adapter can wrap this function without changing the runtime layer.
    Tests and smoke checks can pass a fake runtime to avoid loading models.
    """
    storage = JobStorage(storage_root)
    runtime = ModelRuntime() if runtime is None else runtime
    jobs = JobManager(runtime=runtime, storage=storage)
    return {
        "jobs": jobs,
        "runtime": runtime,
        "storage": storage,
    }
