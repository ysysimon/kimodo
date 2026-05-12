# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime contract used by Kimodo server jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from ..schemas import GenerationRequest, GenerationResult


class Runtime(Protocol):
    """Minimal generation runtime contract used by server jobs."""

    def generate(self, request: GenerationRequest, *, job_dir: str | Path) -> GenerationResult:
        """Generate artifacts for one request under ``job_dir``."""
