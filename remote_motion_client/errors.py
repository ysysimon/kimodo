# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Error types raised by :mod:`remote_motion_client`."""

from __future__ import annotations

from typing import Any


class RemoteMotionError(Exception):
    """Base class for remote motion client errors."""


class RemoteMotionHTTPError(RemoteMotionError):
    """HTTP transport or non-success response error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        url: str | None = None,
        response_text: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.url = url
        self.response_text = response_text


class RemoteMotionJobFailedError(RemoteMotionError):
    """Raised when a job reaches a failed or canceled terminal status."""

    def __init__(self, job: Any) -> None:
        message = getattr(job, "message", "") or getattr(job, "error", "") or "Remote motion job did not succeed."
        status = getattr(job, "status", "unknown")
        job_id = getattr(job, "job_id", "unknown")
        super().__init__(f"Job {job_id} ended with status {status}: {message}")
        self.job = job


class RemoteMotionTimeoutError(RemoteMotionError):
    """Raised when waiting for a job exceeds the requested timeout."""


class RemoteMotionArtifactError(RemoteMotionError):
    """Raised when artifact metadata or download output is invalid."""
