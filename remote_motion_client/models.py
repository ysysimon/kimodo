# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Data models for the remote motion HTTP client."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

TERMINAL_STATUSES = {"succeeded", "failed", "canceled"}


@dataclass(frozen=True)
class ArtifactInfo:
    key: str
    filename: str
    content_type: str
    size_bytes: int
    download_url: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArtifactInfo":
        return cls(
            key=str(data["key"]),
            filename=str(data["filename"]),
            content_type=str(data.get("content_type", "")),
            size_bytes=int(data.get("size_bytes", 0)),
            download_url=data.get("download_url"),
            payload=dict(data),
        )


@dataclass(frozen=True)
class JobInfo:
    job_id: str
    status: str
    progress: float = 0.0
    message: str = ""
    artifacts: dict[str, ArtifactInfo] = field(default_factory=dict)
    error: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JobInfo":
        artifacts = data.get("artifacts") or {}
        return cls(
            job_id=str(data["job_id"]),
            status=str(data["status"]),
            progress=float(data.get("progress", 0.0)),
            message=str(data.get("message", "")),
            artifacts={
                key: ArtifactInfo.from_dict(value) if isinstance(value, dict) else value
                for key, value in artifacts.items()
            },
            error=data.get("error"),
            payload=dict(data),
        )
