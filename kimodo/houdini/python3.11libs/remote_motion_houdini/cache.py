# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cache path helpers for Houdini remote motion downloads."""

from __future__ import annotations

from pathlib import Path


def default_download_dir(job_id: str, hip_dir: str | Path | None = None) -> Path:
    """Return the default local download directory for one job."""
    root = Path(hip_dir) if hip_dir is not None else _houdini_hip_dir()
    return root / "motion_gen" / job_id


def _houdini_hip_dir() -> Path:
    try:
        import hou  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Houdini cache paths require hou or an explicit hip_dir.") from exc

    hip = hou.expandString("$HIP")
    return Path(hip) if hip else Path.cwd()
