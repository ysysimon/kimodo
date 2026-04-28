# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifact export helpers for server jobs."""

from pathlib import Path
from typing import Mapping

from kimodo.exports.motion_io import save_kimodo_npz


def save_npz_artifact(path: str | Path, motion: Mapping) -> str:
    """Save a Kimodo motion dictionary as a job NPZ artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_kimodo_npz(str(path), motion)
    return str(path)
