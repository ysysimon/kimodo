# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Houdini import helpers for downloaded motion artifacts."""

from __future__ import annotations

from pathlib import Path

_FILE_PARMS = ("file", "filename", "filepath", "clip", "source")
_RELOAD_PARMS = ("reload", "reloadfile")


def refresh_mocap_import(node, artifact_path: str | Path, child_name: str = "mocap_import_bvh") -> bool:
    """Set a child Mocap Import node to the artifact path and press reload.

    Returns ``True`` when a child node was found. Missing optional parms are
    tolerated so the callback can work across Houdini node versions.
    """
    try:
        import hou  # noqa: F401  # type: ignore
    except ImportError as exc:
        raise RuntimeError("refresh_mocap_import must be called from Houdini.") from exc

    mocap = node.node(child_name) if node is not None else None
    if mocap is None:
        return False

    path_text = str(Path(artifact_path))
    for parm_name in _FILE_PARMS:
        parm = mocap.parm(parm_name)
        if parm is not None:
            parm.set(path_text)
            break

    for parm_name in _RELOAD_PARMS:
        parm = mocap.parm(parm_name)
        if parm is not None:
            parm.pressButton()
            break

    return True
