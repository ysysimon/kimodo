# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Motion import helpers for Houdini.

This module should stay importable outside Houdini. Functions that require
``hou`` import it lazily so the client can still be tested in regular Python.
"""


def import_motion(path: str):
    """Import a Kimodo artifact into Houdini.

    The concrete KineFX/BVH/NPZ implementation will be added after the server
    contract is in place.
    """
    try:
        import hou  # type: ignore
    except ImportError as exc:
        raise RuntimeError("import_motion must be called from Houdini hython.") from exc

    raise NotImplementedError(f"Houdini import for {path!r} is not implemented yet.")
