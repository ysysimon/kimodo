# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Houdini geometry parsers for Kimodo constraint payloads."""

from __future__ import annotations

from .constants import (
    ALLOWED_END_EFFECTOR_JOINT_NAMES,
    DEFAULT_END_EFFECTOR_CONSTRAINT_NODE,
    DEFAULT_FULLBODY_CONSTRAINT_NODE,
    DEFAULT_HEADING_FORWARD_AXIS,
    DEFAULT_ROOT2D_CONSTRAINT_NODE,
    POSE_CONSTRAINT_NODES,
)
from .errors import PoseConstraintParseError, Root2DConstraintParseError
from .pose import pose_constraints_from_geometry
from .root2d import root2d_constraint_from_geometry
from .skeletons import _G1_ORDER, _KNOWN_SKELETON_ORDERS, _SMPLX22_ORDER, _SOMA30_ORDER, _SOMA77_ORDER

__all__ = [
    "ALLOWED_END_EFFECTOR_JOINT_NAMES",
    "DEFAULT_END_EFFECTOR_CONSTRAINT_NODE",
    "DEFAULT_FULLBODY_CONSTRAINT_NODE",
    "DEFAULT_HEADING_FORWARD_AXIS",
    "DEFAULT_ROOT2D_CONSTRAINT_NODE",
    "POSE_CONSTRAINT_NODES",
    "PoseConstraintParseError",
    "Root2DConstraintParseError",
    "_G1_ORDER",
    "_KNOWN_SKELETON_ORDERS",
    "_SMPLX22_ORDER",
    "_SOMA30_ORDER",
    "_SOMA77_ORDER",
    "pose_constraints_from_geometry",
    "root2d_constraint_from_geometry",
]
