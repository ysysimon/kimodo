# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Constraint parser constants."""

from __future__ import annotations

DEFAULT_ROOT2D_CONSTRAINT_NODE = "OUT_ROOT2D_CONSTRAINTS"
DEFAULT_FULLBODY_CONSTRAINT_NODE = "OUT_FULLBODY_CONSTRAINTS"
DEFAULT_END_EFFECTOR_CONSTRAINT_NODE = "OUT_END_EFFECTOR_CONSTRAINTS"
POSE_CONSTRAINT_NODES = {
    DEFAULT_FULLBODY_CONSTRAINT_NODE: "fullbody",
    DEFAULT_END_EFFECTOR_CONSTRAINT_NODE: "end-effector",
    "OUT_left_hand_CONSTRAINTS": "left-hand",
    "OUT_right-hand_CONSTRAINTS": "right-hand",
    "OUT_left-foot_CONSTRAINTS": "left-foot",
    "OUT_right-foot_CONSTRAINTS": "right-foot",
}
DEFAULT_HEADING_FORWARD_AXIS = "+Z"
ALLOWED_END_EFFECTOR_JOINT_NAMES = ("LeftFoot", "RightFoot", "LeftHand", "RightHand", "Hips")
_ROOT_NAMES = ("Hips", "pelvis_skel", "pelvis")
_ROOT_PARENT_NAMES = ("Root",)
