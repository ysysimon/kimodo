# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Constraint parser exceptions."""

from __future__ import annotations


class Root2DConstraintParseError(ValueError):
    """Raised when root2d constraint geometry is malformed."""


class PoseConstraintParseError(ValueError):
    """Raised when packed KineFX pose constraint geometry is malformed."""
