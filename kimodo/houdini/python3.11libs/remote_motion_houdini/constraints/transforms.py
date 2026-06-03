# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Matrix and transform helpers for constraint parsing."""

from __future__ import annotations

import math
from typing import Any

from .constants import DEFAULT_HEADING_FORWARD_AXIS
from .errors import PoseConstraintParseError, Root2DConstraintParseError


def _vector3(value: Any, name: str) -> tuple[float, float, float]:
    try:
        if len(value) == 3:
            return (float(value[0]), float(value[1]), float(value[2]))
    except Exception as exc:
        raise Root2DConstraintParseError(f"{name} must be a 3D vector.") from exc
    raise Root2DConstraintParseError(f"{name} must be a 3D vector.")


def _matrix4_rows(value: Any, *, point_index: int) -> list[list[float]]:
    as_tuple = getattr(value, "asTuple", None)
    if callable(as_tuple):
        value = as_tuple()

    try:
        if len(value) == 4 and all(hasattr(row, "__len__") and len(row) == 4 for row in value):
            return [[float(value[row][col]) for col in range(4)] for row in range(4)]
        if len(value) == 16:
            flat = [float(item) for item in value]
            return [flat[0:4], flat[4:8], flat[8:12], flat[12:16]]
    except Exception as exc:
        raise PoseConstraintParseError(
            f"Pose point {point_index} localtransform must be a 4x4 matrix or 16-number sequence."
        ) from exc

    raise PoseConstraintParseError(
        f"Pose point {point_index} localtransform must be a 4x4 matrix or 16-number sequence."
    )


def _matrix4_multiply_row_major(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [
        [sum(left[row][k] * right[k][col] for k in range(4)) for col in range(4)]
        for row in range(4)
    ]


def _rotation_without_scale(
    matrix: list[list[float]],
    *,
    allow_scaled_rotation: bool,
    constraint_type: str,
    prim_index: int,
    joint_name: str,
) -> list[list[float]]:
    if not allow_scaled_rotation:
        return matrix

    normalized: list[list[float]] = []
    for row in matrix:
        length = math.sqrt(sum(component * component for component in row))
        if length <= 1e-8:
            raise PoseConstraintParseError(
                f"{constraint_type} packed primitive {prim_index} joint {joint_name!r} local rotation "
                "contains a zero scale axis."
            )
        normalized.append([component / length for component in row])
    return normalized


def _validate_rotation_matrix(
    matrix: list[list[float]],
    constraint_type: str,
    prim_index: int,
    joint_name: str,
    *,
    rotation_label: str = "local rotation",
    tolerance: float = 0.001,
) -> None:
    for row_index in range(3):
        length = math.sqrt(sum(matrix[row_index][col] * matrix[row_index][col] for col in range(3)))
        if abs(length - 1.0) > tolerance:
            raise PoseConstraintParseError(
                f"{constraint_type} packed primitive {prim_index} joint {joint_name!r} {rotation_label} "
                "contains scale or shear."
            )
        for other_index in range(row_index + 1, 3):
            dot = sum(matrix[row_index][col] * matrix[other_index][col] for col in range(3))
            if abs(dot) > tolerance:
                raise PoseConstraintParseError(
                    f"{constraint_type} packed primitive {prim_index} joint {joint_name!r} {rotation_label} "
                    "contains scale or shear."
                )

    determinant = _matrix3_determinant(matrix)
    if abs(determinant - 1.0) > tolerance:
        raise PoseConstraintParseError(
            f"{constraint_type} packed primitive {prim_index} joint {joint_name!r} {rotation_label} "
            "must have determinant 1."
        )


def _matrix3_determinant(matrix: list[list[float]]) -> float:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def _transpose_matrix3(matrix: list[list[float]]) -> list[list[float]]:
    return [[matrix[col][row] for col in range(3)] for row in range(3)]


def _matrix_to_axis_angle(matrix: list[list[float]]) -> list[float]:
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat = [
            0.25 * scale,
            (matrix[2][1] - matrix[1][2]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[1][0] - matrix[0][1]) / scale,
        ]
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        quat = [
            (matrix[2][1] - matrix[1][2]) / scale,
            0.25 * scale,
            (matrix[0][1] + matrix[1][0]) / scale,
            (matrix[0][2] + matrix[2][0]) / scale,
        ]
    elif matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        quat = [
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[0][1] + matrix[1][0]) / scale,
            0.25 * scale,
            (matrix[1][2] + matrix[2][1]) / scale,
        ]
    else:
        scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
        quat = [
            (matrix[1][0] - matrix[0][1]) / scale,
            (matrix[0][2] + matrix[2][0]) / scale,
            (matrix[1][2] + matrix[2][1]) / scale,
            0.25 * scale,
        ]

    length = math.sqrt(sum(component * component for component in quat))
    quat = [component / length for component in quat]
    if quat[0] < 0.0:
        quat = [-component for component in quat]

    vector_length = math.sqrt(quat[1] * quat[1] + quat[2] * quat[2] + quat[3] * quat[3])
    if vector_length < 1e-8:
        return [0.0, 0.0, 0.0]

    angle = 2.0 * math.atan2(vector_length, quat[0])
    scale = angle / vector_length
    return [quat[1] * scale, quat[2] * scale, quat[3] * scale]


def _axis_vector(axis: str) -> tuple[float, float, float]:
    normalized = str(axis or DEFAULT_HEADING_FORWARD_AXIS).strip().upper()
    axes = {
        "+X": (1.0, 0.0, 0.0),
        "-X": (-1.0, 0.0, 0.0),
        "+Z": (0.0, 0.0, 1.0),
        "-Z": (0.0, 0.0, -1.0),
    }
    if normalized not in axes:
        raise Root2DConstraintParseError(
            f"Unsupported heading_forward_axis {axis!r}; expected one of {', '.join(axes)}."
        )
    return axes[normalized]


def _heading_from_transform(
    transform: Any,
    forward_axis: tuple[float, float, float],
    *,
    point_index: int,
) -> list[float]:
    matrix = _matrix3_rows(transform)
    fx, fy, fz = forward_axis
    world_x = fx * matrix[0][0] + fy * matrix[1][0] + fz * matrix[2][0]
    world_z = fx * matrix[0][2] + fy * matrix[1][2] + fz * matrix[2][2]
    length = math.hypot(world_x, world_z)
    if length <= 1e-8:
        raise Root2DConstraintParseError(
            f"Root2D constraint point {point_index} has a transform whose forward axis "
            "cannot be projected onto the XZ plane."
        )
    # Kimodo stores heading as [cos(theta), sin(theta)], where theta=0 faces +Z.
    return [world_z / length, world_x / length]


def _matrix3_rows(value: Any) -> list[list[float]]:
    as_tuple = getattr(value, "asTuple", None)
    if callable(as_tuple):
        value = as_tuple()

    try:
        if len(value) == 3 and all(hasattr(row, "__len__") and len(row) == 3 for row in value):
            return [[float(value[row][col]) for col in range(3)] for row in range(3)]
        if len(value) == 9:
            flat = [float(item) for item in value]
            return [flat[0:3], flat[3:6], flat[6:9]]
    except Exception as exc:
        raise Root2DConstraintParseError("Point transform must be a 3x3 matrix or 9-number sequence.") from exc

    raise Root2DConstraintParseError("Point transform must be a 3x3 matrix or 9-number sequence.")
