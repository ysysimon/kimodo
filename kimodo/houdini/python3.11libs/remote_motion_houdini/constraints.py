# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Houdini geometry parsers for Kimodo constraint payloads."""

from __future__ import annotations

import math
from typing import Any

DEFAULT_ROOT2D_CONSTRAINT_NODE = "OUT_ROOT2D_CONSTRAINTS"
DEFAULT_HEADING_FORWARD_AXIS = "+Z"


class Root2DConstraintParseError(ValueError):
    """Raised when root2d constraint geometry is malformed."""


def root2d_constraint_from_geometry(
    geometry,
    *,
    frame_origin: int | float = 1,
    include_heading: bool = True,
    heading_forward_axis: str = DEFAULT_HEADING_FORWARD_AXIS,
    nonplanar_y_tolerance: float = 0.001,
    output_world_offset: tuple[float, float, float] | list[float] | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Build one Kimodo ``root2d`` constraint from point geometry.

    The parser reads point ``frame`` attributes, point positions, and optional
    point ``transform`` matrix3 attributes. Positions are projected to Kimodo's
    root2d XZ plane; Y is only used to report warnings.
    """
    points = _geometry_points(geometry)
    if not points:
        return None, []

    if not _has_point_attrib(geometry, points, "frame"):
        raise Root2DConstraintParseError("Root2D constraint geometry must have a point 'frame' attribute.")

    read_heading = include_heading and _has_point_attrib(geometry, points, "transform")
    forward_axis = _axis_vector(heading_forward_axis)
    world_offset = _vector3(output_world_offset or (0.0, 0.0, 0.0), "output_world_offset")
    warnings: list[str] = []
    rows = []
    seen_frames: set[int] = set()

    for index, point in enumerate(points):
        frame = int(float(_point_attrib_value(point, "frame")) - float(frame_origin))
        if frame < 0:
            raise Root2DConstraintParseError(
                f"Root2D constraint point {index} maps to negative frame {frame}; "
                f"check frame_origin={frame_origin}."
            )
        if frame in seen_frames:
            raise Root2DConstraintParseError(f"Root2D constraint has duplicate frame {frame}.")
        seen_frames.add(frame)

        world_x, world_y, world_z = _point_position(point)
        x = world_x - world_offset[0]
        y = world_y - world_offset[1]
        z = world_z - world_offset[2]
        if abs(y) > nonplanar_y_tolerance:
            warnings.append(
                f"Root2D constraint point at source frame {_point_attrib_value(point, 'frame')} has "
                f"non-planar canonical Y={y:.6g}; using X/Z only."
            )

        heading = None
        if read_heading:
            transform = _point_attrib_value(point, "transform")
            heading = _heading_from_transform(transform, forward_axis, point_index=index)

        rows.append((frame, [x, z], heading))

    rows.sort(key=lambda row: row[0])
    constraint: dict[str, Any] = {
        "type": "root2d",
        "frame_indices": [frame for frame, _pos, _heading in rows],
        "smooth_root_2d": [pos for _frame, pos, _heading in rows],
    }
    if read_heading:
        constraint["global_root_heading"] = [heading for _frame, _pos, heading in rows]

    return constraint, warnings


def _geometry_points(geometry) -> list:
    points_method = getattr(geometry, "points", None)
    if not callable(points_method):
        raise Root2DConstraintParseError("Root2D constraint source does not provide point geometry.")
    return list(points_method())


def _has_point_attrib(geometry, points: list, name: str) -> bool:
    find_point_attrib = getattr(geometry, "findPointAttrib", None)
    if callable(find_point_attrib):
        try:
            return find_point_attrib(name) is not None
        except Exception:
            pass

    if not points:
        return False
    try:
        _point_attrib_value(points[0], name)
    except Exception:
        return False
    return True


def _point_attrib_value(point, name: str) -> Any:
    attrib_value = getattr(point, "attribValue", None)
    if not callable(attrib_value):
        raise Root2DConstraintParseError("Root2D constraint points do not support attribValue().")
    value = attrib_value(name)
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _point_position(point) -> tuple[float, float, float]:
    position_method = getattr(point, "position", None)
    if callable(position_method):
        value = position_method()
    else:
        value = _point_attrib_value(point, "P")

    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except Exception as exc:
        raise Root2DConstraintParseError("Root2D constraint point position must be a 3D vector.") from exc


def _vector3(value: Any, name: str) -> tuple[float, float, float]:
    try:
        if len(value) == 3:
            return (float(value[0]), float(value[1]), float(value[2]))
    except Exception as exc:
        raise Root2DConstraintParseError(f"{name} must be a 3D vector.") from exc
    raise Root2DConstraintParseError(f"{name} must be a 3D vector.")


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
    world_x = matrix[0][0] * fx + matrix[0][1] * fy + matrix[0][2] * fz
    world_z = matrix[2][0] * fx + matrix[2][1] * fy + matrix[2][2] * fz
    length = math.hypot(world_x, world_z)
    if length <= 1e-8:
        raise Root2DConstraintParseError(
            f"Root2D constraint point {point_index} has a transform whose forward axis "
            "cannot be projected onto the XZ plane."
        )
    return [world_x / length, world_z / length]


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
