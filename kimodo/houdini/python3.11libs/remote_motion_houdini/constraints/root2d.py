# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Root 2D constraint parser."""

from __future__ import annotations

from typing import Any

from .constants import DEFAULT_HEADING_FORWARD_AXIS
from .errors import Root2DConstraintParseError
from .geometry import _geometry_points, _has_point_attrib, _point_attrib_value, _point_position
from .transforms import _axis_vector, _heading_from_transform, _vector3


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
