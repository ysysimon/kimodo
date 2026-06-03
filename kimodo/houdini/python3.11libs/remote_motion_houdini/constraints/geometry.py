# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Houdini geometry access helpers for constraint parsing."""

from __future__ import annotations

from typing import Any

from .errors import PoseConstraintParseError, Root2DConstraintParseError


def _geometry_points(geometry) -> list:
    points_method = getattr(geometry, "points", None)
    if not callable(points_method):
        raise Root2DConstraintParseError("Root2D constraint source does not provide point geometry.")
    return list(points_method())


def _geometry_prims(geometry) -> list:
    prims_method = getattr(geometry, "prims", None)
    if not callable(prims_method):
        raise PoseConstraintParseError("Pose constraint source does not provide primitive geometry.")
    return list(prims_method())


def _geometry_points_for_pose(geometry, constraint_type: str) -> list:
    points_method = getattr(geometry, "points", None)
    if not callable(points_method):
        raise PoseConstraintParseError(f"{constraint_type} embedded pose does not provide point geometry.")
    return list(points_method())


def _packed_point(primitive, prim_index: int):
    points_method = getattr(primitive, "points", None)
    if not callable(points_method):
        raise PoseConstraintParseError(f"Packed primitive {prim_index} does not provide points().")
    points = list(points_method())
    if len(points) != 1:
        raise PoseConstraintParseError(f"Packed primitive {prim_index} must have exactly one packed point.")
    return points[0]


def _packed_embedded_geometry(primitive, prim_index: int):
    embedded_method = getattr(primitive, "getEmbeddedGeometry", None)
    if callable(embedded_method):
        embedded = embedded_method()
        if embedded is not None:
            return embedded

    geometry_method = getattr(primitive, "geometry", None)
    if callable(geometry_method):
        embedded = geometry_method()
        if embedded is not None:
            return embedded

    raise PoseConstraintParseError(
        f"Packed primitive {prim_index} does not expose embedded geometry via getEmbeddedGeometry()."
    )


def _packed_point_attrib(point, name: str, constraint_type: str, prim_index: int) -> Any:
    try:
        return _point_attrib_value(point, name)
    except Exception as exc:
        raise PoseConstraintParseError(
            f"{constraint_type} packed primitive {prim_index} outer point must have {name!r} attribute."
        ) from exc


def _packed_point_raw_attrib(point, name: str, constraint_type: str, prim_index: int) -> Any:
    try:
        return _point_attrib_raw_value(point, name)
    except Exception as exc:
        raise PoseConstraintParseError(
            f"{constraint_type} packed primitive {prim_index} outer point must have {name!r} attribute."
        ) from exc


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
    value = _point_attrib_raw_value(point, name)
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return value[0]
    return value


def _point_attrib_raw_value(point, name: str) -> Any:
    attrib_value = getattr(point, "attribValue", None)
    if not callable(attrib_value):
        raise ValueError("Constraint points do not support attribValue().")
    return attrib_value(name)


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
