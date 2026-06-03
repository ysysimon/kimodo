# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed KineFX pose constraint parser."""

from __future__ import annotations

from typing import Any

from .constants import _ROOT_NAMES, _ROOT_PARENT_NAMES, ALLOWED_END_EFFECTOR_JOINT_NAMES, POSE_CONSTRAINT_NODES
from .errors import PoseConstraintParseError
from .geometry import (
    _geometry_points_for_pose,
    _geometry_prims,
    _packed_embedded_geometry,
    _packed_point,
    _packed_point_attrib,
    _packed_point_raw_attrib,
    _point_attrib_value,
)
from .skeletons import _KNOWN_SKELETON_ORDERS
from .transforms import (
    _matrix4_multiply_row_major,
    _matrix4_rows,
    _matrix_to_axis_angle,
    _rotation_without_scale,
    _transpose_matrix3,
    _validate_rotation_matrix,
    _vector3,
)


def pose_constraints_from_geometry(
    geometry,
    constraint_type: str,
    *,
    frame_origin: int | float = 1,
    output_world_offset: tuple[float, float, float] | list[float] | None = None,
    skeleton_order: tuple[str, ...] | list[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build Kimodo full-body or end-effector constraints from packed KineFX poses."""
    if constraint_type not in set(POSE_CONSTRAINT_NODES.values()):
        raise PoseConstraintParseError(f"Unsupported pose constraint type {constraint_type!r}.")

    primitives = _geometry_prims(geometry)
    if not primitives:
        return [], []

    world_offset = _vector3(output_world_offset or (0.0, 0.0, 0.0), "output_world_offset")
    warnings: list[str] = []
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    inferred_order: tuple[str, ...] | None = tuple(skeleton_order) if skeleton_order is not None else None

    for prim_index, primitive in enumerate(primitives):
        packed_point = _packed_point(primitive, prim_index)
        source_frame = _packed_point_attrib(packed_point, "frame", constraint_type, prim_index)
        frame = int(float(source_frame) - float(frame_origin))
        if frame < 0:
            raise PoseConstraintParseError(
                f"{constraint_type} constraint packed primitive {prim_index} maps to negative frame {frame}; "
                f"check frame_origin={frame_origin}."
            )

        joint_group = ()
        if constraint_type == "end-effector":
            joint_group = _canonical_joint_names(
                _packed_point_raw_attrib(packed_point, "joint_names", constraint_type, prim_index)
            )

        pose_geometry = _packed_embedded_geometry(primitive, prim_index)
        pose_points = _geometry_points_for_pose(pose_geometry, constraint_type)
        raw_joints = _parse_pose_point_transforms(pose_points, constraint_type, prim_index)

        if inferred_order is None:
            inferred_order = _infer_skeleton_order(raw_joints.keys(), constraint_type, prim_index)
        root_parent_name = _root_parent_name(raw_joints)
        ignored_joints = sorted(set(raw_joints) - set(inferred_order) - set(_ROOT_PARENT_NAMES))
        if ignored_joints:
            warnings.append(
                f"{constraint_type} packed primitive {prim_index} has extra non-Kimodo joints "
                f"{', '.join(ignored_joints)}; ignoring them."
            )
        parsed_joints = _parse_expected_pose_joints(
            raw_joints,
            inferred_order,
            constraint_type,
            prim_index,
            root_parent_name=root_parent_name,
        )
        ordered_rotations, root_position = _ordered_pose_values(
            parsed_joints,
            inferred_order,
            world_offset,
            constraint_type,
            prim_index,
        )
        groups.setdefault(joint_group, []).append(
            {
                "frame": frame,
                "local_joints_rot": ordered_rotations,
                "root_position": root_position,
                "smooth_root_2d": [root_position[0], root_position[2]],
            }
        )

    constraints: list[dict[str, Any]] = []
    for joint_group, rows in groups.items():
        rows.sort(key=lambda row: row["frame"])
        _validate_unique_pose_frames(rows, constraint_type, joint_group)
        constraint: dict[str, Any] = {
            "type": constraint_type,
            "frame_indices": [row["frame"] for row in rows],
            "local_joints_rot": [row["local_joints_rot"] for row in rows],
            "root_positions": [row["root_position"] for row in rows],
            "smooth_root_2d": [row["smooth_root_2d"] for row in rows],
        }
        if constraint_type == "end-effector":
            constraint["joint_names"] = list(joint_group)
        constraints.append(constraint)

    constraints.sort(key=lambda item: (item["type"], item["frame_indices"][0], tuple(item.get("joint_names", []))))
    return constraints, warnings


def _canonical_joint_names(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError as exc:
            raise PoseConstraintParseError("end-effector joint_names must be a string array.") from exc

    allowed_by_lower = {name.lower(): name for name in ALLOWED_END_EFFECTOR_JOINT_NAMES}
    canonical_names: set[str] = set()
    for raw_name in values:
        key = str(raw_name).strip().lower()
        if key not in allowed_by_lower:
            raise PoseConstraintParseError(
                f"Unsupported end-effector joint name {raw_name!r}; expected one of "
                f"{', '.join(ALLOWED_END_EFFECTOR_JOINT_NAMES)}."
            )
        canonical_names.add(allowed_by_lower[key])

    output = [name for name in ALLOWED_END_EFFECTOR_JOINT_NAMES if name in canonical_names]
    if not output:
        raise PoseConstraintParseError("end-effector joint_names must contain at least one joint name.")
    return tuple(output)


def _parse_pose_point_transforms(points: list, constraint_type: str, prim_index: int) -> dict[str, list[list[float]]]:
    parsed: dict[str, list[list[float]]] = {}
    if not points:
        raise PoseConstraintParseError(f"{constraint_type} packed primitive {prim_index} contains an empty pose.")

    for point_index, point in enumerate(points):
        joint_name = str(_pose_point_attrib(point, "origin_name", constraint_type, prim_index, point_index))
        if not joint_name:
            raise PoseConstraintParseError(
                f"{constraint_type} packed primitive {prim_index} point {point_index} has an empty origin_name."
            )
        if joint_name in parsed:
            raise PoseConstraintParseError(
                f"{constraint_type} packed primitive {prim_index} has duplicate joint {joint_name!r}."
            )

        localtransform = _matrix4_rows(
            _pose_point_attrib(point, "localtransform", constraint_type, prim_index, point_index),
            point_index=point_index,
        )
        parsed[joint_name] = localtransform
    return parsed


def _parse_expected_pose_joints(
    raw_joints: dict[str, list[list[float]]],
    skeleton_order: tuple[str, ...],
    constraint_type: str,
    prim_index: int,
    *,
    root_parent_name: str | None = None,
) -> dict[str, tuple[list[float], list[float]]]:
    parsed: dict[str, tuple[list[float], list[float]]] = {}
    root_name = skeleton_order[0]
    for joint_name in skeleton_order:
        if joint_name not in raw_joints:
            raise PoseConstraintParseError(
                f"{constraint_type} packed primitive {prim_index} is missing skeleton joint {joint_name!r}."
            )
        localtransform = raw_joints[joint_name]
        allow_scaled_rotation = False
        if joint_name == root_name and root_parent_name is not None:
            localtransform = _matrix4_multiply_row_major(localtransform, raw_joints[root_parent_name])
            allow_scaled_rotation = True

        rotation = _rotation_without_scale(
            [localtransform[0][:3], localtransform[1][:3], localtransform[2][:3]],
            allow_scaled_rotation=allow_scaled_rotation,
            constraint_type=constraint_type,
            prim_index=prim_index,
            joint_name=joint_name,
        )
        _validate_rotation_matrix(rotation, constraint_type, prim_index, joint_name)
        translation = [localtransform[3][0], localtransform[3][1], localtransform[3][2]]
        parsed[joint_name] = (_matrix_to_axis_angle(_transpose_matrix3(rotation)), translation)
    return parsed


def _pose_point_attrib(point, name: str, constraint_type: str, prim_index: int, point_index: int) -> Any:
    try:
        return _point_attrib_value(point, name)
    except Exception as exc:
        raise PoseConstraintParseError(
            f"{constraint_type} packed primitive {prim_index} embedded pose point {point_index} "
            f"must have {name!r} attribute."
        ) from exc


def _infer_skeleton_order(joint_names: Any, constraint_type: str, prim_index: int) -> tuple[str, ...]:
    joint_set = set(joint_names)
    for order in _KNOWN_SKELETON_ORDERS:
        if joint_set == set(order):
            return order
    for order in _KNOWN_SKELETON_ORDERS:
        if set(order).issubset(joint_set):
            return order

    sizes = ", ".join(str(len(order)) for order in _KNOWN_SKELETON_ORDERS)
    raise PoseConstraintParseError(
        f"{constraint_type} packed primitive {prim_index} has {len(joint_set)} joints, but it does not match "
        f"a known complete Kimodo skeleton ({sizes} joints)."
    )


def _root_parent_name(raw_joints: dict[str, list[list[float]]]) -> str | None:
    for name in _ROOT_PARENT_NAMES:
        if name in raw_joints:
            return name
    return None


def _ordered_pose_values(
    parsed_joints: dict[str, tuple[list[float], list[float]]],
    skeleton_order: tuple[str, ...],
    output_world_offset: tuple[float, float, float],
    constraint_type: str,
    prim_index: int,
) -> tuple[list[list[float]], list[float]]:
    names = set(parsed_joints)
    expected = set(skeleton_order)
    missing = [name for name in skeleton_order if name not in names]
    unknown = sorted(names - expected)
    if missing:
        raise PoseConstraintParseError(
            f"{constraint_type} packed primitive {prim_index} is missing skeleton joint {missing[0]!r}."
        )
    if unknown:
        raise PoseConstraintParseError(
            f"{constraint_type} packed primitive {prim_index} has unknown skeleton joint {unknown[0]!r}."
        )

    root_name = skeleton_order[0]
    if root_name not in _ROOT_NAMES:
        raise PoseConstraintParseError(f"Skeleton root {root_name!r} is not recognized by the Houdini plugin.")

    root_translation = parsed_joints[root_name][1]
    root_position = [
        root_translation[0] - output_world_offset[0],
        root_translation[1] - output_world_offset[1],
        root_translation[2] - output_world_offset[2],
    ]
    return [parsed_joints[name][0] for name in skeleton_order], root_position


def _validate_unique_pose_frames(rows: list[dict[str, Any]], constraint_type: str, joint_group: tuple[str, ...]) -> None:
    seen: set[int] = set()
    for row in rows:
        frame = int(row["frame"])
        if frame in seen:
            group = f" for joint_names={list(joint_group)!r}" if joint_group else ""
            raise PoseConstraintParseError(f"{constraint_type} constraint has duplicate frame {frame}{group}.")
        seen.add(frame)
