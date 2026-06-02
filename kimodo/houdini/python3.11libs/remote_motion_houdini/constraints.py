# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Houdini geometry parsers for Kimodo constraint payloads."""

from __future__ import annotations

import math
from typing import Any

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

_SOMA77_ORDER = (
    "Hips",
    "Spine1",
    "Spine2",
    "Chest",
    "Neck1",
    "Neck2",
    "Head",
    "HeadEnd",
    "Jaw",
    "LeftEye",
    "RightEye",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumb1",
    "LeftHandThumb2",
    "LeftHandThumb3",
    "LeftHandThumbEnd",
    "LeftHandIndex1",
    "LeftHandIndex2",
    "LeftHandIndex3",
    "LeftHandIndex4",
    "LeftHandIndexEnd",
    "LeftHandMiddle1",
    "LeftHandMiddle2",
    "LeftHandMiddle3",
    "LeftHandMiddle4",
    "LeftHandMiddleEnd",
    "LeftHandRing1",
    "LeftHandRing2",
    "LeftHandRing3",
    "LeftHandRing4",
    "LeftHandRingEnd",
    "LeftHandPinky1",
    "LeftHandPinky2",
    "LeftHandPinky3",
    "LeftHandPinky4",
    "LeftHandPinkyEnd",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandThumb1",
    "RightHandThumb2",
    "RightHandThumb3",
    "RightHandThumbEnd",
    "RightHandIndex1",
    "RightHandIndex2",
    "RightHandIndex3",
    "RightHandIndex4",
    "RightHandIndexEnd",
    "RightHandMiddle1",
    "RightHandMiddle2",
    "RightHandMiddle3",
    "RightHandMiddle4",
    "RightHandMiddleEnd",
    "RightHandRing1",
    "RightHandRing2",
    "RightHandRing3",
    "RightHandRing4",
    "RightHandRingEnd",
    "RightHandPinky1",
    "RightHandPinky2",
    "RightHandPinky3",
    "RightHandPinky4",
    "RightHandPinkyEnd",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "LeftToeEnd",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
    "RightToeEnd",
)
_SOMA30_ORDER = (
    "Hips",
    "Spine1",
    "Spine2",
    "Chest",
    "Neck1",
    "Neck2",
    "Head",
    "Jaw",
    "LeftEye",
    "RightEye",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumbEnd",
    "LeftHandMiddleEnd",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandThumbEnd",
    "RightHandMiddleEnd",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
)
_G1_ORDER = (
    "pelvis_skel",
    "left_hip_pitch_skel",
    "left_hip_roll_skel",
    "left_hip_yaw_skel",
    "left_knee_skel",
    "left_ankle_pitch_skel",
    "left_ankle_roll_skel",
    "left_toe_base",
    "right_hip_pitch_skel",
    "right_hip_roll_skel",
    "right_hip_yaw_skel",
    "right_knee_skel",
    "right_ankle_pitch_skel",
    "right_ankle_roll_skel",
    "right_toe_base",
    "waist_yaw_skel",
    "waist_roll_skel",
    "waist_pitch_skel",
    "left_shoulder_pitch_skel",
    "left_shoulder_roll_skel",
    "left_shoulder_yaw_skel",
    "left_elbow_skel",
    "left_wrist_roll_skel",
    "left_wrist_pitch_skel",
    "left_wrist_yaw_skel",
    "left_hand_roll_skel",
    "right_shoulder_pitch_skel",
    "right_shoulder_roll_skel",
    "right_shoulder_yaw_skel",
    "right_elbow_skel",
    "right_wrist_roll_skel",
    "right_wrist_pitch_skel",
    "right_wrist_yaw_skel",
    "right_hand_roll_skel",
)
_SMPLX22_ORDER = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
_KNOWN_SKELETON_ORDERS = (_SOMA77_ORDER, _SOMA30_ORDER, _G1_ORDER, _SMPLX22_ORDER)


class Root2DConstraintParseError(ValueError):
    """Raised when root2d constraint geometry is malformed."""


class PoseConstraintParseError(ValueError):
    """Raised when packed KineFX pose constraint geometry is malformed."""


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


def _vector3(value: Any, name: str) -> tuple[float, float, float]:
    try:
        if len(value) == 3:
            return (float(value[0]), float(value[1]), float(value[2]))
    except Exception as exc:
        raise Root2DConstraintParseError(f"{name} must be a 3D vector.") from exc
    raise Root2DConstraintParseError(f"{name} must be a 3D vector.")


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
    tolerance: float = 0.001,
) -> None:
    for row_index in range(3):
        length = math.sqrt(sum(matrix[row_index][col] * matrix[row_index][col] for col in range(3)))
        if abs(length - 1.0) > tolerance:
            raise PoseConstraintParseError(
                f"{constraint_type} packed primitive {prim_index} joint {joint_name!r} local rotation "
                "contains scale or shear."
            )
        for other_index in range(row_index + 1, 3):
            dot = sum(matrix[row_index][col] * matrix[other_index][col] for col in range(3))
            if abs(dot) > tolerance:
                raise PoseConstraintParseError(
                    f"{constraint_type} packed primitive {prim_index} joint {joint_name!r} local rotation "
                    "contains scale or shear."
                )

    determinant = _matrix3_determinant(matrix)
    if abs(determinant - 1.0) > tolerance:
        raise PoseConstraintParseError(
            f"{constraint_type} packed primitive {prim_index} joint {joint_name!r} local rotation "
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
