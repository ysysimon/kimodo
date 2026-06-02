# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HDA callback entry points for remote Kimodo motion generation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from remote_motion_client import RemoteMotionClient

from .cache import default_download_dir
from .config import get_server_url
from .constraints import (
    DEFAULT_ROOT2D_CONSTRAINT_NODE,
    POSE_CONSTRAINT_NODES,
    pose_constraints_from_geometry,
    root2d_constraint_from_geometry,
)
from .importer import format_houdini_path, refresh_mocap_import


def model_menu(kwargs: dict[str, Any] | None = None) -> list[str]:
    """Return model menu items as Houdini token/label pairs."""
    from kimodo.model.registry import AVAILABLE_MODELS, DEFAULT_MODEL, FRIENDLY_NAMES

    names = list(dict.fromkeys(AVAILABLE_MODELS))
    if DEFAULT_MODEL in names:
        names.remove(DEFAULT_MODEL)
        names.insert(0, DEFAULT_MODEL)

    items = []
    for name in names:
        items.extend([name, FRIENDLY_NAMES.get(name, name)])
    return items


def on_created(kwargs: dict[str, Any]) -> None:
    """Initialize HDA parms when a Kimodo remote motion node is created."""
    from kimodo.model.registry import DEFAULT_MODEL

    node = kwargs["node"]
    model_parm = node.parm("model")
    if model_parm is not None:
        model_parm.set(DEFAULT_MODEL)


def generate_motion(kwargs: dict[str, Any]) -> Path:
    """Generate one BVH motion from HDA parms and import it into the node."""
    node = kwargs["node"]
    server_url = _eval_parm(node, "server_url", None) or get_server_url()
    texts, durations = _eval_prompt_segments(node)
    seed = _eval_parm(node, "seed", None)
    model = _eval_menu_token(node, "model", None)
    poll_interval = float(_eval_parm(node, "poll_interval", 1.0))
    wait_timeout = _eval_parm(node, "wait_timeout", None)

    payload: dict[str, Any] = {
        "texts": texts,
        "durations": durations,
        "formats": ["bvh"],
    }
    if seed not in (None, ""):
        payload["seed"] = int(seed)
    if model not in (None, ""):
        payload["model"] = model
    _apply_optional_generation_parms(node, payload)
    output_world_offset = _output_world_offset(node)
    if _is_nonzero_vector(output_world_offset):
        payload["output_world_offset"] = list(output_world_offset)
    _apply_constraints(node, payload, output_world_offset)

    client = RemoteMotionClient(server_url)
    submitted = client.submit(payload)
    job = client.wait(
        submitted.job_id,
        poll_interval=poll_interval,
        timeout=None if wait_timeout in (None, "") else float(wait_timeout),
    )
    download_dir = default_download_dir(job.job_id)
    artifact_path = client.download_artifact(job, "bvh", download_dir)

    artifact_parm = node.parm("artifact_path")
    if artifact_parm is not None:
        artifact_parm.set(str(artifact_path))
    bvh_parm = node.parm("bvhfile")
    if bvh_parm is not None:
        bvh_parm.set(format_houdini_path(artifact_path))

    refresh_mocap_import(node, artifact_path)
    return artifact_path


def _eval_prompt_segments(node) -> tuple[list[str], list[float]]:
    indexed_prompt_names = _indexed_parm_names(node, "prompt")
    if indexed_prompt_names:
        texts = []
        durations = []
        for index in sorted(indexed_prompt_names):
            prompt = str(_eval_parm(node, f"prompt{index}", "")).strip()
            if not prompt:
                continue
            texts.append(prompt)
            durations.append(float(_eval_parm(node, f"duration{index}", 5.0)))
        if texts:
            return texts, durations

    return [str(_eval_parm(node, "prompt", "")).strip()], [float(_eval_parm(node, "duration", 5.0))]


def _indexed_parm_names(node, base_name: str) -> set[int]:
    parms = getattr(node, "parms", None)
    if not callable(parms):
        return set()

    indices = set()
    pattern = re.compile(rf"{re.escape(base_name)}(\d+)$")
    for parm in parms():
        name_method = getattr(parm, "name", None)
        if not callable(name_method):
            continue
        match = pattern.fullmatch(str(name_method()))
        if match is not None:
            indices.add(int(match.group(1)))
    return indices


def _apply_optional_generation_parms(node, payload: dict[str, Any]) -> None:
    diffusion_steps = _eval_parm(node, "diffusion_steps", None)
    if diffusion_steps not in (None, ""):
        payload["diffusion_steps"] = int(diffusion_steps)

    text_weight = _eval_parm(node, "text_weight", None)
    constraint_weight = _eval_parm(node, "constraint_weight", None)
    if text_weight not in (None, "") and constraint_weight not in (None, ""):
        payload["cfg_type"] = "separated"
        payload["cfg_weight"] = [float(text_weight), float(constraint_weight)]
    elif text_weight not in (None, ""):
        payload["cfg_type"] = "regular"
        payload["cfg_weight"] = float(text_weight)

    num_transition_frames = _eval_parm(node, "num_transition_frames", None)
    if num_transition_frames not in (None, ""):
        payload["num_transition_frames"] = int(num_transition_frames)


def _apply_constraints(
    node,
    payload: dict[str, Any],
    output_world_offset: tuple[float, float, float],
) -> None:
    if not _eval_bool_parm(node, "enable_constraints", False):
        return

    constraints: list[dict[str, Any]] = []
    warnings: list[str] = []

    geometry = _constraint_source_geometry(node, DEFAULT_ROOT2D_CONSTRAINT_NODE)
    if geometry is not None:
        constraint, root2d_warnings = root2d_constraint_from_geometry(
            geometry,
            frame_origin=float(_eval_parm(node, "frame_origin", 1)),
            include_heading=_eval_bool_parm(node, "include_heading", True),
            heading_forward_axis=str(_eval_menu_token(node, "heading_forward_axis", "+Z")),
            nonplanar_y_tolerance=float(_eval_parm(node, "nonplanar_y_tolerance", 0.001)),
            output_world_offset=output_world_offset,
        )
        if constraint is not None:
            constraints.append(constraint)
        warnings.extend(root2d_warnings)

    for source_name, constraint_type in POSE_CONSTRAINT_NODES.items():
        geometry = _constraint_source_geometry(node, source_name)
        if geometry is None:
            continue
        pose_constraints, pose_warnings = pose_constraints_from_geometry(
            geometry,
            constraint_type,
            frame_origin=float(_eval_parm(node, "frame_origin", 1)),
            output_world_offset=output_world_offset,
        )
        constraints.extend(pose_constraints)
        warnings.extend(pose_warnings)

    if constraints:
        payload["constraints"] = constraints
    _display_houdini_warnings(warnings)


def _constraint_source_geometry(node, source_name: str):
    source = _constraint_source_node(node, source_name)
    if source is None:
        return None

    geometry_method = getattr(source, "geometry", None)
    if not callable(geometry_method):
        return None
    return geometry_method()


def _constraint_source_node(node, source_name: str):
    child_method = getattr(node, "node", None)
    if callable(child_method):
        try:
            return child_method(source_name)
        except Exception:
            return None
    return None


def _display_houdini_warnings(warnings: list[str]) -> None:
    if not warnings:
        return
    try:
        import hou  # type: ignore
    except ImportError:
        return

    ui = getattr(hou, "ui", None)
    display_message = getattr(ui, "displayMessage", None)
    if not callable(display_message):
        return

    severity_type = getattr(hou, "severityType", None)
    warning = getattr(severity_type, "Warning", None) if severity_type is not None else None
    kwargs = {"severity": warning} if warning is not None else {}
    display_message("\n".join(warnings), **kwargs)


def _eval_parm(node, parm_name: str, default: Any = None) -> Any:
    parm = node.parm(parm_name)
    if parm is None:
        return default
    return parm.eval()


def _eval_bool_parm(node, parm_name: str, default: bool) -> bool:
    value = _eval_parm(node, parm_name, None)
    if value in (None, ""):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _output_world_offset(node) -> tuple[float, float, float]:
    return _eval_vector3_parm(node, "output_world_offset", (0.0, 0.0, 0.0))


def _eval_vector3_parm(node, parm_name: str, default: tuple[float, float, float]) -> tuple[float, float, float]:
    parm_tuple = getattr(node, "parmTuple", None)
    if callable(parm_tuple):
        tuple_parm = parm_tuple(parm_name)
        if tuple_parm is not None:
            value = tuple_parm.eval()
            return _coerce_vector3(value, parm_name)

    value = _eval_parm(node, parm_name, None)
    if value not in (None, ""):
        return _coerce_vector3(value, parm_name)

    components = []
    for suffix in ("x", "y", "z"):
        component = _eval_parm(node, f"{parm_name}{suffix}", None)
        if component in (None, ""):
            components = []
            break
        components.append(component)
    if components:
        return _coerce_vector3(components, parm_name)

    return default


def _coerce_vector3(value: Any, parm_name: str) -> tuple[float, float, float]:
    try:
        if len(value) == 3:
            return (float(value[0]), float(value[1]), float(value[2]))
    except TypeError:
        pass
    except Exception as exc:
        raise ValueError(f"{parm_name} must be a 3D vector.") from exc
    raise ValueError(f"{parm_name} must be a 3D vector.")


def _is_nonzero_vector(value: tuple[float, float, float]) -> bool:
    return any(abs(component) > 1e-8 for component in value)


def _eval_menu_token(node, parm_name: str, default: Any = None) -> Any:
    parm = node.parm(parm_name)
    if parm is None:
        return default

    value = parm.eval()
    if value in (None, ""):
        return default
    if isinstance(value, str):
        return value

    token = _menu_token_from_index(parm, value)
    if token is not None:
        return token

    eval_as_string = getattr(parm, "evalAsString", None)
    if callable(eval_as_string):
        text = eval_as_string()
        if text not in ("", str(value)):
            return text
    return default


def _menu_token_from_index(parm, value: Any) -> str | None:
    if not isinstance(value, int):
        return None

    template_method = getattr(parm, "parmTemplate", None)
    if not callable(template_method):
        return None

    template = template_method()
    menu_items_method = getattr(template, "menuItems", None)
    if not callable(menu_items_method):
        return None

    menu_items = menu_items_method()
    if 0 <= value < len(menu_items):
        return str(menu_items[value])
    return None
