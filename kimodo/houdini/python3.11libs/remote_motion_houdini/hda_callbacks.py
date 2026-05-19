# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HDA callback entry points for remote Kimodo motion generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from remote_motion_client import RemoteMotionClient

from .cache import default_download_dir
from .config import get_server_url
from .importer import refresh_mocap_import


def generate_motion(kwargs: dict[str, Any]) -> Path:
    """Generate one BVH motion from HDA parms and import it into the node."""
    node = kwargs["node"]
    server_url = _eval_parm(node, "server_url", None) or get_server_url()
    prompt = _eval_parm(node, "prompt", "")
    duration = float(_eval_parm(node, "duration", 5.0))
    seed = _eval_parm(node, "seed", None)
    model = _eval_parm(node, "model", None)
    poll_interval = float(_eval_parm(node, "poll_interval", 1.0))
    wait_timeout = _eval_parm(node, "wait_timeout", None)

    payload: dict[str, Any] = {
        "texts": [prompt],
        "durations": [duration],
        "formats": ["bvh"],
    }
    if seed not in (None, ""):
        payload["seed"] = int(seed)
    if model not in (None, ""):
        payload["model"] = model

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

    refresh_mocap_import(node, artifact_path)
    return artifact_path


def _eval_parm(node, parm_name: str, default: Any = None) -> Any:
    parm = node.parm(parm_name)
    if parm is None:
        return default
    return parm.eval()
