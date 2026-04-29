# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Artifact export helpers for server jobs."""

from pathlib import Path
from typing import Mapping
from zipfile import ZIP_DEFLATED, ZipFile

import torch

from kimodo.exports.bvh import save_motion_bvh
from kimodo.exports.motion_io import save_kimodo_npz
from kimodo.skeleton import SOMASkeleton30, global_rots_to_local_rots

from .schemas import ArtifactRecord


def _download_url(job_id: str | None, key: str) -> str | None:
    return f"/jobs/{job_id}/artifacts/{key}" if job_id else None


def _artifact_record(path: Path, *, key: str, content_type: str, job_id: str | None) -> ArtifactRecord:
    return ArtifactRecord(
        key=key,
        filename=path.name,
        content_type=content_type,
        size_bytes=path.stat().st_size,
        download_url=_download_url(job_id, key),
    )


def _sample_motion(motion: Mapping, sample_idx: int, n_samples: int) -> dict:
    return {
        key: (value[sample_idx] if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] == n_samples else value)
        for key, value in motion.items()
    }


def save_npz_artifacts(
    artifacts_dir: str | Path,
    motion: Mapping,
    *,
    job_id: str | None = None,
) -> dict[str, ArtifactRecord]:
    """Save Kimodo motion dictionaries as NPZ artifact(s)."""
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    n_samples = int(motion["posed_joints"].shape[0])

    if n_samples == 1:
        path = artifacts_dir / "motion.npz"
        save_kimodo_npz(str(path), _sample_motion(motion, 0, n_samples))
        return {"npz": _artifact_record(path, key="npz", content_type="application/octet-stream", job_id=job_id)}

    artifacts = {}
    for sample_idx in range(n_samples):
        key = f"npz_{sample_idx:02d}"
        path = artifacts_dir / f"motion_{sample_idx:02d}.npz"
        save_kimodo_npz(str(path), _sample_motion(motion, sample_idx, n_samples))
        artifacts[key] = _artifact_record(path, key=key, content_type="application/octet-stream", job_id=job_id)
    return artifacts


def save_bvh_artifacts(
    artifacts_dir: str | Path,
    motion: Mapping,
    *,
    skeleton,
    fps: float,
    device: str | torch.device,
    job_id: str | None = None,
    standard_tpose: bool = False,
) -> dict[str, ArtifactRecord]:
    """Save SOMA motion as BVH.

    Returns public artifact metadata. Non-SOMA skeletons return an empty dict.
    """
    if "somaskel" not in skeleton.name:
        return {}

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    export_skeleton = skeleton.somaskel77.to(device) if isinstance(skeleton, SOMASkeleton30) else skeleton
    n_samples = int(motion["posed_joints"].shape[0])

    def save_one(sample_idx: int, sample_path: Path) -> Path:
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        joints_pos = torch.as_tensor(motion["posed_joints"][sample_idx], device=device)
        joints_rot = torch.as_tensor(motion["global_rot_mats"][sample_idx], device=device)
        local_rot_mats = global_rots_to_local_rots(joints_rot, export_skeleton)
        root_positions = joints_pos[:, export_skeleton.root_idx, :]
        save_motion_bvh(
            sample_path,
            local_rot_mats,
            root_positions,
            skeleton=export_skeleton,
            fps=fps,
            standard_tpose=standard_tpose,
        )
        return sample_path

    if n_samples == 1:
        path = save_one(0, artifacts_dir / "motion.bvh")
        return {"bvh": _artifact_record(path, key="bvh", content_type="application/octet-stream", job_id=job_id)}

    bvh_paths = []
    for sample_idx in range(n_samples):
        bvh_paths.append(save_one(sample_idx, artifacts_dir / f"motion_{sample_idx:02d}.bvh"))

    return {
        f"bvh_{sample_idx:02d}": _artifact_record(
            bvh_path,
            key=f"bvh_{sample_idx:02d}",
            content_type="application/octet-stream",
            job_id=job_id,
        )
        for sample_idx, bvh_path in enumerate(bvh_paths)
    }


def save_zip_artifact(
    artifacts_dir: str | Path,
    artifacts: Mapping[str, ArtifactRecord],
    *,
    job_id: str | None = None,
) -> ArtifactRecord:
    """Zip all generated artifact files and return one downloadable record."""
    artifacts_dir = Path(artifacts_dir)
    zip_path = artifacts_dir / "artifacts.zip"
    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as zip_file:
        for artifact in artifacts.values():
            zip_file.write(artifacts_dir / artifact.filename, arcname=artifact.filename)
    return _artifact_record(zip_path, key="zip", content_type="application/zip", job_id=job_id)
