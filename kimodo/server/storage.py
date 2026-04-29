# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Filesystem-backed storage for Kimodo server jobs."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from .schemas import ArtifactRecord, GenerationRequest, JobRecord


class JobStorage:
    """Manage per-job directories and JSON metadata."""

    def __init__(self, root: str | None = None) -> None:
        root = root or os.environ.get("KIMODO_JOB_ROOT", os.path.join("~", ".cache", "kimodo", "jobs"))
        self.root = Path(root).expanduser()

    def create_job_id(self) -> str:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{stamp}-{uuid.uuid4().hex[:8]}"

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def artifacts_dir(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "artifacts"

    def create_job_dir(self, job_id: str) -> Path:
        path = self.job_dir(job_id)
        path.mkdir(parents=True, exist_ok=False)
        return path

    def write_request(self, job_id: str, request: GenerationRequest) -> None:
        request.job_id = job_id
        self._write_json(self.job_dir(job_id) / "request.json", request.to_dict())

    def write_status(self, record: JobRecord) -> None:
        self._write_json(self.job_dir(record.job_id) / "status.json", record.to_dict())

    def read_status(self, job_id: str) -> JobRecord:
        with open(self.job_dir(job_id) / "status.json", "r", encoding="utf-8") as f:
            return JobRecord.from_dict(json.load(f))

    def resolve_artifact(self, job_id: str, artifact_key: str) -> tuple[Path, ArtifactRecord]:
        """Resolve an artifact key to a local path for a future HTTP adapter.

        The caller supplies only ``job_id`` and an artifact key recorded in
        ``status.json``. The stored filename is resolved under the job's
        artifacts directory to avoid arbitrary path downloads.
        """
        record = self.read_status(job_id)
        artifact = record.artifacts.get(artifact_key)
        if artifact is None:
            raise KeyError(f"Artifact {artifact_key!r} not found for job {job_id!r}.")
        if not isinstance(artifact, ArtifactRecord):
            raise TypeError(f"Artifact {artifact_key!r} is not a structured ArtifactRecord.")

        artifacts_dir = self.artifacts_dir(job_id).resolve()
        path = (artifacts_dir / artifact.filename).resolve()
        if artifacts_dir != path.parent:
            raise ValueError(f"Artifact filename escapes artifacts directory: {artifact.filename!r}.")
        if not path.is_file():
            raise FileNotFoundError(path)
        return path, artifact

    @staticmethod
    def _write_json(path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, path)
