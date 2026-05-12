# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic runtime for server interface tests and smoke checks."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from ..schemas import ArtifactRecord, GenerationRequest, GenerationResult


def _download_url(job_id: str | None, key: str) -> str | None:
    return f"/jobs/{job_id}/artifacts/{key}" if job_id else None


class FakeRuntime:
    """Fast deterministic runtime for server interface tests and smoke checks.

    This runtime intentionally does not load Kimodo models, touch CUDA, or run
    diffusion. It only writes tiny placeholder artifacts with the same public
    metadata shape as the real runtime.
    """

    def generate(self, request: GenerationRequest, *, job_dir: str | Path) -> GenerationResult:
        if len(request.texts) != len(request.durations):
            raise ValueError("texts and durations must have the same length.")

        artifacts_dir = Path(job_dir) / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        formats = {fmt.lower() for fmt in request.formats}

        artifacts: dict[str, ArtifactRecord] = {}
        for fmt in ("npz", "bvh"):
            if fmt not in formats:
                continue
            artifacts.update(self._write_format_artifacts(artifacts_dir, fmt, request=request))

        if request.zip_output and artifacts:
            zip_path = artifacts_dir / "artifacts.zip"
            with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as zip_file:
                for artifact in artifacts.values():
                    zip_file.write(artifacts_dir / artifact.filename, arcname=artifact.filename)
            artifacts = {
                "zip": self._artifact_record(
                    zip_path,
                    key="zip",
                    content_type="application/zip",
                    job_id=request.job_id,
                )
            }

        return GenerationResult(job_id=request.job_id, artifacts=artifacts)

    def _write_format_artifacts(
        self,
        artifacts_dir: Path,
        fmt: str,
        *,
        request: GenerationRequest,
    ) -> dict[str, ArtifactRecord]:
        extension = fmt
        content_type = "application/octet-stream"
        n_samples = request.num_samples

        if n_samples == 1:
            path = artifacts_dir / f"motion.{extension}"
            self._write_dummy_artifact(path, request=request, key=fmt)
            return {fmt: self._artifact_record(path, key=fmt, content_type=content_type, job_id=request.job_id)}

        artifacts = {}
        for sample_idx in range(n_samples):
            key = f"{fmt}_{sample_idx:02d}"
            path = artifacts_dir / f"motion_{sample_idx:02d}.{extension}"
            self._write_dummy_artifact(path, request=request, key=key)
            artifacts[key] = self._artifact_record(
                path,
                key=key,
                content_type=content_type,
                job_id=request.job_id,
            )
        return artifacts

    @staticmethod
    def _write_dummy_artifact(path: Path, *, request: GenerationRequest, key: str) -> None:
        path.write_text(
            "\n".join(
                [
                    "Kimodo fake server artifact",
                    f"key={key}",
                    f"job_id={request.job_id}",
                    f"model={request.model}",
                    f"num_samples={request.num_samples}",
                    f"texts={request.texts!r}",
                    "",
                ]
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _artifact_record(path: Path, *, key: str, content_type: str, job_id: str | None) -> ArtifactRecord:
        return ArtifactRecord(
            key=key,
            filename=path.name,
            content_type=content_type,
            size_bytes=path.stat().st_size,
            download_url=_download_url(job_id, key),
        )
