# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FastAPI adapter for the Kimodo server core."""

from __future__ import annotations

import re
from contextlib import asynccontextmanager
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Response, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from kimodo.model.registry import DEFAULT_MODEL

from .app import create_app
from .runtime import Runtime, TextEncoderServerConfig
from .schemas import ArtifactRecord, GenerationRequest, JobRecord, JobStatus

_PATH_COMPONENT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class GenerationRequestBody(BaseModel):
    """HTTP request body for creating a Kimodo generation job."""

    model_config = ConfigDict(extra="forbid")

    texts: list[str]
    durations: list[float]
    model: str = DEFAULT_MODEL
    diffusion_steps: int = 100
    num_samples: int = 1
    seed: int | None = None
    cfg_type: Literal["nocfg", "regular", "separated"] | None = None
    cfg_weight: float | list[float] | None = None
    num_transition_frames: int = 5
    first_heading_angle: float | list[float] | None = None
    formats: list[str] = Field(default_factory=lambda: ["npz", "bvh"])
    bvh_standard_tpose: bool = True
    zip_output: bool = False
    postprocess: bool = True
    root_margin: float = 0.04
    constraints: Any | None = None

    def to_generation_request(self) -> GenerationRequest:
        return GenerationRequest(**self.model_dump())


class ArtifactResponse(BaseModel):
    key: str
    filename: str
    content_type: str
    size_bytes: int
    download_url: str | None = None

    @classmethod
    def from_record(cls, record: ArtifactRecord) -> "ArtifactResponse":
        return cls(**record.to_dict())


class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float
    message: str
    artifacts: dict[str, ArtifactResponse] = Field(default_factory=dict)
    error: str | None = None

    @classmethod
    def from_record(cls, record: JobRecord) -> "JobResponse":
        return cls(
            job_id=record.job_id,
            status=record.status,
            progress=record.progress,
            message=record.message,
            artifacts={key: ArtifactResponse.from_record(value) for key, value in record.artifacts.items()},
            error=record.error,
        )


class HealthResponse(BaseModel):
    status: str
    runtime: str


def create_fastapi_app(
    storage_root: str | None = None,
    runtime: Runtime | None = None,
    text_encoder_config: TextEncoderServerConfig | None = None,
) -> FastAPI:
    """Create the FastAPI application that wraps the framework-neutral core."""
    core = create_app(
        storage_root=storage_root,
        runtime=runtime,
        text_encoder_config=text_encoder_config,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        core["jobs"].executor.shutdown(wait=True)
        close = getattr(core["runtime"], "close", None)
        if callable(close):
            close()

    app = FastAPI(title="Kimodo Server", lifespan=lifespan)
    app.state.jobs = core["jobs"]
    app.state.runtime = core["runtime"]
    app.state.storage = core["storage"]

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", runtime=type(app.state.runtime).__name__)

    @app.post("/jobs", response_model=JobResponse, status_code=status.HTTP_202_ACCEPTED)
    def submit_job(request: GenerationRequestBody) -> JobResponse:
        record = app.state.jobs.submit(request.to_generation_request())
        return JobResponse.from_record(record)

    @app.get("/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: str) -> JobResponse:
        _validate_path_component(job_id, "job_id")
        return JobResponse.from_record(_get_job_or_404(app, job_id))

    @app.post("/jobs/{job_id}/cancel", response_model=JobResponse)
    def cancel_job(job_id: str) -> JobResponse:
        _validate_path_component(job_id, "job_id")
        try:
            record = app.state.jobs.cancel(job_id)
        except FileNotFoundError as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.") from error
        return JobResponse.from_record(record)

    @app.get("/jobs/{job_id}/artifacts/{artifact_key}")
    def download_artifact(job_id: str, artifact_key: str) -> Response:
        _validate_path_component(job_id, "job_id")
        _validate_path_component(artifact_key, "artifact_key")
        try:
            path, artifact = app.state.storage.resolve_artifact(job_id, artifact_key)
        except (FileNotFoundError, KeyError) as error:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found.") from error
        except ValueError as error:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(error)) from error
        return FileResponse(path, media_type=artifact.content_type, filename=artifact.filename)

    return app


def _get_job_or_404(app: FastAPI, job_id: str) -> JobRecord:
    try:
        return app.state.jobs.get(job_id)
    except FileNotFoundError as error:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found.") from error


def _validate_path_component(value: str, field_name: str) -> None:
    if value in {"", ".", ".."} or _PATH_COMPONENT_RE.fullmatch(value) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Invalid {field_name}.")
