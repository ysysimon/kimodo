# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Requests-based client for the Kimodo remote motion server."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests

from .errors import (
    RemoteMotionArtifactError,
    RemoteMotionHTTPError,
    RemoteMotionJobFailedError,
    RemoteMotionTimeoutError,
)
from .models import ArtifactInfo, JobInfo

_FILENAME_RE = re.compile(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?')


class RemoteMotionClient:
    """Small HTTP client for remote Kimodo motion generation."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        timeout: float = 30.0,
        *,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self.timeout = timeout
        self.session = session or requests.Session()

    def health(self) -> dict[str, Any]:
        return self._request_json("GET", "health")

    def submit(self, payload: dict[str, Any]) -> JobInfo:
        return JobInfo.from_dict(self._request_json("POST", "jobs", json=payload))

    def get_job(self, job_id: str) -> JobInfo:
        return JobInfo.from_dict(self._request_json("GET", f"jobs/{job_id}"))

    def cancel(self, job_id: str) -> JobInfo:
        return JobInfo.from_dict(self._request_json("POST", f"jobs/{job_id}/cancel"))

    def wait(self, job_id: str, poll_interval: float = 1.0, timeout: float | None = None) -> JobInfo:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            job = self.get_job(job_id)
            if job.status == "succeeded":
                return job
            if job.status in {"failed", "canceled"}:
                raise RemoteMotionJobFailedError(job)
            if deadline is not None and time.monotonic() >= deadline:
                raise RemoteMotionTimeoutError(f"Timed out waiting for job {job_id}.")
            time.sleep(poll_interval)

    def download_artifact(
        self,
        job_or_job_id: JobInfo | str,
        artifact_key: str,
        dst_dir_or_path: str | Path,
    ) -> Path:
        artifact: ArtifactInfo | None = None
        if isinstance(job_or_job_id, JobInfo):
            job_id = job_or_job_id.job_id
            artifact = job_or_job_id.artifacts.get(artifact_key)
            if artifact is None:
                raise RemoteMotionArtifactError(f"Job {job_id} has no artifact {artifact_key!r}.")
            download_path = artifact.download_url or f"/jobs/{job_id}/artifacts/{artifact_key}"
            filename = artifact.filename
        else:
            job_id = job_or_job_id
            download_path = f"/jobs/{job_id}/artifacts/{artifact_key}"
            filename = artifact_key

        response = self._request("GET", download_path, stream=True)
        filename = _content_disposition_filename(response.headers.get("content-disposition")) or filename
        dst_path = _resolve_download_path(Path(dst_dir_or_path), filename)
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(dst_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
        finally:
            response.close()

        return dst_path

    def _request_json(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self._request(method, path, **kwargs)
        try:
            return response.json()
        except ValueError as exc:
            raise RemoteMotionHTTPError(
                f"Expected JSON response from {response.url}.",
                status_code=response.status_code,
                url=response.url,
                response_text=response.text,
            ) from exc

    def _request(self, method: str, path: str, **kwargs: Any) -> requests.Response:
        url = _join_url(self.base_url, path)
        kwargs.setdefault("timeout", self.timeout)
        try:
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.HTTPError as exc:
            response = exc.response
            raise RemoteMotionHTTPError(
                f"HTTP {response.status_code} from {response.url}.",
                status_code=response.status_code,
                url=response.url,
                response_text=response.text,
            ) from exc
        except requests.RequestException as exc:
            raise RemoteMotionHTTPError(f"HTTP request failed for {url}: {exc}", url=url) from exc


def _join_url(base_url: str, path: str) -> str:
    if path.startswith(("http://", "https://")):
        return path
    return urljoin(base_url, path.lstrip("/"))


def _resolve_download_path(dst_dir_or_path: Path, filename: str) -> Path:
    if dst_dir_or_path.exists() and dst_dir_or_path.is_dir():
        return dst_dir_or_path / filename
    if str(dst_dir_or_path).endswith(("/", "\\")):
        return dst_dir_or_path / filename
    if dst_dir_or_path.suffix:
        return dst_dir_or_path
    return dst_dir_or_path / filename


def _content_disposition_filename(value: str | None) -> str | None:
    if not value:
        return None
    match = _FILENAME_RE.search(value)
    if match is None:
        return None
    return Path(match.group(1)).name
