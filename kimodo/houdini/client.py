# SPDX-FileCopyrightText: Copyright (c) 2026 ysysimon. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Thin HTTP client intended for Houdini hython usage."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any
from urllib import request as urlrequest


@dataclass
class ClientJob:
    job_id: str
    status: str
    payload: dict[str, Any]


class KimodoClient:
    """Minimal stdlib-only client for a Kimodo inference server."""

    def __init__(self, base_url: str = "http://127.0.0.1:8765", timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def health(self) -> dict[str, Any]:
        return self._get("/v1/health")

    def submit(self, **payload) -> ClientJob:
        data = self._post("/v1/jobs", payload)
        return ClientJob(job_id=data["job_id"], status=data["status"], payload=data)

    def get_job(self, job_id: str) -> ClientJob:
        data = self._get(f"/v1/jobs/{job_id}")
        return ClientJob(job_id=data["job_id"], status=data["status"], payload=data)

    def wait(self, job_id: str, poll_interval: float = 1.0) -> ClientJob:
        while True:
            job = self.get_job(job_id)
            if job.status in {"succeeded", "failed", "canceled"}:
                return job
            time.sleep(poll_interval)

    def _get(self, path: str) -> dict[str, Any]:
        with urlrequest.urlopen(self.base_url + path, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = urlrequest.Request(
            self.base_url + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlrequest.urlopen(req, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))
