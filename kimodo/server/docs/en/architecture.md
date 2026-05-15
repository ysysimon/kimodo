# Server Architecture

The Kimodo server package provides reusable server-side core building blocks
without binding model execution to a specific HTTP framework. A thin FastAPI
adapter wraps those core objects for HTTP access while keeping model loading,
job lifecycle management, and artifact handling in the core layer.

## Module Responsibilities

| Module | Responsibility |
| --- | --- |
| `app.py` | Composition root that wires `JobStorage`, `Runtime`, and `JobManager`. |
| `schemas.py` | Data contracts for requests, results, job status, and artifact metadata. |
| `jobs.py` | Job submission, background execution, status lookup, and cancellation. |
| `storage.py` | Job directories, `request.json`, `status.json`, and artifact path resolution. |
| `exports.py` | Exports runtime output as NPZ, BVH, or zip artifacts. |
| `runtime/` | Runtime contract, real model runtime, fake runtime, and text encoder service management. |
| `asgi.py` | FastAPI routes, HTTP request/response models, downloads, and lifespan cleanup. |

## Data Flow

The core data flow for one generation job is:

```text
GenerationRequest
  -> JobManager.submit()
  -> JobStorage creates the job directory and writes request/status
  -> Runtime.generate()
  -> exports writes artifacts
  -> JobManager updates JobRecord
  -> JobStorage writes status.json
```

`JobRecord.artifacts` stores downloadable artifact metadata, not arbitrary local
paths. The HTTP adapter resolves real files through
`JobStorage.resolve_artifact(job_id, artifact_key)`.

## HTTP Adapter Boundary

The FastAPI adapter remains thin:

- Convert HTTP requests into `GenerationRequest`.
- Call `JobManager.submit()`, `JobManager.get()`, and `JobManager.cancel()`.
- Use `JobStorage.resolve_artifact()` to serve artifact downloads.
- Return HTTP response models that omit server-local `job_dir`.
- Shut down the job executor and close runtime-owned resources during ASGI
  lifespan cleanup.

The HTTP adapter should not load models directly, manage text encoder
subprocesses, construct artifact local paths by hand, or bypass `JobStorage`
when accessing job files.

The current FastAPI routes are:

- `GET /health`
- `POST /jobs`
- `GET /jobs/{job_id}`
- `POST /jobs/{job_id}/cancel`
- `GET /jobs/{job_id}/artifacts/{artifact_key}`
