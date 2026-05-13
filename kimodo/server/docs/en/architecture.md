# Server Architecture

The Kimodo server package provides reusable server-side core building blocks
without binding them directly to a specific HTTP framework. This keeps model
loading, job lifecycle management, and artifact handling stable when a future
FastAPI, Flask, or other adapter is added.

## Module Responsibilities

| Module | Responsibility |
| --- | --- |
| `app.py` | Composition root that wires `JobStorage`, `Runtime`, and `JobManager`. |
| `schemas.py` | Data contracts for requests, results, job status, and artifact metadata. |
| `jobs.py` | Job submission, background execution, status lookup, and cancellation. |
| `storage.py` | Job directories, `request.json`, `status.json`, and artifact path resolution. |
| `exports.py` | Exports runtime output as NPZ, BVH, or zip artifacts. |
| `runtime/` | Runtime contract, real model runtime, fake runtime, and text encoder service management. |

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
paths. A future HTTP adapter should resolve real files through
`JobStorage.resolve_artifact(job_id, artifact_key)`.

## HTTP Adapter Boundary

A future FastAPI adapter should remain thin:

- Convert HTTP requests into `GenerationRequest`.
- Call `JobManager.submit()`, `JobManager.get()`, and `JobManager.cancel()`.
- Use `JobStorage.resolve_artifact()` to serve artifact downloads.
- Return `JobRecord.to_dict()` or an equivalent response shape to clients.

The HTTP adapter should not load models directly, manage text encoder
subprocesses, construct artifact local paths by hand, or bypass `JobStorage`
when accessing job files.
