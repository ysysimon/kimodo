# Server 模块架构

Kimodo server 模块的目标是提供一组可复用的服务端核心构件，而不把模型执行直接绑定到某个 HTTP 框架。当前 FastAPI adapter 会薄封装这些核心对象，对外提供 HTTP 访问，同时让模型加载、job 生命周期和 artifact 管理继续留在 core 层。

## 模块职责

| 模块 | 职责 |
| --- | --- |
| `app.py` | composition root，负责组装 `JobStorage`、`Runtime` 和 `JobManager`。 |
| `schemas.py` | 定义 request、result、job status 和 artifact metadata 等数据契约。 |
| `jobs.py` | 管理 job 提交、后台执行、状态查询和取消。 |
| `storage.py` | 管理 job 目录、`request.json`、`status.json` 和 artifact 路径解析。 |
| `exports.py` | 把 runtime 输出导出为 NPZ、BVH 或 zip artifact。 |
| `runtime/` | 定义 runtime contract，并提供真实模型 runtime、fake runtime 和 text encoder 服务管理。 |
| `asgi.py` | FastAPI routes、HTTP request/response models、下载接口和 lifespan cleanup。 |

## 数据流

一次 generation job 的核心数据流如下：

```text
GenerationRequest
  -> JobManager.submit()
  -> JobStorage 创建 job 目录并写入 request/status
  -> Runtime.generate()
  -> exports 写入 artifacts
  -> JobManager 更新 JobRecord
  -> JobStorage 写回 status.json
```

`JobRecord.artifacts` 保存的是可下载 artifact 的元数据，不是任意本地路径。HTTP adapter 下载文件时，会通过 `JobStorage.resolve_artifact(job_id, artifact_key)` 解析真实文件。

## HTTP Adapter 边界

FastAPI adapter 只做薄封装：

- 把 HTTP request 转成 `GenerationRequest`。
- 调用 `JobManager.submit()`、`JobManager.get()`、`JobManager.cancel()`。
- 使用 `JobStorage.resolve_artifact()` 实现 artifact 下载。
- 返回不会暴露服务端本地 `job_dir` 的 HTTP response model。
- 在 ASGI lifespan cleanup 阶段关闭 job executor，并释放 runtime 拥有的资源。

HTTP adapter 不应该直接加载模型、管理 text encoder subprocess、拼接 artifact 本地路径，或者绕过 `JobStorage` 访问 job 文件。

当前 FastAPI routes：

- `GET /health`
- `POST /jobs`
- `GET /jobs/{job_id}`
- `POST /jobs/{job_id}/cancel`
- `GET /jobs/{job_id}/artifacts/{artifact_key}`
