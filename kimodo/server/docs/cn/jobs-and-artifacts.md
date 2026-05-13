# Job 与 Artifact

server 模块把一次 generation 请求表示为一个 job。job 的状态和生成文件都由 `JobManager` 与 `JobStorage` 管理。

## Job 生命周期

`JobStatus` 当前包含五种状态：

| 状态 | 含义 |
| --- | --- |
| `queued` | job 已提交，等待后台 worker 执行。 |
| `running` | runtime 正在执行 generation。 |
| `succeeded` | generation 成功完成，并写入 artifact metadata。 |
| `failed` | generation 抛出异常，`error` 字段保存 traceback。 |
| `canceled` | job 在开始执行前被取消。 |

`JobManager` 默认使用单 worker 队列，适合 GPU-backed generation。`cancel()` 只能取消还没有开始运行的 future；已经进入 `running` 的 job 不会被强行中断。

## JobRecord

`JobRecord` 是 job 状态的持久化记录，主要字段如下：

| 字段 | 含义 |
| --- | --- |
| `job_id` | job 唯一标识，由 `JobStorage.create_job_id()` 生成。 |
| `status` | 当前 job 状态。 |
| `job_dir` | job 的服务端目录。client 不应该依赖这个路径。 |
| `progress` | 当前进度。现阶段主要是 `0.0` 或 `1.0`。 |
| `message` | 简短状态描述。 |
| `artifacts` | 生成结果的 artifact metadata。 |
| `error` | 失败时的 traceback，成功时为 `None`。 |

## Storage 目录

默认 storage root 来自 `KIMODO_JOB_ROOT`，如果未设置则使用 `~/.cache/kimodo/jobs`。也可以通过 `create_app(storage_root=...)` 或 `JobStorage(root=...)` 显式指定。

一个 job 目录大致如下：

```text
{storage_root}/
  {job_id}/
    request.json
    status.json
    artifacts/
      motion.npz
      motion.bvh
      artifacts.zip
```

`request.json` 保存提交时的 `GenerationRequest`，`status.json` 保存最新的 `JobRecord`。

## ArtifactRecord

`ArtifactRecord` 描述一个可下载文件：

| 字段 | 含义 |
| --- | --- |
| `key` | artifact key，例如 `npz`、`bvh`、`zip`、`npz_00`。 |
| `filename` | job 的 `artifacts/` 目录下的文件名。 |
| `content_type` | 下载时可使用的 content type。 |
| `size_bytes` | 文件大小。 |
| `download_url` | 未来 HTTP adapter 暴露给 client 的下载路径。 |

生成结果不会把任意本地路径暴露给 client。下载时应由服务端根据 `job_id` 和 `artifact_key` 解析文件。

## 下载模型

未来 HTTP adapter 可以暴露类似接口：

```text
GET /jobs/{job_id}/artifacts/{artifact_key}
```

实现时应调用：

```python
path, artifact = storage.resolve_artifact(job_id, artifact_key)
```

`resolve_artifact()` 会从 `status.json` 读取 artifact metadata，并确保文件位于当前 job 的 `artifacts/` 目录下，避免 client 通过路径参数下载任意本地文件。
