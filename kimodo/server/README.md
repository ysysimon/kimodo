# Kimodo Server 请求参数

server API 复用 CLI 和 demo 使用的同一条生成路径。一次生成请求由
`schemas.py` 里的 `GenerationRequest` 表示。

生成结果里的 `artifacts` 不再是 server 本机路径，而是可下载 artifact 的元数据。
HTTP adapter 接入后，client 应通过 `download_url` 下载文件。

## 请求示例

```json
{
  "texts": ["A person walks forward.", "A person turns around."],
  "durations": [3.0, 2.0],
  "model": "kimodo-soma-rp",
  "diffusion_steps": 100,
  "num_samples": 1,
  "seed": 42,
  "cfg_type": "separated",
  "cfg_weight": [2.0, 2.0],
  "num_transition_frames": 5,
  "first_heading_angle": 0.0,
  "formats": ["npz", "bvh"],
  "zip_output": false,
  "postprocess": true,
  "root_margin": 0.04,
  "constraints": null,
  "job_id": null
}
```

## 响应示例

单 sample 且 `zip_output=false`：

```json
{
  "job_id": "20260429-xxxx",
  "status": "succeeded",
  "artifacts": {
    "npz": {
      "key": "npz",
      "filename": "motion.npz",
      "content_type": "application/octet-stream",
      "size_bytes": 123456,
      "download_url": "/jobs/20260429-xxxx/artifacts/npz"
    },
    "bvh": {
      "key": "bvh",
      "filename": "motion.bvh",
      "content_type": "application/octet-stream",
      "size_bytes": 45678,
      "download_url": "/jobs/20260429-xxxx/artifacts/bvh"
    }
  }
}
```

多 sample 且 `zip_output=true` 时，所有已生成 artifact 会打成一个 zip：

```json
{
  "artifacts": {
    "zip": {
      "key": "zip",
      "filename": "artifacts.zip",
      "content_type": "application/zip",
      "size_bytes": 123456,
      "download_url": "/jobs/20260429-xxxx/artifacts/zip"
    }
  }
}
```

## 参数说明

### `texts`

类型：`list[str]`

必填。每个字符串是一段 prompt。server 调模型时使用 `multi_prompt=True`，所以
`texts` 中的每一项会被当作一个连续动作段。

### `durations`

类型：`list[float]`

必填。每个数字是对应 prompt 段的时长，单位是秒。`durations` 必须和 `texts`
长度一致。runtime 会用 `duration * model.fps` 转成模型需要的帧数。

### `model`

类型：`str`

默认值：`DEFAULT_MODEL`，当前是 `kimodo-soma-rp`。指定要使用的 Kimodo 模型名或别名。

### `diffusion_steps`

类型：`int`

默认值：`100`

每段生成使用的 DDIM denoising steps。数值越高通常越慢，也可能提升质量。

### `num_samples`

类型：`int`

默认值：`1`

同一组输入要生成多少条候选 motion。它不是逐 prompt 参数。

### `seed`

类型：`int | None`

默认值：`None`

随机种子，用于复现生成结果。`None` 表示不主动设置随机状态。

### `cfg_type`

类型：`Literal["nocfg", "regular", "separated"] | None`

默认值：`None`

Classifier-free guidance 的模式：

- `"nocfg"`：不使用 CFG，此时不能传 `cfg_weight`。
- `"regular"`：标准 CFG，此时 `cfg_weight` 必须是一个数字。
- `"separated"`：分开 text 和 constraint CFG，此时 `cfg_weight` 必须是
  `[text_weight, constraint_weight]`。
- `None`：如果 `cfg_weight` 也为 `None`，使用模型默认 CFG 设置；如果只传
  `cfg_weight`，runtime 会按其形状推断 CFG 类型。

### `cfg_weight`

类型：`float | list[float] | None`

默认值：`None`

CFG 的强度参数。单个 float 表示 `regular`，两个 float 的 list 表示
`separated`。无约束生成时，主要起作用的是 `text_weight`。

### `num_transition_frames`

类型：`int`

默认值：`5`

连续 prompt 段之间用于过渡的重叠帧数。只有 `texts` 包含多段时才有意义。

### `first_heading_angle`

类型：`float | list[float] | None`

默认值：`None`

初始身体朝向，单位是 radians。单个 float 会应用到所有 samples；list 可以为每个
sample 分别提供一个值。

### `formats`

类型：`list[str]`

默认值：`["npz", "bvh"]`

本次 job 要导出的 artifact 格式列表，作用于所有 samples。它不是逐 sample 映射；
也就是说 `formats=["npz", "bvh"]` 不是表示 sample 0 导出 NPZ、sample 1 导出 BVH，
而是表示这次 job 的每条候选 motion 都按可支持的方式导出 NPZ 和 BVH。

`num_samples` 控制生成几条候选 motion；`formats` 控制这些候选 motion 要导出哪些格式。

当前支持：

- `"npz"`：Kimodo NPZ。单 sample 返回 `npz`；多 sample 返回 `npz_00`、`npz_01` 等。
- `"bvh"`：SOMA BVH。单 sample 返回 `bvh`；多 sample 返回 `bvh_00`、`bvh_01` 等。
  G1 或 SMPL-X 模型会跳过 BVH artifact，job 仍可成功。

### `zip_output`

类型：`bool`

默认值：`False`

是否把本次 job 生成出的所有 artifact 文件打成一个 zip。

- `False`：每个 artifact 单独返回 metadata。
- `True`：返回一个 `zip` artifact，文件名为 `artifacts.zip`，其中包含本次 job 已生成的所有文件。

### `postprocess`

类型：`bool`

默认值：`True`

是否启用 motion post-processing。G1 模型会自动禁用 post-processing，这和 CLI/demo 行为一致。

### `root_margin`

类型：`float`

默认值：`0.04`

post-processing 中 root 水平位置修正的 margin，单位是米。这个参数主要和有 constraints 的生成有关。

### `constraints`

类型：`Any | None`

默认值：`None`

可选 constraints。无约束生成使用 `None`。如果提供，可以是 constraints JSON 文件路径，
也可以是 `load_constraints_lst` 接受的 JSON-compatible constraint dict 列表。

### `job_id`

类型：`str | None`

默认值：`None`

job id。提交请求时由 `JobStorage` 分配并写回 request。client 通常不需要传。

## Artifact 下载

生成文件统一保存到 job 目录下的 `artifacts/` 子目录。未来 HTTP adapter 应暴露：

```text
GET /jobs/{job_id}/artifacts/{artifact_key}
```

下载实现应通过 `JobStorage.resolve_artifact(job_id, artifact_key)` 查找文件，不要接受
任意本地路径作为下载参数。
