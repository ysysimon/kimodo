# 请求参数

一次生成请求由 `GenerationRequest` 表示。`texts` 和 `durations` 是逐 motion segment 的字段，必须长度一致；其他字段都是整个 job 的全局设置。

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

## 字段总览

| 字段 | 类型 | 默认值 | client 通常需要传 | 说明 |
| --- | --- | --- | --- | --- |
| `texts` | `list[str]` | 无 | 是 | 每个字符串是一段 prompt。 |
| `durations` | `list[float]` | 无 | 是 | 每段 prompt 的时长，单位是秒。 |
| `model` | `str` | `DEFAULT_MODEL` | 否 | Kimodo 模型名或别名，当前默认模型是 `kimodo-soma-rp`。 |
| `diffusion_steps` | `int` | `100` | 否 | DDIM denoising steps。 |
| `num_samples` | `int` | `1` | 否 | 同一组输入生成多少条候选 motion。 |
| `seed` | `int \| None` | `None` | 否 | 随机种子，用于复现结果。 |
| `cfg_type` | `"nocfg" \| "regular" \| "separated" \| None` | `None` | 否 | CFG 模式。 |
| `cfg_weight` | `float \| list[float] \| None` | `None` | 否 | CFG 强度。 |
| `num_transition_frames` | `int` | `5` | 否 | 多段 prompt 之间的过渡重叠帧数。 |
| `first_heading_angle` | `float \| list[float] \| None` | `None` | 否 | 初始身体朝向，单位是 radians。 |
| `formats` | `list[str]` | `["npz", "bvh"]` | 否 | 要导出的 artifact 格式。 |
| `zip_output` | `bool` | `False` | 否 | 是否把 artifact 打成一个 zip。 |
| `postprocess` | `bool` | `True` | 否 | 是否启用 motion post-processing。 |
| `root_margin` | `float` | `0.04` | 否 | post-processing 的 root margin，单位是米。 |
| `constraints` | `Any \| None` | `None` | 否 | 可选 constraints，当前属于 advanced/experimental。 |
| `job_id` | `str \| None` | `None` | 否 | 由 `JobStorage` 分配，client 通常不需要传。 |

## 核心字段说明

### `texts` 与 `durations`

`texts` 必填。server 调模型时使用 `multi_prompt=True`，所以 `texts` 中的每一项会被当作一个连续动作段。

`durations` 必填，长度必须和 `texts` 一致。runtime 会用 `duration * model.fps` 转成模型需要的帧数。

### `cfg_type` 与 `cfg_weight`

Classifier-free guidance 支持以下组合：

- `cfg_type="nocfg"`：不使用 CFG，此时不能传 `cfg_weight`。
- `cfg_type="regular"`：标准 CFG，此时 `cfg_weight` 必须是一个数字。
- `cfg_type="separated"`：分开 text 和 constraint CFG，此时 `cfg_weight` 必须是 `[text_weight, constraint_weight]`。
- `cfg_type=None` 且 `cfg_weight=None`：使用模型默认 CFG 设置。
- `cfg_type=None` 且只传 `cfg_weight`：runtime 会按 `cfg_weight` 的形状推断 CFG 类型。

### `formats`、`num_samples` 与 `zip_output`

`formats` 是 job 级别的全局导出配置，不是逐 prompt segment、逐 sample 或逐 motion 片段的配置。

`num_samples` 控制生成几条候选 motion；`formats` 控制每条候选 motion 要导出哪些格式。`formats=["npz", "bvh"]` 不是表示 sample 0 导出 NPZ、sample 1 导出 BVH，而是表示每条候选 motion 都尝试导出 NPZ 和 BVH。

当前支持：

- `npz`：Kimodo NPZ。单 sample 返回 `npz`；多 sample 返回 `npz_00`、`npz_01` 等。
- `bvh`：SOMA BVH。单 sample 返回 `bvh`；多 sample 返回 `bvh_00`、`bvh_01` 等。G1 或 SMPL-X 模型会跳过 BVH artifact，job 仍可成功。

当 `zip_output=true` 且已有 artifact 时，response 只返回一个 `zip` artifact，文件名为 `artifacts.zip`。

### `postprocess` 与 `root_margin`

`postprocess` 控制是否启用 motion post-processing。G1 模型会自动禁用 post-processing，这和 CLI/demo 行为一致。

`root_margin` 是 post-processing 中 root 水平位置修正的 margin，单位是米，主要和有 constraints 的生成有关。

### `constraints`

`constraints` 当前属于 advanced/experimental 字段。无约束生成使用 `None`。如果提供，可以是 constraints JSON 文件路径，也可以是 `load_constraints_lst` 接受的 JSON-compatible constraint dict 列表。

在正式 HTTP adapter 落地前，不应把这里理解为稳定的 HTTP wire format。
