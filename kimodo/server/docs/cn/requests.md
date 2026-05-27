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
  "bvh_standard_tpose": true,
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

| 字段 | 类型 | 默认值 | client 通常需要传 | 是否 prompt 级别 | 说明 |
| --- | --- | --- | --- | --- | --- |
| `texts` | `list[str]` | 无 | 是 | 是 | 每个字符串是一段 prompt / motion segment。 |
| `durations` | `list[float]` | 无 | 是 | 是 | 每段 prompt 的时长，单位是秒；和 `texts` 按 index 一一对应。 |
| `model` | `str` | `DEFAULT_MODEL` | 否 | 否 | Kimodo 模型名或别名，当前默认模型是 `kimodo-soma-rp`。 |
| `diffusion_steps` | `int` | `100` | 否 | 否 | 整个 job 共用的 DDIM denoising steps。 |
| `num_samples` | `int` | `1` | 否 | 否 | 同一组输入生成多少条候选 motion，不是 prompt 段数。 |
| `seed` | `int \| None` | `None` | 否 | 否 | 整个 job 共用的随机种子，用于复现结果。 |
| `cfg_type` | `"nocfg" \| "regular" \| "separated" \| None` | `None` | 否 | 否 | 整个 job 共用的 CFG 模式。 |
| `cfg_weight` | `float \| list[float] \| None` | `None` | 否 | 否 | 整个 job 共用的 CFG 强度；`regular` 用一个数字，`separated` 用 `[text_weight, constraint_weight]`。 |
| `num_transition_frames` | `int` | `5` | 否 | 否 | 所有相邻 prompt segment 之间共用的过渡重叠帧数。 |
| `first_heading_angle` | `float \| list[float] \| None` | `None` | 否 | 否 | 初始身体朝向，单位是 radians；可按 sample 给值，但不是按 prompt segment。 |
| `formats` | `list[str]` | `["npz", "bvh"]` | 否 | 否 | 整个 job 的 artifact 导出格式。 |
| `bvh_standard_tpose` | `bool` | `True` | 否 | 否 | 导出 BVH 时，是否使用 standard T-pose rest skeleton。 |
| `zip_output` | `bool` | `False` | 否 | 否 | 是否把整个 job 的 artifact 打成一个 zip。 |
| `postprocess` | `bool` | `True` | 否 | 否 | 是否对生成结果启用 motion post-processing。 |
| `root_margin` | `float` | `0.04` | 否 | 否 | post-processing 的 root margin，单位是米。 |
| `constraints` | `Any \| None` | `None` | 否 | 否 | 全局时间线上的可选 constraints；constraint 内部可按帧范围作用到某些 prompt segment。 |
| `job_id` | `str \| None` | `None` | 否 | 否 | 由 `JobStorage` 分配，client 通常不需要传。 |

## 核心字段说明

### `texts` 与 `durations`

`texts` 必填。server 调模型时使用 `multi_prompt=True`，所以 `texts` 中的每一项会被当作一个连续动作段。多段 prompt 最终会按时间拼成一条 motion；不会因为有多段 prompt 就自动导出多个 BVH。

`durations` 必填，长度必须和 `texts` 一致，并和 `texts` 按 index 一一对应。runtime 会用 `duration * model.fps` 转成模型需要的帧数。

### `cfg_type` 与 `cfg_weight`

Classifier-free guidance 支持以下组合：

- `cfg_type="nocfg"`：不使用 CFG，此时不能传 `cfg_weight`。
- `cfg_type="regular"`：标准 CFG，此时 `cfg_weight` 必须是一个数字，表示把 denoising 预测往“同时满足 text 和 constraint 的条件预测”方向推的强度。`0` 接近无条件预测，`1` 使用条件预测，`>1` 会强化 prompt/constraint 的影响，但过大可能降低自然度。
- `cfg_type="separated"`：分开 text 和 constraint CFG，此时 `cfg_weight` 必须是 `[text_weight, constraint_weight]`。
- `cfg_type=None` 且 `cfg_weight=None`：使用模型默认 CFG 设置。
- `cfg_type=None` 且只传 `cfg_weight`：runtime 会按 `cfg_weight` 的形状推断 CFG 类型。

`regular` 的单个权重，以及 `separated` 的 `text_weight` / `constraint_weight` 都是 guidance scale，不是概率或比例，所以不需要加起来等于 `1`。例如 `[2.0, 2.0]`、`[2.0, 4.0]` 都是合法语义，分别表示 text 和 constraint 两路 guidance 的独立强度。

权重为 `0` 表示不使用对应条件相对于 unconditional prediction 的额外 guidance。`separated` 中 `[0.0, 2.0]` 会关闭 text guidance、保留 constraint guidance；`[2.0, 0.0]` 会保留 text guidance、关闭 constraint guidance；`[0.0, 0.0]` 接近无条件生成。

负数权重不推荐使用。负数不是“减弱一点”，它会把预测往对应条件的反方向推，通常会降低 prompt/constraint 符合度并增加不稳定结果风险。

没有传外部 `constraints` 时，`constraint_weight` 不一定必须设为 `0`。单段 text-only 生成中，没有 active constraint mask，constraint guidance 通常不会产生实际约束效果；但多段 prompt 生成时，runtime 会为相邻 segment 构造内部 transition constraints，`constraint_weight` 仍会影响段落衔接。因此默认 `[2.0, 2.0]` 也适用于没有外部 constraints 的 multi-prompt 请求。

### `formats`、`num_samples` 与 `zip_output`

`formats` 是 job 级别的全局导出配置，不是逐 prompt segment、逐 sample 或逐 motion 片段的配置。

`num_samples` 控制生成几条候选 motion；`formats` 控制每条候选 motion 要导出哪些格式。`formats=["npz", "bvh"]` 不是表示 sample 0 导出 NPZ、sample 1 导出 BVH，而是表示每条候选 motion 都尝试导出 NPZ 和 BVH。

当前支持：

- `npz`：Kimodo NPZ。单 sample 返回 `npz`；多 sample 返回 `npz_00`、`npz_01` 等。
- `bvh`：SOMA BVH。单 sample 返回 `bvh`；多 sample 返回 `bvh_00`、`bvh_01` 等。G1 或 SMPL-X 模型会跳过 BVH artifact，job 仍可成功。

`bvh_standard_tpose` 控制 BVH 的 rest pose。默认值 `true` 会用 Kimodo standard T-pose rest skeleton 导出 BVH；传 `false` 时会使用 BONES-SEED 兼容的 rest pose。

当 `zip_output=true` 且已有 artifact 时，response 只返回一个 `zip` artifact，文件名为 `artifacts.zip`。

### `postprocess` 与 `root_margin`

`postprocess` 控制是否启用 motion post-processing。G1 模型会自动禁用 post-processing，这和 CLI/demo 行为一致。

`root_margin` 是 post-processing 中 root 水平位置修正的 margin，单位是米，主要和有 constraints 的生成有关。

### `constraints`

`constraints` 当前属于 advanced/experimental 字段。无约束生成使用 `None`。如果提供，可以是 constraints JSON 文件路径，也可以是 `load_constraints_lst` 接受的 JSON-compatible constraint dict 列表。

`constraints` 不是 prompt 级别字段，而是作用在整条生成 motion 的全局时间线上。多段 prompt 生成时，runtime 会按累计帧范围把 constraint 裁剪到当前 segment；因此 constraint 的帧号应按完整 motion 的时间线来写，而不是每段 prompt 内从 0 重新计数。

这个字段仍属于 advanced/experimental；建议把下面的格式当作当前 server 实现支持的请求格式，而不是长期稳定的公共协议。

## Constraints 请求格式

HTTP 请求中推荐把 `constraints` 作为 JSON-compatible list 直接放进 `POST /jobs` body，而不是传 client 本地文件路径。每个 list item 是一个 constraint set object，并且至少包含 `type` 和 `frame_indices`。不同 `type` 的 constraint set 可以混合放在同一个数组里。

```json
{
  "texts": ["A person walks forward and reaches with the right hand."],
  "durations": [4.0],
  "model": "kimodo-soma-rp",
  "cfg_type": "separated",
  "cfg_weight": [2.0, 2.0],
  "constraints": [
    {
      "type": "root2d",
      "frame_indices": [0, 60, 120],
      "smooth_root_2d": [[0.0, 0.0], [0.0, 1.0], [0.5, 2.0]]
    },
    {
      "type": "right-hand",
      "frame_indices": [90],
      "root_positions": [[0.25, 0.95, 1.5]],
      "smooth_root_2d": [[0.25, 1.5]],
      "local_joints_rot": [
        [
          [0.0, 0.0, 0.0],
          "... repeat until J joints ..."
        ]
      ]
    }
  ]
}
```

下面示例中的 `"... repeat until J joints ..."` / `"... J joints ..."` 只是为了说明形状，不是可直接提交的合法值。真实请求里的 `local_joints_rot` 必须是纯数字嵌套数组，形状为 `[T, J, 3]`，其中 `J` 必须等于目标模型 skeleton 的 joint 数量，例如 SOMA30 是 `30`，SOMA77 是 `77`，G1 是 `34`。

通用坐标规则：

- `frame_indices` 是整条生成 motion 的 0-based 全局帧号，不是每个 prompt segment 内部重新计数。
- 坐标系是 Y-up，XZ 是水平地面平面。
- 位置单位是米。
- `smooth_root_2d` 是 `[x, z]`，相对 canonical origin；通常第 0 帧附近是 `[0.0, 0.0]`。
- `root_positions` 是 `[x, y, z]`，其中 `y` 是 root/hips 高度。
- `global_root_heading` 不是 radians，而是 `[cos(theta), sin(theta)]`。

### Constraint set 总览

| `type` | 必须字段 | Optional 字段 | 说明 |
| --- | --- | --- | --- |
| `root2d` | `frame_indices`, `smooth_root_2d` | `global_root_heading` | 约束 smoothed root 在 XZ 地面平面的路径或 waypoint。 |
| `fullbody` | `frame_indices`, `root_positions`, `local_joints_rot` | `smooth_root_2d` | 通过完整 pose keyframe 约束全身 joint positions。 |
| `left-hand` | `frame_indices`, `root_positions`, `local_joints_rot` | `smooth_root_2d` | `end-effector` 的 shorthand，只约束左手相关 end-effector 和 root/hips。 |
| `right-hand` | `frame_indices`, `root_positions`, `local_joints_rot` | `smooth_root_2d` | `end-effector` 的 shorthand，只约束右手相关 end-effector 和 root/hips。 |
| `left-foot` | `frame_indices`, `root_positions`, `local_joints_rot` | `smooth_root_2d` | `end-effector` 的 shorthand，只约束左脚相关 end-effector 和 root/hips。 |
| `right-foot` | `frame_indices`, `root_positions`, `local_joints_rot` | `smooth_root_2d` | `end-effector` 的 shorthand，只约束右脚相关 end-effector 和 root/hips。 |
| `end-effector` | `frame_indices`, `joint_names`, `root_positions`, `local_joints_rot` | `smooth_root_2d` | 通用 end-effector constraint，可一次指定多个 semantic end-effector group。 |

除 `root2d` 外，pose-based constraint set 都需要 `root_positions` 和完整 skeleton 的 `local_joints_rot`。这些 set 不是直接传某个手/脚的 XYZ target；它们会先用完整 pose 做 FK，再从中抽取对应的全身或 end-effector 目标。

### `root2d`

必须字段：

- `type`: `"root2d"`
- `frame_indices`: `[T]`
- `smooth_root_2d`: `[T, 2]`，每项为 `[x, z]`

Optional 字段：

- `global_root_heading`: `[T, 2]`，每项为 `[cos(theta), sin(theta)]`

```json
{
  "type": "root2d",
  "frame_indices": [0, 30, 60],
  "smooth_root_2d": [
    [0.0, 0.0],
    [0.0, 1.0],
    [0.5, 2.0]
  ],
  "global_root_heading": [
    [1.0, 0.0],
    [0.7071, 0.7071],
    [0.0, 1.0]
  ]
}
```

如果不需要约束朝向，可以省略 `global_root_heading`：

```json
{
  "type": "root2d",
  "frame_indices": [0, 60],
  "smooth_root_2d": [[0.0, 0.0], [1.2, 0.3]]
}
```

### `fullbody`

必须字段：

- `type`: `"fullbody"`
- `frame_indices`: `[T]`
- `root_positions`: `[T, 3]`，每项为 `[x, y, z]`
- `local_joints_rot`: `[T, J, 3]`，axis-angle radians

Optional 字段：

- `smooth_root_2d`: `[T, 2]`，每项为 `[x, z]`。如果省略，runtime 会使用 `root_positions` 的 XZ 分量。

```json
{
  "type": "fullbody",
  "frame_indices": [60],
  "root_positions": [[0.5, 0.95, 2.0]],
  "smooth_root_2d": [[0.5, 2.0]],
  "local_joints_rot": [
    [
      [0.0, 0.0, 0.0],
      [0.02, 0.0, 0.0],
      "... repeat until J joints ..."
    ]
  ]
}
```

### `left-hand` / `right-hand` / `left-foot` / `right-foot`

这四个 set type 是 `end-effector` 的 shorthand。字段与 `fullbody` 相同，但内部只使用完整 pose 中对应 hand/foot end-effector 相关目标。

必须字段：

- `type`: `"left-hand"`、`"right-hand"`、`"left-foot"` 或 `"right-foot"`
- `frame_indices`: `[T]`
- `root_positions`: `[T, 3]`
- `local_joints_rot`: `[T, J, 3]`

Optional 字段：

- `smooth_root_2d`: `[T, 2]`。如果省略，runtime 会使用 `root_positions` 的 XZ 分量。

```json
{
  "type": "right-hand",
  "frame_indices": [90],
  "root_positions": [[0.25, 0.95, 1.5]],
  "smooth_root_2d": [[0.25, 1.5]],
  "local_joints_rot": [
    [
      [0.0, 0.0, 0.0],
      [0.02, 0.0, 0.0],
      "... repeat until J joints ..."
    ]
  ]
}
```

其它 shorthand 只需要替换 `type`：

```json
{ "type": "left-hand", "frame_indices": [90], "root_positions": [[0.25, 0.95, 1.5]], "local_joints_rot": [["... J joints ..."]] }
```

```json
{ "type": "left-foot", "frame_indices": [90], "root_positions": [[0.25, 0.95, 1.5]], "local_joints_rot": [["... J joints ..."]] }
```

```json
{ "type": "right-foot", "frame_indices": [90], "root_positions": [[0.25, 0.95, 1.5]], "local_joints_rot": [["... J joints ..."]] }
```

### `end-effector`

`end-effector` 用于一次约束一个或多个 semantic end-effector group。当前 `joint_names` 使用这些名字：

```json
["LeftFoot", "RightFoot", "LeftHand", "RightHand", "Hips"]
```

必须字段：

- `type`: `"end-effector"`
- `joint_names`: `list[str]`
- `frame_indices`: `[T]`
- `root_positions`: `[T, 3]`
- `local_joints_rot`: `[T, J, 3]`

Optional 字段：

- `smooth_root_2d`: `[T, 2]`。如果省略，runtime 会使用 `root_positions` 的 XZ 分量。

```json
{
  "type": "end-effector",
  "joint_names": ["LeftHand", "RightFoot"],
  "frame_indices": [45, 90],
  "root_positions": [
    [0.0, 0.95, 0.8],
    [0.25, 0.95, 1.5]
  ],
  "smooth_root_2d": [
    [0.0, 0.8],
    [0.25, 1.5]
  ],
  "local_joints_rot": [
    [
      [0.0, 0.0, 0.0],
      [0.02, 0.0, 0.0],
      "... repeat until J joints ..."
    ],
    [
      [0.05, 0.0, 0.0],
      [0.02, 0.01, 0.0],
      "... repeat until J joints ..."
    ]
  ]
}
```

### 混合 constraints

可以在一个请求中混合不同 set type：

```json
{
  "constraints": [
    {
      "type": "root2d",
      "frame_indices": [0, 60, 120],
      "smooth_root_2d": [[0.0, 0.0], [0.0, 1.0], [0.5, 2.0]]
    },
    {
      "type": "right-hand",
      "frame_indices": [90],
      "root_positions": [[0.25, 0.95, 1.5]],
      "local_joints_rot": [
        [
          [0.0, 0.0, 0.0],
          "... repeat until J joints ..."
        ]
      ]
    },
    {
      "type": "fullbody",
      "frame_indices": [120],
      "root_positions": [[0.5, 0.95, 2.0]],
      "local_joints_rot": [
        [
          [0.0, 0.0, 0.0],
          "... repeat until J joints ..."
        ]
      ]
    }
  ]
}
```

混合使用时要避免同一帧或相近帧的约束互相矛盾。例如 `root2d` 指定第 90 帧 root 在 `[0.0, 1.0]`，但某个 pose-based constraint 的 `root_positions` / `smooth_root_2d` 暗示 root 在很远的位置，这会降低生成质量或导致约束被忽略。
