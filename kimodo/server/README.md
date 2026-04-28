# Kimodo Server 请求参数

server API 复用 CLI 和 demo 使用的生成路径。一次生成请求由 `schemas.py` 里的
`GenerationRequest` 表示。

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
  "formats": ["npz"],
  "postprocess": true,
  "root_margin": 0.04,
  "constraints": null,
  "job_id": null
}
```

## 参数说明

### `texts`

类型：`list[str]`

必填。每个字符串是一段 prompt。server 调模型时使用 `multi_prompt=True`，所以
`texts` 中的每一项会被当作一个连续动作段。

示例：

```json
["A person walks forward.", "A person turns around."]
```

这表示先生成一段向前走，再生成一段转身。

### `durations`

类型：`list[float]`

必填。每个数字是对应 prompt 段的时长，单位是秒。`durations` 必须和 `texts`
长度一致。runtime 会用 `duration * model.fps` 转成模型需要的帧数。

示例：

```json
[3.0, 2.0]
```

配合上面的 `texts`，表示第一段 3 秒，第二段 2 秒。

### `model`

类型：`str`

默认值：`DEFAULT_MODEL`，当前是 `kimodo-soma-rp`。

指定要使用的 Kimodo 模型名或别名。例如 `kimodo-soma-rp`、`kimodo-g1-rp`、
`Kimodo-SOMA-RP-v1.1` 等。

### `diffusion_steps`

类型：`int`

默认值：`100`

每段生成使用的 DDIM denoising steps。数值越高通常越慢，也可能提升质量。这个值是
全局参数，会作用到本次请求里的所有 prompt 段。

### `num_samples`

类型：`int`

默认值：`1`

同一组输入要生成多少条候选动作。它不是逐 prompt 参数。比如 `texts` 有两段且
`num_samples=3`，会生成 3 条完整动作序列，每条都包含这两段 prompt。

### `seed`

类型：`int | None`

默认值：`None`

随机种子，用于复现生成结果。

- `int`：生成前调用 `seed_everything(seed)`，尽量复现同样结果。
- `None`：不主动设置随机状态，每次结果可能不同。

### `cfg_type`

类型：`Literal["nocfg", "regular", "separated"] | None`

默认值：`None`

Classifier-free guidance 的模式。

- `"nocfg"`：不使用 CFG。此时不能传 `cfg_weight`。
- `"regular"`：标准 CFG。此时 `cfg_weight` 必须是一个数字。
- `"separated"`：分离 text 和 constraint 的 CFG。此时 `cfg_weight` 必须是
  `[text_weight, constraint_weight]`。
- `None`：如果 `cfg_weight` 也为 `None`，使用模型自身默认 CFG 设置；如果只传了
  `cfg_weight`，runtime 会按 `cfg_weight` 的形状推断 CFG 类型。

### `cfg_weight`

类型：`float | list[float] | None`

默认值：`None`

CFG 的强度参数。不同类型含义不同：

- `float`：表示 `regular` CFG 的单个 guidance weight。例如 `2.5`。
- `list[float]`：表示 `separated` CFG 的两个权重，格式是
  `[text_weight, constraint_weight]`。例如 `[2.0, 2.0]`。
- `None`：不覆盖模型默认 CFG 权重。

如果 `cfg_type` 已指定，`cfg_weight` 必须和该类型匹配。如果 `cfg_type=None` 但传了
`cfg_weight`，runtime 会自动推断：单个 float 推断为 `regular`，两个 float 的 list
推断为 `separated`。

无约束生成时，主要起作用的是 `text_weight`；没有 constraints 时，
`constraint_weight` 基本不会产生有效影响。

### `num_transition_frames`

类型：`int`

默认值：`5`

连续 prompt 段之间用于过渡的重叠帧数。只有 `texts` 包含多段时才有意义。

例如 `texts` 有两段时，模型会拿第一段末尾的 `num_transition_frames` 帧作为第二段
开头的条件，再做短窗口过渡融合。

### `first_heading_angle`

类型：`float | list[float] | None`

默认值：`None`

初始身体朝向，单位是 radians。不同类型含义不同：

- `float`：所有 samples 共用同一个初始朝向。例如 `0.0` 表示面向 canonical +Z。
- `list[float]`：为每个 sample 单独指定初始朝向。list 长度必须是 `1` 或
  `num_samples`。
- `None`：使用模型默认行为，等价于初始朝向 `0.0`。

如果不需要每个 sample 不同朝向，推荐客户端只传单个 float 或不传。

### `formats`

类型：`list[str]`

默认值：`["npz"]`

要保存的 artifact 格式。当前 runtime 实际只保存 `npz`。以后如果 server 接入更多
导出格式，可以在这里扩展。

### `postprocess`

类型：`bool`

默认值：`True`

是否启用 motion post-processing。它可以减少 foot skating，也可以帮助 constrained
generation 更贴近约束。G1 模型会自动禁用 post-processing，这和 CLI/demo 行为一致。

### `root_margin`

类型：`float`

默认值：`0.04`

post-processing 中 root 水平位置修正的 margin，单位是米。数值越小，root 越贴近
目标；数值越大，允许偏离更多。

这个参数主要和有 constraints 的生成有关。无约束基础推理里通常不用调整。

### `constraints`

类型：`Any | None`

默认值：`None`

可选 constraints。

- `None`：无约束生成。
- `str`：constraints JSON 文件路径。
- `list[dict]`：JSON-compatible constraint dict 列表，格式需要能被
  `load_constraints_lst` 解析。

无约束基础推理应传 `null` 或省略。

### `job_id`

类型：`str | None`

默认值：`None`

job id。提交请求时由 `JobStorage` 分配并写回 request。client 通常不需要传这个字段。
