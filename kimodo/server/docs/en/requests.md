# Request Parameters

One generation request is represented by `GenerationRequest`. `texts` and
`durations` are per-motion-segment fields and must have the same length; all
other fields are global settings for the whole job.

## Request Example

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

## Response Example

Single sample with `zip_output=false`:

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

Multiple samples with `zip_output=true`; all generated artifacts are packaged
into one zip:

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

## Field Overview

| Field | Type | Default | Usually set by client | Description |
| --- | --- | --- | --- | --- |
| `texts` | `list[str]` | None | Yes | Prompt segments. |
| `durations` | `list[float]` | None | Yes | Duration for each prompt segment, in seconds. |
| `model` | `str` | `DEFAULT_MODEL` | No | Kimodo model name or alias. The current default is `kimodo-soma-rp`. |
| `diffusion_steps` | `int` | `100` | No | DDIM denoising steps. |
| `num_samples` | `int` | `1` | No | Number of candidate motions to generate for the same input. |
| `seed` | `int \| None` | `None` | No | Random seed for reproducible results. |
| `cfg_type` | `"nocfg" \| "regular" \| "separated" \| None` | `None` | No | CFG mode. |
| `cfg_weight` | `float \| list[float] \| None` | `None` | No | CFG strength. |
| `num_transition_frames` | `int` | `5` | No | Overlap frames used between multiple prompt segments. |
| `first_heading_angle` | `float \| list[float] \| None` | `None` | No | Initial body heading angle in radians. |
| `formats` | `list[str]` | `["npz", "bvh"]` | No | Global artifact formats to export for this job. |
| `bvh_standard_tpose` | `bool` | `True` | No | If exporting BVH, use the standard T-pose rest skeleton. |
| `zip_output` | `bool` | `False` | No | Whether to package artifacts into one zip file. |
| `postprocess` | `bool` | `True` | No | Whether to enable motion post-processing. |
| `root_margin` | `float` | `0.04` | No | Root margin for post-processing, in meters. |
| `constraints` | `Any \| None` | `None` | No | Optional constraints; currently advanced/experimental. |
| `job_id` | `str \| None` | `None` | No | Assigned by `JobStorage`; clients usually do not set this. |

## Core Field Details

### `texts` and `durations`

`texts` is required. The server calls the model with `multi_prompt=True`, so each
entry in `texts` is treated as one continuous motion segment.

`durations` is required and must have the same length as `texts`. The runtime
converts each duration into model frames with `duration * model.fps`.

### `cfg_type` and `cfg_weight`

Classifier-free guidance supports these combinations:

- `cfg_type="nocfg"`: disable CFG. `cfg_weight` must not be passed.
- `cfg_type="regular"`: standard CFG. `cfg_weight` must be one number. It is
  the guidance scale that pushes the denoising prediction toward the
  text-and-constraint conditional prediction. `0` is close to unconditional
  prediction, `1` uses the conditional prediction, and values greater than `1`
  strengthen prompt/constraint influence at the risk of less natural motion.
- `cfg_type="separated"`: separate text and constraint CFG. `cfg_weight` must be
  `[text_weight, constraint_weight]`.
- `cfg_type=None` and `cfg_weight=None`: use the model's default CFG settings.
- `cfg_type=None` with only `cfg_weight`: the runtime infers CFG type from the
  shape of `cfg_weight`.

The single `regular` weight and the `separated` `text_weight` /
`constraint_weight` values are guidance scales, not probabilities or ratios.
They do not need to add up to `1`. For example, `[2.0, 2.0]` and `[2.0, 4.0]`
are both meaningful settings for independent text and constraint guidance
strengths.

A weight of `0` disables the extra guidance from that condition relative to the
unconditional prediction. With `separated`, `[0.0, 2.0]` disables text guidance
and keeps constraint guidance, `[2.0, 0.0]` keeps text guidance and disables
constraint guidance, and `[0.0, 0.0]` is close to unconditional generation.

Negative weights are not recommended. A negative value does not simply weaken a
condition; it pushes the prediction away from that condition and can reduce
prompt/constraint adherence or produce unstable motion.

When no external `constraints` are provided, `constraint_weight` does not always
need to be `0`. For a single text-only prompt there is no active constraint
mask, so constraint guidance usually has no practical constraint effect.
However, multi-prompt generation creates internal transition constraints between
adjacent segments, and `constraint_weight` still affects those transitions. The
default `[2.0, 2.0]` is therefore also valid for multi-prompt requests without
external constraints.

### `formats`, `num_samples`, and `zip_output`

`formats` is a job-level global export setting. It is not a per-prompt-segment,
per-sample, or per-motion-section setting.

`num_samples` controls how many candidate motions are generated; `formats`
controls which artifact formats are exported for each candidate motion.
`formats=["npz", "bvh"]` does not mean sample 0 exports NPZ and sample 1 exports
BVH. It means each generated candidate motion attempts to export both NPZ and
BVH.

Currently supported formats:

- `npz`: Kimodo NPZ. A single sample returns `npz`; multiple samples return
  `npz_00`, `npz_01`, and so on.
- `bvh`: SOMA BVH. A single sample returns `bvh`; multiple samples return
  `bvh_00`, `bvh_01`, and so on. G1 or SMPL-X models skip BVH artifacts while
  allowing the job to succeed.

`bvh_standard_tpose` controls the BVH rest pose. The default `true` exports BVH
with Kimodo's standard T-pose rest skeleton. Set it to `false` to export with
the BONES-SEED-compatible rest pose.

When `zip_output=true` and artifacts were generated, the response contains only
one `zip` artifact named `artifacts.zip`.

### `postprocess` and `root_margin`

`postprocess` controls whether motion post-processing is enabled. G1 models
automatically disable post-processing, matching the CLI/demo behavior.

`root_margin` is the root horizontal position correction margin used by
post-processing, in meters. It mainly matters for constrained generation.

### `constraints`

`constraints` is currently an advanced/experimental field. Use `None` for
unconstrained generation. If provided, it can be a constraints JSON file path or
a JSON-compatible constraint dict list accepted by `load_constraints_lst`.

This field remains advanced/experimental. Treat the format below as the current
server-supported request shape, not as a long-term stable public protocol.

## Constraints Request Format

For HTTP requests, prefer passing `constraints` as a JSON-compatible list
directly in the `POST /jobs` body rather than as a client-local file path. Each
list item is one constraint set object and must include at least `type` and
`frame_indices`. Different constraint set types can be mixed in the same list.

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

The `"... repeat until J joints ..."` / `"... J joints ..."` strings in the
examples are placeholders for explaining shapes, not valid values to submit. In
a real request, `local_joints_rot` must be a pure numeric nested array with
shape `[T, J, 3]`, where `J` must match the target model skeleton joint count,
for example `30` for SOMA30, `77` for SOMA77, and `34` for G1.

Common coordinate rules:

- `frame_indices` are 0-based frame numbers on the full generated motion
  timeline, not frame numbers restarted within each prompt segment.
- Coordinates use a Y-up frame, with XZ as the horizontal ground plane.
- Position units are meters.
- `smooth_root_2d` is `[x, z]` relative to the canonical origin; frame 0 is
  usually near `[0.0, 0.0]`.
- `root_positions` is `[x, y, z]`, where `y` is the root/hips height.
- `global_root_heading` is not radians; it is `[cos(theta), sin(theta)]`.

### Constraint Set Overview

| `type` | Required fields | Optional fields | Description |
| --- | --- | --- | --- |
| `root2d` | `frame_indices`, `smooth_root_2d` | `global_root_heading` | Constrains smoothed root path or waypoints on the XZ ground plane. |
| `fullbody` | `frame_indices`, `root_positions`, `local_joints_rot` | `smooth_root_2d` | Constrains full-body joint positions through complete pose keyframes. |
| `left-hand` | `frame_indices`, `root_positions`, `local_joints_rot` | `smooth_root_2d` | Shorthand for `end-effector`; constrains left-hand related end-effectors and root/hips. |
| `right-hand` | `frame_indices`, `root_positions`, `local_joints_rot` | `smooth_root_2d` | Shorthand for `end-effector`; constrains right-hand related end-effectors and root/hips. |
| `left-foot` | `frame_indices`, `root_positions`, `local_joints_rot` | `smooth_root_2d` | Shorthand for `end-effector`; constrains left-foot related end-effectors and root/hips. |
| `right-foot` | `frame_indices`, `root_positions`, `local_joints_rot` | `smooth_root_2d` | Shorthand for `end-effector`; constrains right-foot related end-effectors and root/hips. |
| `end-effector` | `frame_indices`, `joint_names`, `root_positions`, `local_joints_rot` | `smooth_root_2d` | General end-effector constraint; can specify multiple semantic end-effector groups. |

Except for `root2d`, pose-based constraint sets require `root_positions` and a
complete skeleton `local_joints_rot`. They do not accept direct XYZ targets for
one hand or foot. Kimodo first runs FK from the complete pose, then extracts the
full-body or end-effector targets used by the selected set type.

### `root2d`

Required fields:

- `type`: `"root2d"`
- `frame_indices`: `[T]`
- `smooth_root_2d`: `[T, 2]`, each item is `[x, z]`

Optional fields:

- `global_root_heading`: `[T, 2]`, each item is `[cos(theta), sin(theta)]`

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

Omit `global_root_heading` when heading should not be constrained:

```json
{
  "type": "root2d",
  "frame_indices": [0, 60],
  "smooth_root_2d": [[0.0, 0.0], [1.2, 0.3]]
}
```

### `fullbody`

Required fields:

- `type`: `"fullbody"`
- `frame_indices`: `[T]`
- `root_positions`: `[T, 3]`, each item is `[x, y, z]`
- `local_joints_rot`: `[T, J, 3]`, axis-angle radians

Optional fields:

- `smooth_root_2d`: `[T, 2]`, each item is `[x, z]`. If omitted, the runtime
  uses the XZ components from `root_positions`.

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

These four set types are `end-effector` shorthands. They use the same fields as
`fullbody`, but internally only use the corresponding hand/foot end-effector
targets from the complete pose.

Required fields:

- `type`: `"left-hand"`, `"right-hand"`, `"left-foot"`, or `"right-foot"`
- `frame_indices`: `[T]`
- `root_positions`: `[T, 3]`
- `local_joints_rot`: `[T, J, 3]`

Optional fields:

- `smooth_root_2d`: `[T, 2]`. If omitted, the runtime uses the XZ components
  from `root_positions`.

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

For the other shorthands, only replace `type`:

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

Use `end-effector` to constrain one or more semantic end-effector groups at
once. Current `joint_names` values use these names:

```json
["LeftFoot", "RightFoot", "LeftHand", "RightHand", "Hips"]
```

Required fields:

- `type`: `"end-effector"`
- `joint_names`: `list[str]`
- `frame_indices`: `[T]`
- `root_positions`: `[T, 3]`
- `local_joints_rot`: `[T, J, 3]`

Optional fields:

- `smooth_root_2d`: `[T, 2]`. If omitted, the runtime uses the XZ components
  from `root_positions`.

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

### Mixed Constraints

Different set types can be mixed in one request:

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

When mixing constraints, avoid contradictory targets at the same or nearby
frames. For example, if `root2d` places the root at `[0.0, 1.0]` on frame 90
but a pose-based constraint's `root_positions` / `smooth_root_2d` implies a far
away root location, generation quality can drop or constraints can be ignored.
