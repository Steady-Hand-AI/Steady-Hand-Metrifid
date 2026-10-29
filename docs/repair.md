# Planar controller repair

Available in Metrifid **0.9.0** through the `metrifid repair` commands and `metrifid.repair` SDK.

Metrifid adjusts an explicit starting controller for one fixed planar pushing family, retains the
measured result, and independently replays a selected controller. Every qualification is
**simulation only**, for the stated task and runtime. It is not hardware safety approval, proof of
a global optimum, or a robustness guarantee. A finite unsuccessful search does not prove the task
impossible.

## Prepare the supported runtime

Repair and replay require **Linux x86_64, CPython 3.13, MuJoCo 3.16.0 and NumPy 2.5.4**. Prepare an
isolated environment yourself and install the supported package versions:

```bash
python3.13 -m venv .venv-repair
. .venv-repair/bin/activate
python -m pip install 'metrifid==0.9.0' 'mujoco==3.16.0' 'numpy==2.5.4'
metrifid repair --help
```

These pins apply to repair execution; the core package retains Python >=3.11 and its broader
platform-dependent MuJoCo requirements. Metrifid does not install or discover execution
environments. An unsupported or missing runtime is refused before simulation. Help, request
loading, and saved-result reading require neither MuJoCo nor NumPy; they do not establish support
for execution on that machine.

## Run, read and independently replay

Save [`examples/repair/planar_push.json`](../examples/repair/planar_push.json) as `request.json`.
It contains an explicit nonwinning starting vector, a target and a lower finger rail. Omitted
limits take the documented defaults; no initializer is loaded from elsewhere.

```bash
mkdir repair-results
metrifid repair run request.json --output repair-results/run
metrifid repair show repair-results/run/repair.json

mv repair-results/run repair-results/saved-run
metrifid repair replay repair-results/saved-run --output repair-results/replay
metrifid repair show repair-results/replay/replay.json
```

All output directories must be new, under existing real parents. Existing destinations are
refused, including empty ones. Symbolic-link path components are refused. A run can be a sibling
of its request file. A replay output must be outside its input bundle, and cannot overwrite that
bundle or an ancestor. There is no implicit output location, browser or server.

A run retains `repair.json` and the available generated `nominal.xml`, `finer.xml`, `reset.json`
and selected `policy.json`. Keep these files together when moving a bundle. Replay uses the saved
request, generated scenes, reset and selected controller; it does not reopen the original request
or source files. Replay independently establishes and compares the physical reset, then evaluates
the selected controller at both timesteps, without optimization. Its output is just `replay.json`.
A replay record can be read independently, but cannot be used as a new run bundle for replay.

`repair show` validates the retained bytes and their internal consistency. Its exit `0` means
**read success**, even when the saved run exited `10`, `20` or `70`. It reports the recorded outcome
and qualification separately; it performs no new simulation. Generic `metrifid show` accepts
certification and model-release receipts, not repair records. A saved success does not allow
replay to skip its fresh evaluation.

Add `--json` to any operation for one `metrifid.repair_summary` document followed by a newline.
Diagnostics are on stderr. Successfully parsed refusals and errors also emit this summary.
Syntax errors, including missing required arguments, return `64` with stderr and empty stdout.
There is no `--full` repair option. For example, read the saved result as JSON:

```bash
metrifid repair show repair-results/saved-run/repair.json --json
```

The source distribution also includes an installed-command check using the unchanged public
example. From an extracted source distribution or checkout, with the environment above active:

```bash
python examples/repair/installed_smoke.py --output repair-smoke --expected-version 0.9.0
```

The output parent must not exist. The check invokes the installed console script for one repair
run, saved-result reads and one independent replay after moving the bundle and deleting its copied
request. It requires a changed, confirmed controller and a freshly qualified replay. It is a
bounded check of this one example, not a general success guarantee.

### Python file workflow

```python
from metrifid.repair import (
    read_result_file,
    replay_run_directory,
    run_request_file,
)

run = run_request_file("request.json", "repair-results/run")
print(run.command_status, run.command_exit_code, run.reason)
if run.result_path is not None:
    saved = read_result_file(run.result_path)
    print(saved.command_status, saved.recorded_status, saved.recorded_exit_code)
    if saved.document["selected_parameters"] is not None:
        replay = replay_run_directory("repair-results/run", "repair-results/replay")
        print(replay.command_status, replay.command_exit_code)
```

The three functions share the CLI's admission, writer and evaluator. Their `WorkflowResult`
contains `operation`, `command_status`, `command_exit_code`, `reason`, nullable `recorded_status`
and `recorded_exit_code`, `qualified`, nullable verified `result_path`, verified `artifact_paths`,
nullable admitted `document` and `actual_work`, nullable `diagnostic`, and `partial_output`.
Paths are absolute and returned only after verification. A publication failure returns no
successful path and `qualified=False`, even if physical evaluation had qualified.

The in-memory SDK remains available:

```python
from metrifid.repair import load_request, repair_controller, replay_controller

request = load_request("request.json")
result = repair_controller(request)
replay = replay_controller(result)
```

`load_request` raises `RepairInputError` for invalid input (`reason="INVALID_REQUEST"`) or reader
OS failure (`reason="INPUT_READ_FAILED"`). The file workflow maps these to exits `64` and `70`,
respectively. In-memory results are Python records, not an additional serialized format.

## Supported task and controller

The family `planar_push_timed15_v1` contains one 80 × 40 × 20 mm, 0.1 kg block, one horizontally
guided cylindrical finger, and a fixed square support. There are no walls or added stops. Gravity,
geometry, masses, contacts, nominal initial pose, force limits and physical gates are fixed. This
is not a loader for arbitrary robots, MJCF models or controllers.

The controller has three object-relative approach/push strokes. Each stroke's first four
normalized coordinates describe approach angle, tangential offset and two local push components;
the final three coordinates set stroke durations. The controller observes block pose at stroke
boundaries and finger position at 50 Hz. Its reference speed is bounded by 0.12 m/s and the PD
force command by 2 N Euclidean norm. Timing durations are scaled together if needed and cumulative
deadlines are floored to the 50 Hz grid. The final reference is stationary before the hold window.

Every evaluation covers four seconds at 0.002 s and 0.001 s. Qualification requires all four
predicates at every sample of the final 0.2 s: position error ≤ 0.005 m, yaw error ≤ π/60 rad,
linear speed ≤ 0.01 m/s, and angular speed ≤ 0.1 rad/s. Whole-episode finite-state, warning,
contact/penetration, lift, tilt, workspace, rail, speed, reference and force gates still apply.
The paired objective is the maximum of `max(0, hold_max / limit - 1)` over all eight hold terms.
A zero screening objective requires separate complete paired confirmation.

### Request version 1

Objects reject unknown or duplicate keys. All fields below are required except `limits` and its
members. JSON `null`, numeric strings, booleans as numbers, nonfinite tokens and overflow are
invalid. Integer fields reject decimal tokens such as `1.0`. Other finite integer/decimal tokens
are normalized to binary64.

| Field | Type and admitted value |
| --- | --- |
| `schema` | String `metrifid.repair_request` |
| `version` | Integer `1` |
| `family` | String `planar_push_timed15_v1` |
| `task` | Object with exactly the three task fields below |
| `task.rail_y_lower_m` | Number in `[-0.18, -0.04]`; upper rail stays `0.18` |
| `task.target_xy_m` | Two numbers; center inside `[-0.18,0.18]²` and every rotated block corner inside support `[-0.2,0.2]²` |
| `task.target_yaw_rad` | Number in `[-π,π]`, without wrapping or clipping |
| `controller` | Object with only `parameters_normalized` |
| `controller.parameters_normalized` | Exactly 15 numbers, each strictly inside `(1e-9, 1-1e-9)` |
| `limits` | Optional object; `{}` and omission use independent defaults |
| `limits.max_attempts` | Integer `2`–`1000`; default `128` |
| `limits.max_integration_steps` | Integer ≥ `12200`; default `96000` |
| `limits.max_cpu_seconds` | Finite positive number; default `120` |

Selected/generated controllers use the closed `[0,1]` box. A selected boundary vector is valid
for replay but may be invalid as a new starting request. Neither boundary is weakened.

Request files are UTF-8 without BOM, at most 65,536 bytes including whitespace, depth 8, 128 nodes,
and 128 UTF-8 bytes per key/string. Admission imports no native libraries.

## Outcomes, costs and partial output

| Operation | Status | Exit | Meaning |
| --- | --- | ---: | --- |
| Run | `REPAIRED` | 0 | A changed controller passed complete paired confirmation |
| Run | `ALREADY_QUALIFIED` | 0 | The supplied start passed complete paired confirmation |
| Run | `NO_QUALIFIED_POLICY_FOUND` | 10 | Finite nonqualification, known initial invalidity, or a finite search bound |
| Replay | `QUALIFIED` | 0 | Fresh paired evaluation qualified |
| Replay | `NOT_QUALIFIED` | 10 | Fresh complete physical nonqualification |
| Run/replay | `INCOMPLETE` | 20 | CPU interruption or unknown, missing, nonfinite or adjudication-required evidence |
| All | `REFUSED` | 64 | Malformed/unsupported data, inconsistent bundle, unsupported runtime or invalid destination |
| All | `ERROR` | 70 | File I/O/publication failure or unexpected internal/evaluator exception |
| Show | `READ` | 0 | Record validated; its saved status, exit and qualification are reported separately |

Replay without a selected controller is `REFUSED/64`, reason `NO_SELECTED_CONTROLLER`, without
simulation. A known invalid initial state/policy is finite nonqualification, not proof of
infeasibility. Complete failed confirmation terminates with reason `CONFIRMATION_DID_NOT_QUALIFY`.
KeyboardInterrupt, SystemExit and native process death are not fabricated completed results.

Coordinate polling uses step `0.01`, minimum step `0.00125`, positive-improvement tolerance
`1e-10`, at most 100 sweeps, initial preferred sign `+1`, and halving after a sweep without
movement. A first measured zero receives confirmation and terminates regardless of whether its
improvement is smaller than the positive-improvement tolerance. There are no seeds or restarts.

Attempts include the initial screen and out-of-box proposals, but not confirmation. Out-of-box
proposals use zero physics steps. Before a physical screen, the engine reserves its full possible
pair plus a confirmation pair; it charges actual completed work only. Nominal settling costs
200 steps, a complete pair 6,000 steps, and separate replay 6,200 steps. Partial and hard-rejected
trajectories remain charged. A failed native call can leave an unknown total, with the known
completed count retained separately; counts are never inferred from simulation time.

The process-CPU interval begins on entering the engine invocation, before lazy native imports and
model setup. Search uses at most 75% of the request cap; confirmation uses the same clock through
the full cap. Separate replay receives a fresh full CPU and integration allowance, with no search
attempts. These cooperative bounds are not command wall-time or preemption guarantees. JSON
admission, publication and CLI presentation are outside the reported engine CPU interval.

Available member files are published before `repair.json`, the run commit marker. `replay.json`
is published after replay. Files and public directory identity are reverified before returning
paths. Publication failure returns `ERROR/70`, no successful result path and no command
qualification. Already committed owned files may remain, including a receipt whose final
verification failed. Cleanup removes only still-owned private temporaries. An early refusal may
leave a newly owned empty directory; there is no automatic atomic-directory rollback.

## Saved format, version 1

All listed fields are required unless explicitly nullable. Unknown/duplicate keys and wrong types
are rejected. Floats remain numbers with exact binary64 round trips. Serialization uses sorted
keys, compact separators, `ensure_ascii=True`, `allow_nan=False`, UTF-8 and exactly one trailing
newline. A parameter identity is SHA-256 of that encoding of the vector itself. Artifact hashes
cover exact bytes, including any newline. SHA-256 strings contain 64 lowercase hexadecimal digits. Physical reset and scene identities retain their own
algorithms; they are not vector hashes.

A nonfinite/unavailable measured metric is stored as `null`, with an incomplete/nonfinite reason
and its field path in `unavailable_measurements`. Request and selected-vector invalidity is
refused, never repaired by this replacement. Missing measurement records stay `null`.

The `limitations` array contains these exact three strings:

```json
[
  "Simulation only.",
  "Only planar_push_timed15_v1 on Linux x86_64, CPython 3.13, MuJoCo 3.16.0 and NumPy 2.5.4.",
  "No global-optimum, robustness, authorship or cross-platform bitwise replay guarantee."
]
```

### CLI summary

| Field | Type/value |
| --- | --- |
| `schema`, `version` | `metrifid.repair_summary`, integer `1` |
| `operation` | `run`, `show` or `replay` |
| `command_status`, `command_exit_code` | Command status string and integer exit from the table above |
| `recorded_status`, `recorded_exit_code` | Recorded status string/integer exit, or `null` when unavailable |
| `qualified` | Boolean; saved qualification for `READ`, fresh qualification for evaluation, false on command failure |
| `reason` | String describing the outcome |
| `result_path` | Verified absolute result pathname string, or `null` |

### Run record: `repair.json`

| Field | Type/value |
| --- | --- |
| `schema`, `version`, `operation` | `metrifid.repair_result`, integer `1`, `run` |
| `tool` | Object `{name: "metrifid", version: string}` using actual installed metadata |
| `request` | Admitted request above, with all default limits explicit, or `null` if unavailable before admission |
| `profile` | Supported generated profile below, or `null` when unavailable |
| `timestep_identities` | Object with numeric `nominal_dt_s`, `finer_dt_s` and string `nominal_scene_sha256`, `finer_scene_sha256`, or `null` |
| `status`, `exit_code`, `reason` | Recorded status string, integer exit, reason string |
| `qualified`, `confirmation_complete` | Booleans |
| `diagnostic` | String or `null` |
| `original_parameters`, `original_parameters_sha256` | Starting vector and identity string, or `null` if unavailable |
| `selected_parameters`, `selected_parameters_sha256` | Closed-box selected vector and identity string, or `null` |
| `initialization` | Physical measurement below, or `null` |
| `initial_screening`, `selected_screening`, `confirmation` | Paired measurement below, or `null` |
| `method` | `{name: "coordinate_poll", settings: object}` with exact settings below |
| `costs` | Execution costs below |
| `runtime` | Object with available string fields `implementation`, `python`, `system`, `machine`, `mujoco`, `numpy`; empty if unavailable |
| `artifacts` | Object keyed only by available fixed member filenames, with artifact records below |
| `limitations` | Array of strings stating supported family/runtime, simulation-only scope and no global-optimum/robustness/authorship guarantee |
| `unavailable_measurements` | Sorted exact paths for every null in measurement domains, including missing entire records; dot and `[index]` notation |

Method settings are exactly `initial_step: 0.01`, `minimum_step: 0.00125`,
`actual_improvement_tolerance: 1e-10`, `max_sweeps: 100`, `initial_preferred_sign: 1`, and
`stagnant_step_multiplier: 0.5`. There is no serialized screening history or trajectory.

### Replay record: `replay.json`

| Field | Type/value |
| --- | --- |
| `schema`, `version`, `operation` | `metrifid.repair_replay`, integer `1`, `replay` |
| `tool` | Same tool object as the run record |
| `source_repair_sha256` | SHA-256 string of source `repair.json`; linkage, not a remote dependency |
| `request` | Fully resolved admitted request, or `null` if unavailable before admission |
| `selected_parameters`, `selected_parameters_sha256` | Selected vector and parameter identity string, or `null` when unavailable |
| `status`, `exit_code`, `reason`, `qualified` | Fresh replay status string, integer exit, reason string and boolean |
| `diagnostic` | String or `null` |
| `measurement` | Fresh paired measurement, or `null` |
| `runtime`, `costs`, `limitations`, `unavailable_measurements` | Same field meanings as above, for this replay |

### Artifact, policy and reset records

| Record | Exact fields |
| --- | --- |
| Artifact entry | `filename`: one of `nominal.xml`, `finer.xml`, `reset.json`, `policy.json`; `size_bytes`: nonnegative integer; `sha256`: SHA-256 string |
| `policy.json` | `schema`: `metrifid.repair_policy`; `version`: integer `1`; `parameters_normalized`: 15-number selected vector; `parameters_sha256`: identity string; `qualified`: boolean |
| `reset.json` | `schema`: `metrifid.repair_reset`; `version`: integer `1`; `arrays`: array of entries below; `settled_block_com`: three finite numbers; `settled_block_yaw`: finite number; `nominal_scene_sha256`, `state_sha256`: SHA-256 strings; `settling_failures`: array of strings |
| Reset array entry | `name`: string; `shape`: array of nonnegative integers; `values`: flattened array of finite numbers |

Reset arrays are exactly `qpos` shape `[9]`, `qvel` `[8]`, `act` `[0]`, `mocap_pos` `[0,3]`, and
`mocap_quat` `[0,4]`. The state digest covers their little-endian binary64 values in sorted name
order. Replay validates dimensions, finiteness and identities, then requires exact equality with
an independently established reset in the supported runtime. A recomputed supplied hash alone
cannot satisfy that check. Saved XML is compared to independently generated expected content;
arbitrary XML is never executed.

### Measurements and costs

| Physical measurement field | Type |
| --- | --- |
| `classification` | `COMPLETE`, `HARD_REJECTION` or `INCOMPLETE` |
| `qualified`, `reason` | Boolean and string |
| `hold_maxima` | Four numbers in position/yaw/linear-speed/angular-speed order, or `null`; unavailable elements may be `null` only with explicit incomplete evidence |
| `metrics` | Object containing the available typed measurement fields described below |
| `hard_failures` | Array of strings |
| `completed_steps`, `expected_steps` | Nonnegative integers |
| `dt_s` | Number, or `null` only for incomplete/unavailable evidence |
| `hold_samples`, `expected_hold_samples` | Nonnegative integers |

Initialization metrics contain `settling_failures` (string array) and `physical_state_sha256`
(string). Episode metrics contain numeric maxima `maximum_penetration_m`, `max_command_norm_N`,
`max_finger_speed_m_s`, `max_block_speed_m_s`, `maximum_lift_m`, `maximum_tilt_rad`,
`max_reference_speed_m_s`; numeric `dt_s` and `episode_time_s`; integer `physics_steps`,
`requested_physics_steps`, `hold_samples`, `expected_hold_samples`; string `settled_state_sha256`;
`endpoint_errors` (four numbers or null); the four nullable numeric fields
`hold_max_position_error_m`, `hold_max_yaw_error_rad`, `hold_max_linear_speed_m_s`,
`hold_max_angular_speed_rad_s`; `schedule`; and `warnings`.

`schedule` has numeric arrays `boundaries_s`, `decoded_boundaries_s`, `motion_deadlines_s`,
`scheduled_starts_s`, `timing_parameters_normalized`; `stroke_windows_s` is three two-number
arrays; `stationary_tail_start_s` is numeric. `warnings` is an array of objects with integer
`index`, `number` and `lastinfo`. Unavailable measured numbers use null as explicitly listed in
`unavailable_measurements`; they cannot substantiate success.

| Paired measurement field | Type |
| --- | --- |
| `parameters_normalized` | 15-number vector bound to this evaluation |
| `nominal`, `finer` | Physical measurement or `null` |
| `objective` | Finite nonnegative number or `null` |
| `classification`, `reason` | Strings; classification is `COMPLETE`, `HARD_REJECTION` or `INCOMPLETE` |
| `complete`, `qualified` | Booleans |
| `known_completed_steps`, `unknown_step_calls` | Nonnegative integers |

| Execution cost field | Type |
| --- | --- |
| `attempts` | Nonnegative integer |
| `integration_steps` | Nonnegative integer or `null` for unknown total |
| `known_completed_steps`, `unknown_step_calls` | Nonnegative integers |
| `cpu_seconds` | Finite nonnegative number |
| `limits` | Resolved request limits object, or `null` when unavailable |
| `search_cpu_fraction` | Number `0.75` for admitted repair, `null` for replay or unavailable limits |

### Generated profile

`profile` is an exact supported-family definition, not a user-extensible model. Its root fields
are `family` (string), `scene` (object) and `controller` (object). The reader regenerates and
compares it against the admitted request.

| Object | Exact fields and types |
| --- | --- |
| `scene` | `gravity`: three numbers; `floor`, `block`, `finger`, `contact`, `runtime`, `target`, `whole_episode_gates`: objects below |
| `scene.floor` | `surface_z`: number; `half_sizes_m`: three numbers |
| `scene.block` | `size_m`: three numbers; `mass_kg`: number; `initial_center_m`: three numbers; `initial_yaw_rad`: number |
| `scene.finger` | `radius_m`, `height_m`, `center_z_m`, `mass_kg`: numbers; `initial_xy_m`: two numbers; `rail_limits_xy_m`: two two-number bounds |
| `scene.contact` | `condim`: integer; `sliding_friction`: object with numeric `block_floor`, `finger_block`; `solref`: two numbers; `solimp`: five numbers; `penetration_gate_m`: number |
| `scene.runtime` | Numeric `nominal_dt_s`, `finer_dt_s`, `solver_tolerance`, `control_period_s`, `settle_seconds`, `episode_seconds`; integer `solver_iterations`; string `integrator`, `solver`, `cone` |
| `scene.target` | `xy_m`: two numbers; numeric `yaw_rad`, `position_tolerance_m`, `yaw_tolerance_rad`, `linear_speed_max_m_s`, `angular_speed_max_rad_s`, `stable_hold_s` |
| `scene.whole_episode_gates` | Numeric `lift_above_settled_com_m`, `tilt_max_rad`, `block_speed_max_m_s`, `finger_speed_max_m_s`; `block_xy_com_bounds_m`: two two-number bounds |
| `controller` | Integer `parameter_dimension`; numeric `position_gain_N_m`, `velocity_gain_N_s_m`, `force_norm_bound_N`, `reference_speed_bound_m_s`; `reference_bounds_xy_m`: two two-number bounds |

### Admission and portability limits

Each `repair.json`/`replay.json` is bounded to 1 MiB, depth 16, 16,000 nodes and 64 KiB per
key/string. Each XML and `reset.json` is at most 64 KiB; `policy.json` is at most 16 KiB. Overflow
is refused, not truncated. Members must be bounded regular files with exact fixed names and
verified byte counts/hashes. Undeclared side files in the directory are ignored and are never
included in verified artifact paths. Symbolic links or special files used as declared members,
path escape and inconsistent vector, scene, reset or outcome bindings are refused.

Recorded `REPAIRED`/`ALREADY_QUALIFIED` requires complete confirmation for the selected vector and
the correct changed/unchanged relationship. Unknown or missing evidence cannot support saved
success. An unsuccessful run may contain no selected policy or reset; it remains readable but
cannot meet replay admission.

Records are unsigned. Reading proves internal consistency, not authorship or historical tamper
detection. A self-consistent supported replacement can be admitted and freshly evaluated. A moved
complete bundle remains usable, but there is no cross-platform bitwise replay promise: exact
independent reset matching is required in the supported execution runtime.
