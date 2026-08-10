# Timestep fidelity audit

`metrifid audit-timestep timestep_audit.json` answers one question for one model:

> For **this** model, **this** declared workload, and **these** declared tolerances, which larger
> integration timesteps still reproduce the reference model's monitored joint behaviour?

It is a thin, strict driver over the accepted comparison engine. Every candidate is one ordinary
`compare` run between a reference variant and a candidate variant that differ only in
`option.timestep`. The audit adds no physics, no new metric, and no new status.

## What it is not

The reference timestep is the **comparison reference**. It is not asserted to be physically
correct or a ground truth. The audit does not establish physical correctness, hardware safety,
closed-loop policy quality, global model equivalence, contact/sensor/reward equivalence, or
causality, and it never reports wall-clock timing or throughput. The recommendation is a
statement about the declared workload and tolerances only.

## Configuration

The configuration key set is frozen. A missing key or an unknown key is refused; nothing is
defaulted and nothing is ignored.

```json
{
  "schema_version": 1,
  "model_root": "model",
  "entrypoint": "robot.xml",
  "initial_state": "state.npz",
  "actions": "actions.npz",
  "control_dt": "0.01",
  "repeats": 3,
  "joint_tolerances": {
    "joint_a7": {
      "joint_type": "hinge",
      "angle_rad": "0.005",
      "angular_velocity_rad_s": "0.25"
    }
  },
  "candidate_step_dts": ["0.002", "0.0025", "0.004", "0.005", "0.01"],
  "workload_kind": "SCREENING",
  "workload_label": "generated deterministic screening excitation, not a task policy",
  "output_dir": "audit_out"
}
```

| Field | Meaning |
|---|---|
| `model_root`, `entrypoint` | The complete model closure, resolved relative to the configuration file. |
| `initial_state`, `actions` | Existing canonical `metrifid.state` / `metrifid.actions` schema-version `1` NPZ artifacts. |
| `control_dt` | Exact decimal control interval. Not a float. |
| `repeats` | Passed through to each comparison unchanged. |
| `joint_tolerances` | Declared before the run, in physical units, exactly as `compare` accepts them. |
| `candidate_step_dts` | 1 to 12 exact decimal tokens, unique by **normalized rational value** (`0.002` and `0.0020` collide), each strictly larger than the compiled reference timestep. |
| `workload_kind` | `REAL_PROJECT` or `SCREENING`. `SCREENING` marks a generated excitation, not a task policy. |
| `output_dir` | Must be absent or an existing empty real directory, and must not be the model root or any directory below it. |

## Path admission

The declared model is admitted before the audit creates anything. `model_root` and `entrypoint`
are measured through the same accepted model-closure path the comparison engine uses, so the
entrypoint must be a normal relative POSIX `.xml` path inside the model root. An absolute
entrypoint, a `..` component, a backslash, a NUL, an empty or dot entrypoint, a non-XML suffix, a
symlink, a missing or non-regular entrypoint, and an invalid root are all refused **before** any
output directory, workspace, model copy, or variant write exists. The refusal keeps the model
closure's own reason, role, and evidence and is reported with `operation = "audit-timestep"`.

The output directory is separated from the model root before it is created. An `output_dir` that
resolves to the model root, or to a directory below it, is refused with `OUTPUT_PATH_INVALID`,
`evidence.issue = "output_inside_model_root"`, and exit `64`; nothing is created. The admitted
output, workspace, candidate directories, final files, and cleanup remain bound to retained
directory descriptors. Replacing the public output path cannot redirect publication or deletion.

Candidates are evaluated in ascending exact-rational order regardless of the order written.
Each candidate is identified by the deterministic token `dt_<numerator>_over_<denominator>`
built from its normalized value, so `0.0025` is always `dt_1_over_400`.

## How a candidate is evaluated

1. **The campaign freezes one input set.** Before the reference or any candidate, the audit creates
   one immutable complete source-closure snapshot and reads one bounded exact byte sequence for
   each state and actions NPZ. Every role and candidate consumes those same frozen inputs; no
   generated `.xml`, `.mjb`, or copied model directory survives the run.
2. The candidate timestep is applied **in memory** to the candidate compiled model's
   `mjModel.opt.timestep`. MuJoCo documents `option` as mapping to `mjModel.opt`: it is runtime
   simulation state that does not affect compilation, so the two roles differ only by the timestep
   under audit. The reference model's timestep is never modified.
3. `control_dt / step_dt` is computed exactly. A candidate that does not land on the control grid
   is not resampled and not approximated — the comparison engine refuses it.
4. The comparison runs unchanged and its receipt is published under
   `candidates/<token>/comparison.json` and `.md`.

Because both roles are the same frozen source and workload bytes, every completed receipt records
the same equal baseline/candidate closure hashes, state raw and semantic hashes, and actions raw
and semantic hashes, while role timesteps remain distinct. The aggregate derives those identities
from the frozen campaign rather than rereading live paths or borrowing one candidate receipt.

After every candidate has produced its decision evidence and immediately before aggregate
publication, the audit reverifies the live model closure and exact state/actions bytes. A mismatch
refuses the audit. Exact private workspace artifacts are removed when possible, while already
registered candidate, failure, or aggregate evidence is preserved and the operation never reports
success. The user's model tree is never written to.

## Classifications

| Comparison status | Audit classification |
|---|---|
| `NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD` | `WITHIN_DECLARED_TOLERANCE` |
| `MATERIAL_BEHAVIOR_CHANGE` | `OUTSIDE_DECLARED_TOLERANCE` |
| `COVERAGE_INSUFFICIENT` | `INCONCLUSIVE` |
| `NONDETERMINISTIC_REPLAY` | `INCONCLUSIVE` |
| — (candidate control grid is nonintegral) | `REFUSED` |
| — (any other candidate operational failure) | `INCONCLUSIVE` |

A failed candidate is a completed row, not an abort. Its preserved operational failure is written
verbatim to `candidates/<token>/operational_failure.json`, keeps operation `compare`, and every
later candidate still runs.

Exactly one operational failure is classified `REFUSED`: a **candidate-role
`CONTROL_GRID_NONINTEGRAL`** result whose precomputed steps per control interval is null. That
refusal is a statement about the declared schedule — the candidate timestep does not divide
`control_dt`, so the candidate was never executed and no evidence is missing.

**Every other candidate operational failure is `INCONCLUSIVE`**, including an internal, identity,
model-compile, workload, environment, or output failure, and including a `CONTROL_GRID_NONINTEGRAL`
code carrying another role or a non-null step count. Those candidates produced no trustworthy
comparison, so the audit does not treat them as harmless. An inconclusive row keeps its precomputed
integral step count and step-count factor: those are exact properties of the declared schedule and
are not erased because execution failed later.

## Recommendation policy

`largest_within_tolerance_completed_prefix`.

Candidates are scanned in ascending order. Only an exact nonintegral candidate-grid `REFUSED` row
is skipped, and it never breaks the scan. The **first** `OUTSIDE_DECLARED_TOLERANCE` or
`INCONCLUSIVE` result stops the scan permanently, and `blocked_by_prior_non_within` is set.

The scan applies that skip predicate itself rather than trusting the classification alone: a
`REFUSED` row is skipped only when its operational reason is `CONTROL_GRID_NONINTEGRAL` **and** its
steps per control interval is null. Any other refused row blocks the scan. The audit therefore
never recommends a larger timestep across a candidate whose evidence is missing, even if a future
change misclassified that candidate.

The audit never assumes monotonic behaviour and never performs a binary search: every declared
candidate is attempted and reported. A within-tolerance candidate that appears *after* a
non-within one is reported in the table but is deliberately **not** recommended, because the
evidence between it and the reference is broken.

## Output

```text
audit_out/
├── timestep_audit.json          # canonical, self-hashed under "audit_sha256"
├── timestep_audit.md            # maintainer-readable rendering of the same values
└── candidates/
    ├── dt_1_over_500/comparison.json, comparison.md
    └── dt_1_over_250/operational_failure.json
```

Before a success result is returned, Audit rechecks the frozen live model/state/actions inputs, every retained candidate result, and the aggregate pair. The public output path and exact registered tree are checked last, after the live model context closes, so a path replacement during any earlier verification causes refusal rather than a success result containing attacker-controlled paths.

The aggregate records the tool identity, the configuration hash, the source model tree hash,
the workload identities, the reference timestep and its steps per control interval, one row per
candidate, and the recommendation. Each row carries
`reference_to_candidate_step_count_factor = reference_steps / candidate_steps` as an exact
rational, which is `null` for a refused candidate. This factor is a step-count ratio only; it is
not a speed or throughput claim.

`timestep_audit.md` is a rendering of those same values and adds nothing to them. Its candidate
table carries exactly these columns, in this order:

```text
Candidate timestep | Steps/interval | Step-count factor | Classification |
Operational reason | Maximum tolerance ratio | Worst witness | First crossing
```

A refused candidate shows its `operational_reason`; a completed one shows `—`. A first crossing,
when the receipt recorded one, shows the joint, the metric, the boundary index, the crossing time
as a readable decimal with its exact rational, the error, and the tolerance; otherwise it shows
`—`. Every user- or model-controlled string in the report — workload label, candidate token,
joint, metric, reason — passes through one escaping helper that rewrites `\`, `|`, CR, and LF, so
a hostile name cannot forge table structure or split a row. No raw Python `dict` or `list`
representation is ever emitted, and rendering the same aggregate twice is byte-identical UTF-8
with LF endings.

Repeated runs of the same configuration in the same location produce byte-identical
`timestep_audit.json` and `timestep_audit.md`. The audit's internal workspace is anchored to the
admitted output directory rather than to a random temporary path for exactly this reason, and is
removed before the command returns. Relocating the configuration or reinstalling the
distribution changes the recorded identities, and therefore the aggregate hash.

## Exit codes and refusals

An audit-level failure is emitted as one canonical `OperationalFailure` on stderr with
`operation = "audit-timestep"`. It removes exact private workspace artifacts when possible but may
leave already-published candidate, failure, or aggregate evidence; those files are diagnostic
evidence, not a successful audit. A per-candidate comparison failure keeps `operation = "compare"`
so its bytes stay comparable with an ordinary `compare` run.

```text
0   audit published
64  invalid invocation, configuration, model path, or output path
    (e.g. OUTPUT_DIRECTORY_NOT_EMPTY, OUTPUT_PATH_INVALID, MODEL_CLOSURE_PATH_ESCAPE)
70  internal invariant failed
```

## Worked example

The following illustrative KUKA iiwa14 screening result uses reference timestep `0.001`,
`control_dt` `0.01`, 3 repeats, 0.005 rad and 0.25 rad/s tolerances, and a generated
deterministic excitation. The underlying run artifacts are not shipped in this repository:

| Candidate | Steps/interval | Classification |
|---|---:|---|
| 0.002 | 5 | `WITHIN_DECLARED_TOLERANCE` |
| 0.0025 | 4 | `WITHIN_DECLARED_TOLERANCE` |
| 0.004 | — | `REFUSED` (`CONTROL_GRID_NONINTEGRAL`) |
| 0.005 | 2 | `WITHIN_DECLARED_TOLERANCE` |
| 0.01 | 1 | `OUTSIDE_DECLARED_TOLERANCE` |

Recommendation: `dt_1_over_200` (`0.005` s). `0.004` does not divide `0.01` exactly, so it was
refused rather than approximated, and the candidates after it still ran. This result is about
this model, this generated screening excitation, and these tolerances. It is not a statement
about iiwa14 in general, about other workloads, or about MuJoCo.
