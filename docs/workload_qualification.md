# Workload qualification

`metrifid qualify-workload` answers one question:

> Do three selected workloads detect every declared model perturbation at or above its required
> magnitude, under the existing Metrifid comparison tolerances, and which perturbations remain
> blind?

It answers it with completed `metrifid compare` runs. Nothing is estimated from a sensitivity
model: every cell in the receipt is a real comparison of your baseline model against a probe model
you supplied.

## What you declare

A probe group is one perturbation, one direction, and a ladder of magnitudes you built models for:

```json
{
  "probe_id": "hinge_damping_increase",
  "parameter": "shoulder.damping",
  "direction": "increase",
  "required_detection_magnitude": "0.005",
  "variants": [
    {"magnitude": "0.002", "candidate": {"model_root": "probes/damping_increase/rung_1", "entrypoint": "model.xml", "declared_step_dt": "0.001"}},
    {"magnitude": "0.005", "candidate": {"model_root": "probes/damping_increase/rung_2", "entrypoint": "model.xml", "declared_step_dt": "0.001"}}
  ]
}
```

Metrifid does not mutate your model source. You supply the probe models, and each one is admitted
against the baseline the same way a comparison admits a candidate.

`magnitude_semantics` is required, and it is your declaration of what the exact decimal means.
Metrifid preserves it byte for byte and never parses it as a unit, converts it, or checks it. The
same is true of `parameter`, `direction`, and `magnitude`:

> Metrifid compares the exact supplied admitted model closures and preserves the user's parameter,
> direction, magnitude, and magnitude-semantics labels. It does not independently establish that
> those labels faithfully describe the source edits or that no other source change exists.

Every completed receipt carries `USER_DECLARED_PROBE_SEMANTICS_NOT_VERIFIED` to say exactly that.
The Model Change Gate (`metrifid review-model`) is separate supporting evidence about what changed
between two model sources; it is not run here and is not automatic proof of a magnitude label.

Two bounded checks do run once closure identities are known, so a ladder cannot describe itself
dishonestly: every rung closure must differ from the baseline closure, and every rung closure must
be unique within its group. Neither verifies your edit; they stop one closure from standing in for
several magnitudes.

A workload candidate is one recorded initial state and one recorded action sequence:

```json
{
  "workload_id": "fast_high",
  "initial_state": "workloads/fast_high/state.npz",
  "actions": "workloads/fast_high/actions.npz",
  "control_dt": "0.01"
}
```

Declare between three and sixteen. The budget is exactly three in schema version 1.

### Paths, bases, and identifiers

Every declared path is a normalized, traversal-free, relative POSIX path. Backslashes, absolute
paths, `.`, `..`, and non-normalized spellings are refused rather than repaired. Each field resolves
against exactly one base:

```text
baseline.model_root          relative to the qualification.json directory
probe candidate.model_root   relative to the qualification.json directory
workload.initial_state       relative to the qualification.json directory
workload.actions             relative to the qualification.json directory
aliases                      relative to the qualification.json directory
output_dir                   relative to the qualification.json directory
baseline.entrypoint          relative to baseline.model_root
probe candidate.entrypoint   relative to that candidate.model_root
```

An entrypoint names a file inside its own model root, never a same-named file beside the
configuration.

`workload_id`, `probe_id`, `parameter`, and `magnitude_semantics` are semantic labels. They appear
in the receipt and the report and **never** determine a storage path. Evidence directories use
fixed-width ordinals derived only from admitted sequence position:

```text
qualification_out/evidence/controls/workload_000/
qualification_out/evidence/probes/workload_000/group_000/rung_000/
```

Each record carries the normalized relative locator of its retained files as data, so a locator can
be read back and re-admitted but can never be used as a write target.

## What Metrifid runs

### Output ownership

Ownership of the output root is decided completely before anything is created. Every model root is
resolved canonically and compared against the proposed output in both canonical and symlink-alias
form. The run refuses when the output would be equal to or inside any baseline or probe model root,
and when the output root already exists. Nothing is created until those checks pass, so a
misconfigured output leaves your models byte-for-byte and tree-for-tree unchanged.

The owned root is then created and written only through retained directory descriptors opened
`O_NOFOLLOW`, with no-clobber creation, so a symlink substituted at the parent path after admission
cannot redirect a write.

### Planned and actual comparisons

```text
planned comparisons = workloads + workloads x sum(variants per probe group)
schema-v1 maximum   = 16 + 16 x (16 x 8) = 2064
```

`planned_comparisons` is in the receipt and the report beside the actual counts. The three selected
workloads are the result, not the cost.

For every workload candidate:

```text
one zero-change control   baseline against the same baseline
one comparison per rung   baseline against that probe model
```

The zero-change control is the false-alarm check. A workload whose baseline differs from itself
cannot be evidence about anything, so it is excluded from selection and the exact control receipt
is recorded with the reason.

Completed comparison statuses map onto detection outcomes exactly:

```text
NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD -> NOT_DETECTED
MATERIAL_BEHAVIOR_CHANGE                    -> DETECTED
COVERAGE_INSUFFICIENT                       -> UNRESOLVED
NONDETERMINISTIC_REPLAY                     -> UNRESOLVED
```

An unresolved comparison stays unresolved. It never becomes a non-detection.

## Campaign identity

An individual comparison freezes and reverifies its own sources. What it cannot know is whether the
other comparisons in the campaign saw the same baseline, the same workload artifacts, the same tool
build, and the same runtime. Before any subset is selected, every completed cell is checked as one
campaign:

```text
one tool identity and one runtime environment across every cell
one baseline model closure across every cell
each zero-change control compares the baseline against itself
one candidate closure per probe rung across every workload
one state and actions identity per workload across its control and all its probe cells
one aliases identity across every cell
one comparison contract per workload, up to the expected role closure and candidate timestep
each declared timestep equals the admitted value for that role
```

A probe may legitimately declare its own timestep. There is deliberately no rule that a candidate
timestep must equal the baseline timestep; what is required is that the declared value is the one
the configuration declared for that role and that compare admits it.

## How the answer is decided

Rungs are ordered by increasing magnitude. A **detection floor** exists only when some rung and
every larger rung are detected; the floor is the smallest such rung. A single detection at a small
magnitude with a gap above it establishes nothing, because a perturbation you can see at 0.002 and
cannot see at 0.010 is not a detection threshold.

```text
QUALIFIED     a floor exists and it is at or below the required magnitude
UNRESOLVED    a rung at or above the requirement did not complete as a decision
INSUFFICIENT  those rungs completed, and no floor at or below the requirement exists
```

Every three-workload subset of the eligible candidates is enumerated. Sixteen candidates and a
budget of three is at most 560 subsets, so the search is exhaustive by construction: no sampling,
no optimizer, no tolerance to tune. Subsets are ranked by more qualified groups, then fewer
unresolved groups, then more detected variants, then the lexicographically smaller tuple of
workload identities.

## Statuses and exit codes

```text
QUALIFIED_FOR_DECLARED_PROBES   every group qualified                     exit 0
PARTIALLY_QUALIFIED             some qualified, some insufficient, none unresolved  exit 20
INSUFFICIENT_EXCITATION         none qualified, none unresolved           exit 20
UNRESOLVED                      one or more groups unresolved             exit 30
```

Invalid configuration, failed admission, fewer than three eligible workloads, or a failed
comparison returns the ordinary operational failure surface (`64` for invalid input or admission,
`70` for internal failure) and publishes no qualification receipt.

## What you get back

Two files, sealed and then linked into place without clobbering anything, verified as a pair, and
reported as a success only after both pass that final verification:

```text
<output_dir>/receipt/workload_qualification.json
<output_dir>/receipt/workload_qualification.md
```

and every compare receipt the answer was built from, retained under `<output_dir>/evidence/`,
together with the exact admitted `qualification.json` bytes at `<output_dir>/qualification.json`.

### Witnesses

The status-bearing witness explains the status it is attached to:

```text
QUALIFIED_FOR_DECLARED_PROBES                  first_witness is null
UNRESOLVED                                     first_witness is the first unresolved witness
PARTIALLY_QUALIFIED / INSUFFICIENT_EXCITATION  first_witness is the first blind witness
```

```text
first_unresolved_witness  first unresolved rung at or above that group's required magnitude
first_blind_witness       first not-detected rung at or above that requirement that explains
                          why no qualifying floor exists
first_false_alarm_witness first excluded zero-change workload, in workload_id lexical order
```

Canonical order is probe-group declaration order, then increasing exact magnitude, then
`workload_id` Unicode-code-point lexical order. A rung below the requirement stays visible in the
matrix but never becomes the status-bearing witness, because it cannot explain the decision. A false
alarm is an eligibility warning and never replaces the status-bearing witness.

### What the receipt proves, and what validates it

`load_and_validate_workload_qualification_receipt(path)` does all of this:

```text
1. bounded strict admission of the aggregate JSON from a regular no-follow file
2. strict typed parsing that refuses unknown or missing fields at every level
3. pure semantic reconstruction: every derived claim recomputed from the receipt's own
   configuration and cell records
4. the owned root derived from the published location
5. every registered locator re-admitted as normalized, relative, unique, and confined
6. the retained raw qualification configuration matched by bytes and digest
7. every generated comparison configuration matched by bytes and digest
8. every retained comparison receipt parsed as a typed ComparisonReceipt
9. every retained receipt matched to its aggregate record and to the campaign identity
```

A mismatch rejects the receipt even when `receipt_sha256` was recomputed.

### What `receipt_sha256` is not

```text
receipt_sha256 detects accidental corruption of the canonical receipt content
it is not a digital signature
it does not authenticate an author, a machine, or a campaign
it alone does not validate decision semantics
recomputing it cannot make a contradictory receipt valid
```

It is computed over the canonical self-hash primitive of the receipt object, not over the raw file
bytes. Trust comes from the reconstruction and linked-evidence passes above.

## What it does not claim

```text
DECLARED_PROBES_ONLY
USER_DECLARED_PROBE_SEMANTICS_NOT_VERIFIED
DECLARED_WORKLOAD_CANDIDATES_ONLY
MONITORED_JOINT_COORDINATES_ONLY
NO_GLOBAL_EQUIVALENCE_CLAIM
NO_TASK_SAFETY_OR_REAL_WORLD_TRANSFER_CLAIM
```

This block is the frozen limitation registry, in registry order. Every completed receipt carries all
six codes.

A qualification is about the perturbations you declared, the workloads you declared, and the joint
coordinates you monitor. It is not a global equivalence claim, and it says nothing about task
safety or real-world transfer.

## Complete example

`examples/workload_qualification/` runs end to end:

```bash
python prepare_workloads.py
metrifid qualify-workload qualification.json
```

Four workloads drive one hinge at different amplitudes and rates against a four-rung damping
ladder. Two of them are blind to the smallest perturbations, one is blind to all four, and the
receipt says which. That is the point of the command: it tells you which of your workloads would
have noticed.

## Python

`metrifid.workload_qualification` exports exactly six names, and that list is a compatibility
commitment:

```text
QualificationExitCode                               the completed process exit-code registry
QualificationResult                                 the completed result the run returns
QualificationStatus                                 the completed status registry
WorkloadQualificationOperationError                 the bounded operational refusal
load_and_validate_workload_qualification_receipt    validate a receipt and its linked evidence
qualify_configuration_file                          run one campaign from a strict JSON file
```

Nothing else is public. The strict JSON configuration file is the only supported way to describe a
campaign; configuration, probe, workload, cell, group, limitation and cardinality types are internal
implementation detail and may change without a compatibility guarantee. Importing the submodule does
no filesystem, network, or native-runtime work.

```python
from metrifid.workload_qualification import (
    QualificationStatus,
    WorkloadQualificationOperationError,
    load_and_validate_workload_qualification_receipt,
    qualify_configuration_file,
)

try:
    result = qualify_configuration_file("qualification.json")
except WorkloadQualificationOperationError as refusal:
    # A bounded operational refusal: the campaign did not reach a completed decision.
    raise SystemExit(str(refusal)) from refusal

print(result.status.value, result.exit_code)
if result.status is not QualificationStatus.QUALIFIED_FOR_DECLARED_PROBES:
    print("something your workloads cannot see was declared")

receipt = load_and_validate_workload_qualification_receipt(result.qualification_json)
print(receipt["selected_workload_ids"])
```

A complete runnable script is [`examples/sdk/workload_qualification_api.py`](../examples/sdk/workload_qualification_api.py).
