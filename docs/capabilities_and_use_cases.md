# Capabilities and use cases

## Purpose of this document

This document describes what **Metrifid** currently does, why it is useful, which engineering decisions it can support, and where its claims stop.

It describes the current `metrifid` implementation and its public documentation. It deliberately separates:

- capabilities that are implemented now;
- evidence and safeguards that are implemented now;
- practical uses that follow directly from those capabilities;
- limitations and unsupported uses.

This is not marketing copy and does not claim more than the current implementation supports.

---

## The accepted command surface

Metrifid is a local MuJoCo assurance library and command-line tool. The accepted command surface is
exactly these seven commands:

| Command | The question it answers |
| --- | --- |
| `certify` | Did two MuJoCo source closures compile into the exact same complete compiled model? |
| `review-model` | Which compiled fields changed, and does a declared policy allow, require, or forbid each one? |
| `compare` | Did a candidate behave differently from a baseline on one exact declared workload? |
| `qualify-workload` | Would these declared workloads actually detect a supplied probe perturbation? |
| `audit-timestep` | Which declared larger timestep stays within the tolerances of one exact workload? |
| `review-runtime` | May one exact native MuJoCo profile replace another, given twelve retained evidence cells? |
| `run-runtime-review` | Create those twelve native evidence cells through two prepared profiles and decide immediately. |

## What each command decides

These are separate decisions, and none of them substitutes for another:

```text
certify             exact compiled-artifact identity
review-model        static declared-policy coverage of compiled changes
compare             workload-bounded behavior difference
qualify-workload    workload detection power against supplied probes
audit-timestep      workload-bounded timestep qualification
review-runtime      native-profile replacement over retained evidence
run-runtime-review  native evidence creation plus the same replacement decision
```

Metrifid does not treat source-code similarity, successful compilation, visual inspection, or one
arbitrary simulation run as sufficient proof. It binds each result to exact source bytes, runtime
identity, declared workload inputs, deterministic outputs, and machine-readable receipts.

The strongest current product idea is:

> **Start from the thing that changed.** If a model or asset changed, use `certify`, then
> `review-model` to see exactly which compiled fields moved, then `compare` when you need to know
> whether the difference matters on the workload you care about. If the runtime changed, use
> `run-runtime-review`. Use `audit-timestep` when the question is how far the timestep can be
> increased within your declared fidelity limits.

A completed exit 20, 30, or 40 is a referee decision, not a process failure.

## What Metrifid does not decide

```text
physical correctness or safety of any model
behavioral equivalence from a Certify result
equivalence across simulation backends
whether a declared policy rule is a good idea
whether your tolerances are the right tolerances
which MuJoCo version you should be using
```

Metrifid does not discover, install, or repair MuJoCo environments. It measures the profiles you
prepare and refuses rather than guessing when an input is ambiguous.

Certify makes no behavioral-equivalence claim: it decides whether the compiled artifacts differ, not
whether the robot moves differently.

---

# 1. `metrifid certify`: exact compiled-artifact certification

## The question it answers

```text
Do these two measured MuJoCo source closures, compiled under this recorded runtime,
produce byte-identical complete MJB artifacts?
```

Command:

```bash
metrifid certify BASELINE_MJCF CANDIDATE_MJCF \
  --output OUTPUT_DIRECTORY \
  [--baseline-root BASELINE_ROOT] \
  [--candidate-root CANDIDATE_ROOT]
```

Example:

```bash
metrifid certify old/robot.xml new/robot.xml --output certification-result/
```

## What it actually does

For each model role, Metrifid:

1. verifies the installed Metrifid distribution identity;
2. admits and measures the complete model-root source closure;
3. records exact bytes for every admitted regular file under that root;
4. creates a private immutable same-byte snapshot;
5. separately discovers which files MuJoCo actually reaches while compiling the entrypoint;
6. compiles the snapshot, not the mutable original tree;
7. refuses unbound plugin, callback, or user-implementation surfaces;
8. serializes the complete compiled `mjModel` using MuJoCo's MJB writer;
9. hashes every serialized byte;
10. compares the two complete MJB artifacts;
11. reverifies both live source closures before publishing the result;
12. publishes deterministic JSON and Markdown evidence with no-clobber output semantics.

The compiled identity is:

```text
SHA-256 over every byte emitted by mj_saveModel
```

It is **not**:

- a selected list of important fields;
- an approximate numerical comparison;
- a normalized representation;
- a tolerance-based result;
- a workload result;
- a source-text diff.

## Outcomes

```text
exit 0
CERTIFIED_COMPILED_EQUIVALENCE
Every serialized byte matched.

exit 40
NOT_CERTIFIED_COMPILED_DIFFERS
At least one serialized byte differed.

exit 64
Invalid invocation, input, model closure, environment, or output.

exit 70
Internal invariant or execution failure.
```

## What a successful certificate means

The exact machine claim is:

> The measured baseline and candidate source closures, compiled under the recorded runtime identity, produced byte-identical complete MJB artifacts.

This is workload-free. No initial state, control sequence, monitored joint, tolerance, or simulation stepping is required.

A successful certificate is especially useful because it establishes that **nothing in the complete serialized compiled model differs**, including fields that manual inspection might overlook.

## What it does not mean

A successful certificate does not prove:

- source-text equality;
- license equality;
- visual intent;
- physical correctness;
- hardware safety;
- controller correctness;
- task suitability;
- equivalence under another MuJoCo version or operating system;
- equivalence after a caller mutates `mjModel`;
- equivalence of external callbacks, plugins, controls, applied forces, or other runtime inputs.

The certificate is bound to the runtime identity recorded in the receipt.

## Behavior implication

The receipt may state a separately labelled implication:

> If the certified model bytes are loaded under the recorded runtime, neither model is modified, the complete initial `mjData` state is identical, all controls and external inputs are identical, external implementation code is identical, and execution is deterministic, then the compiled model cannot be the source of divergence.

This implication depends on all listed premises. It is not part of the certificate status itself.

## When compiled artifacts differ

The byte result is decided first. Metrifid then adds descriptive evidence:

- first differing byte offset;
- number of differing bytes;
- up to 100 changed public fields;
- field types, dtypes, and shapes;
- field SHA-256 values;
- changed-element count when computable;
- up to eight deterministic index/value witnesses per field;
- up to 200 omitted fields with explicit reasons;
- a truthful `truncated` indicator when report bounds are reached.

This field report helps locate the difference. It does not decide whether the difference is dynamically material.

Possible result:

```text
NOT_CERTIFIED_COMPILED_DIFFERS
field witness: body_mass[7]
```

But the product does not automatically convert that into:

```text
MATERIAL_BEHAVIOR_CHANGE
```

For that decision, use `metrifid compare` with a declared workload.

## Conservative behavior and known over-sensitivity

Complete MJB identity intentionally includes more than dynamics-only state. Changes such as these may break exact compiled identity even when they do not affect the user's physical behavior of interest:

- visual RGBA changes;
- model-name changes;
- joint-name changes;
- asset filename changes with identical geometry;
- other serialized metadata changes.

In those cases, `certify` correctly says only that the complete compiled artifacts differ. It does not claim the difference is physically important.

## Models it refuses

Certify refuses models whose behavior depends on external implementation code that is not bound by the certificate, including supported categories of:

- engine plugins;
- plugin sensors;
- user actuator implementations;
- user sensor implementations;
- active global MuJoCo callbacks.

It does not apply replay-only restrictions that are irrelevant to compiled artifact identity. For example, the existence of mocap bodies or history state does not by itself prevent certification.

## Useful applications

`certify` is useful for:

- XML formatting and canonicalization changes;
- comments, attribute reordering, and equivalent numeric text;
- moving equivalent values into `<default>` classes;
- include-tree refactors;
- model-directory reorganization;
- exporter and generator changes;
- vendor model revisions;
- model-release checks;
- pre-merge or CI checks for intended behavior-neutral maintenance;
- confirming that two differently written source trees compile to the same model;
- deciding whether a more expensive workload-based comparison is necessary.

## Static model-release review

### The question it answers

> What changed between two compiled robot models, was each change allowed, required, forbidden or
> undeclared by my declared policy, and what is the first unexpected static field?

### The allow/require/forbid policy role

A maintainer writes one bounded policy bound to the exact baseline compiled subject. Each rule
carries an effect and an exact selector:

```text
ALLOW     this exact compiled change is expected and accepted
REQUIRE   this exact compiled change must be present, or the result is outside policy
FORBID    this exact compiled change must not be present
```

Anything observed and not declared is `UNDECLARED`. The gate never infers intent: a change the
policy does not name cannot become an accepted change.

### First unexpected static field

A completed result names the first unexpected change in one deterministic order, so a reviewer has
a single place to start rather than a diff to read. It also names the first missing required change
when a `REQUIRE` rule is unmet. Both witnesses are stable across processes and hash seeds.

### Model-release pull request use case

A robot model repository can gate merges on it. The maintainer records the compiled subjects, writes
a policy describing the intended change, and runs `review-model` in review. A reformatting pull
request that changes no compiled byte returns `NO_COMPILED_CHANGE` at exit 0. A declared mass change
returns `WITHIN_DECLARED_POLICY`. Anything else stops at exit 40 with a named first witness, which
is the case a human should actually look at.

### Static-only limitation

This is a static compiled-structure decision. It steps no model and makes no dynamic-behavior,
safety, deployment-readiness or release-authorization claim. A statically permitted compiled change
may still change behavior; that question belongs to `compare` on a declared workload.

## Resource limits

Be precise about what is and is not guaranteed:

- Model-closure content is bounded by the documented byte limit.
- File-count, directory-count, and traversal-depth limits are **not** separately guaranteed in this
  alpha local release.
- This package is for local trusted-input workflows, not a hosted untrusted-upload service.

Those bounds are the whole guarantee: this local release adds no separate file-count,
directory-count or traversal-depth budget, and it accepts no untrusted hosted upload.

Current fixed limits include:

```text
Maximum complete source closure:        256 MiB
Maximum serialized MJB artifact:        512 MiB
Maximum copied field-witness array:      64 MiB per field
```

The 512 MiB artifact limit applies after compilation. It is not a promise that compilation itself uses no more than 512 MiB.

---

# 2. `metrifid compare`: deterministic workload-bounded behavior comparison

## The question it answers

```text
On this exact initial state, action sequence, time grid, monitored coordinates,
repeat policy, and tolerance policy, did the candidate move materially away from the baseline?
```

Command:

```bash
metrifid compare comparison.json
```

The configuration is strict JSON. There is no YAML form and no automatic inference of missing workload semantics.

## What the workload contains

A comparison requires:

- a baseline model root and MJCF entrypoint;
- a candidate model root and MJCF entrypoint;
- one canonical initial-state NPZ artifact;
- one canonical actions NPZ artifact;
- exact `control_dt`;
- repeat count;
- monitored hinge or slide joints;
- physical-unit tolerances;
- explicit alignment or aliases where accepted and required;
- output location.

Nothing is inferred from a controller, task, policy, video, or arbitrary log.

## Canonical workload artifacts

The public Python writers are:

```python
from metrifid import write_state_artifact, write_actions_artifact
```

The artifacts are:

```text
metrifid.state   schema version 1
metrifid.actions schema version 1
```

They carry both:

- a hash of the exact NPZ file bytes;
- a semantic-content hash.

This allows the receipt to distinguish exact file identity from equivalent declared workload content.

The writers require an absent destination and do not overwrite an existing file, directory, or symbolic link.

## Execution model

For each role and each repeat, Metrifid:

1. creates a fresh `MjData`;
2. restores the canonical initial qpos, qvel, and activation values;
3. clears declared auxiliary inputs;
4. calls `mj_forward`;
5. captures boundary 0;
6. applies each action row at the left edge of its interval;
7. executes exactly the admitted number of MuJoCo steps;
8. captures boundary `k + 1`;
9. repeats the complete role execution to test repeatability;
10. compares baseline and candidate only at equal canonical boundaries.

For `N` action rows, a complete trace contains:

```text
N + 1 boundaries
```

Metrifid performs no:

- interpolation;
- resampling;
- time warping;
- silent truncation;
- fabricated tail filling.

If a trace becomes nonfinite or incomplete, the missing tail remains absent and the receipt records why.

## Exact time grid

Time is declared exactly rather than through approximate binary floating point:

```text
control_dt
baseline step_dt
candidate step_dt
number of controls
```

`control_dt` must be an exact integer multiple of each role's MuJoCo timestep. Metrifid refuses a nonintegral grid rather than rounding the number of internal steps.

Actions use zero-order hold: the control row is applied at the left boundary and held for the complete control interval.

## Supported metrics

The current comparator supports monitored hinge and slide joints only.

```text
hinge position
  abs(atan2(sin(candidate - baseline), cos(candidate - baseline)))

hinge velocity
  abs(candidate - baseline)

slide position
  abs(candidate - baseline)

slide velocity
  abs(candidate - baseline)
```

Hinge position correctly handles the `+pi/-pi` seam through wrapped angular distance.

Tolerance comparison uses the exact rational tolerance. A crossing is strict:

```text
error == tolerance
is not material

error > tolerance
is a crossing
```

## Outcomes

```text
exit 0
NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD

exit 10
MATERIAL_BEHAVIOR_CHANGE

exit 20
COVERAGE_INSUFFICIENT

exit 30
NONDETERMINISTIC_REPLAY

exit 64
Invalid invocation, input, environment, workload, or output

exit 70
Internal failure
```

## Evidence produced

The comparison receipt records:

- exact source-closure identities;
- compiled model dimensions and alignment;
- exact workload identities;
- exact time-grid rationals;
- repeat signatures;
- observed warnings and errors;
- maximum error for every monitored metric;
- exact tolerance;
- error-to-tolerance ratio when defined;
- worst boundary and exact time;
- first tolerance crossing;
- incomplete-trace evidence;
- decision reasons;
- claim limitations;
- installed tool and runtime identity.

The repeatability signature binds:

- boundary indices;
- exact times;
- role-local clock bits;
- qpos bytes;
- qvel bytes;
- activation bytes;
- warnings;
- errors;
- invalid-boundary evidence;
- initial-state preservation.

## Supported and unsupported coordinates

Supported:

- hinge position;
- hinge velocity;
- slide position;
- slide velocity.

Currently refused or not implemented:

- ball-joint metrics;
- free-joint metrics;
- contact-force metrics;
- body-pose metrics;
- site-pose metrics;
- arbitrary sensor metrics;
- reward comparison;
- task-success comparison;
- energy metrics;
- closed-loop policy execution;
- controller-in-the-loop comparison.

## Claim boundary

A green comparison means only:

> No monitored hinge or slide coordinate exceeded its declared tolerance on this exact workload under the recorded runtime and repeat policy.

It does not prove:

- global model equivalence;
- physical correctness;
- hardware safety;
- unmonitored-coordinate equivalence;
- behavior on another initial state or control sequence;
- behavior after the declared horizon;
- causal diagnosis;
- controller or policy correctness.

## Useful applications

`compare` is useful for:

- reviewing a model update after `certify` reports differing artifacts;
- regression checks for inertial, damping, friction, actuator, solver, or asset changes;
- model-vendor update qualification;
- model-conversion checks using one declared replay;
- deterministic pull-request gates;
- MuJoCo asset or parameter release checks;
- controls and MPC model maintenance;
- reproducing and localizing a behavior difference;
- identifying the first boundary and coordinate that exceeds tolerance;
- distinguishing material change, insufficient coverage, and nondeterminism.

## Safety budgets

Before replay, Metrifid enforces:

```text
Maximum total internal MuJoCo steps:      10,000,000
Maximum retained float64 trace storage:   268,435,456 bytes
```

When a request exceeds a limit, Metrifid produces `COVERAGE_INSUFFICIENT` rather than beginning an unbounded replay.

---

# 3. `metrifid audit-timestep`: workload-specific timestep qualification

## The question it answers

```text
For this model, workload, and tolerance policy, which declared larger integration timesteps
remain within the behavior envelope of the reference timestep?
```

Command:

```bash
metrifid audit-timestep timestep_audit.json
```

## What it does

The audit uses one model and one frozen workload. It compares the reference model against candidate variants that differ only in:

```text
mjModel.opt.timestep
```

Every candidate is evaluated through the same accepted comparison engine.

The audit adds no new physics and no new behavior metric.

## Required inputs

A timestep-audit configuration includes:

- model root;
- relative MJCF entrypoint;
- canonical initial-state artifact;
- canonical actions artifact;
- exact control interval;
- repeats;
- joint tolerances;
- one to twelve candidate timesteps;
- workload kind and label;
- output directory.

Candidate timestep values are exact decimal tokens and are unique by normalized rational value. For example:

```text
0.002
0.0020
```

represent the same candidate and cannot both appear.

Each candidate must be strictly larger than the compiled reference timestep.

## One frozen campaign

The audit freezes one complete input set before evaluating the reference or any candidate:

```text
one immutable complete model-closure snapshot
one exact state NPZ byte sequence
one exact actions NPZ byte sequence
```

Every candidate uses those same frozen inputs. The aggregate is not allowed to combine results from different live input bytes.

Immediately before aggregate publication, Metrifid reverifies the live model, state, and action inputs.

## Candidate classifications

```text
WITHIN_DECLARED_TOLERANCE
OUTSIDE_DECLARED_TOLERANCE
INCONCLUSIVE
REFUSED
```

Mapping from comparison results:

```text
NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD
  -> WITHIN_DECLARED_TOLERANCE

MATERIAL_BEHAVIOR_CHANGE
  -> OUTSIDE_DECLARED_TOLERANCE

COVERAGE_INSUFFICIENT
  -> INCONCLUSIVE

NONDETERMINISTIC_REPLAY
  -> INCONCLUSIVE

candidate control grid is nonintegral
  -> REFUSED

other candidate operational failure
  -> INCONCLUSIVE
```

A failed candidate is retained as one completed row. Later candidates still run.

## Exact nonintegral-grid refusal

Suppose:

```text
control_dt = 0.01
candidate timestep = 0.004
```

The candidate does not divide the control interval exactly. Metrifid refuses it rather than using 2.5 steps, rounding to 2 or 3 steps, resampling actions, or changing the control interval.

## Recommendation policy

The policy is:

```text
largest_within_tolerance_completed_prefix
```

Candidates are sorted in ascending exact-rational order.

- A pure nonintegral-grid refusal is skipped for recommendation purposes.
- The first `OUTSIDE_DECLARED_TOLERANCE` candidate stops the recommendation prefix.
- The first `INCONCLUSIVE` candidate also stops the prefix.
- A later passing candidate is still reported but is not recommended across a broken evidence prefix.
- Every candidate is attempted; Metrifid does not assume monotonic behavior and does not binary-search.

Example:

```text
0.002   WITHIN
0.0025  WITHIN
0.004   REFUSED: CONTROL_GRID_NONINTEGRAL
0.005   WITHIN
0.01    OUTSIDE

recommended: 0.005
```

## What it reports

The aggregate includes:

- source model identity;
- frozen workload identities;
- reference timestep;
- reference steps per control interval;
- every candidate timestep;
- candidate steps per interval;
- exact reference-to-candidate step-count factor;
- classification;
- operational reason where applicable;
- maximum tolerance ratio;
- worst witness;
- first crossing;
- recommendation;
- reason the recommendation was blocked when applicable.

Output tree:

```text
audit_out/
  timestep_audit.json
  timestep_audit.md
  candidates/
    dt_<rational>/comparison.json
    dt_<rational>/comparison.md
    or operational_failure.json
```

## What the step-count factor means

The factor is:

```text
reference steps per control interval
------------------------------------
candidate steps per control interval
```

It is an exact simulation-step ratio.

It is **not** a measured claim about:

- wall-clock speed;
- CPU time;
- GPU time;
- reinforcement-learning throughput;
- cloud cost;
- end-to-end training speed.

Those must be measured separately in the user's environment.

## Claim boundary

A recommendation means only:

> This candidate is the largest declared timestep supported by the completed-prefix policy for this exact model, frozen workload, monitored coordinates, tolerances, and recorded runtime.

It does not prove:

- the reference timestep is physically correct;
- the recommended timestep is safe for all states or tasks;
- contact, sensor, reward, energy, or policy equivalence;
- closed-loop stability;
- hardware safety;
- a specific wall-clock speedup.

## Useful applications

`audit-timestep` is useful for:

- reducing simulation step count on a representative workload;
- qualifying a larger timestep before changing a project default;
- creating evidence for timestep changes in a release or pull request;
- identifying the first candidate that leaves a declared fidelity envelope;
- distinguishing an out-of-tolerance candidate from an inconclusive one;
- preserving evidence for every candidate rather than returning only one number;
- giving RL or controls teams a reproducible starting point for performance experiments.

---

# 4. Shared assurance capabilities

## Complete source-closure identity

Metrifid treats a model as more than one XML entrypoint.

The source closure includes every admitted regular file beneath the declared model root. MuJoCo dependency discovery is recorded separately.

Consequences:

- changing an unused regular file changes the source-closure identity;
- it does not change the compiled MJB unless MuJoCo actually resolves that file;
- source identity and compiled identity remain separate facts;
- the original source tree is never modified;
- compilation uses a private immutable snapshot;
- the live source is checked again before successful publication.

This is useful for models containing:

- nested XML includes;
- meshes;
- textures;
- height fields;
- compiler directories;
- defaults and shared assets;
- generated or vendor-supplied model files.

## Descriptor-confined filesystem handling

Metrifid uses retained directory and file descriptors to prevent admitted paths from being redirected during execution.

Protections include:

- no-follow model traversal;
- path-confinement checks;
- refusal of path escapes;
- refusal of symbolic-link model members;
- no-clobber publication;
- retained ownership of product-created artifacts;
- exact-byte verification before success;
- preservation of unrelated or replaced caller files;
- refusal of output inside a model root;
- bounded, nonblocking NPZ input admission;
- public output-path verification at the last success boundary.

This work is important because a correct numerical result is not useful if the source or output paths can silently change underneath it.

## Deterministic, canonical evidence

Metrifid publishes canonical JSON.

Rules include:

- object keys sorted by Unicode code-point order;
- strict UTF-8;
- no arbitrary whitespace;
- no raw JSON floats;
- exact normalized rational objects;
- tagged IEEE-754 binary64 values preserving exact bits;
- no timestamps in canonical receipts;
- self-hashes computed over canonical content.

Examples:

```json
{"numerator":1,"denominator":500}
```

```json
{"kind":"ieee754_binary64","bits":"3ff0000000000000"}
```

The binary64 representation preserves:

- finite values;
- positive and negative zero;
- positive and negative infinity;
- NaN payload bits.

## Self-hashing versus signing

Receipts are self-consistent and self-hashed. The public validators check their internal structure, identities, decision facts, and hashes.

They are **not cryptographically signed**.

A self-hash detects accidental or inconsistent modification when validated, but it does not prove who created the receipt and does not stop someone creating a different self-consistent receipt.

Signed attestations, Sigstore, in-toto, and an enterprise evidence registry are not part of the current product.

## Installed distribution identity

Decision-bearing commands require a normal installed Metrifid distribution.

The current product refuses:

- editable installations for decision-bearing execution;
- source-tree execution on `sys.path`;
- mixed package roots;
- more than one visible Metrifid distribution;
- loaded package members that do not match the installed manifest.

Each receipt records the Metrifid distribution version and digest of the code that produced it.

This supports reproducibility, but it means contributors need a two-lane workflow:

```text
editable development for ordinary coding tools and suitable unit work
noneditable built-wheel environment for decision-bearing integration validation
```

## Refusal instead of guessing

Metrifid does not silently repair incompatible inputs.

It refuses or returns an inconclusive result for conditions such as:

- unsafe or escaping paths;
- symbolic-link model members;
- invalid source mutation;
- unsupported plugins or callbacks;
- inconsistent joint or actuator alignment;
- unsupported monitored joint type;
- nonintegral control grid;
- nonfinite execution;
- nondeterministic replay;
- insufficient trace or step budget;
- malformed canonical JSON;
- duplicate JSON keys;
- unsafe output location;
- nonempty output directory;
- invalid or oversized workload artifact.

Operational refusal exits are distinct from decision outcomes.

---

# 5. Public Python API

The current top-level import is:

```python
import metrifid
```

Public categories include:

## Workload writers

```python
write_state_artifact
write_actions_artifact
```

Use them to create canonical state and action NPZ inputs for `compare` and `audit-timestep`.

## Canonical data and hashing

```python
canonical_json_bytes
canonical_sha256
strict_json_loads
Binary64
ExactRational
```

## Comparison receipts and decisions

```python
validate_receipt
finalize_receipt
ComparisonConfig
ComparisonContractIdentity
ComparisonReceipt
ComparisonStatus
ReasonCode
ReasonRecord
REASON_REGISTRY
STATUS_PRECEDENCE
LimitationCode
```

## Operational failures and identities

```python
OperationalExitCode
OperationalFailure
OperationalReasonCode
OperationalStage
OperationalToolObservation
InputDigestCode
InputDigest
EngineThreadpoolState
```

Certification receipt validation is also available from:

```python
from metrifid.certify import validate_receipt
```

The validators establish internal semantic consistency. They do not provide a digital signature.

---

# 6. Practical workflows

## Workflow A: behavior-neutral model refactor

Use when reformatting XML, reorganizing includes, or moving equivalent values into defaults.

```text
1. Run certify on old and new entrypoints.
2. If exit 0, retain the certificate with the change.
3. If exit 40, inspect the field report.
4. Run compare when workload-specific behavior matters.
```

Useful result:

```text
Source trees differ.
Complete compiled artifacts are identical.
No replay was required.
```

## Workflow B: physical model parameter change

Use when changing mass, inertia, damping, friction, armature, actuator gain, control range, solver options, or geometry.

```text
1. Run certify.
2. Expect exit 40 when compiled bytes change.
3. Review the changed-field evidence.
4. Run compare on the exact operational workload.
5. Review first crossing and worst witness.
```

Useful result:

```text
The artifacts differ.
On the declared workload, the first material crossing occurred at boundary 137,
on joint X, metric Y, at exact time T.
```

## Workflow C: vendor or upstream model update

Use when a robot vendor, model publisher, or shared model repository publishes a new revision.

```text
1. Freeze old and new source roots.
2. Certify exact compiled identity.
3. If different, run one or more representative workload comparisons.
4. Retain receipts beside the accepted revision.
```

This gives downstream users evidence stronger than "the XML changed" or "the viewer still looks right."

## Workflow D: pull-request gate

Use in CI for model-maintenance repositories.

```text
certify exit 0
  intended behavior-neutral change compiled identically

certify exit 40
  compiled model changed; explicit review required

compare exit 0
  no monitored coordinate exceeded tolerance on the declared workload

compare exit 10
  material workload-bounded difference found
```

Exit codes are stable enough for shell or CI gating.

## Workflow E: timestep qualification

Use when considering a larger MuJoCo integration timestep.

```text
1. Select a representative workload.
2. Declare physical-unit tolerances before observing results.
3. List candidate timesteps.
4. Run audit-timestep.
5. Review every candidate, not only the recommendation.
6. Measure actual wall-clock performance separately.
```

Useful result:

```text
0.005 s was the largest candidate supported by the completed-prefix policy.
0.01 s crossed tolerance.
0.004 s was refused because it did not divide the control interval exactly.
```

## Workflow F: reproducible research or benchmark artifact

Use when publishing a model or benchmark with an exact model revision and workload.

Retain:

- source-closure digest;
- runtime identity;
- workload semantic digests;
- comparison or audit receipt;
- human-readable Markdown rendering;
- exact distribution identity.

This helps another engineer establish what was actually tested.

---

# 7. Who Metrifid is useful for

## MuJoCo model maintainers and publishers

They can use `certify` on every revision to prove whether a source maintenance change altered the complete compiled model.

## Robotics simulation-infrastructure engineers

They can use `certify` and `compare` as deterministic pull-request and release gates around shared model repositories.

## Robot OEM and digital-twin teams

They can qualify official model revisions, vendor changes, asset updates, and parameter changes before downstream users consume them.

## Controls and MPC engineers

They can compare a model revision on the exact state and control sequence relevant to their controller-development workflow.

## Reinforcement-learning and humanoid teams

They can use `audit-timestep` to qualify candidate timesteps on a representative excitation before measuring training-throughput gains.

## Framework and benchmark maintainers

They can attach reproducible receipts to model imports, benchmark releases, and MuJoCo asset updates.

## Academic and reproducibility-focused labs

They can preserve exact source, workload, runtime, and result identities beside paper artifacts or shared lab models.

---

# 8. What Metrifid is not currently designed to do

Metrifid is not currently:

- a general simulation cloud;
- a scenario scheduler;
- a dashboard;
- a controller-training system;
- a reinforcement-learning framework;
- a model editor or automatic repair tool;
- a visual-diff tool;
- a hardware safety-certification system;
- a physics truth oracle;
- a task-success evaluator;
- a closed-loop controller validator;
- a cross-simulator parity tool;
- a CPU-versus-GPU backend comparator;
- a MuJoCo-version migration command;
- a signed evidence registry;
- an enterprise approval system;
- a substitute for engineering judgment.

It makes narrow, reproducible decisions and records the boundaries of those decisions.

---

# 9. Current limitations

## Runtime and compatibility

The current source declares:

```text
Python >=3.11, with no metadata upper bound
MuJoCo Python package >=3.9, stable, with no minor-version ceiling
MuJoCo package/native base version and algorithmic native integer must agree
MuJoCo exact validated profiles 3.9.0, 3.10.0, 3.11.0, and 3.12.0
NumPy >=1.26, with no runtime upper bound
```

Every native command uses the same capability-based runtime gate. The gate requires Linux or macOS plus the POSIX descriptor-relative filesystem capabilities used by the security model; it does not reject a machine architecture through an allowlist. Native Windows is unsupported, and WSL is the documented route.

Stable future releases are admitted only when their measured operation capabilities pass and are
reported as capability-compatible, not validated. Package/native mismatch, missing call-graph
capabilities, uncharacterized `review-model` field surfaces, and unrepresentable actuator signatures
fail closed with specific operational evidence. MJB digests are runtime-bound and are not expected
to match across platforms or Python environments.

## Operating systems

The present security design relies on POSIX descriptor-relative filesystem behavior. Native Windows is not currently a proven supported environment.

## Installed-wheel requirement

Decision-bearing execution expects a normal installed distribution. Editable/source-tree execution is deliberately refused for trusted receipts.

## Metrics

Behavior comparison is limited to monitored hinge and slide qpos/qvel metrics.

## Open-loop execution

Comparison uses a predeclared open-loop action sequence. It does not execute a feedback controller or policy.

## No automatic aliases

Name or alignment incompatibilities are refused rather than silently guessed.

## Exact certificate can be conservative

A visual or naming-only serialized difference can cause `NOT_CERTIFIED_COMPILED_DIFFERS` even when a particular workload would show no physical difference.

## Fixed resource budgets

The current product uses fixed limits rather than user-configurable resource policies.

## Unsigned receipts

Self-hashed evidence is not signed provenance.

---

# 10. Best current value proposition

The most faithful current description is:

> **Metrifid is a local trust layer for MuJoCo model changes. It can certify exact compiled-model identity without a workload, compare declared behavior when artifacts differ, and qualify candidate timesteps against an explicit workload and tolerance policy.**

A shorter public description is:

> **Certify what compiled. Compare what changed. Qualify how fast the simulation can step. Keep the evidence.**

The first two sentences must always be read with the product's claim boundaries:

- `certify` is an artifact identity statement;
- `compare` is workload-bounded;
- `audit-timestep` is workload- and tolerance-bounded;
- none of them proves global physical truth or hardware safety.

---

# 11. Recommended command selection

| Situation | Start with | Continue with |
|---|---|---|
| XML formatter or refactor intended to be neutral | `certify` | `compare` only if compiled bytes differ |
| Mass, damping, friction, actuator, solver, or geometry change | `certify` | `compare` on representative workloads |
| Vendor or upstream model revision | `certify` | `compare` and retain receipts |
| Suspected regression | `compare` | inspect first crossing and reasons |
| Candidate timestep increase | `audit-timestep` | measure real throughput separately |
| CI gate for model maintenance | `certify` | require review or workload comparison on exit 40 |
| Reproducible benchmark release | `certify` + `compare` | retain exact receipts and workload artifacts |

---

# 12. Final faithful assessment

Metrifid already provides a coherent three-part technical product:

```text
Certify
  exact complete compiled-artifact identity

Compare
  deterministic workload-bounded behavior evidence

Audit timestep
  deterministic workload-bounded candidate qualification
```

Its strongest qualities are:

- exact source and compiled identities;
- strict workload declarations;
- exact time arithmetic;
- deterministic repeats;
- refusal instead of silent approximation;
- first-divergence evidence;
- canonical self-hashed receipts;
- bounded resource behavior;
- descriptor-confined source and output handling;
- clear separation between artifact equality and behavior equality.

Its current limitations are equally important:

- four retained exact MuJoCo profiles, with later stable releases only capability-admitted and
  `review-model` limited to cataloged public-field surfaces;
- currently narrow measured platform evidence;
- open-loop workload only;
- hinge and slide qpos/qvel metrics only;
- no cross-runtime or cross-backend product surface;
- no cryptographic signing or team control plane.

Within those limits, Metrifid is useful today as a local SDK on the tested runtime, an open-source alpha library, a model-release evidence tool, and a CI gate for MuJoCo assets.

## Workload qualification

`metrifid qualify-workload` answers whether the workloads you already run would notice a model
change you care about.

You declare a baseline model, one or more probe ladders (a perturbation, a direction, and models
you built at increasing magnitudes), a required detection magnitude per ladder, and three to
sixteen recorded workloads. Metrifid runs a zero-change control per workload and one ordinary
comparison per probe rung per workload, then enumerates every three-workload subset and reports the
best one under a fixed order.

It is useful when you have a regression suite of workloads and no evidence about what they can
actually see. The answer names the perturbations that stay invisible, which is the part a coverage
number cannot tell you.

It does not claim anything about perturbations you did not declare, workloads you did not declare,
coordinates you do not monitor, global model equivalence, task safety, or real-world transfer. It
also does not infer detectability from a sensitivity or estimability model: every cell is a
completed comparison.

It does not verify your probe labels. Metrifid compares the exact supplied admitted model closures
and preserves your `parameter`, `direction`, `magnitude`, and `magnitude_semantics` declarations; it
does not establish that they faithfully describe the source edits or that no other source change
exists. Completed receipts carry `USER_DECLARED_PROBE_SEMANTICS_NOT_VERIFIED` to say so. The Model
Change Gate is separate supporting evidence about source change, not automatic proof of a label.

A published receipt is fully checkable: the public loader reparses it strictly, recomputes every
derived claim from its own configuration and cell records, and rebinds it to the retained comparison
configurations and receipts under its owned output root. `receipt_sha256` is corruption detection
only — it is not a signature and recomputing it cannot make a contradictory receipt valid. See
[`docs/workload_qualification.md`](workload_qualification.md).
