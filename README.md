# Metrifid

**Know whether your MuJoCo model actually changed — before you argue about whether the simulation did.**

A source diff cannot tell you. Reordered attributes, reformatted XML and moved includes change
the bytes without changing the model. Running the simulation cannot tell you either: it only
reports on the trajectories you happened to run.

`metrifid certify` compiles both versions and compares every byte MuJoCo emits for the compiled
model.

Project links: [repository](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid) · [documentation](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/docs/getting_started.md) · [issues](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/issues) · [changelog](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/CHANGELOG.md) · [security reporting](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/security/advisories/new)

## Start with the change you made

You do not need to choose between seven commands to get an answer. Two things change in practice —
what you model, and what you run it on — and each has one place to start.

### Your model or asset changed

Someone edited an MJCF file, regenerated a mesh, bumped an importer, or merged a pull request, and
you need to know whether the model MuJoCo actually compiles is still the same model.

```bash
metrifid certify old/robot.xml new/robot.xml --output out/
```

```text
exit 0   CERTIFIED_COMPILED_EQUIVALENCE   every serialized byte matched
exit 40  NOT_CERTIFIED_COMPILED_DIFFERS   at least one byte differed
```

Certify compiles both closures and compares every byte MuJoCo emits. It runs no workload, needs no
initial state, actions, or tolerances, and never steps the simulation. **Certify makes no
behavioral-equivalence claim**: it tells you the compiled artifact differs, not that the robot will
behave differently.

Exit 40 is not a crash. A completed exit 20, 30, or 40 can be a referee decision — Metrifid finished
its work and is reporting what it found.

### Your MuJoCo runtime changed

You are moving to a new MuJoCo native version and need to know whether it may replace the one your
evidence was built on, for your closure and your workload.

```bash
metrifid run-runtime-review runtime_review_run.json
```

This creates the fixed twelve native evidence cells through two explicit, already-prepared Python
profiles and returns the full-horizon Runtime Review decision. **Metrifid does not discover, install,
or repair MuJoCo environments** — you prepare the two profiles, and Metrifid measures what they do.

## Follow the evidence

Both routes hand you a receipt that names the next question. Follow the one you actually have:

| You now know | You still want | Use |
| --- | --- | --- |
| the compiled artifacts differ | which fields changed, and whether your policy allows them | `review-model` |
| a change is allowed statically | whether it moves the trajectories you care about | `compare` |
| your tolerances are set | whether your workloads would notice a perturbation at all | `qualify-workload` |
| the model is settled | which timestep still holds within tolerance | `audit-timestep` |
| twelve evidence cells already exist | whether one native profile may replace another | `review-runtime` |

Every one of these is a completed decision with a receipt you can revalidate independently. None of
them establishes physical correctness or safety.

See [`docs/public_cases.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/docs/public_cases.md) for six complaint-backed cases that walk the whole
journey end to end, and [`docs/getting_started.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/docs/getting_started.md) for the guided tour.

## First use

From an empty directory, with nothing but the installed wheel:

```bash
python -m pip install metrifid
python -m metrifid.demo
metrifid --help
```

`python -m metrifid.demo` needs no checkout, no arguments, and no network. It certifies one
source-different but compiled-identical pair (exit 0) and one physically changed pair (exit 40),
revalidates both published receipts, and prints `Metrifid demo passed`.

Then certify two of your own model revisions:

```bash
metrifid certify old/robot.xml new/robot.xml --output out/
```

Working from a source checkout instead? `python -m pip install .` installs the same package;
integration, security, receipt, and release evidence must run from a noneditable installation.

```text
exit 0   CERTIFIED_COMPILED_EQUIVALENCE   every serialized byte matched
exit 40  NOT_CERTIFIED_COMPILED_DIFFERS   at least one byte differed
```

It runs no workload, needs no initial state, no actions and no tolerances, and never steps the
simulation.



## Try the source-checkout example

From a clone of the [Metrifid repository](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid):

```bash
python examples/certify/run_example.py
```

```text
different source, same compiled model : CERTIFIED_COMPILED_EQUIVALENCE (exit 0)
one changed mass                      : NOT_CERTIFIED_COMPILED_DIFFERS (exit 40)

all 6 checks passed
```

Two models written differently that compile identically, and one model with a single changed
mass. See [`examples/certify/`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/tree/main/examples/certify/).

## Why complete source closure and compiled identity matter

**Closure**, because a model is not one file. The source closure contains every admitted regular
file under the declared model root, measured by exact bytes. Dependency discovery separately
identifies the files MuJoCo reaches while compiling the entrypoint. An unused regular file therefore
changes the source-closure identity, but does not change the compiled MJB unless MuJoCo resolves it.
Your original files are never modified: the measured members are copied into a private immutable
snapshot, compiled from there, and the source is reverified before anything is published.

**Compiled identity**, because the compiled model is what the engine simulates. The identity is
SHA-256 over every byte `mj_saveModel` emits — no projection, no normalization, no rounding, no
tolerance. If the certificate says the artifacts are identical, nothing in the compiled model
differs, including fields nobody thought to check.

When artifacts differ, the receipt locates the difference: the first differing byte offset, the
count, and a bounded field report naming the changed public fields with sorted index and value
witnesses.

## Review a model release

**`metrifid review-model`** — what changed between two compiled robot models, was each change
allowed, required, forbidden or undeclared by your policy, and where is the first unexpected one?
It reads two model sources and one bounded maintainer policy, and links its decision to a Certify
receipt built from the exact same compiled artifacts.

```bash
metrifid review-model baseline/model.xml candidate/model.xml \
  --policy model_release_policy.json \
  --output review-output
```

| Status | Exit | Meaning |
| --- | --- | --- |
| `NO_COMPILED_CHANGE` | 0 | The complete MJBs matched; there is no compiled change row. |
| `WITHIN_DECLARED_POLICY` | 0 | Every observed compiled change is declared by the policy. |
| `REVIEW_REQUIRED` | 40 | At least one observed change is undeclared, or coverage is incomplete. |
| `OUTSIDE_DECLARED_POLICY` | 40 | A change is forbidden, or a required change is missing. |

This is a static policy decision. It steps no model and **does not establish dynamic equivalence**:
a statically permitted compiled change may still change behavior, which is what `compare` measures.

See [`docs/model_release.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/docs/model_release.md) for the complete schema and semantics, and
[`examples/model_release/README.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/examples/model_release/README.md) for a runnable two-pass
example.

## The remaining decisions

- **`metrifid run-runtime-review`** — create the fixed twelve native evidence cells through two
  explicit, already-prepared Python profiles and immediately receive the same full-horizon Runtime
  Review decision. Metrifid measures but does not discover, install, or repair those environments.
  See the copyable [`examples/runtime_review_run`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/tree/main/examples/runtime_review_run/) journey.

- **`metrifid review-runtime`** — whether one exact candidate MuJoCo native profile may replace one
  exact baseline profile for a declared source closure and workload over the complete retained
  horizon. It reads twelve already-produced three-grid/two-repeat evidence cells; it does not run a
  worker or select a method. Both paths use the same referee. See
  [`docs/runtime_review.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/docs/runtime_review.md).

- **`metrifid qualify-workload`** — whether three of your declared workloads would actually
  notice a supplied probe model at or above the magnitude you care about, and which probes stay
  invisible. Every answer is built from ordinary `compare` runs against probe models you supply,
  so it reports detection under your tolerances rather than a sensitivity estimate. The parameter
  name, direction, magnitude and `magnitude_semantics` you write down are preserved declarations
  about *your* probe closures: Metrifid binds the exact bytes it compared and reports what was
  detected, and does not independently establish that those labels describe the source edits or
  that no other source change exists.

- **`metrifid audit-timestep`** — which of your declared candidate timesteps still produces no
  material difference against the reference, for a declared workload and tolerances? Every
  candidate is attempted independently; the recommendation is the largest one backed by an
  unbroken within-tolerance prefix. See [`docs/timestep_audit.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/docs/timestep_audit.md).
- **`metrifid compare`** — under one declared deterministic open-loop workload, did monitored
  hinge or slide traces move beyond tolerances you set in physical units? See
  [`docs/comparison.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/docs/comparison.md).

`compare` is the workload-bounded decision. `certify` is an artifact statement and makes no
behavioral claim.

## Documentation

| Document | What it covers |
| --- | --- |
| [docs/getting_started.md](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/docs/getting_started.md) | a human-friendly first tour, installation, and documentation map |
| [docs/capabilities_and_use_cases.md](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/docs/capabilities_and_use_cases.md) | current capabilities, practical uses, limits, and non-goals |
| [docs/public_cases.md](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/docs/public_cases.md) | six complaint-backed public cases and the external Robosuite Jaco study |
| [docs/reference.md](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/docs/reference.md) | commands, exits, statuses, schemas, Python API |
| [`docs/compiled_certification.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/docs/compiled_certification.md) | the `certify` contract and claim boundary |
| [`docs/model_closure.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/docs/model_closure.md) | how the model closure is measured and confined |
| [`docs/workloads.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/docs/workloads.md) | state, action and exact time-grid declarations |
| [`docs/comparison.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/docs/comparison.md) | the `compare` decision |
| [`docs/timestep_audit.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/docs/timestep_audit.md) | the `audit-timestep` decision |
| [`docs/model_release.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/docs/model_release.md) | the `review-model` static Model Change Gate |
| [`docs/workload_qualification.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/docs/workload_qualification.md) | the `qualify-workload` detection decision |
| [`docs/runtime_review.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/docs/runtime_review.md) | the `run-runtime-review` execution journey and `review-runtime` retained-evidence decision |
| [docs/sdk.md](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/docs/sdk.md) | the supported programmatic execution surfaces, including `review_runtime_configuration_file` |
| [`docs/canonicalization.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/docs/canonicalization.md) | canonical JSON, exact numbers, self-hashing |
| [`docs/menagerie_formatter_case.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/docs/menagerie_formatter_case.md) | a real public case study |

## Requirements

Python 3.11 or newer, with no upper bound and no runtime rejection based on the interpreter name.
Release evidence currently uses CPython because the MuJoCo binary wheels exercised by the release
matrix target CPython. Other Python implementations are not claimed as validated without native
evidence. Runtime admission requires a stable MuJoCo `3.9` or newer and is capability-based, with
no minor-version ceiling on any platform. Package resolution carries one platform exception:
because upstream publishes no Intel macOS wheel for MuJoCo `3.11` or newer, the dependency is
`mujoco>=3.9,<3.11` on Darwin `x86_64` and `mujoco>=3.9` with no minor ceiling everywhere else.
That bound is a packaging-resolution fact about upstream wheel availability, not a narrowing of
what the runtime admits. Exact stable MuJoCo `3.9.0`, `3.10.0`, `3.11.0`, and `3.12.0` profiles are
retained-validated. A later stable release is admitted only when its Python/native identities agree
and the requested operation's measured capabilities pass; it is reported as capability-compatible,
never falsely as validated. On platforms without that packaging exception the newest stable release
is the primary development and release profile—3.12.0 for the frozen 2026-08-22 snapshot—while the
exact retained older profiles remain backward-compatibility evidence. NumPy is `>=1.26`, with no
runtime ceiling and no runtime version gate.

Every native command — `certify`, `review-model`, `compare`, `audit-timestep`,
and `qualify-workload` —
passes through one shared runtime gate. That gate admits Linux and macOS when the required POSIX
capabilities are present, and it never inspects the machine architecture: architecture is receipt
evidence only, so there is no allowlist. The Intel macOS packaging bound above is a dependency
marker read by the installer, not a runtime check; the gate itself remains architecture-neutral. Native Windows is unsupported because those capabilities
are absent; WSL is the documented route. A coherent runtime can still refuse one claim: missing
call-graph capabilities use `MUJOCO_RUNTIME_CAPABILITY_MISSING`, and an unrepresentable feature uses
`MUJOCO_FEATURE_COVERAGE_INCOMPLETE`. In particular, `review-model` requires an exact characterized
public-field surface, while `certify` may still identify complete MJB bytes on a newer admitted
runtime. Receipts continue binding exact package, native-library, and installed-distribution hashes.

`run-runtime-review` instead measures and admits both explicitly declared external native profiles
through its fixed preflight collector before invoking the frozen worker in either profile. It does
not search for or repair an environment, and it does not substitute the installed Metrifid process's
runtime gate for those profile-specific measurements.

The package metadata has no artificial Python upper bound. Release CI runs two complete
installed-wheel suites — one on the primary resolver-latest profile and one on the declared
dependency-floor profile — and bounded smokes on the other default interpreter and platform
profiles. The exact matrix lives in the workflow and changes as upstream support does. A tuple is
described as validated only after that exact lane passes; a future Python minor or another
implementation is not rejected solely by its name or version, but is not described as validated
without native evidence.

Pure helpers — the workload artifact writers, canonical JSON and exact-number helpers, and receipt
parsing and validation — never require the native gate and work without MuJoCo admission.

## What a certificate claims

> The measured baseline and candidate source closures, compiled under the recorded runtime
> identity, produced byte-identical complete MJB artifacts.

That claim is workload-free but runtime-bound: it holds for the runtime identity recorded in the
receipt, which is why that identity is part of the receipt. It does not claim source-text
equality, licensing, visual intent, task suitability, hardware safety, or that the same digest
would appear under another operating system, MuJoCo version or build, and it does not survive a
caller mutating `mjModel` afterwards. Every receipt carries those limitations
explicitly and can be revalidated with `metrifid.certify.validate_receipt`.

## Using Metrifid from Python

Each CLI command is a thin wrapper over one supported function:

```python
from metrifid.certify import certify_models
from metrifid.compare import compare_configuration_file
from metrifid.model_release import review_model_release
from metrifid.timestep_audit import audit_configuration_file
from metrifid.workload_qualification import qualify_configuration_file
from metrifid.runtime_review import (
    review_runtime_configuration_file,
    run_runtime_review_configuration_file,
)
```

See [docs/sdk.md](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/docs/sdk.md) for signatures, return types, the completed-decision versus refusal distinction,
output ownership, the native runtime requirement, and the concurrency statement.
Runnable SDK scripts for `certify`, `compare`, `audit-timestep`, `qualify-workload`, and the
Runtime Review execution journey live in
[`examples/sdk/`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/tree/main/examples/sdk/). The Model Change Gate example lives in [`examples/model_release/`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/tree/main/examples/model_release/).

## License, ownership, and contributions

Copyright 2026 Volodymyr Barylyak. Metrifid is licensed under Apache License 2.0.

- `LICENSE` contains the controlling software license.
- `NOTICE` contains project attribution.
- `DCO` contains the Developer Certificate of Origin 1.1 used for contributed commits.
- `docs/licensing_and_contributions.md` explains the relationship among those files.

Contributions require a DCO sign-off as described in [`CONTRIBUTING.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/CONTRIBUTING.md). The DCO is not a copyright
assignment; contributors retain copyright in their contributions.
