# Metrifid

**Know whether your MuJoCo model actually changed — before you argue about whether the simulation did.**

A source diff cannot tell you. Reordered attributes, reformatted XML and moved includes change
the bytes without changing the model. Running the simulation cannot tell you either: it only
reports on the trajectories you happened to run.

`metrifid certify` compiles both versions and compares every byte MuJoCo emits for the compiled
model.

Project links: [repository](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid) · [documentation](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/0.2.1/docs/getting_started.md) · [issues](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/issues) · [changelog](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/0.2.1/CHANGELOG.md) · [security reporting](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/security/advisories/new)

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

## Using the GitHub Action

You can easily integrate `metrifid certify` into your CI/CD pipeline to prevent compiled-artifact regressions in your pull requests. This repository doubles as a composite GitHub Action.

```yaml
- name: Metrifid Certify
  uses: Steady-Hand-AI/Steady-Hand-Metrifid@v0.2.1
  with:
    baseline_mjcf: baseline/model.xml
    candidate_mjcf: candidate/model.xml
    python_version: "3.11"
```

The action will fail the step if the models compile differently, and it will automatically post the markdown receipt directly into your `$GITHUB_STEP_SUMMARY`. For a real-world example, see [`examples/github_actions/menagerie_pr_check.yml`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/examples/github_actions/menagerie_pr_check.yml).

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
mass. See [`examples/certify/`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/tree/0.2.1/examples/certify/).

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

## The other two decisions

- **`metrifid audit-timestep`** — which of your declared candidate timesteps still produces no
  material difference against the reference, for a declared workload and tolerances? Every
  candidate is attempted independently; the recommendation is the largest one backed by an
  unbroken within-tolerance prefix. See [`docs/timestep_audit.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/0.2.1/docs/timestep_audit.md).
- **`metrifid compare`** — under one declared deterministic open-loop workload, did monitored
  hinge or slide traces move beyond tolerances you set in physical units? See
  [`docs/comparison.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/0.2.1/docs/comparison.md).

`compare` is the workload-bounded decision. `certify` is an artifact statement and makes no
behavioral claim.

## Documentation

| Document | What it covers |
| --- | --- |
| [`docs/getting_started.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/0.2.1/docs/getting_started.md) | a human-friendly first tour, installation, and documentation map |
| [`docs/capabilities_and_use_cases.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/0.2.1/docs/capabilities_and_use_cases.md) | current capabilities, practical uses, limits, and non-goals |
| [`docs/reference.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/0.2.1/docs/reference.md) | commands, exits, statuses, schemas, Python API |
| [`docs/compiled_certification.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/0.2.1/docs/compiled_certification.md) | the `certify` contract and claim boundary |
| [`docs/model_closure.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/0.2.1/docs/model_closure.md) | how the model closure is measured and confined |
| [`docs/workloads.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/0.2.1/docs/workloads.md) | state, action and exact time-grid declarations |
| [`docs/comparison.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/0.2.1/docs/comparison.md) | the `compare` decision |
| [`docs/timestep_audit.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/0.2.1/docs/timestep_audit.md) | the `audit-timestep` decision |
| [`docs/sdk.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/0.2.1/docs/sdk.md) | the supported programmatic execution surface: `certify_models`, `compare_configuration_file`, `audit_configuration_file` |
| [`docs/canonicalization.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/0.2.1/docs/canonicalization.md) | canonical JSON, exact numbers, self-hashing |
| [`docs/menagerie_formatter_case.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/0.2.1/docs/menagerie_formatter_case.md) | a real public case study |

## Requirements

Python 3.11 or newer, with no upper bound and no runtime rejection based on the interpreter name.
Release evidence currently uses CPython because the MuJoCo binary wheels exercised by the release
matrix target CPython. Other Python implementations are not claimed as validated without native
evidence. MuJoCo native engine `3.10.0` exactly; binding-only `3.10.0.postN` packages target that
same engine and are accepted. NumPy `>=1.26`, with no runtime ceiling and no runtime version gate.

Every native command — `certify`, `compare`, `audit-timestep`, and compiled model identity —
passes through one shared runtime gate. That gate admits Linux and macOS when the required POSIX
capabilities are present, and it never inspects the machine architecture: architecture is receipt
evidence only, so there is no allowlist. Native Windows is unsupported because those capabilities
are absent; WSL is the documented route.

The package metadata has no artificial Python upper bound. Release CI is configured for CPython
3.11–3.14 on Linux x86_64, CPython 3.12 and 3.14 on macOS arm64, and CPython 3.12 on macOS
x86_64. A tuple is described as validated only after that exact lane passes; a future Python minor
or another implementation is not rejected solely by its name or version, but is not described as
validated without native evidence.

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

The three CLI commands are thin wrappers over three supported functions:

```python
from metrifid.certify import certify_models
from metrifid.compare import compare_configuration_file
from metrifid.timestep_audit import audit_configuration_file
```

See [`docs/sdk.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/0.2.1/docs/sdk.md) for signatures, return types, the completed-decision versus refusal distinction,
output ownership, the native runtime requirement, and the concurrency statement. Runnable scripts
live in [`examples/sdk/`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/tree/0.2.1/examples/sdk/).

## License, ownership, and contributions

Copyright 2026 Volodymyr Barylyak. Metrifid is licensed under Apache License 2.0.

- `LICENSE` contains the controlling software license.
- `NOTICE` contains project attribution.
- `DCO` contains the Developer Certificate of Origin 1.1 used for contributed commits.
- `docs/licensing_and_contributions.md` explains the relationship among those files.

Contributions require a DCO sign-off as described in [`CONTRIBUTING.md`](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/0.2.1/CONTRIBUTING.md). The DCO is not a copyright
assignment; contributors retain copyright in their contributions.
