# Reference

The stable public surface: commands, exit codes, statuses, schemas and the importable Python API.

## Commands

```bash
metrifid certify BASELINE_MJCF CANDIDATE_MJCF --output OUTPUT_DIRECTORY \
  [--baseline-root BASELINE_ROOT] [--candidate-root CANDIDATE_ROOT]

metrifid compare comparison.json

metrifid qualify-workload QUALIFICATION_JSON
metrifid review-model BASELINE_MJCF CANDIDATE_MJCF --policy POLICY_JSON \
  --output OUTPUT_DIRECTORY \
  [--baseline-root BASELINE_ROOT] [--candidate-root CANDIDATE_ROOT]

metrifid audit-timestep timestep_audit.json
```

## Runtime support

All five native commands share one admission policy: Python 3.11 or newer with no upper bound,
Linux or macOS with the required POSIX capabilities, no architecture allowlist, a stable MuJoCo
package at or above 3.9 whose native string and integer encode the same three-part base version,
and NumPy 1.26 or newer with no upper bound. Exact stable profiles `3.9.0` through `3.12.0` are
retained-validated. A later stable runtime is admitted only by measured operation capabilities and
is labeled capability-compatible rather than validated. The runtime does not reject an interpreter
implementation by name. Native Windows is unsupported; use WSL.

Runtime admission and claim coverage are separate. Missing call-graph capabilities refuse with
`MUJOCO_RUNTIME_CAPABILITY_MISSING`. A runtime may be admitted while `review-model` or a dynamic
action contract refuses with `MUJOCO_FEATURE_COVERAGE_INCOMPLETE`; `review-model` never reuses an
older typed public-field projection for an uncharacterized surface. Both are operational exit 64,
with detected identity, the affected claim, its risk, and a concrete remediation in evidence.

Pure workload writers and canonical JSON, exact-number, and receipt-validation helpers do not invoke
the native runtime gate.

## Exit codes

| Exit | Command | Meaning |
| ---: | --- | --- |
| `0` | `certify` | `CERTIFIED_COMPILED_EQUIVALENCE` — every serialized byte matched |
| `40` | `certify` | `NOT_CERTIFIED_COMPILED_DIFFERS` — at least one byte differed |
| `0` | `compare` | `NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD` |
| `10` | `compare` | `MATERIAL_BEHAVIOR_CHANGE` |
| `20` | `compare` | `COVERAGE_INSUFFICIENT` |
| `30` | `compare` | `NONDETERMINISTIC_REPLAY` |
| `0` | `qualify-workload` | `QUALIFIED_FOR_DECLARED_PROBES` — every declared probe group is detected at or above its required magnitude |
| `20` | `qualify-workload` | `PARTIALLY_QUALIFIED` — some groups qualified, some insufficient, none unresolved |
| `20` | `qualify-workload` | `INSUFFICIENT_EXCITATION` — no group qualified and none unresolved |
| `30` | `qualify-workload` | `UNRESOLVED` — a rung at or above a requirement did not complete as a decision |
| `0` | `review-model` | `NO_COMPILED_CHANGE` — the complete MJBs matched |
| `0` | `review-model` | `WITHIN_DECLARED_POLICY` — every observed change is declared |
| `40` | `review-model` | `REVIEW_REQUIRED` — an observed change is undeclared |
| `40` | `review-model` | `OUTSIDE_DECLARED_POLICY` — forbidden change or missing requirement |
| `0` | `audit-timestep` | the audit completed and published its recommendation |
| `64` | any | invalid invocation, input or output |
| `70` | any | internal failure |

Exit `64` and `70` are refusals: no decision was reached. Every refusal writes one strict
operational-failure JSON document to stderr naming its stage and reason code.

## Statuses

```text
certify           CERTIFIED_COMPILED_EQUIVALENCE
                  NOT_CERTIFIED_COMPILED_DIFFERS

compare           NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD
                  MATERIAL_BEHAVIOR_CHANGE
                  COVERAGE_INSUFFICIENT
                  NONDETERMINISTIC_REPLAY

qualify-workload  QUALIFIED_FOR_DECLARED_PROBES
                  PARTIALLY_QUALIFIED
                  INSUFFICIENT_EXCITATION
                  UNRESOLVED

review-model      NO_COMPILED_CHANGE
                  WITHIN_DECLARED_POLICY
                  REVIEW_REQUIRED
                  OUTSIDE_DECLARED_POLICY

audit-timestep    per candidate: WITHIN, OUTSIDE, INCONCLUSIVE, REFUSED
```

`certify` statuses are its own. They are never mixed into the comparison status registry, and
`certify` never emits a comparison status or exit.

## Schemas

Every published document is canonical JSON with a self-hash over its own content.

| Schema identifier | Published as |
| --- | --- |
| `metrifid.compiled_equivalence_receipt`, version `1` | `certification.json` |
| `metrifid.compiled_field_report`, version `1` | inside a differing certification receipt |
| `metrifid.comparison_receipt`, version `1` | `comparison.json` |
| `metrifid.model_release_receipt`, version `1` | `model_release.json` |
| `metrifid.model_release_policy`, version `1` | the policy you supply to `review-model` |
| `metrifid.workload_qualification_receipt`, version `1` | the receipt `qualify-workload` publishes |
| `metrifid.workload_qualification_config`, version `1` | the `qualification.json` you supply |
| `metrifid.timestep_audit`, version `1` | `timestep_audit.json` |
| `metrifid.operational_failure`, version `1` | stderr on any refusal |

`certify` and `compare` each publish one JSON/Markdown pair into an absent or empty directory. `audit-timestep` publishes its aggregate JSON/Markdown pair plus a `candidates/` evidence tree containing one retained result or operational failure per candidate. A nonempty output directory refuses. Every public file name is acquired with a descriptor-relative no-clobber hard link and exact sealed-byte verification. Failure cleanup removes private temporaries but never unlinks a public final, so a later failure may leave already-linked diagnostic evidence while still refusing the operation. A success exit, the complete command-specific output tree, and the final public-path and byte verification are all required before treating a result as published.

Model traversal and result publication are descriptor-confined: replacing an admitted path cannot redirect reads, writes, publication, or cleanup. Certify and Compare refuse output equal to or below
either model root. State and actions NPZ admission reads from one no-follow descriptor and enforces
the raw-byte bound while reading, before ZIP preflight or array parsing.

### Static model-release review

```python
from metrifid.model_release import (
    load_and_validate_model_release_receipt,
    review_model_release,
)

result = review_model_release(
    "baseline/model.xml",
    "candidate/model.xml",
    "model_release_policy.json",
    "review-output",
    baseline_root="baseline",
    candidate_root="candidate",
)
```

`review_model_release` returns a completed result carrying `status`, `receipt`, `receipt_sha256`,
`model_release_json` and `model_release_markdown`. It publishes `model_release.json` and
`model_release.md` into an absent or empty output directory.

A completed decision returns; a refusal raises. Input refusals and operational failures raise
`metrifid.compare._failure.ComparisonOperationError` carrying the same strict operational-failure
document the CLI writes to stderr, which the CLI maps to exit `64` and `70` respectively. Both
completed exit-0 statuses and both completed exit-40 statuses return normally, so a caller must read
`result.status` rather than treating a return as approval.

`load_and_validate_model_release_receipt` reads published receipt bytes and revalidates the schema,
the embedded Certify receipt, every hash, the rule classification, the ordering, the witnesses and
the claim boundary, without importing MuJoCo or NumPy:

```python
receipt = load_and_validate_model_release_receipt(
    Path("review-output/model_release.json").read_bytes()
)
```

It raises on any inconsistency. Revalidation establishes internal consistency and linkage; it does
not recompile external source artifacts.

## Canonical values

Numbers that must survive a round trip exactly are never written as JSON floats:

- an exact rational is `{"numerator": <int>, "denominator": <int>}`, always normalized;
- an IEEE-754 double is `{"kind": "ieee754_binary64", "bits": "<16 lowercase hex>"}`.

`NaN` and the infinities are representable and survive unchanged. See
[`canonicalization.md`](canonicalization.md).

## Python API

The supported programmatic **execution** surface is documented in [`docs/sdk.md`](sdk.md):
`metrifid.certify.certify_models`, `metrifid.model_release.review_model_release`,
`metrifid.compare.compare_configuration_file`, `metrifid.timestep_audit.audit_configuration_file`,
and `metrifid.workload_qualification.qualify_configuration_file`. Each of the five CLI commands is a
thin wrapper over one of those functions. The table below covers the supporting value types and
helpers.

### `metrifid.workload_qualification`

This submodule declares an exact supported surface. `__all__` is a compatibility commitment and
contains exactly these six names, in this order:

| Name | Purpose |
| --- | --- |
| `QualificationExitCode` | the completed process exit-code registry |
| `QualificationResult` | the completed result `qualify_configuration_file` returns |
| `QualificationStatus` | the completed status registry |
| `WorkloadQualificationOperationError` | the bounded operational refusal |
| `load_and_validate_workload_qualification_receipt` | validate a published receipt and its linked evidence |
| `qualify_configuration_file` | run one campaign from a strict `qualification.json` |

The strict JSON configuration file is the only supported way to describe a campaign. Configuration,
probe, workload, cell, group, limitation and cardinality types are internal implementation detail:
they are not exported, several of their constructors require internal types, and they may change
without a compatibility guarantee.

Importing `metrifid.workload_qualification` performs no filesystem, network, or native-runtime work.
MuJoCo is loaded only when an execution or receipt-validation name is first resolved.


```python
import metrifid
```

Everything in `metrifid.__all__` is public and stable:

| Name | Purpose |
| --- | --- |
| `__version__` | the installed package version |
| `write_state_artifact`, `write_actions_artifact` | write canonical workload NPZ artifacts |
| `canonical_json_bytes`, `canonical_sha256`, `strict_json_loads` | canonical JSON encoding and hashing |
| `Binary64`, `ExactRational` | the two exact numeric forms |
| `validate_receipt`, `finalize_receipt` | validate and self-hash a comparison receipt |
| `ComparisonConfig`, `ComparisonContractIdentity`, `ComparisonReceipt` | comparison schema types |
| `ComparisonStatus`, `ReasonCode`, `ReasonRecord`, `REASON_REGISTRY`, `STATUS_PRECEDENCE` | the comparison decision registries |
| `LimitationCode` | the limitation codes every receipt carries |
| `OperationalExitCode`, `OperationalFailure`, `OperationalReasonCode`, `OperationalStage`, `OperationalToolObservation`, `InputDigestCode`, `InputDigest` | the refusal surface |
| `EngineThreadpoolState` | the observed engine threadpool state |

To validate a certification receipt held as a file or as untrusted bytes, use the raw loader. It
applies bounded strict admission — duplicate member names, raw float tokens, `NaN`/`Infinity`,
malformed UTF-8, and oversized or excessively nested documents are refused before any semantic
check runs — and returns the validated mapping:

```python
from pathlib import Path

from metrifid.certify import load_and_validate_certification_receipt

receipt = load_and_validate_certification_receipt(Path("certification.json").read_bytes())
```

Use `validate_receipt` when you already hold a strictly parsed in-memory mapping:

```python
from metrifid.certify import validate_receipt

validate_receipt(receipt)
```

Both entry points import without MuJoCo or NumPy installed, so a reader can revalidate a
certificate without the simulation engine.

That checks the internal consistency of an unsigned receipt: its exact member sets, frozen claim
and limitation text, the nested runtime identity and its own hash, both role identities, each
artifact's binding to the recorded runtime, the byte comparison against the two artifacts, and
the whole descriptive field report. It is not a signature and does not stop someone publishing a
different but self-consistent receipt.

## Support envelope

`certify`, `compare`, `audit-timestep`, and compiled model identity all share one runtime gate.
It admits Python 3.11 or newer with no upper bound, never inspects the interpreter implementation
name, and never rejects an architecture — the machine string is receipt evidence only. It requires
a POSIX operating system reporting `Linux` or `Darwin` together with the exact `dir_fd`,
`follow_symlinks`, `fd`, callable, and open-flag capabilities the confined filesystem work needs.
Native Windows is unsupported because those capabilities are absent; WSL is the documented route.

The MuJoCo package floor is `mujoco>=3.9`, with no minor ceiling. Only stable three-component
versions (optionally carrying a post or local suffix) are considered; prerelease and development
forms refuse. The package base, native version string, and algorithmically encoded native integer
must agree exactly. The retained validated profiles are 3.9.0, 3.10.0, 3.11.0, and 3.12.0. A later
stable release can be capability-admitted but is explicitly unvalidated. NumPy is `>=1.26` with no
runtime upper bound and no runtime version gate.

The resolver-selected newest stable MuJoCo is the primary development and release authority—3.12.0
for the frozen 2026-08-22 snapshot. Exact 3.9.0 remains the minimum profile; 3.10.0 and 3.11.0 are
retained backward-compatibility profiles. Evidence binds the exact admitted package/native identity.

Capability admission is operation-specific. `certify` needs only compiled-artifact capabilities;
`review-model` additionally requires a cataloged complete public-field surface; and dynamic replay
commands require their stepping/data capabilities plus a representable one-input/one-output action
signature. Missing capabilities and unsupported feature coverage fail closed before a completed
report or receipt is published.

The package metadata supports CPython 3.11 and newer without an artificial upper bound. Release CI
runs two complete installed-wheel suites — one on the primary resolver-latest profile and one on the
declared dependency-floor profile — and bounded smokes on the other default interpreter and platform
profiles; the exact matrix lives in the workflow, which changes as upstream support does. A tuple is
described as validated only after that exact lane passes; a future minor is not rejected solely
because its minor number is newer.

The workload artifact writers, the canonical JSON and exact-number helpers, and receipt parsing and
validation are outside this gate and work without native MuJoCo admission.

A certification receipt is runtime-bound. It records the runtime identity that produced the
artifacts and claims only what that runtime produced; it does not claim that another operating
system, MuJoCo version or build would produce the same digest.

## Installed identity

The tool refuses to run from an editable install, from a source tree on `sys.path`, or when more
than one `metrifid` distribution is visible. Every receipt records the version and the
distribution digest of the code that produced it.
