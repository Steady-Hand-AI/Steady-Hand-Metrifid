# Reference

The stable public surface: commands, exit codes, statuses, schemas and the importable Python API.

## Commands

```bash
metrifid certify BASELINE_MJCF CANDIDATE_MJCF --output OUTPUT_DIRECTORY \
  [--baseline-root BASELINE_ROOT] [--candidate-root CANDIDATE_ROOT]

metrifid compare comparison.json

metrifid audit-timestep timestep_audit.json
```

## Runtime support

All three native commands share one admission policy: Python 3.11 or newer with no upper bound,
Linux or macOS with the required POSIX capabilities, no architecture allowlist, MuJoCo native
engine 3.10.0 exactly, a stable `3.10.0` or binding-only `3.10.0.postN` Python package, and NumPy
1.26 or newer with no upper bound. The runtime does not reject an interpreter implementation by
name. Release evidence currently uses CPython because the MuJoCo binary wheels exercised by the
release matrix target CPython; another implementation is not claimed as validated without native
evidence. Native Windows is unsupported; use WSL.

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
| `metrifid.timestep_audit`, version `1` | `timestep_audit.json` |
| `metrifid.operational_failure`, version `1` | stderr on any refusal |

`certify` and `compare` each publish one JSON/Markdown pair into an absent or empty directory. `audit-timestep` publishes its aggregate JSON/Markdown pair plus a `candidates/` evidence tree containing one retained result or operational failure per candidate. A nonempty output directory refuses. Every public file name is acquired with a descriptor-relative no-clobber hard link and exact sealed-byte verification. Failure cleanup removes private temporaries but never unlinks a public final, so a later failure may leave already-linked diagnostic evidence while still refusing the operation. A success exit, the complete command-specific output tree, and the final public-path and byte verification are all required before treating a result as published.

Model traversal and result publication are descriptor-confined: replacing an admitted path cannot redirect reads, writes, publication, or cleanup. Certify and Compare refuse output equal to or below
either model root. State and actions NPZ admission reads from one no-follow descriptor and enforces
the raw-byte bound while reading, before ZIP preflight or array parsing.

## Canonical values

Numbers that must survive a round trip exactly are never written as JSON floats:

- an exact rational is `{"numerator": <int>, "denominator": <int>}`, always normalized;
- an IEEE-754 double is `{"kind": "ieee754_binary64", "bits": "<16 lowercase hex>"}`.

`NaN` and the infinities are representable and survive unchanged. See
[`canonicalization.md`](canonicalization.md).

## Python API

The supported programmatic **execution** surface is documented in [`docs/sdk.md`](sdk.md):
`metrifid.certify.certify_models`, `metrifid.compare.compare_configuration_file`, and
`metrifid.timestep_audit.audit_configuration_file`. The three CLI commands are thin wrappers over
those functions. The table below covers the supporting value types and helpers.


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

The MuJoCo native engine must be MuJoCo 3.10.0 — `mj_versionString() == "3.10.0"` and
`mj_version() == 3010000` — exactly. Package versions in the stable
`3.10.0` family are accepted, including binding-only `3.10.0.postN` and local variants; prerelease,
development, `3.10.1`, and `3.11`-or-later packages refuse. NumPy is `>=1.26` with no runtime upper
bound and no runtime version gate.

The package metadata supports CPython 3.11 and newer without an artificial upper bound. Release
CI is configured for CPython 3.11–3.14 on Linux x86_64, CPython 3.12 and 3.14 on macOS arm64, and
CPython 3.12 on macOS x86_64. A tuple is described as validated only after that exact lane passes;
a future minor is not rejected solely because its minor number is newer.

The workload artifact writers, the canonical JSON and exact-number helpers, and receipt parsing and
validation are outside this gate and work without native MuJoCo admission.

A certification receipt is runtime-bound. It records the runtime identity that produced the
artifacts and claims only what that runtime produced; it does not claim that another operating
system, MuJoCo version or build would produce the same digest.

## Installed identity

The tool refuses to run from an editable install, from a source tree on `sys.path`, or when more
than one `metrifid` distribution is visible. Every receipt records the version and the
distribution digest of the code that produced it.
