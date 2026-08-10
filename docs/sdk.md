# Python SDK

Metrifid has three execution entry points. They are the supported programmatic surface, and they do
exactly what the three CLI commands do — the CLI is a thin wrapper over them.

```python
from metrifid.certify import certify_models, CertifyResult
from metrifid.compare import compare_configuration_file, ComparisonRunResult
from metrifid.timestep_audit import audit_configuration_file, AuditRunResult
```

Runnable versions of every example below live in `examples/sdk/`.

## Two kinds of outcome

Read this once; it applies to all three operations.

A **completed decision** is a return value. `NOT_CERTIFIED_COMPILED_DIFFERS` is a completed
decision, not a failure: Metrifid looked, and the artifacts differ. Your code inspects the result.

An **operational refusal** is an exception. It means Metrifid did not return a completed decision —
for example because a model root was unreadable, an output location was occupied, a configuration
did not parse, or the native runtime was unsupported. A late refusal can leave diagnostic evidence
that Metrifid had already linked before the final verification boundary. Treat the returned result
or raised exception as authoritative; the presence of an output filename alone never means the
operation completed. Catch `metrifid.certify.CertifyOperationError` or
`metrifid.compare.ComparisonOperationError` (and `AuditAbort` for audits) when you want to translate
a refusal into application-specific error handling. The carried failure object contains the typed
reason code.

## Native runtime requirement

All three operations compile or step a real model, so they require the exact supported runtime:

```text
Python language   >= 3.11, with no upper bound and no interpreter-name allowlist
Operating system  Linux or macOS with the required POSIX descriptor capabilities
MuJoCo package    3.10.0 (binding-only 3.10.0.postN accepted)
MuJoCo native     3.10.0 / mj_version() == 3010000
NumPy             >= 1.26, with no runtime upper bound
```

Release evidence currently uses CPython because the exercised MuJoCo wheels target CPython. Other
Python implementations are not claimed as validated, but Metrifid does not refuse solely because of
the interpreter name. Architecture is recorded in receipts and is not checked against an allowlist.
The runtime refuses when the minimum language level, required POSIX capabilities, operating-system
family, or exact MuJoCo engine profile is not satisfied. Receipt *reading* has no native requirement:
`metrifid.certify.load_and_validate_certification_receipt` imports and runs without MuJoCo or NumPy
installed.

Metrifid must also be installed normally. A decision-bearing operation refuses to run from an
editable install, because every receipt binds the identity of the installed distribution that
produced it.

## Concurrency

**Metrifid does not promise thread-safe or reentrant native execution.** Use separate processes for
parallel jobs. Do not call these functions concurrently from threads in one process, and do not
call them reentrantly from inside a Metrifid callback.

## Output ownership

Each operation publishes into a directory you name. That directory must be absent or empty.
Metrifid never overwrites an existing entry, and it verifies the published path again after writing,
so replacing the directory mid-run causes a refusal rather than a result pointing at someone else's
bytes.

---

## `certify_models`

```python
def certify_models(
    baseline_mjcf: str,
    candidate_mjcf: str,
    output_directory: str,
    *,
    baseline_root: str | None = None,
    candidate_root: str | None = None,
) -> CertifyResult
```

Paths are accepted as `str`. When a role's root is omitted, the entrypoint's parent directory is
used as that model's source-closure root.

**Required inputs.** Two MJCF entrypoints and an absent-or-empty output directory. The output
directory may not sit inside either model root.

**Returns** `CertifyResult` with:

| Field | Meaning |
| --- | --- |
| `status` | `CERTIFIED_COMPILED_EQUIVALENCE` or `NOT_CERTIFIED_COMPILED_DIFFERS` — the decision |
| `receipt` | the complete receipt as a canonical mapping |
| `receipt_sha256` | the receipt's self-hash |
| `certification_json` | published machine receipt path |
| `certification_markdown` | published human rendering path |

**Outputs.** `certification.json` and `certification.md` in the output directory.

```python
import tempfile
from pathlib import Path

from metrifid.certify import certify_models, load_and_validate_certification_receipt

with tempfile.TemporaryDirectory() as raw:
    workspace = Path(raw).resolve()
    result = certify_models(
        str(workspace / "baseline" / "model.xml"),
        str(workspace / "candidate" / "model.xml"),
        str(workspace / "out"),
    )
    print(result.status, result.receipt_sha256)
    load_and_validate_certification_receipt(result.certification_json.read_bytes())
```

See `examples/sdk/certify_api.py`.

## `compare_configuration_file`

```python
def compare_configuration_file(config_path: str | Path) -> ComparisonRunResult
```

Accepts `str` or `pathlib.Path`.

**Required inputs.** One strict JSON configuration with exactly these top-level fields:

```text
schema_version  baseline  candidate  initial_state  actions
control_dt      repeats   joint_tolerances  aliases  output_dir
```

`baseline` and `candidate` each declare `model_root`, `entrypoint`, and `declared_step_dt`.
`repeats` is an integer between 2 and 5. `initial_state` and `actions` name canonical NPZ artifacts
written by `write_state_artifact` and `write_actions_artifact`. Relative paths resolve against the
configuration file's own directory. Nothing is inferred; unknown keys are refused.

**Returns** `ComparisonRunResult` with `receipt` (a `ComparisonReceipt`, whose `status` carries the
decision), `comparison_json`, and `comparison_markdown`.

**Outputs.** `comparison.json` and `comparison.md` in `output_dir`.

```python
from metrifid.compare import compare_configuration_file

result = compare_configuration_file("comparison.json")
print(result.receipt.status)
print(result.comparison_json, result.comparison_markdown)
```

See `examples/sdk/compare_api.py`, and `examples/compare/` for a complete runnable configuration.

## `audit_configuration_file`

```python
def audit_configuration_file(config_path: str | Path) -> AuditRunResult
```

Accepts `str` or `pathlib.Path`.

**Required inputs.** One strict JSON configuration with exactly these top-level fields:

```text
schema_version  model_root  entrypoint  initial_state  actions
control_dt      repeats     joint_tolerances  candidate_step_dts
workload_kind   workload_label  output_dir
```

`candidate_step_dts` lists the timesteps to qualify as exact decimal tokens.

**Returns** `AuditRunResult` with `aggregate` (the canonical mapping holding every candidate
classification and the recommendation), `audit_json`, and `audit_markdown`.

Each candidate is classified `WITHIN_DECLARED_TOLERANCE`, `OUTSIDE_DECLARED_TOLERANCE`, or
`REFUSED`. The recommendation follows the completed-prefix policy: it never crosses a candidate
that produced no trustworthy comparison evidence, and it reports
`blocked_by_prior_non_within` when it stopped early.

**Outputs.** `timestep_audit.json`, `timestep_audit.md`, and one `candidates/<token>/` directory per
declared candidate.

```python
from metrifid.timestep_audit import audit_configuration_file

result = audit_configuration_file("timestep_audit.json")
for row in result.aggregate["candidates"]:
    print(row["token"], row["classification"])
print(result.aggregate["recommendation"]["candidate_token"])
```

See `examples/sdk/audit_api.py`.

## Reading receipts without the engine

```python
from metrifid.certify import load_and_validate_certification_receipt

receipt = load_and_validate_certification_receipt(Path("certification.json").read_bytes())
```

The raw loader applies bounded strict admission — duplicate member names, raw float tokens,
`NaN`/`Infinity`, malformed UTF-8, and oversized or excessively nested documents are refused before
any semantic check — then revalidates the receipt and returns the mapping. Use
`metrifid.certify.validate_receipt` when you already hold a strictly parsed mapping, and
`metrifid.validate_receipt` for a `ComparisonReceipt` object.

## Workload writers

`metrifid.write_state_artifact` and `metrifid.write_actions_artifact` produce the canonical NPZ
artifacts that comparison and audit configurations reference. They are pure: they need no native
runtime and work without MuJoCo installed.
