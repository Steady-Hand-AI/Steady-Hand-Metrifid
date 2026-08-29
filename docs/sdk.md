# Python SDK

Metrifid has five execution entry points. They are the supported programmatic surface, and each
does exactly what its CLI command does — the CLI is a thin wrapper over them.

```python
from metrifid.certify import certify_models, CertifyResult
from metrifid.compare import compare_configuration_file, ComparisonRunResult
from metrifid.model_release import review_model_release, ModelReleaseResult
from metrifid.workload_qualification import qualify_configuration_file, QualificationResult
from metrifid.timestep_audit import audit_configuration_file, AuditRunResult
```

Runnable examples for `certify`, `compare`, `audit-timestep`, and `qualify-workload` live in
`examples/sdk/`. The Model Change Gate example lives in `examples/model_release/run_example.py`.

## Two kinds of outcome

Read this once; it applies to all five operations.

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

All five operations compile a real model, and `compare`, `audit-timestep` and `qualify-workload`
also step one, so they require the exact supported runtime:

```text
Python language   >= 3.11, with no upper bound and no interpreter-name allowlist
Operating system  Linux or macOS with the required POSIX descriptor capabilities
MuJoCo package    stable >=3.9, with no minor-version ceiling
MuJoCo identity   package/native base and algorithmic native integer agree
Validated exact   3.9.0, 3.10.0, 3.11.0, and 3.12.0
NumPy             >= 1.26, with no runtime upper bound
```

The resolver-selected newest stable MuJoCo is the primary development and release profile—3.12.0
for the frozen 2026-08-22 snapshot. Exact 3.9.0 is the supported minimum; retained 3.10.0 and 3.11.0
profiles remain backward-compatibility evidence. SDK results bind the exact admitted runtime bytes.

Release evidence currently uses CPython because the exercised MuJoCo wheels target CPython. Other
Python implementations are not claimed as validated, but Metrifid does not refuse solely because of
the interpreter name. Architecture is recorded in receipts and is not checked against an allowlist.
The runtime refuses when the minimum language level, required POSIX capabilities, operating-system
family, coherent MuJoCo identity, or operation capability surface is not satisfied. A later stable
profile can be admitted without being labeled validated. A specific claim can still fail closed:
`review-model` requires an exact cataloged public-field surface, and static/dynamic action contracts
require representable actuator input/output signatures. Receipt *reading* has no native requirement:
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

## `review_model_release`

```python
from metrifid.model_release import ModelReleaseResult, review_model_release

result: ModelReleaseResult = review_model_release(
    "baseline/model.xml",
    "candidate/model.xml",
    "model_release_policy.json",
    "review-output",
    baseline_root="baseline",
    candidate_root="candidate",
)
```

All four positional arguments are required. Model-root resolution and admission match
`certify_models`. The output directory must be absent or empty and must lie outside both model
roots.

The returned `ModelReleaseResult` carries:

```text
status                   ModelReleaseStatus
receipt                  the complete canonical receipt object
receipt_sha256           the validated self-hash of that receipt
model_release_json       the published JSON path
model_release_markdown   the published Markdown path
```

### Completed results versus raised operational failures

A **completed decision returns**, whichever way it decided:

```text
NO_COMPILED_CHANGE        exit 0 at the CLI
WITHIN_DECLARED_POLICY    exit 0 at the CLI
REVIEW_REQUIRED           exit 40 at the CLI
OUTSIDE_DECLARED_POLICY   exit 40 at the CLI
```

Two of those four map to a nonzero CLI exit, so a plain return is not approval. Branch on
`result.status`:

```python
from metrifid.model_release import ModelReleaseStatus

if result.status is ModelReleaseStatus.WITHIN_DECLARED_POLICY:
    ...  # every observed compiled change was declared
elif result.status is ModelReleaseStatus.NO_COMPILED_CHANGE:
    ...  # the compiled artifacts were byte-identical
else:
    ...  # REVIEW_REQUIRED or OUTSIDE_DECLARED_POLICY: a human decides
```

An **operational failure raises**. Invalid policy bytes, an unusable model root, a nonempty output
directory, a policy bound to a different baseline subject, or an internal failure raise
`metrifid.compare._failure.ComparisonOperationError`. Its `failure` member is the same strict
operational-failure document the CLI writes to stderr; the CLI maps those to exit `64` for an input
refusal and exit `70` for an operational failure. A raised failure publishes no completed pair.

`review_model_release` makes a static policy decision only. It allocates no `mjData` and steps no
model, so a `WITHIN_DECLARED_POLICY` result establishes no dynamic equivalence.

## `load_and_validate_model_release_receipt`

```python
from pathlib import Path

from metrifid.model_release import load_and_validate_model_release_receipt

receipt = load_and_validate_model_release_receipt(
    Path("review-output/model_release.json").read_bytes()
)
print(receipt["status"], receipt["receipt_sha256"])
```

It accepts UTF-8 receipt bytes and revalidates the schema, the embedded Certify receipt, every
recorded hash, the rule classification, the deterministic ordering, both first witnesses and the
claim boundary. It imports neither MuJoCo nor NumPy, so a reviewer can check a published receipt on
a machine that cannot run the engine.

It raises on any inconsistency and returns the validated receipt otherwise. Revalidation establishes
internal consistency and linkage; it does not recompile external source artifacts.

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

## `qualify_configuration_file`

The submodule exports exactly six names, and that list is a compatibility commitment:

```python
from metrifid.workload_qualification import (
    QualificationExitCode,
    QualificationResult,
    QualificationStatus,
    WorkloadQualificationOperationError,
    load_and_validate_workload_qualification_receipt,
    qualify_configuration_file,
)

result: QualificationResult = qualify_configuration_file("qualification.json")
```

Nothing else is public. The strict JSON configuration file is the only supported way to describe a
campaign: configuration, probe, workload, cell, group, limitation and cardinality types are internal
implementation detail, several of their constructors require internal types, and they may change
without a compatibility guarantee. Importing the submodule performs no filesystem, network, or
native-runtime work; MuJoCo loads only when one of the six names is first resolved.

A runnable end-to-end script is [`examples/sdk/workload_qualification_api.py`](../examples/sdk/workload_qualification_api.py).

`result.status` is one of `QUALIFIED_FOR_DECLARED_PROBES`, `PARTIALLY_QUALIFIED`,
`INSUFFICIENT_EXCITATION`, or `UNRESOLVED`, and `result.exit_code` is the matching process exit
code. `result.qualification_json` and `result.qualification_markdown` are the two published files.

The call runs one comparison per zero-change control and one per probe rung per workload, so its
cost is the comparison campaign, not the three workloads it selects. `result.receipt` records both.

```python
receipt = load_and_validate_workload_qualification_receipt(result.qualification_json)
receipt["selected_workload_ids"]
receipt["execution_counts"]["total_comparisons"]
receipt["witnesses"]["first_witness"]
```

`load_and_validate_workload_qualification_receipt` performs full validation when called on the
normal published receipt path. It admits the document as bounded strict JSON from a regular
no-follow file, parses it with a strict typed model that refuses unknown or missing fields at every
level, recomputes every derived claim from the receipt's own configuration and cell records, derives
the owned output root from the published location, re-admits every registered locator as normalized,
relative, unique, and confined, and rebinds the retained raw qualification configuration, every
generated comparison configuration, and every retained comparison receipt to their recorded digests
and to the campaign identity.

A mismatch raises even when `receipt_sha256` was recomputed. That hash detects accidental corruption
of the canonical receipt content; it is not a signature, it authenticates nobody, and resealing a
contradictory receipt does not make it valid.
