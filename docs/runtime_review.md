# Runtime Review

Runtime Review answers one narrow replacement question:

> May the exact candidate MuJoCo native profile replace the exact baseline profile for this exact
> source closure, workload, channel set, tolerances, and complete one-second horizon?

New runs use the semantic roles `baseline` and `candidate`. Each role may name any coherent stable
MuJoCo profile at or above 3.9 whose required capabilities are present. The exact package token,
native library, Python, NumPy, host, and worker identities live in evidence rather than in the role
name. Historical schema-version-1 evidence remains limited to `A_3.10.0` / MuJoCo `3.10.0` and
`B_3.11.0` / MuJoCo `3.11.0` and is validated without rewriting its bytes.

Runtime Review applies only the frozen `CONDITIONAL_TAIL_ENVELOPE` method. Neither product path
discovers, creates, installs, upgrades, selects, or repairs an environment, selects a method, or
makes a generic version-compatibility claim.

## Two paths, one referee

- `metrifid run-runtime-review` takes two explicit already-prepared interpreter launchers, measures
  their exact profile identities, proves each profile can reproduce one complete public integration
  state, creates the fixed twelve evidence cells, and immediately calls the existing referee. New
  runs emit only schema-version-2 evidence.
- `metrifid review-runtime` starts from twelve evidence cells that were already retained. It routes
  schema version 1 through immutable historical validation and schema version 2 through the
  role-based production validator; unknown versions refuse.

Both paths finish with the same strict evidence admission, selected method, receipt, Markdown, exit
mapping, and independent portable validation. The execution path adds no second scientific decision.

## Create and immediately decide the evidence

Copy [`examples/runtime_review_run`](../examples/runtime_review_run/), replace its two placeholder
absolute interpreter paths, and run:

```bash
metrifid run-runtime-review runtime_review_run.json
```

The two environments must already exist and must use the same exact NumPy package token and
distribution payload; there is no product-level NumPy version pin. Metrifid runs a baseline
preflight and then a candidate preflight. Each preflight compiles the fixture and runs a same-profile
sentinel over the complete public `mjSTATE_INTEGRATION` state. Both sentinels must pass before the
first of exactly twelve sequential one-shot cells starts in role/grid/repeat order.

Metrifid uses its packaged frozen worker and a fixed environment allowlist, never a shell,
caller-supplied arguments, hidden retries, or fallback interpreters. Every preflight, sentinel,
command, output stream, exit code, identity, and evidence directory is preserved under the new
absent `output_dir`. A sentinel failure starts zero cross-profile cells and publishes no scientific
migration status.

After all twelve cells pass immediate admission, Metrifid derives the retained-evidence
configuration from the first admitted result, strictly reloads it, and calls
`review_runtime_configuration_file`. A partial/refused/timed-out run keeps its diagnostic bytes but
publishes neither `runtime_review_run.json` nor a completed scientific decision.

## Decide evidence retained elsewhere

1. Create two external scientific profiles with the same Python build, exact NumPy distribution,
   host, CPU, and frozen thread settings. Each MuJoCo package/native pair must be coherent, stable,
   at least 3.9, and expose the capabilities the worker uses. Metrifid need not be installed in these
   profiles.
2. Use the separately supplied native-evidence worker to produce all twelve cells: two profiles,
   step sizes `0.004`, `0.002`, and `0.001`, and repeats `0` and `1`. Preserve each six-member output
   directory byte-for-byte.
3. Install the Metrifid wheel in a product environment and review the strict configuration:

   ```bash
   python -m pip install metrifid
   metrifid review-runtime runtime_review.json
   ```

The [retained-evidence example configuration](../examples/runtime_review/runtime_review.json) is an
explicit historical schema-version-1 specimen. It carries the retained smooth-pendulum identities
but intentionally uses placeholder evidence paths. Replace those paths with the matching historical
evidence; do not edit it into a version-2 claim. `run-runtime-review` generates a version-2
configuration only after it measures both exact profiles and their sentinels, so no static example
can truthfully prefill those hashes.

## Configuration and evidence

The configuration schema is `metrifid.runtime_review_config`. Version `1` is the immutable historical
route; version `2` is the role-based production route. Dispatch is explicit, and unknown versions or
alternate field sets refuse. `required_horizon` is the string `"1"`; the step and repeat sets are
fixed; and the twelve cell locators must be relative portable paths beneath the configuration
directory. Absolute paths, traversal, symlinks, duplicate slots, and output/evidence overlap refuse.

Each version-2 profile declaration binds its semantic role, exact package token, native version
string and integer, profile-identity self-hash, and portable identity-file locator. The owned profile
identity also binds the MuJoCo and NumPy distribution payloads, matching native-library bytes,
Python build, host/libc/thread settings, packaged worker, fixture compile smoke, and sentinel.

Each cell contains exactly:

```text
CHECKSUMS.sha256
fixture.xml
input_manifest.json
model.mjb
result.json
trace.npz
```

The checksum file covers the other five members exactly. Runtime Review validates bounded strict
JSON and safe NPZ content, then binds role, grid, repeat, subject, workload, channel/tolerance/time
layout, compiled artifact, and runtime identity. The two repeats must agree exactly. Only the declared
MuJoCo distribution and native-library identities may differ between profiles.

For version 2, all six cells for one role must agree exactly with the role's profile, runtime,
worker, NumPy, and sentinel identities. Baseline and candidate must carry the same exact NumPy
version and payload identity. A same-version baseline/candidate pair is valid when the two explicit
profile identities and every required invariant agree.

## Live support and retained exact validation

A successfully admitted live runtime reports
`ADMITTED_CAPABILITY_COMPATIBLE_PROFILE`. Metrifid does not turn that into
`VALIDATED_EXACT_PROFILE` because a base version happens to have passed an earlier campaign. Post,
local, vendor, and byte-different builds therefore stay capability-compatible.

Release validation may separately retain `VALIDATED_EXACT_PROFILE` for the exact measured tuple it
tested, including the MuJoCo distribution payload, native-library SHA-256, Python build, and platform
identity. This is an evidence record, not a live product registry. Product receipts bind their exact
runtime identities regardless of that label.

## Completed decisions

| Status | Exit | Meaning |
| --- | ---: | --- |
| `WITHIN_DECLARED_MIGRATION_ENVELOPE` | 0 | Every witness is within tolerance across the complete horizon, with no failed gate or suffix. |
| `INSUFFICIENT_EVIDENCE` | 20 | Repeatability, solver, contact topology, asymptotic, or full-prefix evidence is insufficient. |
| `UNRESOLVED_NEAR_BOUNDARY` | 30 | Complete evidence remains too close to a declared boundary to decide. |
| `OUTSIDE_DECLARED_MIGRATION_ENVELOPE` | 40 | A decisive witness rejects replacement, including when a later suffix is unqualified. |

Malformed configuration, identity mismatch, missing or changed bytes, and unsafe filesystem content
are operational refusals (exit `64`). Unexpected product failure is exit `70` and is never a
completed scientific decision.

Exit zero requires all four conditions: status WITHIN, admitted prefix equal to the required horizon,
no first failing gate, and no unqualified suffix. `first_decisive_witness` is the earliest OUTSIDE or
UNRESOLVED witness for those final statuses; it is null for WITHIN and INSUFFICIENT.

## Owned output and validation

The command atomically owns `<output_dir>/runtime_review/` and publishes:

```text
runtime_review.json
runtime_review.md
admitted_runtime_review_config.json
evidence/baseline/<step>/repeat_<id>/...
evidence/candidate/<step>/repeat_<id>/...
profile_identities/baseline.json        # version 2 only
profile_identities/candidate.json       # version 2 only
```

The evidence members are copied, never linked, and rechecked before publication. The public loader
requires the exact closed root/evidence hierarchy, regenerates the Markdown from the canonical
receipt, and reconstructs the decision from the copied raw traces and diagnostics rather than
trusting receipt status fields:

```python
from metrifid.runtime_review import load_and_validate_runtime_review_receipt

receipt = load_and_validate_runtime_review_receipt(
    "runtime_review_output/runtime_review/runtime_review.json"
)
```

Publication uses an atomic no-replace directory operation. Failure handling is deliberately
non-destructive: if admission or prepublication replay fails, `.runtime_review.staging` may remain;
if a failure occurs after publication, `runtime_review` may remain. Neither tree is a completed
decision unless the SDK or CLI returns it successfully. Inspect and manually remove retained
failure evidence, or choose a new absent output path, before retrying. Metrifid never recursively
deletes these trees during failure handling because a concurrent process could have substituted
caller-owned bytes at a pathname.

Original paths remain history only; the owned output stays verifiable after the original input tree
is removed. The receipt self-hash detects corruption but is not a signature or proof of origin.

## Execution record and claim boundary

A completed version-2 execution adds `runtime_review_run.json`, a canonical self-hashed operational
record of the two preflights, two sentinel passes, twelve first attempts, packaged resource
identities, generated configuration, and final receipt. It is written last and is not a second
migration algorithm or proof of origin. There are no automatic retry or replacement attempts.

The decision is limited to the exact two profiles, source closure, compiled artifacts, workload,
initial state, actions, channel layout, observation times, and tolerances recorded in the receipt. It
does not establish universal MuJoCo equivalence, hardware or policy safety, task success, real-world
transfer, another backend, evidence authenticity, or cryptographic authorization. Platform-sensitive
interval diagnostics are explanatory Markdown only; stable classifications and witness identities are
decision-bearing JSON.
