# Static Model Change Gate

`metrifid review-model` classifies the complete, ordered set of statically observed changes between
two compiled MuJoCo models under one bounded maintainer policy. It links that classification to a
Certify receipt built from the exact same private complete-MJB artifacts.

This is a static policy decision, not a release approval. It allocates no `mjData`, steps no model,
and makes no claim of dynamic equivalence, controller equivalence, task suitability, hardware
safety, deployment safety or operational readiness.

## Command and SDK

```bash
metrifid review-model BASELINE CANDIDATE \
  --policy MODEL_RELEASE_POLICY.json \
  --output OUTPUT_DIRECTORY \
  [--baseline-root BASELINE_ROOT] \
  [--candidate-root CANDIDATE_ROOT]
```

All four main arguments are required. Model-root resolution and admission have the same meaning as
for `metrifid certify`. Output must be outside both model roots.

`review-model` inherits Certify's installed-distribution check and native runtime envelope. Both
models are compiled sequentially under the admitted runtime, and the embedded Certify receipt
records that exact compilation and runtime identity. See
[`compiled_certification.md`](compiled_certification.md) for the supported environment and
complete-MJB identity contract.

The public Python entrypoint is:

```python
from metrifid.model_release import review_model_release

result = review_model_release(
    "baseline/model.xml",
    "candidate/model.xml",
    "model_release_policy.json",
    "review-output",
    baseline_root="baseline",
    candidate_root="candidate",
)
```

`result.status`, `result.receipt`, `result.receipt_sha256`, `result.model_release_json` and
`result.model_release_markdown` are public. A completed CLI invocation writes one canonical JSON
summary line containing `status`, `receipt_sha256`, `model_release_json` and
`model_release_markdown`. The two filename values are the path-independent literals
`model_release.json` and `model_release.md`; the SDK result retains the caller's actual output
paths.

## Policy schema

Every field is required; JSON `null` is a value, not an omitted field.

```json
{
  "schema": "metrifid.model_release_policy",
  "schema_version": 1,
  "baseline_compiled_sha256": "<64 lowercase hexadecimal characters>",
  "candidate_compiled_sha256": "<64 lowercase hexadecimal characters or null>",
  "rules": [
    {
      "id": "allow-arm-mass",
      "effect": "ALLOW",
      "selector": {
        "object_type": "body",
        "object_name": "arm_link",
        "field": "mass",
        "change_kind": "MODIFY"
      },
      "before_sha256": "<64 lowercase hexadecimal characters or null>",
      "after_sha256": "<64 lowercase hexadecimal characters or null>"
    }
  ]
}
```

Effects are `ALLOW`, `REQUIRE` and `FORBID`. Change kinds are `ADD`, `REMOVE` and `MODIFY`.
`object_name` is either one exact name or the whole token `"*"`; partial globs do not exist.
Before and after digests constrain the selected value when non-null. A selector match whose digest
constraint does not match is `UNDECLARED`.

A compiled MuJoCo object whose literal name contains `*` cannot be confused with policy wildcard
syntax. It is omitted from the named semantic layer and any differing artifact is forced through
the opaque `REVIEW_REQUIRED` coverage witness.

`REQUIRE` is intentionally stricter: it must use an exact object name. A `MODIFY` requirement binds
both exact value digests; `ADD` binds exact absence before plus the exact value after; `REMOVE`
binds the exact value before plus exact absence after. An unsatisfied `REQUIRE` rule makes the
completed decision outside policy even if no observed change was forbidden.

### Closed selector fields

| `object_type` | Admitted `field` values |
| --- | --- |
| `body` | `presence`, `parent`, `mass`, `inertia` |
| `joint` | `presence`, `body`, `type`, `limited`, `range` |
| `geom` | `presence`, `body`, `mesh` |
| `mesh` | `presence`, `compiled_geometry_sha256` |
| `actuator` | `presence`, `transmission`, `targets` |
| `compiled_field` | `value` |
| `opaque` | `compiled_artifact` |

For `compiled_field`, `object_name` is the exact path in the frozen public-field registry recorded
in the receipt. An opaque residual is always forced to `UNDECLARED`; an `opaque` rule cannot turn
unexplained complete-MJB evidence into an allowed change.

Actuator `targets` contain the shared compiled-descriptor transmission token and ordered typed
object-name references. Raw MuJoCo numeric target IDs are never policy identities, so declaration
reordering cannot conceal a retargeted joint, site, tendon, or body.

Any change to MuJoCo's compiled name-to-ID mapping fields forces an opaque `REVIEW_REQUIRED`
witness. This conservative fence prevents reordered named objects from exchanging an otherwise
unprojected per-ID property while leaving that raw public array byte-identical. Such a mapping
change cannot be converted to a positive result by allowing the `names`, `names_map`, or
`name_*adr` compiled-field rows.

Consequently, a named-object `ADD`, `REMOVE`, rename, or reorder can satisfy an exact `REQUIRE`
rule yet still leave the overall result `REVIEW_REQUIRED`: those operations also change the
compiled name/ID mapping. The presence rule remains useful as a mandatory structural assertion,
but it is not a way to bypass this conservative coverage fence.

### Admission and ambiguity refusal

Policy admission is pure Python and fail-closed. It rejects unknown or missing members, duplicate
JSON member names, raw floating-point tokens, invalid enum values or digests, duplicate rule IDs,
duplicate selectors, and ambiguous wildcard/exact overlap. Two selectors overlap when they have
the same object type, field and change kind and one uses `"*"` where the other names an object.

The admitted bounds are:

| Item | Bound |
| --- | --- |
| Raw policy size | 1 MiB |
| Rules | 4,096 |
| JSON nesting depth | 8 |
| JSON nodes | 100,000 |
| Each string token | 256 UTF-8 bytes |

Accepted rules are canonicalized by selector and ID. The receipt records both the exact raw policy
SHA-256 and a canonical semantic SHA-256.

## Exact compiled-subject binding

The policy baseline SHA-256 must exactly equal the complete baseline MJB measured during this run.
A mismatch is an input refusal, not a completed policy decision.

An exact candidate binding is required for a positive decision about differing compiled artifacts.
When the MJBs differ and `candidate_compiled_sha256` is `null`, the gate adds a forced
`UNDECLARED` opaque residual with reason `candidate_compiled_subject_unbound`. When a non-null
candidate digest does not match, it adds the same fail-closed residual with reason
`candidate_compiled_subject_mismatch`. Thus an unbound discovery policy cannot authorize a
different candidate, and a policy written for one candidate cannot silently authorize another.

Leaving the candidate null is useful for discovery and is harmless for a true no-change result.
It is not an approval mechanism.

## What one change means

The gate combines three static evidence sources:

1. named semantic objects (`body`, `joint`, `geom`, `mesh`, `actuator`);
2. every value change in a frozen registry of public `MjModel` fields; and
3. a fail-closed opaque residual when the complete MJB differs but subject binding or explanation
   coverage is incomplete.

Name every authored body, joint, geom, mesh and actuator. An MJB difference with incomplete
semantic name coverage receives an opaque residual even when other typed changes are available.

One source edit can create several policy-bearing compiled changes. For example, changing a geom's
mass can change the semantic body's `mass` and `inertia` and multiple derived `compiled_field`
values. Policy must cover every observed derived row; declaring only the source-level intention is
not sufficient. Additions and removals use one `presence` row instead of fabricating field pairing.

Every row records its exact selector, source, classification, matching rule ID or null, before and
after digests, bounded values or metadata, and details. Ordering is deterministic. At most 10,000
compiled changes are admitted; exceeding the bound refuses at exit 64 rather than truncating the
decision. Before publication, the producer also round-trips the complete serialized receipt through
the same 64 MiB, one-million-node public reader used by downstream consumers.

## Classification, status and exit precedence

Each observed row receives exactly one classification:

| Classification | Meaning |
| --- | --- |
| `ALLOWED` | One `ALLOW` rule matched the selector and digest constraints. |
| `REQUIRED` | One `REQUIRE` rule matched; the required declaration is satisfied. |
| `FORBIDDEN` | One `FORBID` rule matched. |
| `UNDECLARED` | No rule matched exactly, a digest constraint differed, or the row is a forced opaque residual. |

Completed status is selected in this fixed order:

1. Any `FORBIDDEN` row **or any missing `REQUIRE` rule** yields
   `OUTSIDE_DECLARED_POLICY`.
2. Otherwise, any `UNDECLARED` row yields `REVIEW_REQUIRED`.
3. Otherwise, zero rows yields `NO_COMPILED_CHANGE`.
4. Otherwise, all rows are declared and the result is `WITHIN_DECLARED_POLICY`.

| Completed status | Exit | Meaning |
| --- | --- | --- |
| `NO_COMPILED_CHANGE` | 0 | The complete MJBs matched and no compiled change row exists. |
| `WITHIN_DECLARED_POLICY` | 0 | Every observed change is exactly allowed or required. |
| `REVIEW_REQUIRED` | 40 | At least one change remains undeclared. |
| `OUTSIDE_DECLARED_POLICY` | 40 | A forbidden change or missing required declaration exists. |

Invalid invocation, policy, model, runtime or output is a refusal at the existing exit-64 boundary.
An internal operational failure uses exit 70. Refusals do not publish a completed model-release
receipt.

## Output and independent reading

A completed review publishes, without clobbering existing entries:

```text
model_release.json
model_release.md
```

The JSON schema is `metrifid.model_release_receipt`, version `1`. It contains:

- the completed status and exit code;
- the full linked Certify receipt plus its receipt and decision SHA-256 identities;
- the full admitted policy, rule count, raw identity and semantic identity;
- the frozen public-field registry identity and field count;
- `changes_complete: true`, the complete ordered `changes` array and classification counts;
- satisfied and missing required rules;
- the first unexpected and first missing-required witnesses;
- the static claim, `dynamic_behavior_claim: "NO_DYNAMIC_BEHAVIOR_CLAIM"`, and mandatory
  limitations;
- a decision SHA-256 and whole-receipt SHA-256.

The public `load_and_validate_model_release_receipt` and `validate_model_release_receipt` readers
recheck the schema, embedded Certify receipt, hashes, rule classification, ordering, summaries,
subject linkage, witnesses and claims without importing MuJoCo or NumPy. Revalidation establishes
internal consistency and linkage; it does not recompile external source artifacts.

Canonical receipts contain no absolute model source paths, output paths, temporary paths or
timestamps. Source-closure evidence uses admitted relative entrypoints and content identities.

## Compiled-subject binding

Every decision fact in a completed result comes from the exact compiled objects the embedded
Certify receipt identifies. The byte comparison, the descriptive field report, the semantic
compiled snapshots, the policy classification and both first witnesses all read the same two
subjects.

Each role's complete MJB is serialized into a private mode-`0600` file created exclusively in a
per-run temporary directory. Its length and SHA-256 are measured as it is written, and its only
directory entry is then removed. What survives is a descriptor: the compiled artifact exists as an
open file with no pathname. Every later consumer reads that descriptor, either through positional
reads or, where MuJoCo's loader requires a name, through the operating system's own descriptor
path, which the kernel resolves from this process's descriptor table rather than from any
directory entry.

Removing the name stops one process substituting a different object for the subject. It does not by
itself bind a consumer to the bytes it read. Checking the retained bytes before and after a consumer
proves only that they were correct at two instants: a process running as the same operating-system
user can mutate the object in between and restore it before the second check, and on Linux it can
reach the nameless object through `/proc/<pid>/fd` to do so. On macOS, which has no `/proc`, it
cannot. That is the same reasoning that rejects hashing a pathname before and after its consumers.

The binding therefore runs through the consumers themselves:

> A completed decision binds the digest of the exact loaded MjModel serialization and the exact byte streams consumed by complete-MJB comparison.

Loading a model reserializes that exact `MjModel` and requires the result to reproduce the
receipt-bound size and SHA-256, which bytes restored after the load cannot repair. The complete-MJB
comparison hashes each role's stream as it consumes it, including the non-overlap tail of the longer
artifact, and requires both observed digests to equal their receipt-bound digests before any
comparison is returned. A mismatch at either consumer fails the run closed with no completed pair.

No protection is claimed against a privileged kernel compromise, against arbitrary mutation of this
process's memory, or against an adversary who can modify the installed distribution itself. This
binding covers the compiled artifacts. The admitted source closures carry their own separate
identity and are reverified independently before publication.

Both `/proc/self/fd` and `/dev/fd` are probed, and each is admitted only after it is proven to name
the exact retained object. Linux publishes both; macOS publishes `/dev/fd`. A platform that offers
neither is refused rather than silently sent back to a pathname.

This binding does not change the resource contract described in
[`compiled_certification.md`](compiled_certification.md): no artifact is read into memory, and the
roles are still compiled and serialized strictly one at a time.

Publication uses the shared descriptor-confined, no-clobber two-file primitive. A call succeeds
only after both retained outputs have the expected bytes and identities. If publication begins and
a later verification fails, an already acquired public entry can remain as incomplete evidence;
only a completed return and verified pair constitute a completed result.

## Trustworthy two-pass use

For a new compiled difference:

1. Run public `certify_models` (or `metrifid certify`) to learn the exact baseline and candidate
   complete-MJB SHA-256 identities.
2. Write a discovery policy with the exact baseline digest, `candidate_compiled_sha256: null`, and
   an empty rule list. Run `review-model` into a fresh output directory and expect
   `REVIEW_REQUIRED` for a differing candidate.
3. Review every non-opaque returned row. Investigate any opaque residual other than the deliberate
   unbound-candidate witness; opaque evidence cannot be policy-allowed away.
4. Write a new policy bound to the exact candidate digest. Prefer exact selectors and exact before
   and after digests. Declare every accepted derived compiled change, not only the intended source
   edit.
5. Run again into another fresh output directory. `WITHIN_DECLARED_POLICY` means the observed
   static change set matches that policy; it does not authorize a release or establish behavior or
   safety.

[`examples/model_release/`](../examples/model_release/) implements this sequence using only public
SDK imports and canonical standard-library JSON.
