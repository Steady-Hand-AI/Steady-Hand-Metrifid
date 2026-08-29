# Compiled artifact certification

`metrifid certify` answers exactly one question:

> Do these two MuJoCo source closures, compiled under this recorded runtime, produce byte-identical complete MJB artifacts?

It is not a comparison. It runs no workload, allocates no `mjData`, and steps nothing. It compiles, serializes and compares bytes.

## Command

```bash
metrifid certify BASELINE_MJCF CANDIDATE_MJCF \
  --output OUTPUT_DIRECTORY \
  [--baseline-root BASELINE_ROOT] \
  [--candidate-root CANDIDATE_ROOT]
```

- Both MJCF arguments are positional and required. `--output` is required.
- No JSON configuration is accepted. There is nothing to configure: the command has no tolerances, no workload and no thresholds.
- When a root is omitted, the entrypoint's real parent directory becomes the model root and its basename becomes the relative entrypoint.
- When a root is supplied, the MJCF must resolve to a regular, nonsymlinked member beneath that real root, and is recorded as a normal relative POSIX entrypoint such as `models/arm/model.xml`.
- Baseline and candidate may be the same path.
- Canonical artifacts carry no absolute paths, no temporary paths and no timestamps.

## Support envelope

`certify` shares one runtime gate with `compare`, `audit-timestep`, and compiled model identity.
There is no separate, narrower Certify envelope.

```text
Python           3.11 or newer; no upper bound; implementation name is not a runtime rejection
operating system POSIX reporting Linux or Darwin, with the required POSIX capabilities present
architecture     never rejected; the machine string is receipt evidence only
MuJoCo package   stable >=3.9, with no minor ceiling
MuJoCo identity  package/native three-part base and native integer agree exactly
validated exact  3.9.0, 3.10.0, 3.11.0, and 3.12.0
NumPy            >=1.26, no runtime upper bound
```

The newest stable MuJoCo is the primary development and release profile—3.12.0 for the frozen
2026-08-22 snapshot. Exact 3.9.0 is the minimum profile; retained 3.10.0 and 3.11.0 results provide
backward-compatibility evidence. Compilation always uses and records the exact admitted runtime.

The package metadata has no artificial Python upper bound and the runtime does not reject an
implementation by name. Release evidence currently uses CPython because the MuJoCo binary wheels
exercised by the release matrix target CPython. The interpreters and platforms CI exercises change
as upstream support does; the matrix in `.github/workflows/ci.yml` is the authority. A tuple is
described as validated only after that exact lane passes. Native Windows is unsupported because the
required POSIX capabilities are absent; WSL is the documented route. Anything the gate refuses is
reported through the environment reason codes, which name the measured fact that failed.

`certify` measures only the compiled-artifact capability surface. A later coherent stable MuJoCo
release can therefore be capability-admitted without being mislabeled validated, even when its
typed `review-model` public-field surface has not yet been characterized. The certificate remains
bound to the exact package, native library, distribution, platform, and MJB identities it observed.

Release evidence is produced from a noneditable wheel, never from the source tree.

A certificate is runtime-bound: it states what the recorded runtime identity produced, and it
carries that identity in the receipt. Comparing digests produced by different operating systems,
MuJoCo versions or builds is outside what the certificate says.

## Statuses and exit codes

| Status | Exit | Meaning |
| --- | --- | --- |
| `CERTIFIED_COMPILED_EQUIVALENCE` | 0 | Every serialized byte matched. |
| `NOT_CERTIFIED_COMPILED_DIFFERS` | 40 | At least one serialized byte differed. |

These are Certify's own. They are not `ComparisonStatus` values and exit 40 exists nowhere else in the product. Refusals keep the accepted exits: 64 for invalid invocation, input or output, 70 for an internal failure. Certify never emits `MATERIAL_BEHAVIOR_CHANGE`, `NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD` or exit 10.

## What is measured

For each role, one at a time:

1. The installed project distribution identity is verified.
2. The Certify runtime envelope is required.
3. Every admitted regular file under the exact model root is measured into the source closure.
4. An immutable same-byte source snapshot is created.
5. The MuJoCo-reached dependency set is discovered separately from that complete source identity.
6. The snapshot entrypoint is compiled under the accepted compile lock and warning and global-callback guard.
7. Compile-only admission is applied.
8. The model is serialized while the closure snapshot is still live.
9. After both roles and all decision evidence are complete, both live source closures are reverified
   immediately before output commit.

The original source tree is never modified. The measured closure members are copied into a private immutable snapshot and compiled from that snapshot.

### Artifact identity

```text
method                    MUJOCO_COMPLETE_MJB_SHA256
mjb_sha256                SHA-256 over every byte of mj_saveModel output
mjb_size_bytes            equal to mj_sizeModel
header_words              the five native C integers MuJoCo wrote
magic_decimal/magic_hex   54321 / 0x0000d431
sizeof_mjtNum             the active build's mjtNum width
mujoco_version_integer    exact active-runtime mj_version() value
runtime_identity_sha256   the recorded runtime this artifact came from
```

Nothing is projected, normalized, rounded or tolerated. An artifact whose length disagrees with `mj_sizeModel`, whose magic word is wrong, whose `mjtNum` width disagrees with the active build, or whose version word disagrees with `mj_version()` refuses as `COMPILED_ARTIFACT_INVALID`. An artifact larger than 512 MiB refuses as `COMPILED_ARTIFACT_SIZE_EXCEEDED`.

Header words three and five are recorded exactly as MuJoCo wrote them and treated as opaque build and layout words. They are properties of the MuJoCo build, not of your model.

The 512 MiB check is a post-compilation serialization and diagnostic bound. It is not a promise about how much memory compiling your model needs.

### Compile-only admission

Certify applies `admit_external_implementation_free_model`, shared with replay. It refuses engine plugins, plugin sensors and user actuator or sensor implementation modes, because those name behavior supplied by code the certificate does not bind.

It deliberately does not apply the replay-only refusals. A model with mocap bodies or history state can be certified: neither affects the compiled artifact. Certify also applies no joint or actuator alignment, no workload representability check and no state, action or tolerance admission, because it has no workload.

Active global MuJoCo callbacks are still refused before compilation, through the accepted compile guard.

## The claim

For a certificate, the machine claim is exactly:

> The measured baseline and candidate source closures, compiled under the recorded runtime identity, produced byte-identical complete MJB artifacts.

This is unconditional and workload-free. It is a statement about artifacts. It does not claim source text equality, license equality, visual intent, task suitability, hardware safety or cross-version portability. Two different source trees can certify; identical sources under a different MuJoCo build may not.

### Behavior implication

The receipt carries, separately labelled and excluded from `decision_sha256`:

> If the certified bytes are loaded under the recorded runtime, neither `mjModel` is modified afterward, complete `mjData` initial state and every external input/state/code surface are identical, and execution is deterministic, then the compiled model cannot be the source of divergence.

Every premise is printed with it. It is an aid to reading the certificate, never part of the decision. In particular, post-certification mutation of `mjModel` is outside the claim entirely.

### Limitations

Every receipt carries all five:

```text
EXACT_RECORDED_RUNTIME_ONLY
POST_CERTIFICATION_MJMODEL_MUTATION_OUTSIDE_CLAIM
EXTERNAL_CODE_AND_INPUT_EQUIVALENCE_NOT_ESTABLISHED
NO_SOURCE_TEXT_LICENSE_OR_VISUAL_INTENT_CLAIM
NO_CROSS_MUJOCO_VERSION_CLAIM
```

## When artifacts differ

The status is fixed by the byte comparison before any field is read. The receipt then adds a descriptive field report over the proven public surface: public non-callable scalar and array members of `MjModel`, plus one level under `model.opt`, `model.vis` and `model.stat`. Paths are sorted lexicographically, and members that cannot be compared are listed in `omitted_fields` with a stable reason rather than dropped.

Each changed field reports both types, dtypes and shapes when arrays, both field SHA-256 digests, the changed element count when computable, and up to eight deterministic sorted index and value witnesses. Values are recorded as exact IEEE-754 binary64 bit patterns, so a `NaN` or infinity survives the receipt intact. Bounds: 100 changed fields returned, 8 witnesses per field, 200 omitted fields returned. `truncated` says when a bound was reached. No array is ever dumped in full and no rename mapping is inferred.

The report is descriptive evidence that locates a difference. It is not an identity policy: no field is ever excluded from the artifact claim because the report cannot describe it.

Two consequences follow from the one-level surface. Every member of `model.vis` is itself a struct, so a purely visual difference has no comparable field one level down. When bytes differ and the public surface finds nothing, the report returns:

```text
NO_PUBLIC_FIELD_DIFFERENCE_IDENTIFIED
```

That is still a valid `NOT_CERTIFIED_COMPILED_DIFFERS` result. The artifacts did differ, and the byte offset and count say where.

A difference tells you the compiled artifacts are not the same. It does not tell you whether that matters. For a workload-bounded decision, run `metrifid compare`.

## Output

Certify publishes `certification.json` and `certification.md` through the same descriptor-confined, no-clobber two-file primitive as `compare`. Output equal to or below either model root refuses before creation, and replacing the public output path cannot redirect publication or cleanup. Success is returned only after both final entries identify Metrifid's retained objects, contain the exact expected bytes, both live source closures still match their immutable snapshots, the snapshot contexts have closed, and one final check confirms that the public path still names the admitted directory and the complete pair still contains the expected bytes. If publication or a later source check fails, Metrifid removes only still-owned private temporaries and returns an error; any public entry already acquired by no-clobber linking is preserved as incomplete evidence and must not be interpreted as a successful receipt pair.

Repeated identical invocations in the same environment publish byte-identical JSON and Markdown. The receipt schema is `metrifid.compiled_equivalence_receipt` with schema version `1`, carries a `decision_sha256` over the decision-bearing members alone and a `receipt_sha256` over the whole document.

## Resource behavior

The two roles are processed strictly sequentially while both lightweight source snapshots remain alive through the decision. Real Menagerie artifacts measured 70.9–119.8 MB in earlier work, so a role's compiled model and its serialized buffer are released before the next role compiles, artifact bytes live in private mode-`0600` files rather than in memory, and the comparison streams fixed-size chunks and accumulates counters instead of collecting differing offsets. Every private file and scratch directory is removed on success and on failure.

## Compiled-artifact confinement

A private artifact's directory entry is removed as soon as its bytes are written, fsynced and measured, so the compiled subject survives only as an open descriptor with no pathname. The byte comparison reads it positionally through that descriptor, and MuJoCo reaches it through the operating system's own descriptor path (`/proc/self/fd` on Linux, `/dev/fd` on Linux and macOS), each admitted only after it is proven to name the exact retained object. A completed decision binds the digest of the exact loaded MjModel serialization and the exact byte streams consumed by complete-MJB comparison. Loading reserializes the exact loaded model and requires its digest to match; the comparison hashes each consumed stream, including the non-overlap tail, and requires both to match. Checking the retained bytes around a consumer would not establish this, because a same-user process can mutate the object between two checks and restore it.

This exists so that every fact in a receipt describes the artifact whose digest that receipt publishes, even when another process running as the same operating-system user can create, rename and replace entries in the temporary directory. Substituting a different object is prevented outright, because no name survives to redirect. Mutating the retained object in place is detected rather than prevented — on Linux a same-user process can still reach it through `/proc/<pid>/fd` — and the reverification makes such a run fail closed instead of completing. It is not a defense against a privileged kernel compromise or against mutation of this process's memory. Reading through the descriptor does not change the resource behavior above: nothing additional becomes resident.
