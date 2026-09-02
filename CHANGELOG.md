# Changelog

Notable changes to `metrifid`. Starting with `0.2.0`, release identifiers use exactly three
dot-separated integer components: `MAJOR.MINOR.PATCH`. Historical development entries below
remain as an audit trail and do not define the current release version.

## 0.7.2

### Added

- The Metrifid Certify composite action accepts an optional `strict` input. It defaults to `"true"`
  and fails the step when the compiled models differ. Setting it to `"false"` softens only that one
  completed outcome, exit 40, so the caller can decide from the `exit_code` and `status` outputs.
  An invalid `strict` value, an invocation refusal (exit 64), an operational error (exit 70), and
  any unexpected exit remain fatal in both modes.
- The README documents the action's inputs, outputs, and strict-mode boundary.

### Changed

- The source distribution no longer ships `tests/contract/test_ci_quality_environment.py`. That
  contract reads repository-root files that are deliberately not packaged, so it could never run
  from an extracted archive. Every test that ships is executable there.

### Unchanged

- The Python API, the command-line surface, receipt schemas and content, runtime admission, MuJoCo
  dependency resolution, and model-comparison behavior are all unchanged.

## 0.7.1

A packaging-only correction to dependency resolution on Intel macOS. No API, schema, receipt
format, runtime-admission rule, filesystem policy, CLI surface, or product behavior changes, and
the set of supported operations is unchanged.

### Fixed

- An ordinary `pip install metrifid` on Intel macOS (`Darwin`, `x86_64`) read the unconditional
  `mujoco>=3.9` requirement, selected a MuJoCo release for which upstream publishes no Intel macOS
  wheel, fell back to the MuJoCo source distribution, and failed because that source build requires
  `MUJOCO_PATH`. The MuJoCo requirement is now expressed as two complementary environment-marked
  requirements: `>=3.9,<3.11` on Darwin `x86_64`, and `>=3.9` with no ceiling everywhere else. The
  ceiling reflects upstream Intel macOS wheel availability and still permits compatible 3.9 and 3.10
  patch releases.

### Unchanged

- Runtime admission remains capability-based and architecture-neutral for coherent stable MuJoCo
  3.9 or newer. The new bound applies to automatic dependency resolution on Darwin `x86_64` only,
  and does not narrow which runtimes Metrifid will admit or validate against. A Python process
  running as `x86_64` under Rosetta follows the Intel branch, because that is the architecture its
  wheels must match.

## 0.7.0

The first release since `0.2.1`. It adds four commands, replaces the exact MuJoCo version wall with
rolling runtime admission, and rebuilds the evidence boundaries so a published receipt can be
revalidated against the bytes it was actually decided from.

### New commands

- **`metrifid review-model`** and `metrifid.model_release` classify the complete, ordered set of
  statically observed changes between two compiled models under one bounded maintainer policy, and
  link that classification to a Certify receipt built from the exact same private complete-MJB
  artifacts. It is a static policy decision, not a release approval: it allocates no `mjData`, steps
  no model, and claims no dynamic, controller, task, hardware, deployment or operational
  equivalence.
- **`metrifid qualify-workload`** and `metrifid.workload_qualification` answer whether the workloads
  you selected detect every declared model perturbation at or above its required magnitude, under
  the existing comparison tolerances, and which perturbations stay blind. Every detection cell is a
  completed `metrifid compare` run against a probe model you supply. Metrifid never mutates model
  source and never infers detectability from a sensitivity or estimability model.
- **`metrifid review-runtime`** and `metrifid.runtime_review` read one exact twelve-cell,
  three-grid/two-repeat owned evidence set and apply the selected conditional tail envelope to one
  full-horizon native-profile replacement question. The command runs no MuJoCo worker and makes no
  claim outside the declared profiles, source closure, workload, and tolerances.
- **`metrifid run-runtime-review`** and the lazy `run_runtime_review_configuration_file` SDK produce
  that evidence set. You supply two already-prepared Python profile launchers and one strict
  self-contained subject/workload manifest; Metrifid measures both profiles and runs the twelve
  cells with the packaged frozen worker, then calls the same referee. It never discovers, creates,
  selects, installs, upgrades, or repairs an environment.

### MuJoCo runtime support

- The exact MuJoCo 3.10 dependency wall is replaced by `mujoco>=3.9` with rolling runtime
  admission. Every live coherent profile is reported as capability-compatible. Exact validation is
  a separate retained release-evidence claim over the measured package payload, native library,
  Python, and platform tuple; it is never inferred from the package's base version.
- The newest stable resolver-selected MuJoCo is the active development, quality, clean-install and
  full-suite authority, with exact 3.9.0 as the support floor and 3.10.0 and 3.11.0 retained as
  backward-compatibility profiles. The frozen current authority is 3.12.0.
- The native-library digest is bound to the exact version coherent runtime admission reports. Stale
  older libraries are ignored; a missing, linked-only, or duplicate exact match refuses.
- Runtime admission is separate from claim coverage. Missing call-graph capabilities and
  unsupported actuator or public-field feature surfaces produce typed, teachable operational
  refusals before any completed result is published. `certify` may still identify complete compiled
  bytes when the narrower typed `review-model` projection is not characterized.
- `review-model` field registries are bound to the exact runtime base version. Historical schema-v1
  receipts still validate purely, while future or cross-version registry substitution is rejected.

### Workload Qualification decisions

- A probe ladder is adjudicated by suffix: a detection floor exists only when a rung and every
  larger rung are detected, so a lone detection with a gap above it establishes no threshold. Probe
  groups resolve to `QUALIFIED`, `UNRESOLVED` or `INSUFFICIENT`; a run resolves to
  `QUALIFIED_FOR_DECLARED_PROBES` (exit 0), `PARTIALLY_QUALIFIED` (exit 20),
  `INSUFFICIENT_EXCITATION` (exit 20) or `UNRESOLVED` (exit 30).
- The workload subset is selected by exact enumeration. Schema version 1 admits three to sixteen
  candidates and freezes the budget at three, so the search evaluates at most 560 subsets and needs
  no sampling, optimizer, or host-timed bound.
- A workload whose zero-change control is not a no-material-difference result is excluded, with the
  exact control receipt and reason recorded. `COVERAGE_INSUFFICIENT` and `NONDETERMINISTIC_REPLAY`
  are never converted into a non-detection.
- `magnitude_semantics` is required on every probe group, preserved exactly and never interpreted.
  Each rung closure must differ from the baseline closure and be unique within its group. The
  `USER_DECLARED_PROBE_SEMANTICS_NOT_VERIFIED` limitation replaces any claim that a supplied probe
  is necessarily the perturbation you meant.
- The status-bearing witness explains its status. A green run publishes none; an unresolved run
  publishes the first unresolved rung at or above the requirement; an insufficient or partially
  qualified run publishes the first blind rung. A zero-change false alarm is a separate eligibility
  warning, and rungs below the requirement no longer become the first witness.
- `planned_comparisons` is reported beside the actual execution counts, with the formula and the
  schema-v1 maximum of 2064 documented.
- `metrifid.workload_qualification.__all__` is frozen to six names: `qualify_configuration_file`,
  `QualificationResult`, `QualificationStatus`, `QualificationExitCode`,
  `WorkloadQualificationOperationError` and `load_and_validate_workload_qualification_receipt`. The
  configuration, probe, workload, cell, group, limitation and cardinality types are not public:
  several of their constructors required internal types, so exporting them would have committed the
  distribution to a surface it could not support. The strict JSON configuration file remains the
  only supported campaign input.

### Evidence, receipts and output ownership

- The public workload-qualification receipt loader parses a strict typed aggregate model, recomputes
  every derived claim from the receipt's own configuration and cell records, and rebinds the
  document to the retained comparison configurations and receipts under its owned output root.
  Editing a decision-bearing field and recomputing `receipt_sha256` no longer produces an accepted
  receipt. The self-hash detects accidental corruption; it is neither a signature nor a semantic
  check, and the documentation says so plainly.
- The typed `ComparisonReceipt` is kept for every completed cell, validated immediately, and bound
  to the exact generated configuration bytes and retained receipt bytes. Cross-cell campaign
  invariants — one tool build, one runtime, one baseline closure, one candidate closure per rung,
  one artifact identity per workload, one contract per workload, and per-role declared timesteps —
  are checked before any subset is selected, and the retained files are re-read from the owned tree
  immediately before publication.
- Semantic identifiers are no longer storage paths. Evidence lives under fixed-width ordinal
  directories derived only from admitted sequence position, and each record carries a normalized
  relative locator as data, so an absolute or traversing identifier cannot place evidence outside
  the owned output root.
- Output ownership is decided before anything is created. A proposed output equal to or inside any
  baseline or probe model root is refused in both canonical and symlink-alias form, a pre-existing
  root is refused, and the owned tree is created and written only through retained directory
  descriptors with no-clobber creation.
- The owned root is bound one component at a time, requiring the object named and the object opened
  to be the same, so a directory replaced between those two observations is refused rather than
  written into. Publication is handed that retained descriptor rather than a pathname, so replacing
  a public path can cause a refusal but cannot redirect a write.
- Reading published evidence back binds the owned root first and reads the aggregate receipt and
  every registered member through that one descriptor. Retained reads are bounded by the caller's
  declared limit, admit regular files only, open without blocking, and compare the file observation
  before and after the read, so a member replaced, truncated, grown or rewritten mid-read is
  refused. Recorded absolute paths are historical coherence metadata, so an honest output tree
  copied elsewhere and renamed still revalidates.
- The qualification configuration, the published receipt and every linked comparison document are
  admitted as bounded strict JSON from regular no-follow files, and every declared path is admitted
  as a normalized traversal-free relative POSIX path resolved at its declared base. An entrypoint
  resolves against its own model root.
- `workload_qualification.json` and `workload_qualification.md` are sealed, linked into place
  without clobbering anything, and verified as a pair; the run reports success only after both
  expected files pass that final verification. A later refusal may leave already linked public
  diagnostic evidence behind, so filenames alone never establish success. Every compare receipt the
  decision used is retained, and the real execution counts are reported rather than the count of
  selected workloads.

### Runtime Review evidence

- Schema version 1 is preserved as the immutable historical exact 3.10/3.11 route. New runs emit
  schema version 2, use only `baseline` and `candidate` roles, admit coherent stable MuJoCo profiles
  at or above 3.9, and bind exact package, native, Python, NumPy, host and worker identities instead
  of encoding versions in role names.
- Each v2 profile preflight must pass a same-profile complete-integration-state sentinel using
  public `mjSTATE_INTEGRATION`, `mj_getState` and `mj_setState` before any cross-profile cell
  starts. A failed sentinel retains operational evidence, starts zero cells, and publishes no
  scientific migration verdict.
- The fixed NumPy completion wall is removed for v2. Baseline and candidate must instead expose the
  same exact NumPy version and distribution payload identity; Metrifid never installs or changes
  NumPy.
- Every profile preflight, one-shot worker command, stream, exit code, fresh identity and six-member
  evidence directory is preserved in a no-clobber run root. A self-hashed operational run record is
  published last, only after the portable scientific receipt validates. Partial failures remain
  diagnostic evidence and never become a completed decision.
- The canonical receipt and its owned evidence snapshot can be independently revalidated after the
  original input tree is removed. Completed partial-prefix and failed-gate evidence is never
  reported as a green replacement decision.
- The completed tree is published with an atomic no-replace directory operation. Failure handling is
  deliberately non-destructive: a private staging tree or an already published tree may remain as
  incomplete evidence for inspection, and cleanup never recursively deletes bytes another process
  could have substituted.
- Final publication verification and standalone receipt replay both enforce the complete closed
  output hierarchy, single-link regular members, and an exact Markdown rendering of the canonical
  receipt. Extra, missing, linked or substituted output content refuses.
- IEEE-754 signed zero is preserved in the complete-integration-state evidence contract. The
  schema-version-2 worker encodes `-0.0` as the canonical token `-0.0` instead of collapsing it to
  `0`, and the independent validator accepts exactly the two canonical zero tokens while refusing
  noncanonical spellings. Complete-state boundaries compare declared shape and exact little-endian
  binary64 bytes rather than `numpy.array_equal`, which treats `+0.0` and `-0.0` as equal even
  though their retained bytes and SHA-256 identities differ. A one-ULP nonzero difference is still
  rejected, and the immutable schema-version-1 worker identity and its historical evidence route are
  unchanged.
- Private selected-method refusals use the case's declared finite horizon rather than a fixed
  one-second endpoint.

### Model Review

- Compiled identity-map deletions remain a fail-closed review in the complaint-backed gallery, case
  expectations reflect the accepted Model Review boundary, one verified retained-MJB load is retried
  after an observed transient descriptor-path failure, and runtime-version tamper tests stay
  effective on the minimum supported MuJoCo profile.

### Public cases, examples and documentation

- Six complaint-backed public cases under `examples/public_cases/`, each an independently authored
  MJCF mechanism analogue built around a mechanism a real person publicly reported: collision-mask
  flattening, explicit contact-pair loss, contact-exclusion loss, mesh-inertia mode change, actuator
  transmission frame change, and force/torque sensor reattachment. Each case runs the complete
  product journey — Certify, a discovery Model Review with an empty policy, and a declared review
  built mechanically from that discovery receipt — alongside an independent direct MuJoCo control
  that observes the same compiled mechanism without using Metrifid. The cases are analogues, not
  upstream reproductions, and they execute no Newton, USD, Isaac or Robosuite code.
- `examples/public_cases/run_all.py` runs all six against an installed Metrifid distribution rather
  than a source-tree import, and publishes per-case receipts, a stable cross-runtime projection, a
  human-readable summary and a checksum manifest. It refuses an output path that already exists,
  runs every case to completion before judging any of them, and reports a divergence without ever
  printing a passing token.
- `examples/sdk/workload_qualification_api.py` is a complete installed-wheel SDK journey that builds
  its own inputs, runs one bounded campaign, and revalidates the published receipt.
- `docs/public_cases.md` documents what each case exercises, what its independent control measures,
  and what none of them claims, together with the external Robosuite Jaco study measured at one
  pinned upstream commit on the newest characterized runtime.
- The public documentation is organised around the change you actually made rather than around the
  command list. The README, the getting-started guide and the capabilities document open with two
  primary routes — a model or asset changed, or the MuJoCo runtime changed — and lead from the first
  result to the command that answers the next question. The capabilities document describes the
  seven-command surface, and `project.urls` Documentation and Changelog point at the main branch.
- Markdown-safe rendering of user labels is centralized, so a semantic label cannot become a table
  cell, heading, fence terminator, raw HTML block or filesystem locator. Canonical JSON still
  carries the exact admitted string.
- `pytest-check` and `pytest-xdist` are part of the development extra. The Workload Qualification
  suites use soft checks with semantic messages, alongside a seeded 5,000-campaign independent
  decision oracle.

## Superseded development entries — 0.4.0.dev0 candidate

- Begin the non-release Metrifid 0.4.0 Model Change Gate candidate. This development identity does
  not authorize publication, tagging, or replacement of the immutable public 0.2.1 release.
- Bind every Model Change Gate and Certify decision consumer to the exact compiled subjects the
  embedded receipt identifies. A private complete MJB is now sealed and its directory entry removed
  as soon as it is written, so the artifact survives only as a retained descriptor with no pathname
  a same-user process could rename or replace. Byte comparison, the descriptive field report and
  the semantic compiled snapshots all read that descriptor. Previously these consumers reopened a
  mutable pathname after the receipt had already measured it, so a completed decision could be
  derived from bytes the receipt never identified. Public statuses, exit codes, receipt schemas and
  CLI/SDK behavior are unchanged, as is the documented one-model, one-buffer resource contract.
- Bind each decision consumer to what it actually consumed. A completed decision binds the digest of
  the exact loaded MjModel serialization and the exact byte streams consumed by complete-MJB
  comparison. Loading reserializes the exact loaded model and requires the receipt-bound size and
  SHA-256; the comparison hashes each role's consumed stream, now including the non-overlap tail of
  the longer artifact, and requires both digests to match before returning. Checking the retained
  bytes around a consumer did not establish this: a same-user process can mutate the object between
  two checks and restore it, so the surrounding checks pass while the consumer reads other bytes.
- Add a deterministic same-UID substitution regression for the private compiled artifacts, three
  exact-consumer regressions covering transient mutate/restore at the loaded-model and consumed-
  stream boundaries plus non-overlap tail coverage, and a cross-process regression requiring
  identical ordered change records, first witnesses and canonical receipt bytes under two explicit
  `PYTHONHASHSEED` values.
- Expose the Model Change Gate on the normal user path: `review-model` now appears in the README,
  the getting-started guide, the capability guide, the command reference, the SDK guide, the CLI
  description and the distribution summary.

## 0.2.1

- Keep Apache License 2.0 as the controlling license for the open-source Metrifid repository.
- Add project attribution through `NOTICE`.
- Add the unmodified Developer Certificate of Origin 1.1 and require signed-off contributed commits.
- Clarify that DCO sign-off is contribution provenance, not copyright assignment.
- Package `LICENSE` and `NOTICE` as PEP 639 legal files and carry `DCO` in the source distribution.
- Identify the copyright holder in package metadata and remove the deprecated Apache license Trove
  classifier.

## 0.2.0

- Add `python -m metrifid.demo`, a self-contained first-use demonstration that runs from a wheel
  installation alone with no checkout, arguments, network access, or packaged assets.
- Add `docs/sdk.md` documenting the supported programmatic execution surface — `certify_models`,
  `compare_configuration_file`, and `audit_configuration_file` — with signatures, return types, the
  completed-decision versus refusal distinction, output ownership, the native runtime requirement,
  and an explicit statement that native execution is neither thread-safe nor reentrant.
- Add four runnable SDK example scripts under `examples/sdk/` that use only public APIs and run
  from any directory against an installed wheel.
- Add a complete, runnable comparison example under `examples/compare/` with all ten configuration
  fields, both models, and a workload preparation script that never overwrites an existing artifact.
- Make the README first-use path wheel-only, and use inline code instead of relative Markdown links
  so the PyPI description has no broken links before owner-bound URLs exist.
- State in `docs/model_closure.md` and `docs/capabilities_and_use_cases.md` that closure content is
  bounded by the documented byte limit, that file-count, directory-count, and traversal-depth limits
  are not separately guaranteed in this pre-alpha release, and that this package targets local
  trusted-input workflows rather than hosted untrusted uploads.
- Replace hand-maintained test-count claims with a statement that release evidence is generated by
  CI, and remove stale `0.2.0a13` version text.
- Ship `SECURITY.md`, `CODE_OF_CONDUCT.md`, `.pre-commit-config.yaml`, and
  `.github/quality-constraints.txt` in the source distribution so it contains everything its own
  contributor and security documentation references.
- Build release artifacts through isolated PEP 517, install and test the direct sdist in a separate
  clean environment, rebuild a wheel from the extracted sdist, and check distribution equivalence
  semantically in CI.

## 0.2.0a14

- Add one bounded strict JSON admission module and apply it to comparison configurations, alias
  files, and timestep-audit configurations. Duplicate member names, raw float tokens, `NaN`,
  `Infinity`, malformed UTF-8, and oversized, over-deep, over-wide, or over-long documents are now
  refused at every file-based trust boundary, and audit configurations no longer use permissive
  JSON parsing.
- Read configuration and alias files through a single no-follow descriptor confirmed regular by
  `fstat`, with the byte ceiling applied before parsing.
- Add `metrifid.certify.load_and_validate_certification_receipt` for receipt files and untrusted
  bytes. It rejects duplicate names and noncanonical numeric tokens before the semantic validator
  runs. `validate_receipt` is unchanged for callers holding a parsed mapping.
- Make certification receipt parsing and validation importable without MuJoCo or NumPy by moving
  the runtime-identity schema to `certify/_runtime_schema.py` and the field-report schema to
  `certify/_field_schema.py`, and by deferring native imports in `certify/_artifact.py` and
  `certify/_bytes.py` to their producer functions.
- Report the real operation on distribution-identity failures, so a `certify` refusal says
  `certify` and an `audit-timestep` refusal says `audit-timestep`.
- Restore normal Python submodule attribute behavior, so `import metrifid.errors` then
  `metrifid.errors` works.
- Install MuJoCo in the CI quality environment so strict MyPy passes on a first push.

## 0.2.0a13

- Add a human-friendly getting-started guide and a faithful capabilities/use-cases reference.
- Make source-checkout installation the primary pre-publication path in README and examples.
- Remove stale release numbers and avoid promising a vulnerability-reporting channel before it is enabled.
- Preserve the open-ended CPython and NumPy compatibility policy introduced in 0.2.0a12.
- Make release quality checks reproducible with CI-only constraints that do not narrow public
  runtime or development dependencies.
- Correct three cross-version/platform test assumptions and the Audit retained-evidence expectation.
- Keep runtime admission capability-based and architecture-neutral.

## [0.2.0a12]

### Changed

- Python metadata is `>=3.11` with no upper bound, and the classifiers list 3.11 through 3.14.
- One shared native-runtime gate now admits Certify, Compare, Audit Timestep, and compiled model
  identity. The separate, narrower Certify envelope is removed.
- The gate rejects only Python below 3.11. It never inspects the interpreter implementation name,
  uses no finite Python-minor allowlist or upper bound, and never rejects an architecture; the
  machine string remains receipt evidence only.
- Platform admission is capability-based: a POSIX operating system reporting `Linux` or `Darwin`
  plus the exact `dir_fd`, `follow_symlinks`, `fd`, callable, and open-flag facilities the confined
  filesystem work requires. Native Windows remains unsupported; WSL is the documented route.
- The MuJoCo runtime dependency is `mujoco==3.10.0.*`, so binding-only `3.10.0.postN` packages that
  target the same native engine are accepted. Prerelease, development, `3.10.1`, and `3.11`-or-later
  package versions still refuse, and the native engine must remain 3.10.0 / 3010000 exactly.
- NumPy is `>=1.26` with no runtime upper bound and no runtime version gate.
- The development extra keeps its package names, drops NumPy, and uses minimum-only specifiers.
- The build-system requirement is `hatchling>=1.27` with no upper bound.
- CI builds one wheel and one sdist once and installs those exact wheel bytes noneditably in every
  compatibility lane. MJB digests are never compared across lanes.
- `tools/mjb_characterization.py` is included in the sdist and runs in every blocking lane. It stays
  outside the public import package.

### Unchanged

- Public APIs, schemas, statuses, reason codes, exit codes, numerical methods, output names, receipt
  bytes, output ownership, source-closure behavior, and the complete-MJB identity rule.
- The workload artifact writers, canonical JSON and exact-number helpers, and receipt parsing and
  validation remain outside the native gate and require no MuJoCo admission.

## [0.2.0a11]

### Fixed

- Certify and Compare perform their final public-path and exact-byte verification after the retained source contexts close, so replacing the public output pathname during final source cleanup causes refusal instead of returning paths to unrelated bytes.
- Timestep Audit verifies the aggregate pair before making the public path and registered output tree its final success boundary, closing the same path-replacement window for aggregate results.
- Private temporary names are relinquished permanently after finalization or cleanup. Reusing an old random temporary name later can no longer cause Metrifid cleanup to remove the caller's new directory entry.
- Comparison and Audit tests now assert the no-clobber preservation contract instead of the obsolete replace-and-delete behavior.

### Unchanged

- Public APIs, schemas, statuses, reason codes, exit codes, numerical methods, output names, receipt bytes for successful unaffected runs, and the complete-MJB identity rule.

## [0.2.0a10]

### Fixed

- Public output ownership is recorded only after a descriptor-relative no-clobber hard link exists
  and still identifies the retained sealed bytes. A conflicting hardlink can no longer be mistaken
  for product-owned output and deleted during failure cleanup.
- Generic paired-result and workload-writer cleanup removes only retained private temporaries.
  Already-linked public evidence is preserved on every later failure, while timestep-audit cleanup
  still removes exact private workspace artifacts and never deletes registered public evidence.

### Unchanged

- Successful Compare, Certify, workload-writer, and timestep-audit bytes; public APIs; schemas;
  statuses; reason codes; exits; numerical methods; and output names.

## [0.2.0a8]

### Fixed

- A timestep audit freezes one model-closure snapshot and one bounded state/actions byte set for
  the entire campaign. Every candidate consumes those inputs, the aggregate derives identities
  from them, and final live-input divergence removes all audit output and refuses publication.
- Model traversal, paired-result publication, audit cleanup, and workload writers are confined to
  retained directory descriptors. NPZ size limits are enforced during the read, Certify retains and
  reverifies both source snapshots, and Certify/Compare refuse output at or below either model root.
- Source-closure documentation now matches execution: every admitted regular file under the model
  root affects source identity, while separate dependency discovery identifies files MuJoCo reaches.
  An unused regular file does not affect compiled-MJB bytes unless MuJoCo resolves it.

### Unchanged

- Public APIs, schemas, statuses, exit codes, output names, receipt meanings, numerical methods,
  exact-time behavior, and the package architecture.

## [0.2.0a7]

### Changed

- The project adopts its public name. The distribution, import package, console
  command, repository slug, schema prefix and runtime key prefix are all `metrifid`, and the
  display name is `Metrifid`. This is a pre-public namespace reset, so no alias for the former
  internal name is shipped in any form.
- The model-closure documentation states the part it previously left out: a substitution that is
  visible in file metadata refuses before the member is read at all. The admitted-replacement,
  final-divergence and byte-identical cases are unchanged and the content-addressed claim is not
  broadened.

### Unchanged

- Every command, decision, status, exit code, refusal, metric, tolerance, time-grid rule,
  source-closure algorithm, receipt hashing rule, canonical encoding, output filename, supported
  platform and public Python export. Only the namespace and the version moved.

## [0.2.0a6]

### Changed

- The source-closure contract is documented as what it computes: content addressing. A member is
  named by its relative path, its exact byte count and the SHA-256 of the exact bytes read. Those
  bytes are the ones hashed, copied into the private immutable snapshot and compiled, and the
  source is re-measured against that snapshot before publication. The documentation no longer
  implies that a byte-identical replacement of a member's underlying file object is detected; no
  published identity depends on distinguishing it.

### Unchanged

- Every command, status, exit code, schema identifier, canonical encoding, output filename,
  supported platform and public Python export. Symlink, path-escape, non-regular-member,
  unreadable-member and byte-budget refusals remain fail-closed, and source content that differs
  from the snapshot at final verification still refuses with `MODEL_CLOSURE_MUTATED`.

## [0.2.0a5]

### Added

- `certify` is validated on Linux x86_64 as well as Darwin x86_64, both with CPython 3.12 and
  MuJoCo Python and native `3.10.0` exactly. The supported set is explicit; every other
  environment still refuses rather than guessing, and the refusal now names the complete
  supported set.
- Continuous integration runs the bundled `examples/certify/run_example.py` against the installed
  wheel, so the documented example is exercised on every change.

### Changed

- Continuous integration installs the built wheel with its bounded `[dev]` extra instead of
  installing validation tools separately, and verifies that both the import and the console
  script resolve from the installed environment.
- Documentation states plainly that a complete-MJB certificate is bound to the runtime identity
  recorded in the receipt. Support statements name the two measured platforms.

### Unchanged

- Every command, status, exit code, schema identifier, canonical encoding, output filename,
  atomic publication rule, closure and snapshot behavior, installed-distribution identity check,
  top-level Python export and runtime dependency. `compare` and `audit-timestep` keep the
  environment envelope they already had.

## [0.2.0a4]

### Changed

- The repository is shaped for outside contributors: development-history artifacts and the tests
  that asserted file inventories, source line counts and documentation prose are gone. Every
  behavior, security, receipt, closure, path-confinement, workload, comparison, audit and
  certification test is retained.
- Two long modules were split along their real seams with no import path changed:
  `_model_admission` is a facade over `_model_compile` and `_model_descriptors`;
  `timestep_audit` is a facade over `_audit_config`, `_audit_execution` and `_audit_reporting`.
- Documentation is organized around the three decisions a user makes, with an API reference, a
  model-closure guide, a workload guide and a public case study. A runnable example ships in
  `examples/certify/`.

### Fixed

- Descriptive field evidence is bounded. A changed field larger than 64 MiB per side keeps its
  path, types, dtypes, shapes and digests but reports no element count and no witnesses, and the
  report's `truncated` flag records it. Changed fields are described in sorted path order and no
  two fields' copied arrays are ever held at the same time.
- Receipt revalidation completes its descriptive type checks: a present field side must name a
  nonempty type and a null-or-nonempty dtype, an absent side must be null throughout, and a
  witness value may only be one of the canonical forms the reporter emits.

## [0.2.0a3]

### Added

- `metrifid certify`: compile two MuJoCo source closures and state, over every byte of the
  serialized compiled model, whether they are identical. Exit `0` certifies; exit `40` does not.
  It runs no workload and never steps the simulation. On a difference the receipt reports the
  first differing byte offset, the differing byte count, and a bounded descriptive report naming
  the changed public fields with sorted index and value witnesses.
- Receipt revalidation for an unsigned certification receipt: exact member sets, frozen claim and
  limitation text, the nested runtime identity and its own hash, both role identities, each
  artifact's binding to the recorded runtime, and the byte comparison against the two artifacts.

### Fixed

- Test isolation no longer depends on suite order: helpers that swap modules record and restore
  every `sys.modules` entry and parent-package attribute they change.

## Earlier development

`0.1.x` through `0.2.0a2` established the foundations the shipped decisions rest on: canonical
JSON with exact rationals and IEEE-754 bit-exact doubles, self-hashed receipts, measured model
closures with bounded no-follow reads and mutation detection, snapshot compilation, semantic
joint and actuator alignment, strict workload artifacts with an exact time grid, the `compare`
decision and its four statuses, the `audit-timestep` decision, and the installed-distribution
identity check. None of it was published.
