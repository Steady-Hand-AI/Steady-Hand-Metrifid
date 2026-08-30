# Getting started

Welcome to **Metrifid**.

This guide is for someone opening the project for the first time and asking:

- What is Metrifid?
- Why is it called Metrifid?
- Which command should I use?
- How do I install and try it?
- Where are the detailed docs?
- What does a result prove—and what does it not prove?

The official spelling is **Metrifid**. The Python package and command-line name are both lowercase:

```text
metrifid
```

This guide is version-neutral. Check the version you actually installed with:

```bash
python -c "import metrifid; print(metrifid.__version__)"
```

---

## Choose your starting point

You do not have to know which of the seven commands you need. Start from the thing that actually
changed, and the first result will tell you where to go next.

### A model or asset changed

An MJCF file was edited, a mesh was regenerated, an importer was upgraded, or a pull request landed,
and you need to know whether MuJoCo still compiles the same model.

```bash
metrifid certify old/robot.xml new/robot.xml --output out/
```

Certify compiles both source closures and compares every byte MuJoCo emits for the compiled model.
It runs no workload and never steps the simulation.

```text
exit 0   CERTIFIED_COMPILED_EQUIVALENCE   every serialized byte matched
exit 40  NOT_CERTIFIED_COMPILED_DIFFERS   at least one byte differed
```

Exit 40 here means the compiled artifacts differ. It does not mean the behavior is wrong, and it does
not mean anything crashed.

### The MuJoCo runtime changed

You are moving to a different MuJoCo native version and need to know whether it may replace the one
your existing evidence was built on.

```bash
metrifid run-runtime-review runtime_review_run.json
```

This creates the fixed twelve native evidence cells through two explicit, already-prepared Python
profiles and returns the full-horizon Runtime Review decision. You prepare the two profiles yourself:
Metrifid measures them, and does not discover, install, or repair MuJoCo environments.

## Continue from the first result

Whichever route you started on, the receipt names the next question. Follow the one you have:

| What you want to know next | Command |
| --- | --- |
| the exact compiled difference, and whether your policy declares it | `review-model` |
| the behavioral consequence under a declared workload and tolerances | `compare` |
| workload detectability — whether your workloads would notice a perturbation at all | `qualify-workload` |
| the largest timestep with an unbroken trustworthy prefix | `audit-timestep` |
| whether pre-existing native evidence supports one profile replacing another | `review-runtime` |

Each of these is a completed decision backed by a receipt you can revalidate independently, and none
of them establishes physical correctness or safety.

## Run the complaint-backed public cases

Six small MJCF pairs, each built around a mechanism someone publicly complained about, run end to end
through the same journey:

```bash
python examples/public_cases/run_all.py --output /absolute/absent/output
```

They are independently authored mechanism analogues, not upstream reproductions. See
[`docs/public_cases.md`](public_cases.md) for what each one exercises and what it does not claim.

---

## 1. Metrifid in one sentence

**Metrifid is a local assurance tool for MuJoCo model changes.**

It helps you answer six different engineering questions:

1. **Did the complete compiled MuJoCo model change?**
2. **What changed, and was the compiled change declared by policy?**
3. **Did behavior change on this exact declared workload?**
4. **Which candidate timestep remains acceptable under this workload and tolerance policy?**
5. **Would my declared workloads even notice the model changes I care about?**
6. **Can I create the complete native-profile migration evidence and decide it in one command?**

Metrifid answers those questions through seven commands, including two paths to the same Runtime
Review referee:

```text
metrifid certify
metrifid review-model
metrifid compare
metrifid audit-timestep
metrifid qualify-workload
metrifid run-runtime-review
metrifid review-runtime
```

A useful product description is:

> **Metrifid — simulation change assurance for MuJoCo models.**

---

## 2. Why the name is Metrifid

**Metrifid is a coined product name, not a standard scientific term and not an acronym.**

The intended meaning combines three ideas:

- **Metric / metrology** — measuring a difference precisely rather than relying on visual inspection.
- **Fiducial** — a trusted reference or fixed basis used for comparison.
- **Fidelity** — asking whether a new model or simulation configuration preserves the behavior you care about.

That fits the product’s workflow:

```text
trusted baseline
→ measured candidate
→ exact decision
→ reproducible receipt
```

The name is broader than “MuJoCo diff.” Metrifid covers both model assurance and runtime assurance: it certifies compiled-model identity, compares declared behavior, qualifies workloads and timesteps against explicit policies, and reviews the native runtime that produced a result.

The name does **not** imply that Metrifid automatically proves physical correctness, hardware safety, or universal equivalence. Its claims are deliberately narrower and are written into every receipt.

---

## 3. The fastest way to understand Metrifid

Use this decision map:

| Your question | Command | What you must provide |
| --- | --- | --- |
| “Did these two MJCF source trees compile to the exact same model?” | `metrifid certify` | Two MJCF entrypoints and an output directory |
| “What compiled fields changed, and did my policy allow them?” | `metrifid review-model` | Two MJCF entrypoints, a model-release policy, and an output directory |
| “Did the candidate behave differently on my exact replay workload?” | `metrifid compare` | A strict comparison JSON configuration, state artifact, action artifact, models, and tolerances |
| “Which larger timestep remains acceptable for this workload?” | `metrifid audit-timestep` | One model, one frozen workload, candidate timesteps, and tolerances |
| “Would my workloads notice these probe models, and which probes stay invisible?” | `metrifid qualify-workload` | One baseline model, probe models you supply at declared magnitudes, three to sixteen workloads, and tolerances. The parameter, direction, magnitude and `magnitude_semantics` labels are preserved user declarations, not findings. |
| “Can I create and decide one exact native-profile migration?” | `metrifid run-runtime-review` | Two explicit already-prepared Python profile launchers and one strict self-contained manifest. |
| “I already retained the twelve native evidence cells; can I decide them?” | `metrifid review-runtime` | One strict retained-evidence Runtime Review configuration. |

A practical progression is:

```text
Start with certify.

If certify reports a compiled difference:
    use review-model to classify the change against your policy, and
    use compare when you need a workload-bounded behavioral decision.

If your goal is simulation-step reduction:
    use audit-timestep.

Before you trust any of the above to catch a future change:
    use qualify-workload to find out whether your workloads can see it at all.

For a native-profile migration, use run-runtime-review when Metrifid should create the fixed twelve
cells, or review-runtime when those cells already exist.
```

---

## 4. What Metrifid is good for

Metrifid is useful when you maintain or review MuJoCo assets and need stronger evidence than “the XML looks okay” or “the simulation still runs.”

Typical uses include:

- verifying XML formatter or whitespace-only changes;
- reviewing changes to `<default>` inheritance;
- reviewing changes to includes, meshes, assets, and compiler directories;
- detecting changes in masses, inertias, damping, armature, friction, actuator parameters, solver options, or timestep;
- checking a vendor or model-publisher update;
- qualifying a model revision before merge or release;
- attaching deterministic evidence to a pull request;
- replaying one accepted state/action workload against baseline and candidate models;
- finding the first monitored coordinate and exact boundary where a tolerance is crossed;
- classifying an inconclusive run rather than silently returning green;
- qualifying larger candidate timesteps without rounding an incompatible control grid.

The first users most likely to benefit are:

- MuJoCo model maintainers;
- robot-model publishers;
- simulation-infrastructure engineers;
- robot OEM simulation teams;
- controls and MPC engineers;
- reinforcement-learning and humanoid teams;
- reproducibility-focused robotics labs.

---

## 5. Requirements and supported environment

The current source declares:

```text
Python:          3.11 or newer, with no artificial upper bound
MuJoCo runtime:  stable MuJoCo >=3.9, capability-based, with no minor-version ceiling
MuJoCo package:  >=3.9,<3.11 on Darwin x86_64; >=3.9 with no ceiling elsewhere
MuJoCo identity: package base, native string, and native integer must agree
Validated exact: 3.9.0, 3.10.0, 3.11.0, and 3.12.0
NumPy:           1.26 or newer, with no runtime upper bound
Operating system: Linux or macOS with the required POSIX filesystem capabilities
Architecture:    recorded as evidence, not rejected through an architecture allowlist
```

On platforms without the packaging exception below, the newest stable MuJoCo is the primary
development and release profile—3.12.0 for the frozen 2026-08-22 snapshot. Exact 3.9.0 is the
supported minimum, and retained 3.10.0 and 3.11.0 results are backward-compatibility evidence.
A normal installation there keeps the `mujoco>=3.9` resolver policy and therefore encounters a
newer stable release automatically.

Package resolution carries one platform exception: upstream publishes no Intel macOS wheel for
MuJoCo 3.11 or newer, so the declared dependency is `mujoco>=3.9,<3.11` on Darwin `x86_64` and
`mujoco>=3.9` with no minor ceiling on every other supported platform. That bound is read by
the installer when resolving dependencies; it does not narrow runtime admission, which stays
capability-based for stable MuJoCo 3.9 or newer on every architecture.
On Intel macOS a normal installation therefore resolves the newest stable MuJoCo below 3.11.

Native Windows is not currently supported because Metrifid’s confined filesystem operations rely on POSIX descriptor-relative behavior. Windows users should use WSL.

Metrifid records the exact runtime identity in its evidence. A result produced under one operating system or MuJoCo build is not automatically a claim about another runtime.

---

## 6. Installation

### Option A: install from a wheel

This is the preferred way to share a controlled SDK build:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install /path/to/metrifid-<version>-py3-none-any.whl
```

On Windows through WSL, use the normal Linux activation command inside WSL.

### Option B: install from an extracted source repository

From the repository root containing `pyproject.toml`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

This builds and installs the project normally. It is not an editable installation.

### Option C: install a published version from PyPI

For a version that is available on the index:

```bash
python -m pip install metrifid
```

Check the project page on the index for the versions published there. Use the wheel or source installation route for any version that is not.

### Do not use an editable install for native decisions

Avoid this for normal use:

```bash
python -m pip install -e .
```

Metrifid intentionally requires a normal installed distribution for decision-bearing native commands so receipts can bind the installed package identity. Contributors should follow [`CONTRIBUTING.md`](../CONTRIBUTING.md), which separates contributor-quality work from installed-wheel product validation.

---

## 7. Verify your installation

Run:

```bash
metrifid --help
metrifid certify --help
metrifid review-model --help
metrifid compare --help
metrifid audit-timestep --help
metrifid qualify-workload --help
metrifid review-runtime --help
metrifid run-runtime-review --help
```

You can also verify the import:

```bash
python -c "import metrifid; print(metrifid.__version__)"
```

When a native command refuses the environment, it exits with `64` and prints a structured operational-failure JSON document to stderr. Read the `stage` and `reason.code` values; do not treat a refusal as a model result.

---

## 8. Your first 10 minutes

The best first experience is the bundled Certify example.

From the repository root:

```bash
python examples/certify/run_example.py
```

Expected result:

```text
different source, same compiled model : CERTIFIED_COMPILED_EQUIVALENCE (exit 0)
one changed mass                      : NOT_CERTIFIED_COMPILED_DIFFERS (exit 40)

all 6 checks passed
```

This example demonstrates both sides of the core decision:

- different source text can compile to the exact same model;
- one physical parameter change can alter the compiled artifact.

Read the example guide here:

[`examples/certify/README.md`](../examples/certify/README.md)

---

## 9. Command 1: `metrifid certify`

Use `certify` when your first question is:

> Do these two MuJoCo source closures compile to byte-identical complete MJB artifacts under this recorded runtime?

### Command

```bash
metrifid certify \
  old/robot.xml \
  new/robot.xml \
  --output certification_result/
```

When the entrypoints belong to larger model roots, provide them explicitly:

```bash
metrifid certify \
  old/models/robot.xml \
  new/models/robot.xml \
  --baseline-root old/ \
  --candidate-root new/ \
  --output certification_result/
```

### Outputs

A successful completed command publishes:

```text
certification_result/
├── certification.json
└── certification.md
```

Use:

- `certification.json` for automation and exact evidence;
- `certification.md` for human review.

### Statuses and exit codes

| Exit | Status | Meaning |
| ---: | --- | --- |
| `0` | `CERTIFIED_COMPILED_EQUIVALENCE` | Every byte in the two complete serialized compiled artifacts matched. |
| `40` | `NOT_CERTIFIED_COMPILED_DIFFERS` | At least one serialized byte differed. |
| `64` | refusal | The invocation, input, environment, or output was not admissible. |
| `70` | internal failure | Metrifid could not complete the operation safely. |

### What Certify does internally

For each role, Metrifid:

1. verifies the installed distribution identity;
2. measures every admitted regular file beneath the declared model root;
3. creates a private same-byte snapshot;
4. discovers the MuJoCo-reached dependency set separately;
5. compiles the snapshot;
6. serializes the complete compiled model using `mj_saveModel`;
7. hashes every serialized byte;
8. compares the two complete artifacts;
9. reverifies both live source closures before publication.

There is no workload, no state artifact, no action artifact, no tolerance, and no simulation stepping.

### What a certificate proves

The exact machine claim is:

> The measured baseline and candidate source closures, compiled under the recorded runtime identity, produced byte-identical complete MJB artifacts.

### What it does not prove

It does not prove:

- source-text equality;
- license equality;
- physical correctness;
- visual intent;
- task suitability;
- hardware safety;
- behavior under another MuJoCo version or operating system;
- equality after a caller mutates `mjModel`;
- global behavioral equivalence when artifacts differ.

When artifacts differ and you need a workload-specific decision, use `compare`.

Detailed guide:

[`docs/compiled_certification.md`](compiled_certification.md)

---

## 10. Command 2: `metrifid compare`

Use `compare` when your question is:

> On this exact deterministic open-loop workload, did the candidate cross any declared tolerance relative to the baseline?

### Command

```bash
metrifid compare comparison.json
```

The configuration is strict JSON. Metrifid does not accept YAML, silently default unknown keys, or infer a workload.

A comparison normally declares:

- baseline and candidate model roots and entrypoints;
- one canonical initial-state NPZ artifact;
- one canonical actions NPZ artifact;
- exact control timing;
- repeat count;
- monitored hinge or slide joints;
- position and velocity tolerances;
- an output directory.

Read [`docs/workloads.md`](workloads.md) before constructing your first comparison.

### Outputs

A completed comparison publishes:

```text
comparison.json
comparison.md
```

`comparison.json` is the authoritative machine receipt. `comparison.md` is a deterministic human rendering of the same decision.

### Statuses and exits

| Exit | Status | Meaning |
| ---: | --- | --- |
| `0` | `NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD` | No monitored metric crossed its declared tolerance. |
| `10` | `MATERIAL_BEHAVIOR_CHANGE` | At least one monitored metric crossed its declared tolerance. |
| `20` | `COVERAGE_INSUFFICIENT` | The requested evidence could not be completed within the admitted coverage contract. |
| `30` | `NONDETERMINISTIC_REPLAY` | Repeats did not produce stable replay evidence. |
| `64` / `70` | refusal/failure | No comparison decision was reached. |

### Supported behavioral metrics

Current behavioral comparison is intentionally narrow:

- hinge position, using wrapped angular distance;
- hinge velocity;
- slide position;
- slide velocity.

Current comparison does **not** implement:

- ball or free-joint tolerance metrics;
- contact equivalence;
- body or site pose metrics;
- sensor equivalence;
- reward or task-outcome metrics;
- closed-loop policy comparison;
- hardware safety validation.

### What Compare is especially good at

- exact replay against one accepted workload;
- strict time-grid admission;
- no hidden resampling or interpolation;
- `N` controls producing `N + 1` captured boundaries;
- repeated execution to detect nondeterminism;
- first-tolerance-crossing localization;
- exact joint, metric, boundary, and time evidence;
- preserving inconclusive outcomes instead of returning a false green.

Detailed guide:

[`docs/comparison.md`](comparison.md)

---

## 11. Command 3: `metrifid audit-timestep`

Use `audit-timestep` when your question is:

> For this exact model, workload, and tolerance policy, which declared candidate timestep has the strongest completed evidence?

### Command

```bash
metrifid audit-timestep timestep_audit.json
```

The audit uses the existing comparison engine. It does not add new physics or assume that fidelity changes monotonically with timestep.

### Important behavior

- One model and one exact workload are frozen for the campaign.
- Every declared candidate is attempted independently.
- A candidate that does not divide the control interval exactly is refused; it is never rounded.
- Later candidates still run after an exact nonintegral-grid refusal.
- The first `OUTSIDE` or `INCONCLUSIVE` result blocks recommending across the broken evidence prefix.
- A later passing candidate can be reported without being recommended.

### Recommendation policy

```text
largest_within_tolerance_completed_prefix
```

This is not simply “pick the largest candidate that happened to pass.” It selects the largest candidate supported by an unbroken prefix of trustworthy evidence, while skipping only exact nonintegral-grid refusals.

### Outputs

```text
audit_out/
├── timestep_audit.json
├── timestep_audit.md
└── candidates/
    └── one comparison receipt or operational-failure document per candidate
```

The audit reports a **step-count factor**, not measured wall-clock speed or training-cost savings.

Detailed guide:

[`docs/timestep_audit.md`](timestep_audit.md)

---

## 11b. Command 4: `metrifid review-model`

Use `review-model` when your question is:

> What changed between these two compiled models, was each change allowed, required, forbidden or
> undeclared by my policy, and where is the first unexpected one?

### Command

```bash
metrifid review-model BASELINE_MJCF CANDIDATE_MJCF \
  --policy MODEL_RELEASE_POLICY.json \
  --output OUTPUT_DIRECTORY \
  [--baseline-root BASELINE_ROOT] [--candidate-root CANDIDATE_ROOT]
```

### Run the worked example

The example performs the trustworthy two-pass workflow end to end: it learns both compiled subjects
with `certify`, runs a rule-free discovery review, converts each explained change into an exact
`ALLOW` rule, and reviews again against that declared policy.

```bash
python examples/model_release/run_example.py --workspace /tmp/metrifid-model-release-example
```

It prints three statuses in this order:

```text
NOT_CERTIFIED_COMPILED_DIFFERS
REVIEW_REQUIRED
WITHIN_DECLARED_POLICY
```

The first is Certify reporting that the compiled artifacts differ. The second is the discovery pass
refusing to bless an unbound candidate. The third is the declared review completing at exit 0.

### Statuses and exit codes

| Status | Exit |
| --- | ---: |
| `NO_COMPILED_CHANGE` | 0 |
| `WITHIN_DECLARED_POLICY` | 0 |
| `REVIEW_REQUIRED` | 40 |
| `OUTSIDE_DECLARED_POLICY` | 40 |

An input refusal exits `64`; an operational failure exits `70`.

### What it does not prove

`review-model` is static. It allocates no `mjData` and steps no model, so a
`WITHIN_DECLARED_POLICY` result does not establish dynamic equivalence. Use `compare` for the
workload-bounded behavioral question. See [`model_release.md`](model_release.md).

## 12. Understanding receipts

Metrifid publishes canonical, self-hashed JSON documents.

The key schemas are:

| Output | Schema |
| --- | --- |
| `certification.json` | `metrifid.compiled_equivalence_receipt`, schema version `1` |
| field details inside a differing certificate | `metrifid.compiled_field_report`, schema version `1` |
| `comparison.json` | `metrifid.comparison_receipt`, schema version `1` |
| `timestep_audit.json` | `metrifid.timestep_audit`, schema version `1` |
| refusal written to stderr | `metrifid.operational_failure`, schema version `1` |

The JSON files contain exact rational values and tagged IEEE-754 Binary64 representations where ordinary JSON floats would lose identity.

Read:

[`docs/canonicalization.md`](canonicalization.md)

A self-hash establishes internal consistency. It is not a digital signature and does not prove who created the document.

---

## 13. Model roots, source closure, and dependencies

A MuJoCo model is often more than one XML file. It can include:

- nested XML files;
- meshes;
- textures;
- height fields;
- compiler directories;
- other regular assets.

Metrifid distinguishes two ideas:

### Source closure

Every admitted regular file beneath the declared model root, measured by exact bytes.

An unused regular file can therefore change the source-closure identity.

### MuJoCo-reached dependency set

The subset of the measured closure that MuJoCo actually reaches while compiling the chosen entrypoint.

An unused file changes the source identity but does not change the compiled MJB unless MuJoCo resolves it.

Metrifid compiles from a private same-byte snapshot and reverifies the live source before publishing.

Read:

[`docs/model_closure.md`](model_closure.md)

---

## Documentation map

Start at the repository root.

| File | Read it when you want to… |
| --- | --- |
| [`README.md`](../README.md) | understand the product, install it, and see the main commands |
| [`examples/certify/README.md`](../examples/certify/README.md) | run the fastest first example |
| [`docs/sdk.md`](sdk.md) | the supported programmatic execution surface for all seven operations |
| [`docs/reference.md`](reference.md) | look up commands, exits, statuses, schemas, support, and Python API |
| [`docs/compiled_certification.md`](compiled_certification.md) | understand the exact `certify` contract |
| [`docs/model_closure.md`](model_closure.md) | understand source identity, roots, dependencies, snapshots, and confinement |
| [`docs/workloads.md`](workloads.md) | create state/action artifacts and understand exact timing |
| [`docs/comparison.md`](comparison.md) | configure and interpret `compare` |
| [`docs/timestep_audit.md`](timestep_audit.md) | configure and interpret `audit-timestep` |
| [`docs/canonicalization.md`](canonicalization.md) | understand canonical JSON, exact numbers, and self-hashes |
| [`docs/public_cases.md`](public_cases.md) | run and interpret the six complaint-backed public cases |
| [`docs/menagerie_formatter_case.md`](menagerie_formatter_case.md) | read a real public MuJoCo Menagerie case study |
| [`CONTRIBUTING.md`](../CONTRIBUTING.md) | modify Metrifid or contribute code |
| [`SECURITY.md`](../SECURITY.md) | understand security scope and report a vulnerability |
| [`CHANGELOG.md`](../CHANGELOG.md) | see release changes |
| [`LICENSE`](../LICENSE) | read the Apache License 2.0 terms |

---

## 15. A recommended exploration path

### Path A: you maintain an MJCF repository

1. Install Metrifid normally.
2. Run the bundled Certify example.
3. Certify two revisions of one real model.
4. Read `certification.md`.
5. Inspect `certification.json`.
6. Add the command to a local release checklist or CI job.
7. When artifacts differ, decide whether you need a `compare` workload.

### Path B: you maintain simulation infrastructure

1. Learn `certify` first.
2. Read `docs/workloads.md`.
3. Create one accepted state and action artifact.
4. Run baseline versus baseline to establish stable replay.
5. Run baseline versus candidate.
6. Inspect the first crossing and repeatability evidence.
7. Preserve the receipt with the pull request or release.

### Path C: you train policies or tune simulation speed

1. Choose one model and representative workload.
2. Declare tolerances you can defend.
3. Choose candidate timesteps before looking at results.
4. Run `audit-timestep`.
5. Read every candidate row, not only the recommendation.
6. Measure real throughput separately before claiming cost savings.

### Path D: you want to contribute

1. Read `CONTRIBUTING.md`.
2. Keep user-facing product execution in a noneditable wheel environment.
3. Run the existing behavior and security tests.
4. Preserve statuses, exits, schemas, and receipt semantics unless a change is deliberately planned and reviewed.
5. Keep new functions focused, documented, and easy for a new contributor to inspect.

---

## 16. Common first-user questions

### “Why not just diff the XML?”

Because different XML can compile to the same model, while a small indirect change in defaults or an included file can change the compiled model. `certify` compares the complete compiled artifact.

### “Why not just run one simulation?”

A simulation reports only on the trajectory you executed. `certify` is workload-free; `compare` is explicitly workload-bounded.

### “Does exit 40 mean the behavior is wrong?”

No. It means the compiled artifacts differ. It does not say the difference is harmful or materially visible on your workload.

### “Does exit 0 from `compare` prove global equivalence?”

No. It means no monitored metric crossed tolerance on the exact declared workload.

### “Can Metrifid decide my tolerances?”

No. The user or organization declares the tolerance policy. Metrifid applies it exactly.

### “Does `audit-timestep` prove a timestep is physically correct?”

No. The reference timestep is a comparison reference, not guaranteed ground truth. The recommendation is scoped to the model, workload, observations, and tolerances.

### “Can I use Windows?”

Use WSL. Native Windows does not provide the POSIX filesystem capabilities the current safety model requires.

### “Can I use a newer Python?”

The package has no artificial Python upper bound. A future Python version is not rejected solely because it is newer, but it is not described as validated until that exact environment passes native evidence.

### “Can I use a newer stable MuJoCo?”

Yes, when the package/native identity is coherent and the requested operation's measured
capabilities are present. Exact 3.9.0 through 3.12.0 profiles are retained-validated. A later stable
release is reported as capability-compatible, not validated. Runtime admission is not blanket claim
support: `review-model` refuses an uncharacterized public-field surface, and static or dynamic
one-actuator projections refuse a zero/multi-input or multi-output signature before publishing a
completed result. The refusal evidence names the missing coverage and a concrete remediation.

### “Why did Metrifid refuse an editable install?”

Decision-bearing native commands bind the installed distribution identity. Install normally with `pip install .` or install a wheel.

### “Why must the output directory be absent or empty?”

Metrifid uses strict no-clobber publication and preserves evidence ownership. It refuses to mix a completed result with unrelated existing files.

---

## 17. What Metrifid does not currently do

Metrifid currently does not:

- repair or rewrite models automatically;
- determine whether a model is physically correct;
- prove hardware safety;
- certify legal or licensing compliance;
- compare every possible state or workload when compiled artifacts differ;
- compare contact, reward, task success, body/site poses, or arbitrary sensors in the current `compare` command;
- execute closed-loop controller or policy equivalence;
- benchmark wall-clock speed in `audit-timestep`;
- provide a hosted dashboard, team registry, signing service, or evidence-retention platform;
- support native Windows;
- claim equality across different MuJoCo versions or builds.

Those boundaries are part of the product’s trust model.

---

## 18. Sharing Metrifid with other people

For a source review or private SDK trial, share a clean repository archive containing:

```text
README.md
LICENSE
pyproject.toml
src/metrifid/
docs/
examples/
tests/
CONTRIBUTING.md
SECURITY.md
CHANGELOG.md
```

Tell a new user:

1. Open `README.md` first.
2. Install with `python -m pip install .` or install the provided wheel.
3. Run `python examples/certify/run_example.py`.
4. Use this cold-start guide to choose the next command.
5. Use the command-specific docs for exact configuration and claim boundaries.

Share the repository source or a built wheel/source distribution. Auxiliary local validation files are not required by SDK users.

---

## 19. Licensing and ownership

Metrifid is distributed under the **Apache License 2.0**. Copyright 2026 Volodymyr Barylyak.

In practical terms:

- the copyright owner keeps copyright in the original Metrifid work;
- recipients receive permission to use, modify, and redistribute the software under the license;
- redistributed copies must retain the license and notices required by Apache-2.0;
- contributors retain copyright in their contributions and submit them under Apache-2.0;
- each contributed commit requires the DCO 1.1 sign-off described in `CONTRIBUTING.md`;
- the software is provided without warranty as described in `LICENSE`.

Read [`LICENSE`](../LICENSE), [`NOTICE`](../NOTICE), [`DCO`](../DCO), and
[`licensing_and_contributions.md`](licensing_and_contributions.md).

---

## 20. The shortest possible cold start

```bash
# 1. Create an environment.
python -m venv .venv
source .venv/bin/activate

# 2. Install Metrifid normally from the repository.
python -m pip install .

# 3. Confirm the command exists.
metrifid --help

# 4. Run the bundled example.
python examples/certify/run_example.py

# 5. Certify your own two model revisions.
metrifid certify old/model.xml new/model.xml --output metrifid_result/

# 6. Read the human result.
cat metrifid_result/certification.md
```

Then decide:

```text
Need artifact identity?       Stay with certify.
Need workload behavior?       Read docs/workloads.md and use compare.
Need timestep qualification?  Read docs/timestep_audit.md and use audit-timestep.
```

---

## 21. Final orientation

Metrifid is built around a simple discipline:

```text
Do not guess what changed.
Measure the exact admitted inputs.
Make a narrowly scoped decision.
Preserve the evidence.
Refuse when the evidence is not trustworthy.
```

Start with `certify`, because it has the lowest setup cost and the strongest workload-free statement. Move to `compare` only when compiled artifacts differ or you need a workload-specific behavior decision. Use `audit-timestep` when you are making a declared fidelity-versus-step-count decision.

For the complete command and API surface, continue with:

[`docs/reference.md`](reference.md)
