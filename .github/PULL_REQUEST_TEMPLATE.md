## What this changes

Describe the change and the files it affects.

## Why

Explain the problem this solves, and link any related issue.

## Validation

CI builds one wheel and one sdist once in a Linux / Python 3.11 job that also runs Ruff check,
Ruff format check, strict MyPy over `src/metrifid` and the CI evidence helper, and a strict Twine
check. That job publishes an original artifact manifest with each filename, size and SHA-256, and
every other lane verifies its entry before installing those exact bytes.

Two complete installed-wheel suites run by default, and they are causally different:

- `linux_x64_py312_full` on Ubuntu 24.04 / CPython 3.12 with the resolver-latest supported runtime;
- `linux_x64_py311_numpy_min` on Ubuntu 24.04 / CPython 3.11 pinned to `numpy==1.26.4` and
  `mujoco==3.9.0`, installed before the wheel with `--no-deps`.

Four boundary smokes cover CPython 3.13 and 3.14 on Linux x64, CPython 3.14 on macOS arm64, and
CPython 3.12 on macOS Intel. Each installs the same original wheel, runs `pip check`, records its
measured architecture and runtime identity, exercises all seven command surfaces, the demo, the
29-check `tools/mjb_characterization.py`, and a focused semantic test list. They are smokes, not
complete suites, and the summary rejects any lane that claims otherwise.

Retained exact MuJoCo 3.10.0 and 3.11.0 lanes and the direct-sdist lane run focused suites. The
broader interpreter and operating-system matrix is available on demand as nonblocking
`expanded_full_diagnostics`, and only through a manual `workflow_dispatch` with
`expanded_full_matrix: true`. It never runs on a push or a pull request, and its artifacts never
count towards the default summary.

`release compatibility summary` is the single decision authority. It runs under exactly `always()`
with no event condition, because a conditionally skipped job is reported to branch protection as a
successful status and a required check must never be skippable. It rejects any required job whose
result is `skipped`, `failure` or `cancelled`, and delegates every judgement to
`.github/scripts/validate_ci_evidence.py`.

The build job records `quality-evidence/checkout_identity.json`, which names the event, the
checked-out commit and tree, and — on a pull request — the head commit and tree fetched from
`refs/pull/<number>/head`. GitHub checks out a synthetic merge commit, so both are recorded and the
summary checks them against the trusted values of the run. MJB digests stay lane-local and are never
compared across runtimes.

List anything you ran locally beyond that, and its outcome.

## Claim boundary

If this changes what a certificate or receipt claims, say so explicitly. A complete-MJB
certificate is bound to the runtime that produced it and makes no behavioral claim.

## Checklist

- [ ] Every commit is signed off with `Signed-off-by:` and satisfies the DCO in `DCO`.
- [ ] I have the right to contribute all code, documentation, and assets in this pull request under
      Apache License 2.0.
- [ ] Tests cover the changed behavior without weakening existing tests.
- [ ] `python -m ruff check .`, `python -m ruff format --check .` and
      `python -m mypy --strict src/metrifid .github/scripts/validate_ci_evidence.py` pass locally,
      matching what CI runs.
- [ ] Documentation is updated when the public surface or a claim changes.
- [ ] No secrets, proprietary robot assets or confidential model files are included.
- [ ] Workflow permissions stay least-privilege.
