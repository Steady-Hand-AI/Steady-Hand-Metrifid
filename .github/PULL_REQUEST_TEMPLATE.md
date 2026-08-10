## What this changes

Describe the change and the files it affects.

## Why

Explain the problem this solves, and link any related issue.

## Validation

CI builds one wheel and one sdist once in a Linux / Python 3.11 job that also runs Ruff check,
Ruff format check, strict MyPy over `src/metrifid`, and Twine check. Every compatibility lane then
downloads and installs those exact wheel bytes as a noneditable wheel and runs `pip check`, the
complete installed-wheel test suite, the bundled `examples/certify/run_example.py`, a CLI help
smoke, and the 29-check `tools/mjb_characterization.py`.

Blocking lanes are CPython 3.11, 3.12, 3.13, and 3.14 on Linux x64, CPython 3.12 and 3.14 on macOS
arm64, CPython 3.12 on macOS x64, and a Linux x64 / Python 3.11 minimum-dependency lane pinned to
`numpy==1.26.4`. Linux arm64 is experimental and non-blocking while its hosted runner is public
preview. MJB digests are lane-local and are never compared across lanes.

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
      `python -m mypy --strict src/metrifid` pass locally.
- [ ] Documentation is updated when the public surface or a claim changes.
- [ ] No secrets, proprietary robot assets or confidential model files are included.
- [ ] Workflow permissions stay least-privilege.
