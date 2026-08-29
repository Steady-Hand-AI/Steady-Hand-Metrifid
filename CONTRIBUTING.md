# Contributing

Thanks for looking. This is a small, deliberately narrow tool, and the bar for changes is
"a user can tell the difference".

## Contribution license and DCO sign-off

Metrifid is licensed under Apache License 2.0. Contributions intentionally submitted for inclusion
are accepted under that license. Every commit in a pull request must certify the Developer
Certificate of Origin 1.1 in [`DCO`](DCO).

Use `git commit -s` to add this line to each commit message:

```text
Signed-off-by: Your Name <you@example.com>
```

The sign-off certifies that you have the right to submit the contribution under Apache-2.0. It is
not a copyright assignment or a contributor license agreement. Contributors retain copyright in
their contributions. Do not submit material you do not have the right to contribute. A pull request
with an unsigned commit will not be merged. The DCO also explains that the public sign-off record is
retained and may be redistributed with the project.

Read [`docs/licensing_and_contributions.md`](docs/licensing_and_contributions.md) before opening a
pull request.

## Set up

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"
```

The editable install above is for editing, linting, typing, and pure/unit work only. Integration,
security, receipt, command, and release evidence must run against a noneditable wheel, because the
product is the installed distribution rather than the source tree.

Metrifid compiles with the exact admitted stable MuJoCo runtime and binds that engine build into
the compiled-artifact identity. The newest stable MuJoCo is the primary development and release
profile; exact retained older profiles remain compatibility-tested, with 3.9 as the support floor.

Python 3.11 or newer is supported, with no upper bound and no runtime rejection based on the
interpreter name. Release evidence currently uses CPython because the MuJoCo binary wheels exercised
by CI target CPython. The package metadata has no artificial upper bound. Linux and macOS are
admitted when the shared POSIX capability gate passes, with no architecture whitelist. Native
Windows is unsupported; use WSL.

CI runs two complete installed-wheel suites, because they answer different questions: one on the
primary resolver-latest profile and one on the declared dependency-floor profile. The other default
interpreter and platform profiles get bounded smokes. The exact interpreters and runner images
change as upstream support does, so the workflow itself is the authority rather than a list repeated
here: <https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/blob/main/.github/workflows/ci.yml>.

## Before you open a pull request

Use the editable development environment for static quality gates:

```bash
python -m ruff check .
python -m ruff format --check .
python -m mypy --strict src/metrifid
```

Run product, integration, security, receipt, command, and release tests from a separate noneditable
wheel environment:

```bash
python -m build --no-isolation
python -m venv .wheel-test-venv
source .wheel-test-venv/bin/activate
python -m pip install dist/metrifid-*.whl
python -m pip install -c .github/quality-constraints.txt   editables hatchling hypothesis mypy pytest pytest-randomly
python -m pytest -q tests
```

CI follows the same separation: it builds one wheel and one sdist once, then installs those exact
wheel bytes noneditably in every compatibility lane. Decision-bearing release evidence never comes
from the editable source tree.

## What the tests are for

Tests here assert *behavior*: statuses, exit codes, refusal reasons, published bytes, hash
stability, path confinement, read bounds and receipt validation. Please add tests of that kind.

Please do not add tests that assert source line counts, repository file inventories,
documentation wording, or frozen distribution hashes. Those pin the shape of the repository
rather than the behavior of the product, and they were removed for that reason.

## Things that are deliberately frozen

Changing any of these is a breaking change and needs its own discussion first:

- the command surface and each command's exit codes;
- the completed statuses;
- the complete-MJB identity method;
- the published schema identifiers;
- canonical JSON encoding, exact-number forms and self-hashing;
- output filenames and atomic publication;
- the top-level `metrifid` exports.

## Style

Ruff decides formatting and lint; there is no separate style guide to read. Keep functions small
enough to read in one go, and prefer a named helper over a comment explaining a long block. A few
functions are longer than that where splitting them would scatter one decision or one descriptor
lifetime across several frames; length is a smell to justify, not a limit enforced by a test.
Prefer a comment explaining *why* over one restating *what*.

## Reporting a security issue

See [`SECURITY.md`](SECURITY.md). Please do not open a public issue for a vulnerability.
