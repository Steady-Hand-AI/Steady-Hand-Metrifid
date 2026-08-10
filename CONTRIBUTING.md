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

The MuJoCo native engine `3.10.0` exactly is a hard requirement — the compiled-artifact identity is
only meaningful against a fixed engine build. A binding-only `3.10.0.postN` package targets that
same engine and is accepted; any other engine version refuses rather than guessing.

Python 3.11 or newer is supported, with no upper bound and no runtime rejection based on the
interpreter name. Release evidence currently uses CPython because the MuJoCo binary wheels exercised
by CI target CPython. The package metadata has no artificial upper bound. CI is configured for
CPython 3.11–3.14 on Linux x86_64, CPython 3.12 and 3.14 on macOS arm64, and CPython 3.12 on
macOS x86_64. Linux and macOS are admitted when the shared POSIX capability gate passes, with no
architecture whitelist. Native Windows is unsupported; use WSL.

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

- the three command shapes and their exit codes;
- the completed statuses;
- the complete-MJB identity method;
- the published schema identifiers;
- canonical JSON encoding, exact-number forms and self-hashing;
- output filenames and atomic publication;
- the top-level `metrifid` exports.

## Style

Ruff decides formatting and lint; there is no separate style guide to read. Keep functions small
enough to read in one screen — nothing in `src/metrifid` exceeds 80 lines. Prefer a named helper
over a comment explaining a long block, and prefer a comment explaining *why* over one restating
*what*.

## Reporting a security issue

See [`SECURITY.md`](SECURITY.md). Please do not open a public issue for a vulnerability.
