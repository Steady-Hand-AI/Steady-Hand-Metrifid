# Certify example

Two questions this directory answers in one command:

- Do two differently-written MuJoCo sources compile to the *same* model?
- Does a one-line physical edit actually change the compiled model?

## Run it

From the repository root, install the source normally and run the example:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python examples/certify/run_example.py
```

Expected output:

```text
different source, same compiled model : CERTIFIED_COMPILED_EQUIVALENCE (exit 0)
one changed mass                      : NOT_CERTIFIED_COMPILED_DIFFERS (exit 40)

all 6 checks passed
```

The script runs the installed `metrifid` executable from a noneditable wheel, writes into a
temporary directory, uses no network, and exits non-zero if any expectation is unmet.

It needs an environment the shared runtime gate admits: Python 3.11 or newer on Linux or macOS with
the required POSIX capabilities and a stable MuJoCo runtime at or above the 3.9 support floor.
Metrifid binds the exact admitted runtime into the result; the newest stable runtime is the primary
release profile and retained older profiles remain compatibility-tested. Architecture is never
rejected. Native Windows is unsupported; run it under WSL. Elsewhere `certify` refuses with exit 64
rather than guessing.

## The models

| File | What it is |
| --- | --- |
| `equivalent/baseline.xml` | one hinge-driven link, 1.5 kg |
| `equivalent/candidate.xml` | the same model with reordered attributes, comments and whitespace |
| `changed.xml` | `baseline.xml` with the link mass changed to 2.0 kg |

`equivalent/candidate.xml` differs from the baseline in source bytes but not in anything MuJoCo
compiles, which is exactly the case a source diff cannot answer and `certify` can. `changed.xml`
differs by one physical property, and the receipt's field report names `body_mass` and shows the
changed index.

## What the exit codes mean

| Exit | Status | Meaning |
| --- | --- | --- |
| `0` | `CERTIFIED_COMPILED_EQUIVALENCE` | every byte of both compiled artifacts matched |
| `40` | `NOT_CERTIFIED_COMPILED_DIFFERS` | at least one byte differed |
| `64` / `70` | refusal | the inputs or the environment were not admissible |

A certificate is a runtime-bound statement about compiled artifacts: it holds for the runtime
identity recorded in the receipt. It is not a claim about behavior, source text, licensing or
task suitability, and not a claim that another platform or MuJoCo version produces the same
bytes. See
[`docs/compiled_certification.md`](../../docs/compiled_certification.md).
