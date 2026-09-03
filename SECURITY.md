# Security Policy

## Supported Versions

Currently, Metrifid is beta software under active development. The only supported version is the current release (`0.7.x`).

| Version | Supported          |
| ------- | ------------------ |
| 0.7.x   | :white_check_mark: |
| < 0.7.0 | :x:                |

## Reporting a Vulnerability

Please do not report security vulnerabilities through public GitHub issues.

Instead, please report vulnerabilities responsibly via [GitHub Private Vulnerability Reporting](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid/security/advisories/new). If you prefer, you can also reach out to the maintainers directly.

We ask that you wait until we have reviewed and addressed the vulnerability before disclosing it publicly. We will try to address reports as quickly as possible.

## Scope

In scope: the `metrifid` package, its command-line entrypoints, and the repository automation in
this tree.

Out of scope: vulnerabilities in MuJoCo itself (report those upstream), the models or assets you
supply as input, and the security of the environment you choose to run the tool in.

## What the tool does with your files

`certify` measures every admitted regular file under each declared model root and copies those
members into private immutable snapshots. Dependency discovery separately identifies the files
MuJoCo reaches during compilation. An unused regular file changes source-closure identity, but not
the compiled MJB unless MuJoCo resolves it. The tool never modifies your source tree, opens a
network connection, or sends telemetry.

The closure identity is content-addressed: each member is named by its relative path, its exact
byte count and the SHA-256 of the exact bytes read. Whatever bytes are read are the bytes that are
hashed, copied into the snapshot and compiled, and the source is re-measured against that snapshot
before anything is published. A source whose content no longer matches the snapshot refuses with
`MODEL_CLOSURE_MUTATED` instead of publishing.

These remain fail-closed refusals: a symlinked member, a path that escapes the model root, a
directory, FIFO, socket or other non-regular member, a missing or unreadable member, and a closure
over the byte budget.

Security-sensitive model traversal, output publication, cleanup, and workload-artifact publication
remain bound to admitted directory descriptors. NPZ admission enforces its byte limit while reading.
Replacing an admitted path may cause refusal, but cannot redirect a read, write, publication, or cleanup. Immediately before a successful command returns, Metrifid rechecks the public output path and every retained output byte after all source or input contexts have closed.
Public output names are acquired with no-clobber hard links and are recorded only after exact
retained-object and sealed-byte verification. Failure cleanup never unlinks a public final; it may
remove only a still-owned private temporary or exact private timestep-audit workspace artifact.
Consequently, a failed command can leave already-linked evidence and must never be interpreted as a
successful publication from filenames alone.

The tool does not claim to detect a byte-identical replacement of the underlying file object. Such
a replacement carries the same content, so the closure identity, the snapshot and the compiled
artifact are unchanged by it, and no published claim depends on distinguishing it.
