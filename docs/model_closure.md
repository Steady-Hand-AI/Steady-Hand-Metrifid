# Model closure

Before this tool compiles anything, it decides exactly which files *are* the model. That set is
the model closure, and it is what every identity, certificate and comparison is anchored to.

## What gets measured

Given a model root and an entrypoint, the source closure contains every admitted regular file
under that declared root. Every admitted regular file contributes its relative path, byte count,
and content hash even when the entrypoint never references it.

Dependency discovery is separate evidence. It asks MuJoCo which includes, meshes, height fields,
textures, and other assets compilation actually reaches, using MuJoCo 3.10.0 path semantics. An
unused regular file changes the source-closure identity, but it does not change compiled-MJB bytes
unless MuJoCo resolves it.

Every member is recorded with its relative POSIX path, its exact byte count and the SHA-256 of the
exact bytes read. The closure identity is a hash over that whole sorted set, so it changes if any
byte of any member changes.

The closure identity is content-addressed. Exact bytes and their hashes define it — not the
filesystem object those bytes happened to live in. Directory metadata is used to enforce the root
boundary, to refuse symlinks and non-regular members, to apply the byte budget, and to keep an
open from blocking; it is not part of the published identity.

## Declared bounds in this alpha release

Be precise about what is and is not guaranteed:

- Model-closure content is bounded by the documented byte limit.
- File-count, directory-count, and traversal-depth limits are **not** separately guaranteed in this
  alpha local release.
- This package is for local trusted-input workflows, not a hosted untrusted-upload service.

Adding count and depth budgets is a prerequisite for any future hosted tier that accepts untrusted
uploads. It is deliberately not claimed here.

## What is refused

A member that is not a regular file, a symlink, a path that escapes the model root, a member that
is missing or unreadable, a closure that exceeds the size budget, or an entrypoint whose first
complete top-level element is not `mujoco`. Each refusal names its own reason code and exits `64`;
nothing is guessed and nothing is silently skipped.

Traversal and reads are descriptor-confined. The root and every child directory are opened without
following links, and each regular file is opened relative to its admitted parent descriptor. Reads
are bounded to at most one byte beyond measured size, so replacement cannot redirect traversal and
a file that grew is detected rather than truncated. Final verification also requires the public
root path to still name the retained root object.

## Your files are not modified

The original source tree is never written to. The measured members are copied into a private
immutable snapshot, compilation happens from that snapshot, and the original source is re-measured
against the snapshot before anything is published. If the source content no longer matches the
snapshot, the run refuses with `MODEL_CLOSURE_MUTATED` rather than publishing.

This is why a certificate can name exactly what it compiled: whatever bytes the reader actually
read are the bytes it hashed, copied into the snapshot, compiled from, and finally checked the
source against.

## What content addressing does and does not distinguish

A substitution is often visible before any byte is read. Each member is measured with its device,
inode, mode and exact byte count, and the opened descriptor is checked against that identity. A
member that is no longer that exact regular file refuses with `MODEL_CLOSURE_MUTATED` without
reading replacement bytes.

If a replacement is admitted and read, the closure names those replacement bytes — exactly the
bytes that are hashed, copied into the snapshot and compiled — and never the earlier ones. The
run then continues only while the source still matches the snapshot; restoring the earlier bytes,
or any other divergence from the snapshot at final verification, refuses publication with
`MODEL_CLOSURE_MUTATED`.

If a member is replaced by a file containing *byte-identical* content, the content is the same and
the certificate does not distinguish that replacement. The tool makes no claim to detect a
byte-identical file-object replacement, and no published identity depends on doing so.

## Why the closure is the unit

Two source trees can differ in bytes and compile to the same model; the same tree can compile
differently under a different MuJoCo build. Anchoring identity to the measured closure keeps
those two facts separate: the closure identity says what went in, the compiled artifact identity
says what came out. `certify` reports both.

See [`compiled_certification.md`](compiled_certification.md) for the artifact side and
[`reference.md`](reference.md) for the published schemas.
