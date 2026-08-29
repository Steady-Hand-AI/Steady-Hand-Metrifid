# Case study: a formatting-only change in MuJoCo Menagerie

This is the first real-world case `certify` was run against. Everything below comes from receipts
published by the tool itself and reproduced by an independent direct-MuJoCo measurement.

## The question

Two revisions of a public robot model. The source files differ. Does the *model* differ?

A source diff cannot answer that: whitespace, attribute order, comments and include structure all
change the bytes without changing anything MuJoCo compiles. Running a simulation cannot answer it
either — a workload only tells you about the trajectories you happened to run.

## What was measured

```text
repository            https://github.com/google-deepmind/mujoco_menagerie
parent revision       43b5e628efe40a7d92eea398e703093ca5df27b1
candidate revision    ac6b2b09983786f3036cab1000221017fa2193b4
model directory       unitree_g1
runtime               MuJoCo Python and native 3.10.0, CPython 3.12, Darwin x86_64
```

Both worktrees were verified clean before and after the run.

## Result

All four entrypoints certified: every byte of both compiled artifacts matched.

| Entrypoint | Closure members | Compiled artifact | Complete-MJB SHA-256 |
| --- | ---: | ---: | --- |
| `g1.xml` | 63 | 75,077,600 B | `b2afa4563fe1ef2e…` |
| `g1_mjx.xml` | 63 | 70,874,177 B | `7798436d1781b39b…` |
| `g1_with_hands.xml` | 63 | 119,818,255 B | `4a461a760be39895…` |
| `scene_mjx.xml` | 63 | 75,873,689 B | `bbad449a4e48f3b4…` |

For every entrypoint the two revisions have **different source closures** and **identical
compiled artifacts**. That is the whole result: the change between these two revisions does not
reach the compiled model for these four entrypoints.

An independent script that imports no part of this package compiled the same eight artifacts
directly with MuJoCo and produced the same eight digests.

## The same four entrypoints, measured again on Linux

The measurement above was made on Darwin x86_64. When `certify` was validated on Linux x86_64, the
four entrypoints were measured there too, at the same two revisions, with CPython 3.12 and MuJoCo
Python and native 3.10.0:

```text
runtime               MuJoCo Python and native 3.10.0, CPython 3.12, Linux x86_64
result                all four entrypoints certified, parent against candidate
```

That is a second measurement, made on its own platform and reported on its own terms. It repeats
the finding that these two revisions do not reach the compiled model for these four entrypoints;
it does not extend either run's certificate beyond the runtime identity recorded in it. Each
receipt states the runtime that produced it, and a different operating system, MuJoCo version or
build may serialize a model differently.

## What was *not* needed

```text
no workload
no initial state
no action sequence
no tolerances
no joint or actuator name alignment
```

`certify` compiles and serializes. It never allocates `mjData` and never steps.

## Control: a real physical change

The same command on a one-property edit — the pelvis inertial mass changed from 3.813 kg to
3.913 kg — returned `NOT_CERTIFIED_COMPILED_DIFFERS` with exit 40, 795 differing bytes, and a
field report naming `body_mass` at the changed index along with the derived fields the change
propagates into. The tool distinguishes "the source moved" from "the model moved".

## What this claims, and what it does not

The claim for each certified pair is exactly:

> The measured baseline and candidate source closures, compiled under the recorded runtime
> identity, produced byte-identical complete MJB artifacts.

Limits that come with it:

- It holds for the recorded runtime only. Another MuJoCo build may serialize the same model
  differently.
- It says nothing about behavior. Identical artifacts plus identical inputs plus deterministic
  execution imply identical simulation, but this command verifies none of those premises.
- It says nothing about source text, licensing or visual intent.
- It covers **these four entrypoints at these two revisions** and nothing else. No claim is made
  about any other Menagerie model or any other revision, and the case makes no repository-wide
  coverage claim.

For the full contract see [`compiled_certification.md`](compiled_certification.md).
