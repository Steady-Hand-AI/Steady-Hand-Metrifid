# The `compare` decision

## Command and configuration

```bash
metrifid compare comparison.json
```

The configuration file must be strict JSON accepted by `ComparisonConfig`. There is no YAML
form, no provider plugin, no capture command and no alternate configuration schema.

## Execution sequence

For each baseline/candidate role and each repeat, the comparator creates a fresh `MjData`,
restores the declared canonical state through role-local aligned qpos/qvel/activation slices,
clears declared auxiliary inputs, calls `mj_forward`, and captures boundary 0. For action row
`k`, it maps canonical actuator columns through the role-local control addresses, assigns
`data.ctrl` at the left boundary, calls `mj_step` exactly the admitted number of internal steps,
and captures boundary `k + 1`.

A successful run therefore has `N + 1` boundaries for `N` controls, including boundary 0 and
terminal boundary `N`. Baseline and candidate are compared only at equal canonical boundary
indices. The comparator performs no interpolation, resampling, time warping, tail filling or
silent truncation. The first nonfinite state stops that role with an incomplete captured prefix
and an explicit numerical/trace reason.

## Supported metrics

Monitored hinge and slide joints only:

```text
hinge position = abs(atan2(sin(candidate - baseline), cos(candidate - baseline)))
hinge velocity = abs(candidate - baseline)
slide position = abs(candidate - baseline)
slide velocity = abs(candidate - baseline)
```

The binary64 error is compared to the exact rational tolerance without first rounding the
tolerance to binary64. A crossing is strict: equality to the exact tolerance is not material.
Evidence records the maximum error, exact tolerance, ratio where defined, worst boundary/time,
and first crossing.

A monitored ball or free joint is refused before rollout using the controlled tolerance-unit
reason with the joint identity in evidence. Contact, body, site, sensor, reward, task-outcome and
closed-loop metrics are not implemented.

## Status and receipt

`compare` publishes one of four statuses under a fixed status precedence. Repeatability
signatures bind every captured boundary index, exact time, observed role-local clock bits,
qpos/qvel/activation bytes, warning snapshots, error logs, invalid-boundary evidence, and the
initial-state preservation flag. Numerical-invalid, trace-integrity, nondeterminism and
metric-crossing conditions produce reason records and finalize through `ComparisonReceipt`.

## Output safety

The declared output directory must be absent or an empty real directory. Files and symlinks
refuse as `OUTPUT_PATH_INVALID`; nonempty directories refuse as `OUTPUT_DIRECTORY_NOT_EMPTY`.
An output equal to or below either model root refuses before creation, including equivalent paths
spelled through `..` or symlink aliases. Both source snapshots remain alive and are reverified after decision evidence is complete. Temporary creation, descriptor-relative no-clobber linking, fsync, and cleanup stay relative to the retained output-directory descriptor; replacing its public path cannot redirect them. Each no-clobber final is recorded only after it identifies the retained sealed bytes. After the live source context closes, the final success check revalidates the public output path and both retained byte strings; a replaced public path therefore refuses instead of returning paths to unrelated files. A failed publication removes private temporaries but preserves every already-linked public
final; the command still refuses, and only a success exit with both expected files is a published
comparison result.

`comparison.json` is canonical strict JSON and remains the authoritative machine evidence,
including exact binary64 bits and full rational primitives.

`comparison.md` is a deterministic readable presentation of that same completed receipt. It
renders, in order: the status line, a decision summary, the claim boundary, repeatability, a
metric summary, the first crossing, incomplete-trace evidence when present, reasons when present,
identities, and the command used for the run. The renderer never mutates, re-finalizes or
recalculates the receipt, and two renders of one receipt are byte-identical.

Presentation rules are fixed. Tagged binary64 values are decoded and formatted with `.12g`. Exact
rationals render as a `.12g` decimal followed by the reduced `numerator/denominator` in
parentheses. Missing values render as an em dash, booleans render as `yes` or `no`, and table
text escapes backslash, pipe, carriage return and newline. Markdown never contains Python or wire
primitive representations; consume `comparison.json` for exact values. Markdown formatting is
presentation only and is not part of the stable interface.

## Claim boundary

A completed receipt applies only to the declared workload and monitored hinge/slide coordinates.
It does not establish global equivalence, physical correctness, hardware safety, causal
diagnosis, or behavior outside the declared state/action/time/tolerance contract.

## Fixed preexecution budgets

After model alignment, workload loading, exact time-grid admission, monitored-joint selection and
comparison-contract construction, the comparator calculates two fixed safety budgets before
replay:

```text
maximum total internal steps = 10,000,000
maximum retained float64 trace bytes = 268,435,456
```

The internal-step request is
`control_intervals * repeats * (baseline_substeps_per_control + candidate_substeps_per_control)`.
The retained trace request is `boundary_count * repeats * 2 roles * 8 bytes * (complete canonical
qpos width + complete canonical qvel width + complete canonical activation width)`.

Both calculations use exact Python integer arithmetic. Equality with a maximum is admitted; a
strictly larger request produces `COVERAGE_INSUFFICIENT` with `INTERNAL_STEP_BUDGET_EXCEEDED`,
`TRACE_MEMORY_BUDGET_EXCEEDED`, or both. The completed receipt remains bound to the exact models,
workload, alignment, time contract, environment, monitored joints and tolerances, but replay is
not entered and no boundary is advanced. These are fixed safety limits, not user-configurable
tolerances.
