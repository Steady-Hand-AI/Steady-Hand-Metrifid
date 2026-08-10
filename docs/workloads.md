# Workloads

`compare` and `audit-timestep` need to know exactly what to run. A workload is that declaration:
one initial state, one action sequence, and an exact time grid. Nothing is inferred.

## The two artifacts

Both are NPZ files with a frozen schema, written by the public writers:

```python
from metrifid import write_state_artifact, write_actions_artifact
```

- **State** (`metrifid.state`, schema version `1`) — the initial `qpos`, `qvel` and activation values, with the
  per-joint and per-actuator offsets that say which block belongs to which name.
- **Actions** (`metrifid.actions`, schema version `1`) — one control row per control interval.

Both are admitted strictly: exact dtypes, exact shapes, offsets that cover every declared width,
finite values only, and canonical names in the exact order the compiled model aligns them. A
mismatch refuses rather than truncating or padding.

Each artifact carries two hashes: one over the raw file bytes, and one over its semantic content.
The semantic hash is what receipts bind to, so re-writing the same values does not change the
declared workload.

The public writers require an absent destination and never overwrite an existing file, directory,
or symlink. They acquire the final name with a descriptor-relative no-clobber hard link. If a later
verification fails after that link exists, the writer refuses but preserves the linked artifact;
failure cleanup removes only its private temporary.

## The time grid

Time is exact, never floating point:

```text
control_dt          how long one action is held
step_dt             the compiled model timestep for a role
control_intervals   how many actions the run applies
```

`control_dt` must be an exact integer multiple of each role's `step_dt`. If it is not, the run
refuses with a nonintegral control grid rather than rounding. Actions are held at the left
boundary of each interval — zero-order hold — and state is sampled at every boundary from 0
through the terminal boundary inclusive. Nothing is interpolated and no tail is fabricated: if a
run fails partway, the boundaries after the failure are absent, not filled in.

## What a comparison decides

`compare` replays both roles on the same grid, repeats the whole run to check determinism, and
compares monitored hinge and slide coordinates against tolerances you declare in physical units.
The result is one of four statuses, and a green result means only that no monitored coordinate
crossed its tolerance on *this* declared workload.

## What a timestep audit decides

`audit-timestep` evaluates every declared candidate timestep independently against one compiled
reference. It never assumes larger timesteps behave monotonically and never binary-searches. The
recommendation is the largest candidate supported by an unbroken ascending prefix of
within-tolerance results — a candidate that refuses or is inconclusive blocks itself and every
larger candidate.

The reference timestep is the comparison reference. It is not asserted to be correct.

See [`comparison.md`](comparison.md) and [`timestep_audit.md`](timestep_audit.md) for the full
configuration shapes and [`reference.md`](reference.md) for exits and schemas.
