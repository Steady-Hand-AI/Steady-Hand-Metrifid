# Complete `metrifid compare` example

A runnable comparison of one model against a differently-written copy of itself. Everything the
configuration references is either in this directory or created by the preparation script, so the
example works after copying this directory anywhere.

## Run it

```bash
python prepare_workload.py
metrifid compare comparison.json
```

The first command writes `state.npz` and `actions.npz` through the public workload writers. It
refuses to overwrite an artifact that already exists, because a recorded result may have been
measured against it.

The second command publishes `comparison_out/comparison.json` and `comparison_out/comparison.md`
and exits 0 with status `NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD`. The output directory must be
absent or empty; delete `comparison_out/` before re-running.

## What is here

| Path | What it is |
| --- | --- |
| `comparison.json` | the complete strict configuration, all ten top-level fields |
| `baseline/model.xml` | one hinge joint, one motor |
| `candidate/model.xml` | the same physics with different attribute order and number spelling |
| `prepare_workload.py` | writes the canonical initial-state and actions artifacts |

## The configuration

Every top-level field is required and nothing is inferred:

| Field | Value here | Notes |
| --- | --- | --- |
| `schema_version` | `1` | frozen |
| `baseline`, `candidate` | `model_root`, `entrypoint`, `declared_step_dt` | roots resolve relative to this file |
| `initial_state`, `actions` | `state.npz`, `actions.npz` | canonical NPZ artifacts |
| `control_dt` | `"0.01"` | exact decimal token, never a float |
| `repeats` | `2` | integer between 2 and 5 |
| `joint_tolerances` | one `hinge` entry | requires exactly `angle_rad` and `angular_velocity_rad_s` |
| `aliases` | `null` | no name mapping needed; both models use the same names |
| `output_dir` | `"comparison_out"` | must be absent or empty |

Tolerances are exact decimal tokens, not floats. `angle_rad` is in radians and
`angular_velocity_rad_s` in radians per second.

## Trying a real difference

Edit `candidate/model.xml` and change the geom `mass` from `1.5` to `1.6`, delete `comparison_out/`,
then run the comparison again. The status becomes `MATERIAL_BEHAVIOR_CHANGE` and the receipt reports
the first boundary where the monitored joint crossed its tolerance.

For the compiled-artifact question — *did the model change at all?* — see `examples/certify/` and
`metrifid certify`. For the Python API, see `docs/sdk.md`.
