# Run Runtime Review example

This data-only example asks Metrifid to create the fixed twelve-cell native evidence set for one
self-contained pendulum, then immediately pass that evidence to the existing Runtime Review
referee.

Before running it, edit only the two interpreter paths in `runtime_review_run.json`:

- `baseline_python`: an absolute executable launcher for the prepared baseline profile;
- `candidate_python`: an absolute executable launcher for the prepared candidate profile.

Metrifid does not search for, create, install, upgrade, or repair either environment. The declared
paths must already satisfy the measured profile contract: coherent stable MuJoCo at or above 3.9,
the same exact NumPy version and distribution payload in both profiles, the same Python build and
host, and the frozen deterministic thread environment. The paths may name the same MuJoCo version,
but they must remain lexically distinct explicit launchers. Keep `manifest.json` and `model.xml`
beside the run configuration, then execute from this directory:

```bash
metrifid run-runtime-review runtime_review_run.json
```

The command runs two one-shot profile preflights. Each preflight compiles this exact fixture and must
pass a same-profile sentinel over the complete public MuJoCo integration state. Only then does the
command start twelve sequential one-shot evidence cells. It preserves every preflight, sentinel,
command, stream, exit code, identity, and six-member evidence directory.

It generates the role-based schema-version-2 retained-evidence configuration and calls the same
referee used by `metrifid review-runtime`. The generated configuration and owned receipt bind
`baseline` and `candidate` roles to the measured package/native/Python/NumPy/host/worker identities;
versions are evidence, not role names. A sentinel failure starts zero cells and publishes no
scientific decision. A completed status may legitimately be non-green; exits `20`, `30`, and `40`
are scientific decisions, while `64` is a bounded operational refusal.

The result is limited to the exact interpreters, measured native profiles, fixture closure,
workload, channels, tolerances, three grids, two repeats, and one-second horizon in the receipt. It
does not establish universal compatibility, hardware safety, task success, or cryptographic origin.
