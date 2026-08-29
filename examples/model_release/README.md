# Static Model Change Gate example

This example shows the trustworthy two-pass workflow for a compiled-model difference. The
candidate changes one geom mass from `1.0` to `1.25`; every authored body, joint, geom and
actuator is named so the static gate can produce semantic witnesses.

## Run it

From the repository root, install Metrifid normally (not editable) and run the public SDK example:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install .
python examples/model_release/run_example.py
```

The default workspace is a new retained temporary directory printed by the script. To select the
location, pass an absent or empty directory:

```bash
python examples/model_release/run_example.py --workspace /tmp/metrifid-model-release-example
```

Expected statuses are:

```text
certify         : NOT_CERTIFIED_COMPILED_DIFFERS
discovery       : REVIEW_REQUIRED
declared review : WITHIN_DECLARED_POLICY
```

The exact artifact paths and the number of generated rules are printed at the end. The script uses
only public `metrifid.certify` and `metrifid.model_release` imports. Policy files are serialized
with sorted, compact standard-library JSON, and the script asserts that product receipts contain
neither absolute model entrypoint nor model root.

## Why two passes?

Certify first supplies the exact baseline and candidate complete-MJB SHA-256 identities. The
discovery policy binds the baseline, deliberately leaves the candidate unbound and contains no
rules. A differing unbound candidate is fail-closed, so the first review returns every explained
change plus an opaque candidate-binding residual and asks for review.

For demonstration, the script converts every non-opaque discovery change into an exact `ALLOW`
rule: exact selector, exact before digest and exact after digest. It binds the second policy to the
candidate MJB and writes the second review into a fresh output directory. A real maintainer should
inspect and justify those changes before writing policy; generation alone is not approval.

The mass edit produces more than a single row. MuJoCo derives body inertia and public compiled
fields from it, and every returned derived change remains independently policy-bearing. Neither
status makes a dynamic-behavior, safety, deployment-readiness or release-authorization claim.

See [`docs/model_release.md`](../../docs/model_release.md) for the complete schema and semantics.
