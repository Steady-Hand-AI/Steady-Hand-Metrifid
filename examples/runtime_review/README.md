# Historical retained Runtime Review example

`runtime_review.json` is an explicit schema-version-1 specimen for the immutable historical
MuJoCo 3.10 baseline and MuJoCo 3.11 candidate evidence contract. Its version-coded profile IDs are
history, not the role convention for new runs.

Replace `PLACEHOLDER_EVIDENCE` only with the matching twelve-cell historical evidence tree. Do not
rewrite the file into a schema-version-2 claim or substitute other runtime identities: the v1 route
validates the original exact contract.

New evidence is created with `metrifid run-runtime-review` and semantic `baseline` / `candidate`
roles. That command measures the exact MuJoCo, native-library, Python, NumPy, host, and worker
identities, requires both complete-integration-state sentinels to pass, then generates the truthful
schema-version-2 configuration and owned receipt. See `../runtime_review_run/README.md` for that
path.
