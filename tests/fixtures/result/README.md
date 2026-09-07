# Result presentation fixtures

Real receipts, produced by running the installed distribution over the two model pairs in
`examples/certify/`. They are recorded here so the presentation tests can assert against evidence a
real run actually produced, without needing MuJoCo to be importable.

| File | Produced by | Records |
|---|---|---|
| `changed_model_release.json` | `metrifid diff examples/certify/equivalent/baseline.xml examples/certify/changed.xml` | `REVIEW_REQUIRED`, exit 40, twelve canonical changes, nine of them joined to descriptive witnesses |
| `equal_model_release.json` | `metrifid diff examples/certify/equivalent/baseline.xml examples/certify/equivalent/candidate.xml` | `NO_COMPILED_CHANGE`, exit 0, no changes, no descriptive field report |
| `changed_certification.json` | `metrifid certify examples/certify/equivalent/baseline.xml examples/certify/changed.xml` | `NOT_CERTIFIED_COMPILED_DIFFERS`, nine changed public fields |

These are recorded runs, not constructed documents: every digest, bit pattern and runtime identity
in them is what the producer wrote. Regenerate them with the three commands above when the receipt
schemas change; do not hand-edit them.
