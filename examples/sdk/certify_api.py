"""Run `certify_models` from Python and read its completed decision.

Certify answers one question: do these two model sources compile to byte-identical artifacts? The
answer is a completed decision, not an exception — a differing pair returns a
``NOT_CERTIFIED_COMPILED_DIFFERS`` result just as normally as an identical pair returns
``CERTIFIED_COMPILED_EQUIVALENCE``. Exceptions are reserved for refusals: an unusable model root,
an occupied output directory, or an unsupported runtime.

    python certify_api.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from _shared import CHANGED_MJCF, EQUIVALENT_MJCF, MODEL_MJCF, write_model

from metrifid.certify import (
    CertifyResult,
    certify_models,
    load_and_validate_certification_receipt,
)


def _run(workspace: Path) -> tuple[CertifyResult, CertifyResult]:
    """Certify one equivalent pair and one physically changed pair."""
    baseline = write_model(workspace / "baseline", MODEL_MJCF)
    equivalent = write_model(workspace / "equivalent", EQUIVALENT_MJCF)
    changed = write_model(workspace / "changed", CHANGED_MJCF)
    return (
        certify_models(str(baseline), str(equivalent), str(workspace / "out_equivalent")),
        certify_models(str(baseline), str(changed), str(workspace / "out_changed")),
    )


def main() -> int:
    """Certify both pairs, revalidate both receipts, and report the two decisions."""
    with tempfile.TemporaryDirectory(prefix="metrifid-sdk-certify-") as raw:
        # resolve(): Metrifid refuses to publish through a symlinked output path, and the
        # platform temporary directory is often reached through one.
        workspace = Path(raw).resolve()
        equivalent, changed = _run(workspace)

        for result in (equivalent, changed):
            # The raw loader is the same strict admission path an independent reader would use.
            load_and_validate_certification_receipt(result.certification_json.read_bytes())

        print(f"equivalent pair : {result_status(equivalent)}")
        print(f"changed pair    : {result_status(changed)}")
        print(f"receipt digest  : {equivalent.receipt_sha256}")
        print(
            f"published       : {equivalent.certification_json.name}, "
            f"{equivalent.certification_markdown.name}"
        )
    return 0


def result_status(result: CertifyResult) -> str:
    """Return one completed certification status as plain text."""
    return str(result.status)


if __name__ == "__main__":
    sys.exit(main())
