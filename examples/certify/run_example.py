#!/usr/bin/env python3
"""Run `metrifid certify` twice against the bundled example models.

    python examples/certify/run_example.py

The first pair is the same model written two different ways. Its compiled artifacts are
identical, so certify exits 0. The second pair changes one mass, so the artifacts differ and
certify exits 40.

This drives the installed `metrifid` executable, not the source tree, so it exercises exactly
what a user gets from the wheel. It needs no network and writes only into a temporary directory.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from metrifid.certify import (
    load_and_validate_certification_receipt,
    validate_receipt,
)

EXAMPLES = Path(__file__).resolve().parent


def executable() -> str:
    """Locate the installed console script, failing loudly if the package is not installed."""
    found = shutil.which("metrifid")
    if found is None:
        raise SystemExit(
            "metrifid is not on PATH. Install the package first. From the repository root:\n"
            "    python -m pip install ."
        )
    return found


def certify(baseline: Path, candidate: Path, output: Path) -> tuple[int, dict[str, object]]:
    """Run one certification and return its exit code and published receipt."""
    completed = subprocess.run(
        [executable(), "certify", str(baseline), str(candidate), "--output", str(output)],
        capture_output=True,
        text=True,
        check=False,
    )
    receipt_path = output / "certification.json"
    if not receipt_path.is_file():
        raise SystemExit(
            f"certify did not publish a receipt (exit {completed.returncode}):\n{completed.stderr}"
        )
    receipt = load_and_validate_certification_receipt(receipt_path.read_bytes())
    return completed.returncode, receipt


def main() -> int:
    """Confirm equivalent and mass-changed models yield exits 0 and 40 with valid receipts."""
    checks: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory() as raw:
        # resolve(): the output path may contain no symlinked component, and the
        # platform temporary directory is often reached through one.
        workspace = Path(raw).resolve()

        equivalent_exit, equivalent = certify(
            EXAMPLES / "equivalent" / "baseline.xml",
            EXAMPLES / "equivalent" / "candidate.xml",
            workspace / "equivalent",
        )
        changed_exit, changed = certify(
            EXAMPLES / "equivalent" / "baseline.xml",
            EXAMPLES / "changed.xml",
            workspace / "changed",
        )

        print(
            f"different source, same compiled model : {equivalent['status']} (exit {equivalent_exit})"
        )
        print(f"one changed mass                      : {changed['status']} (exit {changed_exit})")

        for label, receipt in (("equivalent", equivalent), ("changed", changed)):
            try:
                # The receipt was already admitted and revalidated by the raw loader when it was
                # read; revalidating the parsed mapping keeps the example's check explicit.
                validate_receipt(receipt)
                checks.append((f"{label} receipt validates", True))
            except Exception as exc:
                checks.append((f"{label} receipt validates: {exc}", False))

        checks.append(("equivalent pair exits 0", equivalent_exit == 0))
        checks.append(
            (
                "equivalent pair is certified",
                equivalent["status"] == "CERTIFIED_COMPILED_EQUIVALENCE",
            )
        )
        checks.append(("changed pair exits 40", changed_exit == 40))
        checks.append(
            ("changed pair is not certified", changed["status"] == "NOT_CERTIFIED_COMPILED_DIFFERS")
        )

    failures = [label for label, ok in checks if not ok]
    for label in failures:
        print(f"FAILED: {label}", file=sys.stderr)
    if failures:
        return 1
    print(f"\nall {len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
