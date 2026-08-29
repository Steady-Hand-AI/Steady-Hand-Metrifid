"""Generate and review role-based native evidence through the installed lazy SDK."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    """Execute one strict run configuration and report its completed owned artifacts."""
    parser = argparse.ArgumentParser(
        description=(
            "Preflight two explicit profiles, pass their complete-state sentinels, create twelve "
            "native evidence cells, and immediately run Runtime Review."
        )
    )
    parser.add_argument("configuration", type=Path)
    arguments = parser.parse_args()

    from metrifid.runtime_review import run_runtime_review_configuration_file

    result = run_runtime_review_configuration_file(arguments.configuration)
    print(f"status={result.status.value}")
    print(f"reason_code={None if result.reason_code is None else result.reason_code.value}")
    print(f"receipt_sha256={result.receipt_sha256}")
    print(f"run_sha256={result.run_sha256}")
    print(f"runtime_review_json={result.runtime_review_json}")
    print(f"runtime_review_markdown={result.runtime_review_markdown}")
    print(f"runtime_review_run_json={result.runtime_review_run_json}")
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
