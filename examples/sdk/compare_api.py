"""Run `compare_configuration_file` from Python and read its completed receipt.

Compare answers a different question from Certify: under one declared open-loop workload and one
declared set of joint tolerances, did anything monitored move outside those tolerances? The
configuration is strict JSON; nothing is inferred. A completed run returns a
``ComparisonRunResult`` whose receipt carries the decision, and a configuration or environment
problem raises instead of returning a result.

    python compare_api.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from _shared import MODEL_MJCF, write_model, write_workload

from metrifid.compare import ComparisonRunResult, compare_configuration_file


def _configuration(workspace: Path) -> Path:
    """Write one complete comparison configuration and everything it references."""
    write_model(workspace / "baseline", MODEL_MJCF)
    write_model(workspace / "candidate", MODEL_MJCF)
    write_workload(workspace)
    config = {
        "schema_version": 1,
        "baseline": {
            "model_root": "baseline",
            "entrypoint": "model.xml",
            "declared_step_dt": "0.001",
        },
        "candidate": {
            "model_root": "candidate",
            "entrypoint": "model.xml",
            "declared_step_dt": "0.001",
        },
        "initial_state": "state.npz",
        "actions": "actions.npz",
        "control_dt": "0.01",
        "repeats": 2,
        "joint_tolerances": {
            "shoulder": {
                "joint_type": "hinge",
                "angle_rad": "0.000001",
                "angular_velocity_rad_s": "0.0001",
            }
        },
        "aliases": None,
        "output_dir": "comparison_out",
    }
    path = workspace / "comparison.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def _describe(result: ComparisonRunResult) -> str:
    """Return the completed comparison status as plain text."""
    return str(result.receipt.status)


def main() -> int:
    """Run one comparison of a model against itself and report the completed status."""
    with tempfile.TemporaryDirectory(prefix="metrifid-sdk-compare-") as raw:
        workspace = Path(raw).resolve()
        config = _configuration(workspace)
        result = compare_configuration_file(config)
        print(f"comparison status : {_describe(result)}")
        print(
            f"published         : {result.comparison_json.name}, {result.comparison_markdown.name}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
