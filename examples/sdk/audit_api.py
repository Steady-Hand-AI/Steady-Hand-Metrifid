"""Run `audit_configuration_file` from Python and read its recommendation.

Audit Timestep answers: of the candidate timesteps I declared, which still produce no material
difference against the reference timestep, for this workload and these tolerances? Every declared
candidate is classified, and the recommendation may not cross a candidate that produced no
trustworthy evidence. A completed campaign returns an ``AuditRunResult``; a configuration or
environment problem raises instead.

    python audit_api.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from _shared import MODEL_MJCF, write_model, write_workload

from metrifid.timestep_audit import AuditRunResult, audit_configuration_file


def _configuration(workspace: Path) -> Path:
    """Write one complete timestep-audit configuration and everything it references."""
    write_model(workspace / "model", MODEL_MJCF)
    write_workload(workspace)
    config = {
        "schema_version": 1,
        "model_root": "model",
        "entrypoint": "model.xml",
        "initial_state": "state.npz",
        "actions": "actions.npz",
        "control_dt": "0.01",
        "repeats": 3,
        "joint_tolerances": {
            "shoulder": {
                "joint_type": "hinge",
                "angle_rad": "0.005",
                "angular_velocity_rad_s": "0.25",
            }
        },
        "candidate_step_dts": ["0.002", "0.005"],
        "workload_kind": "SCREENING",
        "workload_label": "sdk example screening sweep",
        "output_dir": "audit_out",
    }
    path = workspace / "timestep_audit.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def _report(result: AuditRunResult) -> None:
    """Print each candidate classification and the completed-prefix recommendation."""
    aggregate = result.aggregate
    candidates = aggregate.get("candidates")
    if isinstance(candidates, list):
        for row in candidates:
            if isinstance(row, dict):
                print(f"  {row.get('token')} : {row.get('classification')}")
    recommendation = aggregate.get("recommendation")
    if isinstance(recommendation, dict):
        print(f"recommended token : {recommendation.get('candidate_token')}")


def main() -> int:
    """Run one screening audit over two candidate timesteps and report the outcome."""
    with tempfile.TemporaryDirectory(prefix="metrifid-sdk-audit-") as raw:
        workspace = Path(raw).resolve()
        result = audit_configuration_file(_configuration(workspace))
        print("candidate classifications:")
        _report(result)
        print(f"published         : {result.audit_json.name}, {result.audit_markdown.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
