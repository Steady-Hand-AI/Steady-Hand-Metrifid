"""Run one complete Workload Qualification campaign through the public Python API.

The question this answers, for one baseline model and the probe models you supply:

    Do three of my declared workloads detect every declared probe at or above the magnitude I care
    about, and which probes stay invisible?

Everything the campaign needs is created here in a temporary directory: the baseline model, one
probe ladder derived from it, four recorded workloads, and the strict JSON configuration that
declares them. Nothing is read from the Metrifid repository and ``sys.path`` is never modified, so
this script runs from any directory against an installed distribution.

The campaign is deliberately small - four workloads, one probe group, two rungs - so it finishes in
seconds while still exercising the real decision: exact enumeration of every three-workload subset,
suffix-based detection floor, and the published receipt with its linked evidence.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from _shared import ACTUATOR_NAMES, MODEL_MJCF, write_model, write_workload

from metrifid.workload_qualification import (
    QualificationExitCode,
    QualificationResult,
    QualificationStatus,
    WorkloadQualificationOperationError,
    load_and_validate_workload_qualification_receipt,
    qualify_configuration_file,
)

# The probe ladder: the same model with the shoulder joint damped a little more at each rung. The
# magnitudes below describe those edits; Metrifid compares the exact closure bytes and reports what
# was detected, and does not independently verify that these labels describe the edits.
_BASELINE_DAMPING = "0.02"
_RUNGS = (("0.03", "0.05"), ("0.06", "0.08"))
_REQUIRED_DETECTION_MAGNITUDE = "0.03"
_MAGNITUDE_SEMANTICS = (
    "absolute increase in the declared parameter, in the source model's native units"
)

# Four candidate workloads driving the joint progressively harder. A gentle workload cannot reveal a
# small damping change; the qualification is what tells you so instead of you guessing.
_WORKLOADS = ("idle", "gentle", "moderate", "vigorous")


def _damped(text: str, damping: str) -> str:
    """Return the shared model with an explicit damping value on the shoulder joint."""
    marker = '<joint name="shoulder" type="hinge" axis="0 1 0"/>'
    if text.count(marker) != 1:
        raise SystemExit("the shared example model no longer has one shoulder joint element")
    return text.replace(marker, marker.replace("/>", f' damping="{damping}"/>'))


def _configuration(root: Path) -> Path:
    """Create every declared input beside a strict configuration file and return its path."""
    write_model(root / "baseline", _damped(MODEL_MJCF, _BASELINE_DAMPING))

    variants = []
    for index, (magnitude, damping) in enumerate(_RUNGS, start=1):
        write_model(root / "probes" / f"rung_{index}", _damped(MODEL_MJCF, damping))
        variants.append(
            {
                "magnitude": magnitude,
                "candidate": {
                    "model_root": f"probes/rung_{index}",
                    "entrypoint": "model.xml",
                    "declared_step_dt": "0.001",
                },
            }
        )

    for workload_id in _WORKLOADS:
        write_workload(root / "workloads" / workload_id)

    configuration = {
        "schema_version": 1,
        "baseline": {
            "model_root": "baseline",
            "entrypoint": "model.xml",
            "declared_step_dt": "0.001",
        },
        "probe_groups": [
            {
                "probe_id": "shoulder_damping_increase",
                "parameter": "shoulder.damping",
                "direction": "increase",
                "magnitude_semantics": _MAGNITUDE_SEMANTICS,
                "required_detection_magnitude": _REQUIRED_DETECTION_MAGNITUDE,
                "variants": variants,
            }
        ],
        "workloads": [
            {
                "workload_id": workload_id,
                "initial_state": f"workloads/{workload_id}/state.npz",
                "actions": f"workloads/{workload_id}/actions.npz",
                "control_dt": "0.01",
            }
            for workload_id in _WORKLOADS
        ],
        "repeats": 2,
        "joint_tolerances": {
            "shoulder": {
                "joint_type": "hinge",
                "angle_rad": "0.0005",
                "angular_velocity_rad_s": "0.005",
            }
        },
        "aliases": None,
        "budget": 3,
        "output_dir": "qualification_out",
    }
    path = root / "qualification.json"
    path.write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    return path


def _report(result: QualificationResult) -> None:
    """Print the completed decision and the two files the campaign published."""
    print(f"status:            {result.status.value}")
    print(f"exit code:         {result.exit_code}")
    print(f"selected workloads: {', '.join(result.receipt['selected_workload_ids'])}")
    print(f"published JSON:     {result.qualification_json.name}")
    print(f"published Markdown: {result.qualification_markdown.name}")


def main() -> int:
    """Run one bounded synthetic campaign and revalidate the receipt it published."""
    print(f"actuators driven: {', '.join(ACTUATOR_NAMES)}")
    with tempfile.TemporaryDirectory(prefix="metrifid-sdk-qualification-") as scratch:
        root = Path(scratch).resolve()
        configuration = _configuration(root)

        try:
            result = qualify_configuration_file(configuration)
        except WorkloadQualificationOperationError as refusal:
            # A bounded operational refusal is a completed, typed outcome, not a crash.
            print(f"the campaign was refused: {refusal}")
            return int(QualificationExitCode.UNRESOLVED)

        _report(result)

        # The receipt is only worth reading because it can be checked. The loader re-admits the
        # retained raw configuration, re-reads every retained comparison configuration and receipt
        # under the published output root, rebinds each to the digest the aggregate recorded for it,
        # and recomputes the decision from those cells. It is a set of enumerated bindings, not a
        # blanket guarantee: see docs/workload_qualification.md for exactly which bindings it makes.
        document = load_and_validate_workload_qualification_receipt(result.qualification_json)
        print(f"receipt revalidated: {document['receipt_sha256']}")
        print(f"comparisons planned: {document['planned_comparisons']}")

        if result.status is QualificationStatus.QUALIFIED_FOR_DECLARED_PROBES:
            print("every declared probe was detected at or above its required magnitude")
        else:
            witness = document["witnesses"]["first_witness"]
            print(f"first witness: {witness}")
        return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
