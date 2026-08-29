"""End-to-end workload qualification against real compiled MuJoCo models.

These run the installed product: real model admission, real comparisons, a real published receipt
pair. The fixtures are synthetic in the sense that the models are small and authored here, but every
status in the receipt comes from a completed comparison, not from a stub.
"""

from __future__ import annotations

import json
import runpy
import shutil
from pathlib import Path

import pytest
import pytest_check as check

from metrifid.workload_qualification import (
    QualificationStatus,
    WorkloadQualificationOperationError,
    load_and_validate_workload_qualification_receipt,
    qualify_configuration_file,
)

_EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "workload_qualification"

pytestmark = pytest.mark.skipif(
    not _EXAMPLE.is_dir(), reason="the workload qualification example is required"
)


def _prepare(tmp_path: Path, *, tolerance: str | None = None) -> Path:
    """Copy the example, write its workload artifacts, and optionally retune the tolerance."""
    root = tmp_path / "case"
    shutil.copytree(_EXAMPLE, root)
    # Run the shipped preparation script the way a reader would, but call its entry point
    # directly so its process-exit convention does not end the test.
    namespace = runpy.run_path(str(root / "prepare_workloads.py"))
    assert namespace["main"]() == 0, (
        "the shipped workload preparation script did not complete, so there are no workload "
        "artifacts for the campaign to qualify"
    )
    if tolerance is not None:
        config_path = root / "qualification.json"
        primitive = json.loads(config_path.read_text(encoding="utf-8"))
        primitive["joint_tolerances"]["shoulder"]["angle_rad"] = tolerance
        primitive["joint_tolerances"]["shoulder"]["angular_velocity_rad_s"] = tolerance
        config_path.write_text(json.dumps(primitive, indent=2) + "\n", encoding="utf-8")
    return root / "qualification.json"


def test_a_complete_qualified_run_publishes_and_revalidates_all_evidence(
    tmp_path: Path,
) -> None:
    """Run the qualified campaign once and check its result, evidence, reload, and no-clobber."""
    configuration = _prepare(tmp_path)
    result = qualify_configuration_file(configuration)

    check.is_true(
        result.status is QualificationStatus.QUALIFIED_FOR_DECLARED_PROBES,
        f"the shipped example completed as {result.status} instead of qualifying for the probe "
        "ladder it declares",
    )
    check.equal(
        result.exit_code,
        0,
        "a qualified run does not exit with the code that status is frozen to",
    )
    assert result.qualification_json.is_file(), (
        "the published JSON receipt is required before the reader can revalidate the campaign"
    )
    check.is_true(
        result.qualification_markdown.is_file(),
        "the run reported a published Markdown receipt that is not a file on disk",
    )

    receipt = result.receipt
    check.equal(
        len(receipt["selected_workload_ids"]),
        3,
        "the published receipt does not select the three workloads the declared budget admits",
    )
    check.equal(
        receipt["excluded_workload_ids"],
        [],
        "a workload was excluded from a run whose zero-change controls should all stay quiet",
    )
    counts = receipt["execution_counts"]
    check.equal(
        counts["zero_change_comparisons"],
        4,
        "the campaign did not run one zero-change control per declared workload candidate",
    )
    check.equal(
        counts["probe_comparisons"],
        16,
        "the campaign did not run every declared workload against every rung of the probe ladder",
    )
    check.equal(
        counts["unresolved_cells"],
        0,
        "a qualified run reports cells whose comparison never resolved a detection outcome",
    )
    check.equal(
        receipt["subsets_evaluated"],
        4,
        "the selector did not evaluate every three-workload subset of the four candidates",
    )
    check.is_none(
        receipt["witnesses"]["first_witness"],
        "a qualified run published an explanatory witness, which only a non-green status carries",
    )
    group = receipt["selection"]["groups"][0]
    check.equal(
        group["status"],
        "QUALIFIED",
        "the selected subset's probe group is not adjudicated QUALIFIED in a qualified run",
    )
    check.equal(
        group["detection_signature"],
        ["DETECTED"] * 4,
        "the selected workloads do not detect every rung of the declared damping ladder",
    )

    reloaded = load_and_validate_workload_qualification_receipt(result.qualification_json)
    check.equal(
        reloaded["receipt_sha256"],
        result.receipt_sha256,
        "the receipt read back from disk carries a different self-hash than the run reported",
    )
    check.equal(
        reloaded["status"],
        result.status.value,
        "the receipt read back from disk reports a different status than the run reported",
    )
    check.equal(
        reloaded["completed_exit_code"],
        result.exit_code,
        "the receipt read back from disk reports a different exit code than the run reported",
    )
    check.equal(
        reloaded,
        result.receipt,
        "the published receipt does not round-trip: reading it back changes its content",
    )

    records = (*reloaded["probe_cells"], *reloaded["zero_change_controls"])
    for record in records:
        locator = record["comparison_receipt_locator"]
        check.equal(
            len(str(record["comparison_receipt_sha256"])),
            64,
            f"the evidence record for {locator} cites a canonical comparison receipt digest that "
            "is not a SHA-256 hex digest",
        )
        check.equal(
            len(str(record["comparison_receipt_raw_sha256"])),
            64,
            f"the evidence record for {locator} cites a raw comparison receipt file digest that "
            "is not a SHA-256 hex digest",
        )
        check.equal(
            len(str(record["comparison_config_raw_sha256"])),
            64,
            f"the evidence record for {locator} cites a raw comparison configuration file digest "
            "that is not a SHA-256 hex digest",
        )

    evidence_root = configuration.parent / "qualification_out" / "evidence"
    published = {str(record["comparison_receipt_sha256"]) for record in records}
    on_disk = {
        str(json.loads(path.read_text(encoding="utf-8"))["receipt_sha256"])
        for path in evidence_root.rglob("comparison_out/comparison.json")
    }
    check.is_true(
        published <= on_disk,
        "the qualification receipt cites comparison digests that no retained evidence file on "
        "disk carries",
    )
    check.equal(
        len(on_disk),
        20,
        "the retained evidence root does not hold one comparison receipt per planned comparison",
    )

    refused_second_run = False
    try:
        qualify_configuration_file(configuration)
    except WorkloadQualificationOperationError:
        refused_second_run = True
    check.is_true(
        refused_second_run,
        "a second campaign silently overwrote the first campaign's published evidence",
    )


def test_a_tolerance_no_probe_can_cross_reports_insufficient_excitation(tmp_path: Path) -> None:
    """When nothing is detected and nothing is unresolved, the answer is insufficient excitation."""
    configuration = _prepare(tmp_path, tolerance="1000")
    result = qualify_configuration_file(configuration)

    check.is_true(
        result.status is QualificationStatus.INSUFFICIENT_EXCITATION,
        f"a tolerance no probe can cross completed as {result.status} instead of reporting "
        "insufficient excitation",
    )
    check.equal(
        result.exit_code,
        20,
        "an insufficient-excitation run does not exit with the code that status is frozen to",
    )
    group = result.receipt["selection"]["groups"][0]
    check.equal(
        group["status"],
        "INSUFFICIENT",
        "the probe group is not adjudicated INSUFFICIENT even though no rung was detected",
    )
    check.equal(
        group["detection_signature"],
        ["NOT_DETECTED"] * 4,
        "a rung was detected under a tolerance wide enough to swallow the whole ladder",
    )
    check.is_none(
        group["floor_magnitude"],
        "a detection floor was published for a probe group that detected nothing",
    )
    witness = result.receipt["witnesses"]["first_witness"]
    assert witness is not None, (
        "an insufficient-excitation run published no explanatory witness, so there is nothing to "
        "read a witness kind or magnitude from"
    )
    check.equal(
        witness["kind"],
        "BLIND_RUNG",
        "the witness explaining an insufficient-excitation run does not name a blind rung",
    )
    # The first witness is the required rung, not the smallest rung in the ladder.
    check.equal(
        witness["magnitude"],
        "0.005",
        "the first witness names a rung other than the required detection magnitude",
    )
