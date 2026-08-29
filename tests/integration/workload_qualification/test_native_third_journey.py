"""The third native journey: a real excluded-zero-change campaign, with nothing injected.

This is an acceptance gate, so it deliberately avoids every shortcut that would make it easy: no
patched comparison, no hand-built typed receipt, no edited receipt, no bare status string, and no
operational failure reclassified as a completed comparison. One workload is simply large enough that
its own baseline-against-itself control exceeds compare's retained-trace memory budget, so the
ordinary installed comparison returns COVERAGE_INSUFFICIENT and the workload is excluded.
"""

from __future__ import annotations

from pathlib import Path

import pytest_check as check

from metrifid.errors import ComparisonStatus
from metrifid.workload_qualification import (
    QualificationStatus,
    load_and_validate_workload_qualification_receipt,
    qualify_configuration_file,
)
from tests._support.workload_qualification import (
    WIDE_JOINTS,
    WIDE_OVERSIZED_INTERVALS,
    write_wide_case,
)


def test_an_oversized_workload_is_excluded_by_its_own_zero_change_control(
    tmp_path: Path,
) -> None:
    """The complete third journey, asserted end to end against the installed product."""
    case = tmp_path / "wide"
    result = qualify_configuration_file(write_wide_case(case))
    receipt = result.receipt

    controls = {c["workload_id"]: c for c in receipt["zero_change_controls"]}
    oversized = controls["oversized"]
    check.equal(
        oversized["comparison_status"],
        ComparisonStatus.COVERAGE_INSUFFICIENT.value,
        "the oversized workload's own zero-change control did not report insufficient coverage",
    )
    check.is_false(
        oversized["eligible"],
        "the oversized workload stayed eligible after its own zero-change control failed",
    )
    check.is_in(
        "COVERAGE_INSUFFICIENT",
        str(oversized["exclusion_reason"]),
        "the recorded exclusion reason does not name the insufficient coverage that caused it",
    )

    check.equal(
        list(receipt["excluded_workload_ids"]),
        ["oversized"],
        "the receipt excludes a different set of workloads than the single oversized one",
    )
    check.equal(
        list(receipt["eligible_workload_ids"]),
        ["normal_low", "normal_mid", "normal_high"],
        "the receipt reports a different eligible set than the three normally sized workloads",
    )
    check.equal(
        len(receipt["selected_workload_ids"]),
        3,
        "the campaign selected a subset that is not the declared three-workload budget",
    )
    check.is_not_in(
        "oversized",
        receipt["selected_workload_ids"],
        "the excluded oversized workload was still selected into the qualifying subset",
    )

    check.equal(
        receipt["subsets_evaluated"],
        1,
        "a different number of subsets was evaluated than the one the eligible set allows",
    )
    check.equal(
        receipt["planned_comparisons"],
        4 + 4 * 2,
        "the plan does not cover one zero-change control plus two probe rungs per workload",
    )
    counts = receipt["execution_counts"]
    check.equal(
        counts["zero_change_comparisons"],
        4,
        "a zero-change control did not run once for every declared workload",
    )
    check.equal(
        counts["probe_comparisons"],
        8,
        "the probe rungs did not run twice for each of the four workloads",
    )
    check.equal(
        counts["total_comparisons"],
        12,
        "the reported total does not account for every zero-change and probe comparison",
    )

    check.equal(
        QualificationStatus(str(receipt["status"])),
        result.status,
        "the published receipt reports a status the returned result does not carry",
    )
    check.equal(
        result.exit_code,
        int(receipt["completed_exit_code"]),
        "the returned exit code disagrees with the completed exit code the receipt records",
    )

    witnesses = receipt["witnesses"]
    alarm = witnesses["first_false_alarm_witness"]
    assert alarm is not None, (
        "the campaign recorded no false-alarm witness for the excluded workload"
    )
    check.equal(
        alarm["kind"],
        "ZERO_CHANGE_FALSE_ALARM",
        "the false-alarm witness is not the zero-change false alarm that caused the exclusion",
    )
    check.equal(
        alarm["workload_id"],
        "oversized",
        "the false-alarm witness names a workload other than the excluded oversized one",
    )
    first = witnesses["first_witness"]
    if first is not None:
        check.is_true(
            first["kind"] != "ZERO_CHANGE_FALSE_ALARM",
            f"a false alarm ({first['kind']}) leads the witnesses, so it explains the status",
        )

    root = (case / "qualification_out").resolve()
    for key in ("comparison_config_locator", "comparison_receipt_locator"):
        check.is_true(
            (root / str(oversized[key])).is_file(),
            f"the excluded workload's {key} points at evidence the campaign did not retain",
        )

    reloaded = load_and_validate_workload_qualification_receipt(result.qualification_json)
    check.equal(
        reloaded,
        result.receipt,
        "the published receipt read back through the loader differs from the one returned",
    )


def test_the_case_is_the_documented_budget_construction() -> None:
    """Pin the sizing so a future budget change makes this case fail loudly rather than silently.

    Sixty-four monitored hinges alone request exactly the budget; the actuator's single activation
    unit is what carries the request over it.
    """
    from metrifid.compare._budget import MAX_TRACE_FLOAT64_BYTES

    boundaries = WIDE_OVERSIZED_INTERVALS + 1
    without_activation = boundaries * 2 * 2 * (WIDE_JOINTS * 2) * 8
    with_activation = boundaries * 2 * 2 * (WIDE_JOINTS * 2 + 1) * 8
    check.is_true(
        without_activation <= MAX_TRACE_FLOAT64_BYTES,
        "the monitored hinges alone now exceed the trace budget, so the case no longer isolates it",
    )
    check.greater(
        with_activation,
        MAX_TRACE_FLOAT64_BYTES,
        "the actuator's activation unit no longer carries the oversized control over the budget",
    )
