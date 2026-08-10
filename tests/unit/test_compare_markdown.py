"""Maintainer-readable Markdown rendering of completed comparison comparison receipts."""

from __future__ import annotations

import dataclasses
from pathlib import Path

from metrifid import (
    ComparisonStatus,
    ExactRational,
    ReasonCode,
    ReasonRecord,
    finalize_receipt,
)
from metrifid.compare._markdown import render_markdown
from metrifid.json_values import Binary64
from metrifid.schemas import (
    AlignmentSummary,
    CanonicalSummary,
    ComparisonReceipt,
    MetricEvidenceSummary,
    MonitoredJoint,
    NumericalEvidenceSummary,
    RepeatabilitySummary,
)
from tests.conftest import build_test_reason_record, frozen_object, make_receipt_candidate

CONFIG = Path("/workspace/comparison.json")

_HINGE = MonitoredJoint(
    "hinge",
    "hinge",
    {
        "angle_rad": ExactRational(1, 1000000),
        "angular_velocity_rad_s": ExactRational(1, 100000),
    },
)
_SLIDER = MonitoredJoint(
    "slider",
    "slide",
    {
        "translation_m": ExactRational(1, 1000000),
        "linear_velocity_m_s": ExactRational(1, 100000),
    },
)
_TOLERANCES = CanonicalSummary.from_primitive(
    {
        "hinge": {
            "joint_type": "hinge",
            "angle_rad": {"numerator": 1, "denominator": 1000000},
            "angular_velocity_rad_s": {"numerator": 1, "denominator": 100000},
        },
        "slider": {
            "joint_type": "slide",
            "translation_m": {"numerator": 1, "denominator": 1000000},
            "linear_velocity_m_s": {"numerator": 1, "denominator": 100000},
        },
    }
)

_MICRO = {"numerator": 1, "denominator": 1000000}
_MILLI = {"numerator": 1, "denominator": 100000}


def _binary64(value: float) -> dict[str, object]:
    """Construct the binary64 fixture used by compare markdown scenarios.

    Deterministic setup isolates compare markdown without bypassing the contract boundary under
    assertion.
    """
    return Binary64.from_float(value).to_primitive()


def _metric(
    *,
    error: float,
    ratio: float,
    tolerance: dict[str, int],
    worst_boundary: int,
    worst_time: dict[str, int],
    crossing_boundary: int | None,
    crossing_time: dict[str, int] | None,
) -> dict[str, object]:
    """Construct the metric fixture used by compare markdown scenarios.

    Deterministic setup isolates compare markdown without bypassing the contract boundary under
    assertion.
    """
    return {
        "maximum_error": _binary64(error),
        "maximum_ratio": _binary64(ratio),
        "tolerance": tolerance,
        "worst_boundary_index": worst_boundary,
        "worst_time": worst_time,
        "first_crossing_boundary_index": crossing_boundary,
        "first_crossing_time": crossing_time,
    }


def _metrics(*, crossing: bool) -> MetricEvidenceSummary:
    """Construct the metrics fixture used by compare markdown scenarios.

    Deterministic setup isolates compare markdown without bypassing the contract boundary under
    assertion.
    """
    boundary = 1 if crossing else None
    time = {"numerator": 1, "denominator": 25} if crossing else None
    scale = 1.0 if crossing else 0.0
    return MetricEvidenceSummary.from_primitive(
        {
            "schema": "metrifid.metric_evidence",
            "schema_version": 1,
            "compared_boundary_count": 65,
            "evaluated": True,
            "joints": [
                _hinge_metric_row(scale, boundary, time),
                _slider_metric_row(scale, boundary, time),
            ],
        }
    )


def _hinge_metric_row(
    scale: float, boundary: int | None, time: dict[str, int] | None
) -> dict[str, object]:
    """Build hinge metric evidence for Markdown rendering tests."""
    return {
        "canonical_name": "hinge",
        "joint_type": "hinge",
        "metrics": {
            "angle_rad": _metric(
                error=0.25 * scale,
                ratio=250000.0 * scale,
                tolerance=_MICRO,
                worst_boundary=36,
                worst_time={"numerator": 36, "denominator": 25},
                crossing_boundary=boundary,
                crossing_time=time,
            ),
            "angular_velocity_rad_s": _metric(
                error=0.5 * scale,
                ratio=50000.0 * scale,
                tolerance=_MILLI,
                worst_boundary=34,
                worst_time={"numerator": 34, "denominator": 25},
                crossing_boundary=boundary,
                crossing_time=time,
            ),
        },
    }


def _slider_metric_row(
    scale: float, boundary: int | None, time: dict[str, int] | None
) -> dict[str, object]:
    """Build slide metric evidence for Markdown rendering tests."""
    return {
        "canonical_name": "slider",
        "joint_type": "slide",
        "metrics": {
            "linear_velocity_m_s": _metric(
                error=0.125 * scale,
                ratio=12500.0 * scale,
                tolerance=_MILLI,
                worst_boundary=7,
                worst_time={"numerator": 7, "denominator": 25},
                crossing_boundary=boundary,
                crossing_time=time,
            ),
            "translation_m": _metric(
                error=0.0625 * scale,
                ratio=62500.0 * scale,
                tolerance=_MICRO,
                worst_boundary=35,
                worst_time={"numerator": 7, "denominator": 5},
                crossing_boundary=boundary,
                crossing_time=time,
            ),
        },
    }


def _repeatability() -> RepeatabilitySummary:
    """Construct the repeatability fixture used by compare markdown scenarios.

    Deterministic setup isolates compare markdown without bypassing the contract boundary under
    assertion.
    """
    role = {
        "stable": True,
        "complete_repeats": 3,
        "repeat_count": 3,
        "captured_boundary_counts": [65, 65, 65],
        "signatures": [],
    }
    return RepeatabilitySummary.from_primitive(
        {
            "schema": "metrifid.repeatability",
            "schema_version": 1,
            "baseline": dict(role),
            "candidate": dict(role),
        }
    )


def _numerical(*, complete: bool) -> NumericalEvidenceSummary:
    """Construct the numerical fixture used by compare markdown scenarios.

    Deterministic setup isolates compare markdown without bypassing the contract boundary under
    assertion.
    """
    role: dict[str, object] = {
        "complete": complete,
        "captured_boundary_count": 65 if complete else 41,
        "expected_boundary_count": 65,
        "initial_state_preserved": True,
        "invalid_kind": None if complete else "NONFINITE_STATE",
        "invalid_boundary_index": None if complete else 41,
        "first_warning": None,
        "error_logs": [],
    }
    return NumericalEvidenceSummary.from_primitive(
        {
            "schema": "metrifid.numerical_evidence",
            "schema_version": 1,
            "baseline": dict(role),
            "candidate": dict(role),
        }
    )


def _draft(
    status: ComparisonStatus,
    *,
    reasons: tuple[ReasonRecord, ...] = (),
    reason_codes: tuple[ReasonCode, ...] = (),
    crossing: bool = False,
    complete: bool = True,
    metrics: MetricEvidenceSummary | None = None,
) -> ComparisonReceipt:
    """Construct the draft fixture used by compare markdown scenarios.

    Deterministic setup isolates compare markdown without bypassing the contract boundary under
    assertion.
    """
    base = make_receipt_candidate(status)
    contract = dataclasses.replace(base.comparison_contract, monitored_joints=(_HINGE, _SLIDER))
    first_crossing = (
        CanonicalSummary.from_primitive(
            {
                "joint_name": "hinge",
                "metric": "angle_rad",
                "boundary_index": 1,
                "time": {"numerator": 1, "denominator": 25},
                "error": _binary64(0.00238841942451),
                "tolerance": _MICRO,
                "ratio": _binary64(2388.41942451),
            }
        )
        if crossing
        else None
    )
    return dataclasses.replace(
        base,
        reasons=reasons,
        reason_codes=reason_codes,
        monitored_joints=(_HINGE, _SLIDER),
        tolerances=_TOLERANCES,
        comparison_contract=contract,
        inputs=dataclasses.replace(base.inputs, comparison_contract_sha256=contract.sha256()),
        alignment=AlignmentSummary(None, ("hinge", "slider"), ("slide",), ()),
        repeatability=_repeatability(),
        numerical_evidence=_numerical(complete=complete),
        metrics=_metrics(crossing=crossing) if metrics is None else metrics,
        first_crossing=first_crossing,
    )


def _same_model() -> ComparisonReceipt:
    """Construct the same model fixture used by compare markdown scenarios.

    Deterministic setup isolates compare markdown without bypassing the contract boundary under
    assertion.
    """
    return finalize_receipt(_draft(ComparisonStatus.NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD))


def _controlled_change() -> ComparisonReceipt:
    """Apply the targeted controlled change mutation to otherwise valid evidence.

    Resealing the result isolates the semantic contradiction exercised by compare markdown.
    """
    reason = dataclasses.replace(
        build_test_reason_record(ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED),
        object_name="hinge",
        metric="angle_rad",
        boundary_index=1,
    )
    return finalize_receipt(
        _draft(
            ComparisonStatus.MATERIAL_BEHAVIOR_CHANGE,
            reasons=(reason,),
            reason_codes=(ReasonCode.JOINT_METRIC_TOLERANCE_EXCEEDED,),
            crossing=True,
        )
    )


def _section(text: str, title: str) -> str:
    """Construct the section fixture used by compare markdown scenarios.

    Deterministic setup isolates compare markdown without bypassing the contract boundary under
    assertion.
    """
    marker = f"## {title}\n"
    start = text.index(marker) + len(marker)
    rest = text[start:]
    end = rest.find("\n## ")
    return rest if end < 0 else rest[:end]


def _metric_rows(text: str) -> list[str]:
    """Construct the metric rows fixture used by compare markdown scenarios.

    Deterministic setup isolates compare markdown without bypassing the contract boundary under
    assertion.
    """
    return [
        line for line in _section(text, "Metric summary").splitlines() if line.startswith("| `")
    ]


def test_same_model_renders_four_metric_rows_and_the_exact_no_crossing_sentence() -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises same model renders four metric rows and the exact no crossing
    sentence; status, numerical evidence, and artifact publication must remain stable for the
    declared workload.
    """
    text = render_markdown(_same_model(), CONFIG)
    rows = _metric_rows(text)
    assert len(rows) == 4
    assert [row.split("|")[1].strip() for row in rows] == [
        "`hinge`",
        "`hinge`",
        "`slider`",
        "`slider`",
    ]
    assert [row.split("|")[3].strip() for row in rows] == [
        "`angle_rad`",
        "`angular_velocity_rad_s`",
        "`linear_velocity_m_s`",
        "`translation_m`",
    ]
    assert "1e-06 (1/1000000)" in text
    assert "1.44 (36/25) s" in text
    assert _section(text, "First crossing").strip() == (
        "No monitored tolerance crossing was recorded."
    )
    for required in (
        "## Decision summary",
        "## Claim boundary",
        "## Repeatability",
        "## Metric summary",
        "## First crossing",
        "## Identities",
        "## Command used for this run",
    ):
        assert required in text


def test_controlled_change_renders_four_rows_and_decoded_first_crossing_values() -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises controlled change renders four rows and decoded first crossing
    values; status, numerical evidence, and artifact publication must remain stable for the
    declared workload.
    """
    text = render_markdown(_controlled_change(), CONFIG)
    assert len(_metric_rows(text)) == 4
    crossing = _section(text, "First crossing")
    assert "| Joint | Metric | Boundary | Time | Error | Tolerance | Ratio |" in crossing
    assert "`hinge`" in crossing
    assert "`angle_rad`" in crossing
    assert "0.00238841942451" in crossing
    assert "2388.41942451" in crossing
    assert "0.04 (1/25) s" in crossing
    assert "1e-06 (1/1000000)" in crossing
    reasons = _section(text, "Reasons")
    assert "| Code | Role | Boundary | Object | Metric |" in reasons
    assert "`JOINT_METRIC_TOLERANCE_EXCEEDED`" in reasons


def test_repeatability_reports_captured_boundary_counts() -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises repeatability reports captured boundary counts; status, numerical
    evidence, and artifact publication must remain stable for the declared workload.
    """
    text = render_markdown(_same_model(), CONFIG)
    section = _section(text, "Repeatability")
    assert "| Role | Stable | Complete repeats | Captured boundaries |" in section
    assert "| baseline | yes | 3/3 | 65, 65, 65 |" in section
    assert "| candidate | yes | 3/3 | 65, 65, 65 |" in section


def test_incomplete_numerical_evidence_remains_visible() -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises incomplete numerical evidence remains visible; status, numerical
    evidence, and artifact publication must remain stable for the declared workload.
    """
    receipt = _draft(ComparisonStatus.NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD, complete=False)
    text = render_markdown(receipt, CONFIG)
    section = _section(text, "Incomplete-trace evidence")
    for role in ("baseline", "candidate"):
        assert f"- {role}: captured 41 of 65 boundaries" in section
    assert "NONFINITE_STATE" in section
    assert "invalid boundary: 41" in section

    complete_text = render_markdown(
        _draft(ComparisonStatus.NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD), CONFIG
    )
    assert "## Incomplete-trace evidence" not in complete_text


def test_both_budget_reasons_retain_requested_and_maximum_values() -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises both budget reasons retain requested and maximum values; status,
    numerical evidence, and artifact publication must remain stable for the declared workload.
    """
    step = ReasonRecord(
        code=ReasonCode.INTERNAL_STEP_BUDGET_EXCEEDED,
        role="comparison",
        object_type=None,
        object_name=None,
        metric=None,
        boundary_index=None,
        evidence=frozen_object(
            {
                "requested_total_internal_steps": 9000,
                "maximum_total_internal_steps": 4096,
            }
        ),
    )
    memory = ReasonRecord(
        code=ReasonCode.TRACE_MEMORY_BUDGET_EXCEEDED,
        role="comparison",
        object_type=None,
        object_name=None,
        metric=None,
        boundary_index=None,
        evidence=frozen_object(
            {
                "requested_trace_float64_bytes": 8192,
                "maximum_trace_float64_bytes": 2048,
            }
        ),
    )
    receipt = _draft(
        ComparisonStatus.COVERAGE_INSUFFICIENT,
        reasons=(step, memory),
        reason_codes=(
            ReasonCode.INTERNAL_STEP_BUDGET_EXCEEDED,
            ReasonCode.TRACE_MEMORY_BUDGET_EXCEEDED,
        ),
    )
    text = render_markdown(receipt, CONFIG)
    assert "`INTERNAL_STEP_BUDGET_EXCEEDED`" in text
    assert "`TRACE_MEMORY_BUDGET_EXCEEDED`" in text
    assert "- Internal steps requested: `9000`; maximum: `4096`." in text
    assert "- Trace float64 bytes requested: `8192`; maximum: `2048`." in text


def test_table_text_escaping_handles_pipe_backslash_and_line_breaks() -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises table text escaping handles pipe backslash and line breaks; status,
    numerical evidence, and artifact publication must remain stable for the declared workload.
    """
    hostile = "we|ird\\name\r\ntail"
    metrics = MetricEvidenceSummary.from_primitive(
        {
            "schema": "metrifid.metric_evidence",
            "schema_version": 1,
            "compared_boundary_count": 65,
            "evaluated": True,
            "joints": [
                {
                    "canonical_name": hostile,
                    "joint_type": "hinge",
                    "metrics": {
                        "angle_rad": _metric(
                            error=0.0,
                            ratio=0.0,
                            tolerance=_MICRO,
                            worst_boundary=0,
                            worst_time={"numerator": 0, "denominator": 1},
                            crossing_boundary=None,
                            crossing_time=None,
                        )
                    },
                }
            ],
        }
    )
    receipt = _draft(ComparisonStatus.NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD, metrics=metrics)
    text = render_markdown(receipt, CONFIG)
    row = _metric_rows(text)[0]
    assert "we\\|ird" in row
    assert "\\\\name" in row
    assert "\\r\\n" in row
    assert row.count("\n") == 0
    assert "\r" not in row


def test_two_renders_of_the_same_receipt_are_byte_identical() -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises two renders of the same receipt are byte identical; status,
    numerical evidence, and artifact publication must remain stable for the declared workload.
    """
    for receipt in (_same_model(), _controlled_change()):
        first = render_markdown(receipt, CONFIG)
        second = render_markdown(receipt, CONFIG)
        assert first == second
        assert first.encode("utf-8") == second.encode("utf-8")


def test_output_contains_no_raw_python_primitive_representation() -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises output contains no raw python primitive representation; status,
    numerical evidence, and artifact publication must remain stable for the declared workload.
    """
    for receipt in (_same_model(), _controlled_change()):
        primitive_before = receipt.to_primitive()
        text = render_markdown(receipt, CONFIG)
        assert "{'" not in text
        assert '"kind":' not in text
        assert "'kind'" not in text
        assert "ieee754_binary64" not in text
        assert "numerator" not in text
        assert "denominator" not in text
        assert "[65, 65, 65]" not in text
        assert receipt.to_primitive() == primitive_before
        assert receipt.receipt_sha256 is not None
