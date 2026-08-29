"""The qualification configuration refuses everything it does not explicitly admit."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import pytest_check as check

from metrifid.workload_qualification._config import QualificationConfig

_EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "workload_qualification"


def _configuration() -> dict:
    return json.loads((_EXAMPLE / "qualification.json").read_text(encoding="utf-8"))


def test_the_shipped_example_configuration_is_admitted() -> None:
    """The example in the repository is a valid configuration, not just documentation."""
    config = QualificationConfig.from_primitive(_configuration())
    check.equal(
        config.budget,
        3,
        "the shipped example does not declare the three-workload budget schema version 1 freezes",
    )
    check.equal(
        config.schema_version,
        1,
        "the shipped example declares a schema version this release does not admit",
    )
    check.equal(
        len(config.workloads),
        4,
        "the shipped example does not offer the four workload candidates it documents",
    )


def test_an_unknown_top_level_field_is_refused() -> None:
    """A field the schema does not declare is a claim the receipt could not carry."""
    primitive = _configuration()
    primitive["tolerance_profile"] = "loose"
    with pytest.raises((TypeError, ValueError)):
        QualificationConfig.from_primitive(primitive)


def test_an_unknown_probe_group_field_is_refused() -> None:
    """Unknown fields are refused at every level, not only the top."""
    primitive = _configuration()
    primitive["probe_groups"][0]["units"] = "N*m*s/rad"
    with pytest.raises((TypeError, ValueError)):
        QualificationConfig.from_primitive(primitive)


def test_a_budget_other_than_three_is_refused() -> None:
    """Schema version 1 freezes the budget at three so the search stays exhaustively bounded."""
    primitive = _configuration()
    primitive["budget"] = 4
    with pytest.raises((TypeError, ValueError)):
        QualificationConfig.from_primitive(primitive)


def test_fewer_than_three_declared_workloads_is_refused() -> None:
    """A budget of three cannot be met by two candidates."""
    primitive = _configuration()
    primitive["workloads"] = primitive["workloads"][:2]
    with pytest.raises(ValueError, match="workloads must contain"):
        QualificationConfig.from_primitive(primitive)


def test_more_than_sixteen_workloads_is_refused() -> None:
    """The candidate ceiling is what keeps the subset count at or below 560."""
    primitive = _configuration()
    template = primitive["workloads"][0]
    extra = []
    for index in range(17):
        item = copy.deepcopy(template)
        item["workload_id"] = f"candidate_{index:02d}"
        extra.append(item)
    primitive["workloads"] = extra
    with pytest.raises(ValueError, match="workloads must contain"):
        QualificationConfig.from_primitive(primitive)


def test_duplicate_workload_identities_are_refused() -> None:
    """Two workloads with the same identity would make the receipt ambiguous."""
    primitive = _configuration()
    primitive["workloads"][1]["workload_id"] = primitive["workloads"][0]["workload_id"]
    with pytest.raises(ValueError, match="workload_id values must be unique"):
        QualificationConfig.from_primitive(primitive)


def test_magnitudes_must_be_strictly_increasing() -> None:
    """A ladder is only a ladder when its rungs are ordered and distinct."""
    primitive = _configuration()
    variants = primitive["probe_groups"][0]["variants"]
    variants[0], variants[1] = variants[1], variants[0]
    with pytest.raises(ValueError, match="strictly increasing"):
        QualificationConfig.from_primitive(primitive)


def test_the_required_magnitude_must_be_one_of_the_declared_rungs() -> None:
    """A requirement that names no rung could never be adjudicated."""
    primitive = _configuration()
    primitive["probe_groups"][0]["required_detection_magnitude"] = "0.007"
    with pytest.raises(ValueError, match="required_detection_magnitude"):
        QualificationConfig.from_primitive(primitive)


def test_a_single_variant_ladder_is_refused() -> None:
    """One rung cannot establish a floor, so a ladder needs at least two."""
    primitive = _configuration()
    group = primitive["probe_groups"][0]
    group["variants"] = group["variants"][:1]
    group["required_detection_magnitude"] = group["variants"][0]["magnitude"]
    with pytest.raises(ValueError, match="variants must contain"):
        QualificationConfig.from_primitive(primitive)


def test_an_undeclared_direction_is_refused() -> None:
    """One group carries exactly one declared direction."""
    primitive = _configuration()
    primitive["probe_groups"][0]["direction"] = "both"
    with pytest.raises(ValueError, match="direction must be"):
        QualificationConfig.from_primitive(primitive)


def test_the_configuration_round_trips_through_its_primitive_form() -> None:
    """What the receipt echoes back is what the user declared."""
    primitive = _configuration()
    config = QualificationConfig.from_primitive(primitive)
    again = QualificationConfig.from_primitive(config.to_primitive())
    check.equal(
        again.to_primitive(),
        config.to_primitive(),
        "re-reading the configuration's own echoed form did not reproduce what the user declared",
    )
