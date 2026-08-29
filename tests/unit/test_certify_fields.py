"""The descriptive public field report: localization, bounds, and its lack of authority."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from metrifid.certify._artifact import serialize_complete_artifact
from metrifid.certify._bytes import compare_retained_artifacts
from metrifid.certify._fields import (
    FIELD_DIFFERENCES_IDENTIFIED,
    MAX_CHANGED_FIELDS_RETURNED,
    MAX_OMITTED_FIELDS_RETURNED,
    MAX_WITNESSES_PER_FIELD,
    NO_PUBLIC_FIELD_DIFFERENCE_IDENTIFIED,
    build_field_report,
)
from metrifid.json_values import Binary64

_TEMPLATE = """
<mujoco model="fixture">
  <option timestep="{timestep}"/>
  <worldbody>
    <body name="b0" pos="0 0 1">
      <geom name="g0" type="sphere" size="0.1" rgba="{rgba}" friction="{friction}" mass="{mass}"/>
      <joint name="j0" type="hinge" axis="0 0 1" damping="{damping}"/>
    </body>
    <body name="b1" pos="1 0 1">
      <geom name="g1" type="sphere" size="0.1"/>
      <joint name="j1" type="hinge" axis="0 1 0"/>
    </body>
  </worldbody>
</mujoco>
"""

_BASE = {
    "timestep": "0.002",
    "rgba": "1 0 0 1",
    "friction": "1 0.005 0.0001",
    "mass": "2",
    "damping": "0.5",
}


def _report(tmp_path: Path, **overrides: str) -> dict:
    """Construct the report fixture used by certification fields scenarios.

    Deterministic setup isolates certification fields without bypassing the contract boundary
    under assertion.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    baseline_xml = _TEMPLATE.format(**_BASE)
    candidate_xml = _TEMPLATE.format(**{**_BASE, **overrides})
    baseline = serialize_complete_artifact(
        mujoco.MjModel.from_xml_string(baseline_xml), "baseline", tmp_path
    )
    candidate = serialize_complete_artifact(
        mujoco.MjModel.from_xml_string(candidate_xml), "candidate", tmp_path
    )
    comparison = compare_retained_artifacts(baseline.retained, candidate.retained)
    return build_field_report(baseline.retained, candidate.retained, comparison)


def _entry(report: dict, path: str) -> dict:
    """Construct the entry fixture used by certification fields scenarios.

    Deterministic setup isolates certification fields without bypassing the contract boundary
    under assertion.
    """
    matches = [item for item in report["changed_fields"] if item["path"] == path]
    assert matches, f"{path} was not reported as changed"
    return matches[0]


@pytest.mark.parametrize(
    ("override", "expected_path"),
    [
        ({"mass": "3"}, "body_mass"),
        ({"damping": "0.7"}, "dof_damping"),
        ({"friction": "0.9 0.005 0.0001"}, "geom_friction"),
        ({"timestep": "0.004"}, "opt.timestep"),
        ({"rgba": "0 1 0 1"}, "geom_rgba"),
    ],
)
def test_each_required_fixture_localizes_to_its_field(
    tmp_path: Path, override: dict[str, str], expected_path: str
) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises each required fixture localizes to its field; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    report = _report(tmp_path, **override)
    assert report["field_report_status"] == FIELD_DIFFERENCES_IDENTIFIED
    entry = _entry(report, expected_path)
    assert entry["baseline_sha256"] != entry["candidate_sha256"]


@pytest.mark.parametrize(
    ("override", "expected_path"),
    [
        ({"mass": "3"}, "body_mass"),
        ({"damping": "0.7"}, "dof_damping"),
        ({"friction": "0.9 0.005 0.0001"}, "geom_friction"),
        ({"rgba": "0 1 0 1"}, "geom_rgba"),
    ],
)
def test_each_array_fixture_carries_a_changed_index_witness(
    tmp_path: Path, override: dict[str, str], expected_path: str
) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises each array fixture carries a changed index witness; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    entry = _entry(_report(tmp_path, **override), expected_path)
    assert entry["changed_element_count"] is not None
    assert entry["changed_element_count"] >= 1
    witness = entry["witnesses"][0]
    assert witness["index"], "an array witness must name the changed index"
    assert witness["baseline_value"] != witness["candidate_value"]
    assert len(entry["witnesses"]) <= MAX_WITNESSES_PER_FIELD


def test_the_timestep_witness_carries_the_exact_scalar_values(tmp_path: Path) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises the timestep witness carries the exact scalar values; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    entry = _entry(_report(tmp_path, timestep="0.004"), "opt.timestep")
    witness = entry["witnesses"][0]
    assert witness["index"] == []
    assert Binary64.from_primitive(witness["baseline_value"]).to_float() == 0.002
    assert Binary64.from_primitive(witness["candidate_value"]).to_float() == 0.004
    assert witness["baseline_text"] == "0.002"
    assert witness["candidate_text"] == "0.004"


def test_witnesses_keep_every_bit_of_a_float(tmp_path: Path) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises witnesses keep every bit of a float; accepting a contradictory or
    noncanonical value would make the signed decision evidence ambiguous.
    """
    entry = _entry(_report(tmp_path, mass="3"), "body_mass")
    for witness in entry["witnesses"]:
        for side in ("baseline_value", "candidate_value"):
            value = witness[side]
            assert set(value) == {"kind", "bits"}
            assert value["kind"] == "ieee754_binary64"


def test_a_multi_axis_index_is_reported_for_a_two_dimensional_field(tmp_path: Path) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises a multi axis index is reported for a two dimensional field;
    accepting a contradictory or noncanonical value would make the signed decision evidence
    ambiguous.
    """
    entry = _entry(_report(tmp_path, rgba="0 1 0 1"), "geom_rgba")
    assert entry["baseline_shape"] == [2, 4]
    assert all(len(witness["index"]) == 2 for witness in entry["witnesses"])
    assert [witness["index"] for witness in entry["witnesses"]] == sorted(
        witness["index"] for witness in entry["witnesses"]
    )


def test_the_report_covers_the_documented_surface_and_sorts_every_path(tmp_path: Path) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises the report covers the documented surface and sorts every path;
    accepting a contradictory or noncanonical value would make the signed decision evidence
    ambiguous.
    """
    report = _report(tmp_path, mass="3")
    changed = [item["path"] for item in report["changed_fields"]]
    omitted = [item["path"] for item in report["omitted_fields"]]
    assert changed == sorted(changed)
    assert omitted == sorted(omitted)
    assert report["fields_compared_count"] > 100
    assert report["fields_compared_count"] == len(_compared_paths(tmp_path / "probe"))


def _compared_paths(tmp_path: Path) -> list[str]:
    """Every path the report compares, taken from a run against one artifact and itself."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    from metrifid.certify._fields import _facts_from_artifact

    artifact = serialize_complete_artifact(
        mujoco.MjModel.from_xml_string(_TEMPLATE.format(**_BASE)), "baseline", tmp_path
    )
    facts, _ = _facts_from_artifact(artifact.retained)
    return sorted(facts)


def test_one_level_under_each_container_is_read_exactly_one_level_deep(tmp_path: Path) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises one level under each container is read exactly one level deep;
    accepting a contradictory or noncanonical value would make the signed decision evidence
    ambiguous.
    """
    paths = _compared_paths(tmp_path)
    assert "opt.timestep" in paths
    assert any(path.startswith("stat.") for path in paths)
    assert "body_mass" in paths
    assert not any(path.count(".") > 1 for path in paths), "the surface stays one level deep"


def test_every_member_one_level_under_vis_is_a_struct_and_is_reported_as_omitted(
    tmp_path: Path,
) -> None:
    """`model.vis` holds only sub-structs, so one level down yields nothing comparable.

    That is the dispatched surface, not an exclusion: each member is still listed in
    `omitted_fields` with a stable reason, and the fallback status below covers the case where
    a real difference lives below it.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    from metrifid.certify._fields import _facts_from_artifact

    artifact = serialize_complete_artifact(
        mujoco.MjModel.from_xml_string(_TEMPLATE.format(**_BASE)), "baseline", tmp_path
    )
    facts, omitted = _facts_from_artifact(artifact.retained)
    assert not any(path.startswith("vis.") for path in facts)
    vis_omissions = {path: reason for path, reason in omitted if path.startswith("vis.")}
    assert len(vis_omissions) == 12
    assert set(vis_omissions.values()) <= {"UNSUPPORTED_MEMBER_TYPE", "CALLABLE_MEMBER"}


def test_a_difference_below_the_public_surface_falls_back_without_changing_the_outcome(
    tmp_path: Path,
) -> None:
    """A visual-only change alters the artifact but is invisible one level under `vis`."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    xml = (
        "<mujoco><visual><global fovy='{fovy}'/></visual>"
        "<worldbody><body><geom size='1'/></body></worldbody></mujoco>"
    )
    baseline = serialize_complete_artifact(
        mujoco.MjModel.from_xml_string(xml.format(fovy=45)), "baseline", tmp_path
    )
    candidate = serialize_complete_artifact(
        mujoco.MjModel.from_xml_string(xml.format(fovy=60)), "candidate", tmp_path
    )
    comparison = compare_retained_artifacts(baseline.retained, candidate.retained)
    assert comparison.equal is False
    report = build_field_report(baseline.retained, candidate.retained, comparison)
    assert report["field_report_status"] == NO_PUBLIC_FIELD_DIFFERENCE_IDENTIFIED
    assert report["changed_fields_total"] == 0
    assert report["differing_byte_count"] == comparison.differing_byte_count
    assert report["first_differing_byte_offset"] == comparison.first_differing_byte_offset


def test_the_three_expanded_containers_are_read_one_level_down(tmp_path: Path) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises the three expanded containers are read one level down; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    (tmp_path / "probe").mkdir(parents=True, exist_ok=True)
    xml = _TEMPLATE.format(**_BASE)
    artifact = serialize_complete_artifact(
        mujoco.MjModel.from_xml_string(xml), "baseline", tmp_path / "probe"
    )
    comparison = compare_retained_artifacts(artifact.retained, artifact.retained)
    report = build_field_report(artifact.retained, artifact.retained, comparison)
    omitted = {item["path"]: item["reason"] for item in report["omitted_fields"]}
    for container in ("opt", "stat", "vis"):
        assert omitted[container] == "EXPANDED_ONE_LEVEL_BELOW"
    assert report["fields_compared_count"] > 0
    assert report["field_report_status"] == NO_PUBLIC_FIELD_DIFFERENCE_IDENTIFIED
    assert report["changed_fields"] == []


def test_identical_artifacts_report_no_public_field_difference(tmp_path: Path) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises identical artifacts report no public field difference; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    xml = _TEMPLATE.format(**_BASE)
    artifact = serialize_complete_artifact(
        mujoco.MjModel.from_xml_string(xml), "baseline", tmp_path
    )
    comparison = compare_retained_artifacts(artifact.retained, artifact.retained)
    report = build_field_report(artifact.retained, artifact.retained, comparison)
    assert report["field_report_status"] == NO_PUBLIC_FIELD_DIFFERENCE_IDENTIFIED
    assert report["changed_fields_total"] == 0
    assert report["truncated"] is False


def test_the_report_is_byte_deterministic_across_repeated_builds(tmp_path: Path) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises the report is byte deterministic across repeated builds; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    first = _report(tmp_path / "a", mass="3", rgba="0 1 0 1")
    second = _report(tmp_path / "b", mass="3", rgba="0 1 0 1")
    assert first == second


def test_the_returned_changed_fields_are_bounded_and_marked_truncated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises the returned changed fields are bounded and marked truncated;
    accepting a contradictory or noncanonical value would make the signed decision evidence
    ambiguous.
    """
    from metrifid.certify import _fields as fields_module

    monkeypatch.setattr(fields_module, "MAX_CHANGED_FIELDS_RETURNED", 2)
    report = _report(tmp_path, mass="3", rgba="0 1 0 1", damping="0.7", friction="0.9 0.005 0.0001")
    assert report["changed_fields_returned"] == 2
    assert report["changed_fields_total"] > 2
    assert report["truncated"] is True
    assert len(report["changed_fields"]) == 2


def test_the_documented_limits_are_the_frozen_values() -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises the documented limits are the frozen values; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    assert MAX_CHANGED_FIELDS_RETURNED == 100
    assert MAX_WITNESSES_PER_FIELD == 8
    assert MAX_OMITTED_FIELDS_RETURNED == 200


def test_no_field_is_ever_dumped_in_full(tmp_path: Path) -> None:
    """A changed field reports counts, digests and bounded witnesses, never whole arrays."""
    entry = _entry(_report(tmp_path, rgba="0 1 0 1"), "geom_rgba")
    assert set(entry) == {
        "path",
        "baseline_type",
        "candidate_type",
        "baseline_dtype",
        "candidate_dtype",
        "baseline_shape",
        "candidate_shape",
        "baseline_sha256",
        "candidate_sha256",
        "changed_element_count",
        "witnesses",
    }
    assert len(entry["witnesses"]) <= MAX_WITNESSES_PER_FIELD


def test_a_nonfinite_element_is_still_represented_exactly() -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises a nonfinite element is still represented exactly; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    from metrifid.certify._fields import _element_evidence

    left = np.array([1.0, 2.0], dtype=np.float64)
    right = np.array([np.nan, np.inf], dtype=np.float64)
    evidence = _element_evidence(left, right)
    assert evidence["changed_element_count"] == 2
    values = [
        Binary64.from_primitive(w["candidate_value"]).to_float() for w in evidence["witnesses"]
    ]
    assert np.isnan(values[0])
    assert np.isinf(values[1])


def test_byte_identical_nan_elements_are_not_reported_as_changed() -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises byte identical nan elements are not reported as changed; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    from metrifid.certify._fields import _element_evidence

    payload = np.array([np.nan, 1.0], dtype=np.float64)
    evidence = _element_evidence(payload, payload.copy())
    assert evidence["changed_element_count"] == 0
    assert evidence["witnesses"] == []


# Truthful witness truncation.


def _many_body_xml(count: int, mass: str) -> str:
    """Construct the many body xml fixture used by certification fields scenarios.

    Deterministic setup isolates certification fields without bypassing the contract boundary
    under assertion.
    """
    bodies = "".join(
        f"<body name='b{index}' pos='0 0 {index}'>"
        f"<geom size='0.1' mass='{mass}'/>"
        f"<joint name='j{index}' type='hinge' axis='0 0 1'/></body>"
        for index in range(count)
    )
    return f"<mujoco><worldbody>{bodies}</worldbody></mujoco>"


def _report_for(tmp_path: Path, baseline_xml: str, candidate_xml: str) -> dict:
    """Construct the report for fixture used by certification fields scenarios.

    Deterministic setup isolates certification fields without bypassing the contract boundary
    under assertion.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    baseline = serialize_complete_artifact(
        mujoco.MjModel.from_xml_string(baseline_xml), "baseline", tmp_path
    )
    candidate = serialize_complete_artifact(
        mujoco.MjModel.from_xml_string(candidate_xml), "candidate", tmp_path
    )
    comparison = compare_retained_artifacts(baseline.retained, candidate.retained)
    return build_field_report(baseline.retained, candidate.retained, comparison)


def test_more_than_eight_changed_elements_truncates_witnesses_and_says_so(
    tmp_path: Path,
) -> None:
    """A field with twelve changed elements returns eight witnesses and reports truncation."""
    report = _report_for(tmp_path, _many_body_xml(12, "2"), _many_body_xml(12, "3"))
    entry = _entry(report, "body_mass")
    assert entry["changed_element_count"] == 12
    assert entry["changed_element_count"] > MAX_WITNESSES_PER_FIELD
    assert len(entry["witnesses"]) == MAX_WITNESSES_PER_FIELD == 8
    assert report["truncated"] is True
    # The other two bounds are deliberately not reached, so the flag can only come from witnesses.
    assert report["changed_fields_total"] == report["changed_fields_returned"]
    assert report["fields_omitted_count"] <= MAX_OMITTED_FIELDS_RETURNED


def test_truncation_is_exactly_the_disjunction_of_the_three_bounds(tmp_path: Path) -> None:
    """`truncated` is true if and only if some configured bound was actually reached.

    Stating the invariant rather than a hand-picked element count keeps the test honest: a mass
    change also moves `body_inertia`, which carries three elements per body, so a fixture chosen
    to sit "exactly at eight" for one field is above the bound for another.
    """
    fixtures = (
        (_many_body_xml(4, "2"), _many_body_xml(4, "2")),
        (_many_body_xml(2, "2"), _many_body_xml(2, "3")),
        (_many_body_xml(12, "2"), _many_body_xml(12, "3")),
    )
    for index, (baseline_xml, candidate_xml) in enumerate(fixtures):
        report = _report_for(tmp_path / f"case{index}", baseline_xml, candidate_xml)
        witnesses_truncated = any(
            (entry["changed_element_count"] or 0) > len(entry["witnesses"])
            for entry in report["changed_fields"]
        )
        expected = (
            report["changed_fields_total"] > report["changed_fields_returned"]
            or report["fields_omitted_count"] > len(report["omitted_fields"])
            or witnesses_truncated
        )
        assert report["truncated"] is expected, report["changed_fields_total"]


def test_a_small_difference_still_reports_no_truncation(tmp_path: Path) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises a small difference still reports no truncation; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    report = _report_for(tmp_path, _many_body_xml(2, "2"), _many_body_xml(2, "3"))
    assert (
        max(entry["changed_element_count"] or 0 for entry in report["changed_fields"])
        <= MAX_WITNESSES_PER_FIELD
    )
    assert report["truncated"] is False


def test_identical_artifacts_report_no_truncation(tmp_path: Path) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises identical artifacts report no truncation; accepting a contradictory
    or noncanonical value would make the signed decision evidence ambiguous.
    """
    report = _report_for(tmp_path, _many_body_xml(4, "2"), _many_body_xml(4, "2"))
    assert report["changed_fields_total"] == 0
    assert report["truncated"] is False


def test_truncation_is_true_when_the_changed_field_bound_is_reached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises truncation is true when the changed field bound is reached;
    accepting a contradictory or noncanonical value would make the signed decision evidence
    ambiguous.
    """
    from metrifid.certify import _fields as fields_module

    monkeypatch.setattr(fields_module, "MAX_CHANGED_FIELDS_RETURNED", 1)
    report = _report_for(tmp_path, _many_body_xml(3, "2"), _many_body_xml(3, "3"))
    assert report["changed_fields_total"] > report["changed_fields_returned"]
    assert report["truncated"] is True


def test_truncation_is_true_when_the_omitted_field_bound_is_reached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises truncation is true when the omitted field bound is reached;
    accepting a contradictory or noncanonical value would make the signed decision evidence
    ambiguous.
    """
    from metrifid.certify import _fields as fields_module

    monkeypatch.setattr(fields_module, "MAX_OMITTED_FIELDS_RETURNED", 1)
    report = _report_for(tmp_path, _many_body_xml(3, "2"), _many_body_xml(3, "3"))
    assert report["fields_omitted_count"] > 1
    assert len(report["omitted_fields"]) == 1
    assert report["truncated"] is True


# Bounded descriptive evidence.


def test_the_copy_budget_is_the_documented_value() -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises the copy budget is the documented value; accepting a contradictory
    or noncanonical value would make the signed decision evidence ambiguous.
    """
    from metrifid.certify._fields import MAX_FIELD_WITNESS_COPY_BYTES

    assert MAX_FIELD_WITNESS_COPY_BYTES == 64 * 1024 * 1024


def test_a_field_over_the_copy_budget_keeps_identity_and_withholds_elements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A budget-declined field publishes path, types, shapes and digests but no element evidence.

    The budget is lowered rather than allocating a real 64 MiB field, so this stays a fast unit
    test with no giant fixture.
    """
    from metrifid.certify import _fields as fields_module

    monkeypatch.setattr(fields_module, "MAX_FIELD_WITNESS_COPY_BYTES", 8)
    report = _report_for(tmp_path, _many_body_xml(4, "2"), _many_body_xml(4, "3"))
    entry = _entry(report, "body_mass")
    assert entry["path"] == "body_mass"
    assert entry["baseline_type"] == "ndarray"
    assert entry["baseline_dtype"] == entry["candidate_dtype"]
    assert entry["baseline_shape"] == entry["candidate_shape"]
    assert entry["baseline_sha256"] != entry["candidate_sha256"]
    assert entry["changed_element_count"] is None
    assert entry["witnesses"] == []
    assert report["truncated"] is True


def test_a_field_within_the_copy_budget_keeps_its_existing_behaviour(tmp_path: Path) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises a field within the copy budget keeps its existing behaviour;
    accepting a contradictory or noncanonical value would make the signed decision evidence
    ambiguous.
    """
    report = _report_for(tmp_path, _many_body_xml(4, "2"), _many_body_xml(4, "3"))
    entry = _entry(report, "body_mass")
    assert entry["changed_element_count"] is not None
    assert entry["witnesses"]


def test_no_two_changed_fields_hold_copied_arrays_at_the_same_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Copies from one changed field must be unreachable before the next field is copied.

    Weak references are the direct evidence: at the moment field N is copied, every earlier
    field's copies must already be collectable, and once the report is built none may survive.
    """
    import gc
    import weakref

    from metrifid.certify import _fields as fields_module

    real_bounded_pair = fields_module._bounded_pair
    previous: list[weakref.ref[object]] = []
    live_at_each_step: list[int] = []

    def tracking_pair(baseline_model, candidate_model, path):  # type: ignore[no-untyped-def]
        """Construct the tracking pair fixture used by certification fields scenarios.

        Deterministic setup isolates no two changed fields hold copied arrays at the same time
        without bypassing the contract boundary under assertion.
        """
        gc.collect()
        live_at_each_step.append(sum(1 for reference in previous if reference() is not None))
        baseline_value, candidate_value = real_bounded_pair(baseline_model, candidate_model, path)
        previous.clear()
        for value in (baseline_value, candidate_value):
            if isinstance(value, np.ndarray):
                previous.append(weakref.ref(value))
        return baseline_value, candidate_value

    monkeypatch.setattr(fields_module, "_bounded_pair", tracking_pair)
    report = _report_for(tmp_path, _many_body_xml(6, "2"), _many_body_xml(6, "3"))

    assert len(live_at_each_step) == report["changed_fields_returned"] > 1
    assert live_at_each_step[0] == 0
    assert all(count == 0 for count in live_at_each_step), live_at_each_step
    monkeypatch.undo()
    gc.collect()
    assert all(reference() is None for reference in previous), "a copied field outlived the report"


def test_changed_fields_are_described_in_sorted_path_order(tmp_path: Path) -> None:
    """Keep published assurance receipts independently verifiable.

    This scenario exercises changed fields are described in sorted path order; accepting a
    contradictory or noncanonical value would make the signed decision evidence ambiguous.
    """
    report = _report_for(tmp_path, _many_body_xml(6, "2"), _many_body_xml(6, "3"))
    paths = [entry["path"] for entry in report["changed_fields"]]
    assert paths == sorted(paths)
