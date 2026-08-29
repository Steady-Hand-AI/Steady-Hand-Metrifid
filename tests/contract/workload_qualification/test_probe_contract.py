"""The probe declaration contract: honest labels, distinct closures, and timestep generality."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import pytest_check as check

from metrifid.compare._failure import ComparisonOperationError
from metrifid.operational import OperationalReasonCode
from metrifid.workload_qualification import qualify_configuration_file
from metrifid.workload_qualification._config import QualificationConfig
from metrifid.workload_qualification._status import QUALIFICATION_LIMITATIONS
from tests._support.workload_qualification import SEMANTICS, write_case


def _config(case: Path) -> dict:
    return json.loads((case / "qualification.json").read_text(encoding="utf-8"))


def _write(case: Path, config: dict) -> Path:
    path = case / "qualification.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def test_missing_magnitude_semantics_refuses(tmp_path: Path) -> None:
    """A magnitude with no stated meaning is not a reviewable declaration."""
    case = tmp_path / "case"
    write_case(case)
    config = _config(case)
    del config["probe_groups"][0]["magnitude_semantics"]
    with pytest.raises((TypeError, ValueError)):
        QualificationConfig.from_primitive(config)


@pytest.mark.parametrize("value", ["", "   ", "\t\n"])
def test_empty_magnitude_semantics_refuses(tmp_path: Path, value: str) -> None:
    """Whitespace is not a declaration."""
    case = tmp_path / "case"
    write_case(case)
    config = _config(case)
    config["probe_groups"][0]["magnitude_semantics"] = value
    with pytest.raises((TypeError, ValueError)):
        QualificationConfig.from_primitive(config)


def test_an_unknown_probe_field_refuses(tmp_path: Path) -> None:
    """Adding magnitude semantics did not loosen the strict probe field set."""
    case = tmp_path / "case"
    write_case(case)
    config = _config(case)
    config["probe_groups"][0]["units"] = "N*m*s/rad"
    with pytest.raises((TypeError, ValueError)):
        QualificationConfig.from_primitive(config)


def test_magnitude_semantics_round_trips_exactly(tmp_path: Path) -> None:
    """The admitted string is preserved without trimming or normalizing."""
    case = tmp_path / "case"
    write_case(case)
    declared = "  absolute increase,  in  N·m·s/rad  "
    config = _config(case)
    config["probe_groups"][0]["magnitude_semantics"] = declared
    parsed = QualificationConfig.from_primitive(config)
    check.equal(
        parsed.probe_groups[0].magnitude_semantics,
        declared,
        "the admitted probe group rewrote the author's stated magnitude meaning instead of "
        "keeping the declaration exactly as written",
    )
    check.equal(
        parsed.to_primitive()["probe_groups"][0]["magnitude_semantics"],
        declared,
        "re-emitting the configuration changed the stated magnitude meaning, so a round trip "
        "would not reproduce the reviewer's own declaration",
    )


def test_a_rung_closure_equal_to_the_baseline_refuses(tmp_path: Path) -> None:
    """A rung that admits the baseline closure declares a perturbation that is not one."""
    case = tmp_path / "case"
    write_case(case)
    config = _config(case)
    config["probe_groups"][0]["variants"][0]["candidate"]["model_root"] = "baseline"
    with pytest.raises(ComparisonOperationError) as caught:
        qualify_configuration_file(_write(case, config))
    assert caught.value.failure.reason.code is OperationalReasonCode.INTERNAL_INVARIANT_FAILED
    assert caught.value.failure.reason.evidence["message"] == (
        "probe group 'hinge_damping_increase' rung 0 admits the same model closure as the "
        "baseline, so it declares a perturbation that is not one"
    )


def test_duplicate_declared_probe_models_across_two_rungs_refuse(tmp_path: Path) -> None:
    """Two magnitudes in one group must not declare the same model root and entrypoint."""
    case = tmp_path / "case"
    write_case(case)
    config = _config(case)
    first = config["probe_groups"][0]["variants"][0]["candidate"]["model_root"]
    config["probe_groups"][0]["variants"][1]["candidate"]["model_root"] = first
    with pytest.raises(ComparisonOperationError) as caught:
        qualify_configuration_file(_write(case, config))
    assert caught.value.failure.reason.code is OperationalReasonCode.CONFIGURATION_PARSE_FAILED
    assert caught.value.failure.reason.evidence["message"] == (
        "probe variants in one group must declare distinct model_root + entrypoint pairs"
    )


def test_distinct_declared_models_with_one_compiled_closure_refuse(tmp_path: Path) -> None:
    """Distinct source locations do not rescue two rungs that compile to one closure."""
    case = tmp_path / "case"
    path = write_case(case)
    first = case / "probes" / "damping_increase" / "rung_1" / "model.xml"
    second = case / "probes" / "damping_increase" / "rung_2" / "model.xml"
    second.write_bytes(first.read_bytes())
    with pytest.raises(ComparisonOperationError) as caught:
        qualify_configuration_file(path)
    assert caught.value.failure.reason.code is OperationalReasonCode.INTERNAL_INVARIANT_FAILED
    assert caught.value.failure.reason.evidence["message"] == (
        "probe group 'hinge_damping_increase' rungs 0 and 1 admit the same model closure, so two "
        "declared magnitudes name one model"
    )


def test_distinct_closures_pass(tmp_path: Path) -> None:
    """The shipped ladder has distinct closures and completes."""
    case = tmp_path / "case"
    result = qualify_configuration_file(write_case(case))
    closures = {
        rung["closure_sha256"]
        for rung in result.receipt["probe_model_closures"]["hinge_damping_increase"]
    }
    check.equal(
        len(closures),
        2,
        "the shipped two-rung ladder published fewer than two distinct model closures, so its "
        "rungs do not name different models",
    )
    check.is_not_in(
        result.receipt["campaign_identity"]["baseline_model_closure_sha256"],
        closures,
        "a probe rung published the baseline's own model closure, so it declares a perturbation "
        "that perturbs nothing",
    )


def test_a_candidate_timestep_different_from_the_baseline_is_allowed(tmp_path: Path) -> None:
    """A probe may legitimately declare its own timestep when the model agrees.

    Schema version 1 deliberately has no rule requiring candidate and baseline timesteps to match;
    what is required is that the declared value is the model's own and that compare admits it.
    """
    case = tmp_path / "case"
    write_case(case)
    rung = case / "probes" / "damping_increase" / "rung_1" / "model.xml"
    rung.write_text(
        rung.read_text(encoding="utf-8").replace('timestep="0.001"', 'timestep="0.002"'),
        encoding="utf-8",
    )
    config = _config(case)
    config["probe_groups"][0]["variants"][0]["candidate"]["declared_step_dt"] = "0.002"
    result = qualify_configuration_file(_write(case, config))
    check.is_in(
        result.status.value,
        {
            "QUALIFIED_FOR_DECLARED_PROBES",
            "PARTIALLY_QUALIFIED",
            "INSUFFICIENT_EXCITATION",
            "UNRESOLVED",
        },
        "a probe declaring the timestep its own compiled model carries did not reach a completed "
        "qualification status, so a legitimate candidate timestep was rejected",
    )


def test_a_candidate_timestep_inconsistent_with_its_model_refuses(tmp_path: Path) -> None:
    """A declared timestep the compiled model does not carry is refused by compare's contract."""
    case = tmp_path / "case"
    write_case(case)
    config = _config(case)
    config["probe_groups"][0]["variants"][0]["candidate"]["declared_step_dt"] = "0.002"
    with pytest.raises(ComparisonOperationError):
        qualify_configuration_file(_write(case, config))


def test_the_probe_semantics_limitation_is_published(tmp_path: Path) -> None:
    """The receipt states that probe labels are declarations, not verified facts."""
    case = tmp_path / "case"
    result = qualify_configuration_file(write_case(case))
    check.is_in(
        "USER_DECLARED_PROBE_SEMANTICS_NOT_VERIFIED",
        result.receipt["limitations"],
        "the published receipt does not state the limitation that probe labels are the author's "
        "declarations rather than verified facts",
    )
    check.is_in(
        "USER_DECLARED_PROBE_SEMANTICS_NOT_VERIFIED",
        [code.value for code in QUALIFICATION_LIMITATIONS],
        "the product's declared limitation vocabulary carries no code for unverified probe "
        "semantics, so the receipt would be publishing a limitation nothing defines",
    )
    joined = " ".join(str(item) for item in result.receipt["not_claimed"])
    check.is_in(
        "faithfully describe the source edits",
        joined,
        "the receipt's not-claimed section never disclaims that the probe labels faithfully "
        "describe the source edits",
    )
    markdown = result.qualification_markdown.read_text(encoding="utf-8")
    check.is_in(
        SEMANTICS,
        markdown,
        "the human-readable report does not quote the author's declared magnitude semantics, so "
        "a reader cannot see what the magnitudes were claimed to mean",
    )
    check.is_in(
        "does not establish that these labels",
        markdown,
        "the human-readable report does not warn the reader that the campaign leaves the probe "
        "labels unverified",
    )
