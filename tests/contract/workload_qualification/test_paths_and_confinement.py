"""Declared paths, semantic labels, and output ownership are all confined.

Two distinct defects motivate this file. Semantic identifiers were used as directory components, so
an absolute identifier placed evidence wherever it liked. And an output overlapping a model root
created staging directories inside that model before the nested comparison refused.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import pytest
import pytest_check as check

from metrifid.compare._failure import ComparisonOperationError
from metrifid.workload_qualification import (
    load_and_validate_workload_qualification_receipt,
    qualify_configuration_file,
)
from metrifid.workload_qualification._config import QualificationConfig
from metrifid.workload_qualification._paths import (
    PathAdmissionError,
    admit_relative_path,
    control_locator,
    probe_locator,
)
from tests._support.workload_qualification import write_case


def _config(case: Path) -> dict:
    return json.loads((case / "qualification.json").read_text(encoding="utf-8"))


def _write(case: Path, config: dict) -> Path:
    path = case / "qualification.json"
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def _escapees(case: Path) -> list[str]:
    """Return each path created under the output root that does not live inside the case."""
    return [
        str(created)
        for created in (case / "qualification_out").rglob("*")
        if not (case in created.resolve().parents or created.resolve() == case)
    ]


HOSTILE_IDS = (
    "/absolute/workload",
    "../escape",
    "with/separator",
    "./dot",
    "line\nbreak",
    "pipe|cell",
    "back`tick",
    "<b>html</b>",
    "C:\\windows",
)


@pytest.mark.parametrize("label", HOSTILE_IDS)
def test_a_hostile_workload_id_never_becomes_a_storage_path(tmp_path: Path, label: str) -> None:
    """A semantic identifier is report data; storage names come from sequence position only."""
    case = tmp_path / "case"
    config = _config(Path(write_case(case)).parent)
    config["workloads"][0]["workload_id"] = label
    path = _write(case, config)
    outside = tmp_path / "escaped"

    try:
        qualify_configuration_file(path)
    except (ComparisonOperationError, PathAdmissionError, ValueError):
        pass
    check.is_false(
        outside.exists(),
        "a hostile workload identifier steered the campaign into writing beside the case",
    )
    check.equal(
        _escapees(case),
        [],
        "a hostile workload identifier became a storage component and placed evidence "
        "outside the case directory",
    )


@pytest.mark.parametrize("label", HOSTILE_IDS)
def test_a_hostile_probe_id_never_becomes_a_storage_path(tmp_path: Path, label: str) -> None:
    """The same holds for probe identifiers."""
    case = tmp_path / "case"
    config = _config(Path(write_case(case)).parent)
    config["probe_groups"][0]["probe_id"] = label
    path = _write(case, config)
    try:
        qualify_configuration_file(path)
    except (ComparisonOperationError, PathAdmissionError, ValueError):
        pass
    check.equal(
        _escapees(case),
        [],
        "a hostile probe identifier became a storage component and placed evidence "
        "outside the case directory",
    )


def test_storage_components_are_ordinals_independent_of_labels() -> None:
    """Storage names derive from admitted position, never from label bytes."""
    check.equal(
        control_locator(0),
        PurePosixPath("evidence/controls/workload_000"),
        "the control evidence directory is not the ordinal for its workload position",
    )
    check.equal(
        probe_locator(2, 1, 3),
        PurePosixPath("evidence/probes/workload_002/group_001/rung_003"),
        "the probe cell directory is not the ordinal triple for its admitted position",
    )


@pytest.mark.parametrize(
    "declared",
    ["/absolute/root", "../outside", "a//b", "./here", "a\\b", "a/../b", "trailing/"],
)
def test_a_non_normalized_declared_path_refuses(declared: str) -> None:
    """Declared paths are admitted as normalized traversal-free relative POSIX paths."""
    with pytest.raises(PathAdmissionError):
        admit_relative_path(declared, "model_root")


def test_an_entrypoint_resolves_against_its_model_root(tmp_path: Path) -> None:
    """An entrypoint names a file inside its own model root, not beside qualification.json.

    A decoy of the same name sits next to the configuration. If the entrypoint were resolved against
    the configuration directory the campaign would read the decoy, so the baseline closure digest is
    checked against the real model file.
    """
    import hashlib

    case = tmp_path / "case"
    path = write_case(case)
    (case / "model.xml").write_text("<mujoco model='decoy'/>", encoding="utf-8")

    result = qualify_configuration_file(path)
    closure = result.receipt["baseline_model_closure"]
    members = closure["members"]
    assert len(members) == 1, (
        "the baseline model closure does not name exactly one file, so there is no single "
        "digest left to compare against the declared model"
    )
    expected = hashlib.sha256((case / "baseline" / "model.xml").read_bytes()).hexdigest()
    check.equal(
        members[0]["sha256"],
        expected,
        "the baseline closure digests the decoy beside qualification.json instead of the "
        "model inside the declared model root",
    )


@pytest.mark.parametrize(
    "output_dir",
    ["baseline", "baseline/out", "probes/damping_increase/rung_1/out"],
)
def test_output_inside_a_model_root_refuses_before_any_write(
    tmp_path: Path, output_dir: str
) -> None:
    """An output under an observed model would modify the tree the campaign is measuring."""
    case = tmp_path / "case"
    write_case(case)
    config = _config(case)
    config["output_dir"] = output_dir
    path = _write(case, config)

    before = {p: p.stat().st_mtime_ns for p in case.rglob("*") if p.is_file()}
    with pytest.raises(ComparisonOperationError):
        qualify_configuration_file(path)
    after = {p: p.stat().st_mtime_ns for p in case.rglob("*") if p.is_file()}
    check.equal(
        after,
        before,
        "a refused qualification created or modified files in the model tree it measures",
    )


def test_a_symlink_alias_of_a_model_root_refuses(tmp_path: Path) -> None:
    """The alias spelling of an overlap is the same overlap."""
    case = tmp_path / "case"
    write_case(case)
    (case / "alias").symlink_to(case / "baseline")
    config = _config(case)
    config["output_dir"] = "alias/out"
    path = _write(case, config)
    with pytest.raises(ComparisonOperationError):
        qualify_configuration_file(path)
    check.is_false(
        (case / "baseline" / "out").exists(),
        "the alias spelling let the campaign stage output inside the model root it observes",
    )


def test_a_pre_existing_output_root_refuses(tmp_path: Path) -> None:
    """Published evidence is never overwritten."""
    case = tmp_path / "case"
    path = write_case(case)
    (case / "qualification_out").mkdir()
    with pytest.raises(ComparisonOperationError):
        qualify_configuration_file(path)


def test_a_second_run_into_the_same_output_directory_refuses(tmp_path: Path) -> None:
    """The no-overwrite contract holds for a completed campaign too."""
    case = tmp_path / "case"
    path = write_case(case)
    qualify_configuration_file(path)
    with pytest.raises(ComparisonOperationError):
        qualify_configuration_file(path)


def test_every_retained_locator_is_normalized_relative_unique_and_confined(
    tmp_path: Path,
) -> None:
    """Locators are recorded data that re-admit cleanly under the owned root."""
    case = tmp_path / "case"
    result = qualify_configuration_file(write_case(case))
    receipt = result.receipt
    root = (case / "qualification_out").resolve()
    locators = [str(receipt["configuration_locator"])]
    for record in (*receipt["zero_change_controls"], *receipt["probe_cells"]):
        locators.append(str(record["comparison_config_locator"]))
        locators.append(str(record["comparison_receipt_locator"]))

    inadmissible: list[str] = []
    unconfined: list[str] = []
    for locator in locators:
        try:
            admit_relative_path(locator, "locator")
        except PathAdmissionError:
            inadmissible.append(locator)
            continue
        if not (root / locator).resolve().is_relative_to(root):
            unconfined.append(locator)

    check.equal(
        len(set(locators)),
        len(locators),
        "the receipt points two retained cells at the same evidence locator",
    )
    check.equal(
        inadmissible,
        [],
        "the receipt records evidence locators that do not re-admit as normalized "
        "traversal-free relative paths",
    )
    check.equal(
        unconfined,
        [],
        "the receipt records evidence locators that resolve outside the owned output root",
    )


def test_an_unknown_configuration_field_still_refuses(tmp_path: Path) -> None:
    """Path admission did not loosen the strict field set."""
    case = tmp_path / "case"
    write_case(case)
    config = _config(case)
    config["unexpected"] = True
    with pytest.raises((TypeError, ValueError)):
        QualificationConfig.from_primitive(config)


def test_a_stable_alias_publishes_a_receipt_its_own_public_loader_accepts(tmp_path: Path) -> None:
    """A declared output reached through a stable link completes and replays immediately.

    Ownership admits a canonical output, so the campaign publishes at the canonical location rather
    than at the declared spelling. A loader that rebuilt the expected evidence location from that
    declaration rejected the product's own honest receipt; a loader that consulted the link's
    current target would instead let a later retarget decide what a published receipt means. Both
    are checked here: the receipt replays immediately, and again after the alias is retargeted.
    """
    case = tmp_path / "case"
    safe = tmp_path / "safe"
    (safe / "sub").mkdir(parents=True)
    write_case(case, output_dir="alias/sub/out")
    (case / "alias").symlink_to(safe)
    model_root = (case / "baseline").resolve()
    model_root_before = sorted(p.name for p in model_root.iterdir())

    result = qualify_configuration_file(case / "qualification.json")

    published = Path(result.qualification_json)
    check.is_true(
        published.is_relative_to(safe.resolve()),
        "the campaign did not publish at the canonical output ownership accepted",
    )
    load_and_validate_workload_qualification_receipt(published)

    (case / "alias").unlink()
    (case / "alias").symlink_to(tmp_path / "elsewhere")
    load_and_validate_workload_qualification_receipt(published)

    check.equal(
        sorted(p.name for p in model_root.iterdir()),
        model_root_before,
        "the campaign wrote inside the model root it was measuring",
    )
    check.is_false(
        (tmp_path / "elsewhere").exists(),
        "retargeting the declared alias created something at its new target",
    )
