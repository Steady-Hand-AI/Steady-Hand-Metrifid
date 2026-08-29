"""The complaint-backed public case gallery must keep deciding exactly what it claims to decide.

The gallery is the public evidence that Metrifid answers a real release question end to end, so it
runs against an installed distribution rather than a source-tree import, and this test drives it the
same way a reader would: as a command, from a working directory outside the checkout.

Every case is reported independently. One case whose control regresses must not hide the five that
still hold, and a stage that changes decision must name the case and the stage it changed for.

Two cases are known to stop short of the declared decision the gallery asks for. Deleting a compiled
element shifts MuJoCo's compiled name identity mapping, and the product answers that fail-closed with
an opaque residual no policy rule can clear. Those two stages are marked as expected failures
strictly, so the day the product gains a way to declare an identity-shifting change, this test fails
and has to be updated rather than silently continuing to expect the old answer.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import pytest_check as check

import metrifid
from metrifid.certify import load_and_validate_certification_receipt
from metrifid.json_values import canonical_json_bytes, compute_self_hash
from metrifid.model_release import load_and_validate_model_release_receipt
from metrifid.model_release._receipt_validation import model_release_decision_sha256

_SUCCESS_TOKEN = "METRIFID_PUBLIC_CASE_GALLERY_PASSED"
_FAILURE_TOKEN = "METRIFID_PUBLIC_CASE_GALLERY_FAILED"

_FROZEN_CASE_ORDER = (
    "collision_filtering.mask_flattening",
    "collision_filtering.explicit_pair_loss",
    "collision_filtering.exclusion_loss",
    "mesh_inertia.mode_change",
    "actuator_transmission.frame_change",
    "sensor_attachment.site_change",
)

_COMMON_JOURNEY = {
    "certify": ("NOT_CERTIFIED_COMPILED_DIFFERS", 40),
    "discovery": ("REVIEW_REQUIRED", 40),
}
_FROZEN_JOURNEY_BY_CASE = {
    case_id: {**_COMMON_JOURNEY, "declared": declared}
    for case_id, declared in (
        ("collision_filtering.mask_flattening", ("WITHIN_DECLARED_POLICY", 0)),
        ("collision_filtering.explicit_pair_loss", ("REVIEW_REQUIRED", 40)),
        ("collision_filtering.exclusion_loss", ("REVIEW_REQUIRED", 40)),
        ("mesh_inertia.mode_change", ("WITHIN_DECLARED_POLICY", 0)),
        ("actuator_transmission.frame_change", ("WITHIN_DECLARED_POLICY", 0)),
        ("sensor_attachment.site_change", ("WITHIN_DECLARED_POLICY", 0)),
    )
}

_FROZEN_CONTROL = {
    "collision_filtering.mask_flattening": "ELIGIBLE_PAIR_SET_DIFFERS",
    "collision_filtering.explicit_pair_loss": "BASELINE_PRESENT_CANDIDATE_ABSENT",
    "collision_filtering.exclusion_loss": "BASELINE_PRESENT_CANDIDATE_ABSENT",
    "mesh_inertia.mode_change": "MASS_OR_INERTIA_DIFFERS",
    "actuator_transmission.frame_change": "ACTUATOR_MOMENT_OR_GENERALIZED_FORCE_DIFFERS",
    "sensor_attachment.site_change": "BASELINE_SENSOR_FRAME_CANDIDATE_GRIP_FRAME",
}

_RECEIPT_LOADERS = {
    "certify": load_and_validate_certification_receipt,
    "discovery": load_and_validate_model_release_receipt,
    "declared": load_and_validate_model_release_receipt,
}


def _repository_root() -> Path:
    """Return the source checkout that holds the tracked gallery."""
    return Path(__file__).resolve().parents[2]


def _runner() -> Path:
    """Return the tracked gallery entry point."""
    return _repository_root() / "examples" / "public_cases" / "run_all.py"


def _run_gallery(output: Path, working_directory: Path) -> subprocess.CompletedProcess[str]:
    """Run the gallery command from a working directory outside the checkout."""
    return subprocess.run(
        [sys.executable, str(_runner()), "--output", str(output)],
        capture_output=True,
        text=True,
        cwd=str(working_directory),
        check=False,
    )


@pytest.fixture(scope="module")
def gallery(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the gallery twice on this runtime, into two separate absent workspaces."""
    if not _runner().is_file():
        pytest.skip(f"the tracked gallery runner is absent at {_runner()}")
    root = tmp_path_factory.mktemp("gallery")
    elsewhere = tmp_path_factory.mktemp("cwd")
    runs = []
    for index in (1, 2):
        output = root / f"run_{index}"
        completed = _run_gallery(output, elsewhere)
        assert (output / "gallery_result.json").is_file(), (
            f"run {index} published no gallery result\n{completed.stdout}\n{completed.stderr}"
        )
        runs.append(
            {
                "completed": completed,
                "output": output,
                "result": json.loads((output / "gallery_result.json").read_text(encoding="utf-8")),
            }
        )
    return {"root": root, "elsewhere": elsewhere, "runs": runs}


def _first(gallery: dict[str, Any]) -> dict[str, Any]:
    """Return the first gallery run."""
    return cast("dict[str, Any]", gallery["runs"][0])


def _case_directory(gallery: dict[str, Any], case_id: str) -> Path:
    """Return the directory one case published into during the first run."""
    run = _first(gallery)
    index = int(run["result"]["case_order"].index(case_id))
    output = cast("Path", run["output"])
    return (output / str(run["result"]["case_results"][index])).parent


def _case_result(gallery: dict[str, Any], case_id: str) -> dict[str, Any]:
    """Return one published case result from the first run."""
    path = _case_directory(gallery, case_id) / "case_result.json"
    return cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))


def test_metrifid_resolves_outside_the_repository_root() -> None:
    """The gallery is only public evidence if it runs against an installed distribution."""
    location = Path(metrifid.__file__ or "").resolve()
    assert location.is_file(), "metrifid did not resolve to a file"
    assert _repository_root() not in location.parents, (
        f"metrifid resolved inside the repository at {location}; run this from an installed wheel"
    )


def test_the_gallery_publishes_the_six_frozen_cases_in_order(gallery: dict[str, Any]) -> None:
    """The case order is frozen because cross-runtime projections are compared row by row."""
    result = _first(gallery)["result"]
    assert tuple(result["case_order"]) == _FROZEN_CASE_ORDER
    assert len(result["case_results"]) == len(_FROZEN_CASE_ORDER)
    assert len(result["stable_projection"]) == len(_FROZEN_CASE_ORDER)
    assert tuple(row["case_id"] for row in result["stable_projection"]) == _FROZEN_CASE_ORDER


@pytest.mark.parametrize(
    ("case_id", "stage"),
    [
        pytest.param(case_id, stage, id=f"{case_id}-{stage}")
        for case_id in _FROZEN_CASE_ORDER
        for stage in _FROZEN_JOURNEY_BY_CASE[case_id]
    ],
)
def test_each_stage_decides_the_frozen_status_and_exit(
    gallery: dict[str, Any], case_id: str, stage: str
) -> None:
    """A changed status or exit code changes what the published evidence means."""
    row = _case_result(gallery, case_id)["product_results"][stage]
    expected_status, expected_exit = _FROZEN_JOURNEY_BY_CASE[case_id][stage]
    observed = (row["status"], row["completed_exit_code"])
    assert observed == (expected_status, expected_exit), (
        f"{case_id} {stage} decided {observed[0]} (exit {observed[1]}) instead of the frozen "
        f"{expected_status} (exit {expected_exit})"
    )


@pytest.mark.parametrize("case_id", _FROZEN_CASE_ORDER)
def test_all_three_receipts_independently_validate(gallery: dict[str, Any], case_id: str) -> None:
    """A receipt a reader cannot revalidate is not evidence."""
    case_directory = _case_directory(gallery, case_id)
    result = _case_result(gallery, case_id)
    for stage, loader in _RECEIPT_LOADERS.items():
        row = result["product_results"][stage]
        # Each receipt path is published relative to the case directory that owns it, which is the
        # directory the case result itself sits in.
        path = case_directory / row["relative_receipt_path"]
        check.is_true(path.is_file(), f"{case_id}: {stage} receipt is missing at {path}")
        if not path.is_file():
            continue
        receipt = loader(path.read_bytes())
        check.equal(
            receipt["status"], row["status"], f"{case_id}: {stage} receipt status disagrees"
        )
        check.equal(
            receipt["receipt_sha256"],
            row["receipt_sha256"],
            f"{case_id}: {stage} receipt hash disagrees with the published result",
        )


def _reseal_model_release_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """Recompute both receipt hashes after one semantic contradiction is injected."""
    resealed = copy.deepcopy(receipt)
    resealed["decision_sha256"] = model_release_decision_sha256(resealed)
    resealed["receipt_sha256"] = None
    resealed["receipt_sha256"] = compute_self_hash(resealed, "receipt_sha256")
    return resealed


@pytest.mark.parametrize(
    "case_id",
    [
        "collision_filtering.explicit_pair_loss",
        "collision_filtering.exclusion_loss",
    ],
)
def test_identity_shifting_deletion_receipts_require_the_opaque_residual(
    gallery: dict[str, Any], case_id: str
) -> None:
    """A canonical reseal cannot suppress the fail-closed complete-MJB identity witness."""
    case_directory = _case_directory(gallery, case_id)
    result = _case_result(gallery, case_id)
    receipt_path = case_directory / result["product_results"]["declared"]["relative_receipt_path"]
    receipt = cast("dict[str, Any]", json.loads(receipt_path.read_text(encoding="utf-8")))
    opaque = [row for row in receipt["changes"] if row["selector"]["object_type"] == "opaque"]
    assert len(opaque) == 1
    assert opaque[0]["details"]["reasons"] == ["compiled_name_identity_mapping_changed"]

    forged = copy.deepcopy(receipt)
    forged["changes"] = [
        row for row in forged["changes"] if row["selector"]["object_type"] != "opaque"
    ]
    forged["change_count"] = len(forged["changes"])
    forged["classification_counts"]["UNDECLARED"] = 0
    forged["status"] = "WITHIN_DECLARED_POLICY"
    forged["completed_exit_code"] = 0
    forged["first_unexpected_witness"] = None

    with pytest.raises(ValueError, match="requires a fail-closed opaque residual"):
        load_and_validate_model_release_receipt(
            canonical_json_bytes(_reseal_model_release_receipt(forged))
        )


@pytest.mark.parametrize("case_id", _FROZEN_CASE_ORDER)
def test_the_required_public_fields_were_observed(gallery: dict[str, Any], case_id: str) -> None:
    """Each case exists to make specific compiled fields move; silence would be a regression."""
    result = _case_result(gallery, case_id)
    manifest = _manifest_for(case_id)
    observed = set(result["changed_public_fields"])
    for field in manifest["required_compiled_fields"]:
        check.is_in(field, observed, f"{case_id}: discovery did not report {field}")


@pytest.mark.parametrize("case_id", _FROZEN_CASE_ORDER)
def test_the_required_semantic_selectors_were_observed(
    gallery: dict[str, Any], case_id: str
) -> None:
    """A case that names semantic selectors must actually produce them."""
    result = _case_result(gallery, case_id)
    manifest = _manifest_for(case_id)
    observed = {json.dumps(item, sort_keys=True) for item in result["semantic_selectors"]}
    for selector in manifest["required_semantic_selectors"]:
        check.is_in(
            json.dumps(selector, sort_keys=True),
            observed,
            f"{case_id}: discovery did not report the selector {selector}",
        )


@pytest.mark.parametrize("case_id", _FROZEN_CASE_ORDER)
def test_the_independent_control_reaches_its_classification(
    gallery: dict[str, Any], case_id: str
) -> None:
    """The control observes the mechanism without Metrifid, so it must stand on its own."""
    control = _case_result(gallery, case_id)["independent_control"]
    check.is_in("kind", control, f"{case_id}: the control published no kind")
    check.equal(
        control["classification"],
        _FROZEN_CONTROL[case_id],
        f"{case_id}: the independent control reached a different classification",
    )


def test_the_runner_refuses_an_output_path_that_already_exists(
    gallery: dict[str, Any], tmp_path: Path
) -> None:
    """Publishing into an existing directory could merge into or overwrite a reader's own files."""
    existing = tmp_path / "already_here"
    existing.mkdir()
    completed = _run_gallery(existing, gallery["elsewhere"])
    assert completed.returncode != 0, "the runner accepted an output path that already existed"
    combined = completed.stdout + completed.stderr
    assert "absent" in combined, f"the refusal did not explain the requirement:\n{combined}"
    assert not (existing / "gallery_result.json").exists(), (
        "the runner published into an existing directory"
    )


def test_a_second_run_into_another_absent_workspace_succeeds(
    gallery: dict[str, Any],
) -> None:
    """The gallery must be repeatable; a run that only works once is not evidence."""
    second = gallery["runs"][1]
    assert (second["output"] / "gallery_result.json").is_file()
    assert (second["output"] / "gallery_summary.md").is_file()
    assert (second["output"] / "CHECKSUMS.sha256").is_file()


def test_the_two_runs_agree_on_the_stable_projection(gallery: dict[str, Any]) -> None:
    """The stable projection is the only thing compared across workspaces, so it must be stable."""
    first, second = (run["result"]["stable_projection"] for run in gallery["runs"])
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


@pytest.mark.parametrize("index", [0, 1])
def test_the_checksum_manifest_covers_every_published_file(
    gallery: dict[str, Any], index: int
) -> None:
    """A manifest that omits files cannot detect a published artifact being altered."""
    import hashlib

    run = gallery["runs"][index]
    output = run["output"]
    manifest_path = output / "CHECKSUMS.sha256"
    recorded: dict[str, str] = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, relative = line.partition("  ")
        recorded[relative] = digest
    published = {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file() and path != manifest_path
    }
    assert set(recorded) == published, "the checksum manifest does not cover every published file"
    for relative, digest in recorded.items():
        actual = hashlib.sha256((output / relative).read_bytes()).hexdigest()
        check.equal(actual, digest, f"{relative} does not match its recorded digest")


def test_the_runner_reports_a_token_that_matches_its_exit_status(
    gallery: dict[str, Any],
) -> None:
    """A passing token printed beside a divergence would be a false green."""
    for index, run in enumerate(gallery["runs"], start=1):
        completed = run["completed"]
        output = completed.stdout
        if completed.returncode == 0:
            check.is_in(_SUCCESS_TOKEN, output, f"run {index} exited 0 without the passing token")
            check.is_not_in(
                _FAILURE_TOKEN, output, f"run {index} exited 0 while reporting a failure"
            )
        else:
            check.is_not_in(
                _SUCCESS_TOKEN,
                output,
                f"run {index} printed the passing token while exiting {completed.returncode}",
            )
            check.is_in(
                _FAILURE_TOKEN,
                output,
                f"run {index} exited {completed.returncode} without naming the failure",
            )


def _manifest_for(case_id: str) -> dict[str, Any]:
    """Return the tracked manifest that declares one case."""
    gallery_root = _repository_root() / "examples" / "public_cases"
    for path in sorted(gallery_root.rglob("case_manifest.json")):
        manifest = cast("dict[str, Any]", json.loads(path.read_text(encoding="utf-8")))
        if manifest["case_id"] == case_id:
            return manifest
    raise AssertionError(f"no tracked manifest declares {case_id!r}")
