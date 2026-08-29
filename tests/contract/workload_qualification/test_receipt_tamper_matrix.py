"""Every decision-bearing field is rehashed and must still be rejected.

The premise of the whole correction: an unkeyed self-hash detects accidental corruption and nothing
else. Each case below edits one class of claim, recomputes `receipt_sha256` so the envelope is
internally consistent, and requires the public loader to refuse anyway.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
import pytest_check as check

from metrifid.json_values import canonical_json_bytes, compute_self_hash
from metrifid.workload_qualification import load_and_validate_workload_qualification_receipt
from metrifid.workload_qualification._aggregate import AggregateSchemaError
from metrifid.workload_qualification._evidence import raw_digest
from metrifid.workload_qualification._receipt import LinkedEvidenceError
from metrifid.workload_qualification._reconstruct import ReconstructionError

from .conftest import RECEIPT_RELATIVE, relocate, reseal

# The three ways a published receipt can be refused: it is not the declared shape, it
# does not follow from its own evidence, or it does not bind to the retained files.
REFUSALS = (AggregateSchemaError, ReconstructionError, LinkedEvidenceError)


def _reject(case: Path, mutate: Callable[[dict], None]) -> str:
    """Reseal one mutated receipt and require the loader to refuse it."""
    receipt_path = case / RECEIPT_RELATIVE
    reseal(receipt_path, mutate)
    with pytest.raises(REFUSALS) as caught:
        load_and_validate_workload_qualification_receipt(receipt_path)
    return str(caught.value)


def _duplicate_compiled_probe_closure(receipt: dict) -> None:
    """Make two reported rungs name one compiled closure without duplicating their declarations."""
    rungs = next(iter(receipt["probe_model_closures"].values()))
    rungs[1]["closure"] = rungs[0]["closure"]
    rungs[1]["closure_sha256"] = rungs[0]["closure_sha256"]


def test_the_honest_receipt_is_accepted(case_copy: Path) -> None:
    """A positive control, so a loader that refused everything would not read as a pass."""
    receipt = load_and_validate_workload_qualification_receipt(case_copy / RECEIPT_RELATIVE)
    check.equal(
        receipt["status"],
        "QUALIFIED_FOR_DECLARED_PROBES",
        "the untampered published receipt does not report the qualified status its own retained "
        "evidence supports, so the refusals below would prove nothing",
    )


_MUTATIONS: dict[str, Callable[[dict], None]] = {
    "canonical configuration declaration": lambda r: r["configuration"]["probe_groups"][
        0
    ].__setitem__("parameter", "forged.parameter"),
    "raw configuration hash": lambda r: r.__setitem__("configuration_raw_sha256", "a" * 64),
    "raw configuration locator": lambda r: r.__setitem__("configuration_locator", "elsewhere.json"),
    "baseline closure": lambda r: r["baseline_model_closure"].__setitem__("entrypoint", "x.xml"),
    "probe rung closure": lambda r: r["probe_model_closures"]["hinge_damping_increase"][
        0
    ].__setitem__("closure_sha256", "b" * 64),
    "workload identity": lambda r: next(
        iter(r["workload_artifact_identities"].values())
    ).__setitem__("actions_raw_sha256", "c" * 64),
    "workload aliases raw identity": lambda r: next(
        iter(r["workload_artifact_identities"].values())
    ).__setitem__("aliases_raw_sha256", "c" * 64),
    "workload aliases semantic identity": lambda r: next(
        iter(r["workload_artifact_identities"].values())
    ).__setitem__("aliases_semantic_sha256", "c" * 64),
    "control status": lambda r: r["zero_change_controls"][0].__setitem__(
        "comparison_status", "MATERIAL_BEHAVIOR_CHANGE"
    ),
    "control eligibility": lambda r: r["zero_change_controls"][0].__setitem__("eligible", False),
    "control exclusion reason": lambda r: r["zero_change_controls"][0].__setitem__(
        "exclusion_reason", "invented"
    ),
    "control receipt hash": lambda r: r["zero_change_controls"][0].__setitem__(
        "comparison_receipt_sha256", "d" * 64
    ),
    "control locator": lambda r: r["zero_change_controls"][0].__setitem__(
        "comparison_receipt_locator",
        "evidence/controls/workload_001/comparison_out/comparison.json",
    ),
    "probe cell key": lambda r: r["probe_cells"][0].__setitem__("variant_index", 1),
    "probe cell status": lambda r: r["probe_cells"][0].__setitem__(
        "comparison_status", "COVERAGE_INSUFFICIENT"
    ),
    "probe cell outcome": lambda r: r["probe_cells"][0].__setitem__(
        "outcome",
        "DETECTED" if r["probe_cells"][0]["outcome"] != "DETECTED" else "NOT_DETECTED",
    ),
    "probe cell raw hash": lambda r: r["probe_cells"][0].__setitem__(
        "comparison_receipt_raw_sha256", "e" * 64
    ),
    "missing cell": lambda r: r.__setitem__("probe_cells", r["probe_cells"][:-1]),
    "duplicate cell": lambda r: r.__setitem__(
        "probe_cells", [*r["probe_cells"], r["probe_cells"][0]]
    ),
    "reordered cells": lambda r: r.__setitem__("probe_cells", list(reversed(r["probe_cells"]))),
    "eligible list": lambda r: r.__setitem__("eligible_workload_ids", ["gentle", "medium"]),
    "excluded list": lambda r: r.__setitem__("excluded_workload_ids", ["gentle"]),
    "selected subset": lambda r: r.__setitem__(
        "selected_workload_ids", ["gentle", "medium", "gentle"]
    ),
    "subsets_evaluated": lambda r: r.__setitem__("subsets_evaluated", 99),
    "subset ranking": lambda r: r.__setitem__("subset_ranking", []),
    "group signature": lambda r: r["selection"]["groups"][0].__setitem__(
        "detection_signature", ["NOT_DETECTED", "NOT_DETECTED"]
    ),
    "group floor": lambda r: r["selection"]["groups"][0].__setitem__("floor_magnitude", "0.08"),
    "group no-floor reason": lambda r: r["selection"]["groups"][0].__setitem__(
        "no_floor_reason", "invented"
    ),
    "group status": lambda r: r["selection"]["groups"][0].__setitem__("status", "INSUFFICIENT"),
    "execution counts": lambda r: r["execution_counts"].__setitem__("probe_comparisons", 999),
    "planned comparisons": lambda r: r.__setitem__("planned_comparisons", 7),
    "overall status": lambda r: r.__setitem__("status", "INSUFFICIENT_EXCITATION"),
    "exit code": lambda r: r.__setitem__("completed_exit_code", 20),
    "witness field": lambda r: r["witnesses"].__setitem__(
        "first_witness",
        {
            "kind": "BLIND_RUNG",
            "probe_id": "x",
            "parameter": "y",
            "magnitude": "0.03",
            "variant_index": 0,
            "workload_id": "gentle",
            "detail": "invented",
        },
    ),
    "witness ordering declaration": lambda r: r["witnesses"].__setitem__(
        "witness_order", "whatever order suits the conclusion"
    ),
    "limitations registry": lambda r: r.__setitem__("limitations", r["limitations"][:-1]),
    "not-claimed registry": lambda r: r.__setitem__("not_claimed", []),
    "tool identity": lambda r: r["campaign_identity"]["tool"].__setitem__("version", "9.9.9"),
    "runtime identity": lambda r: r["campaign_identity"]["environment"].__setitem__(
        "python_version", "1.0.0"
    ),
    "unknown nested field": lambda r: r["selection"].__setitem__("extra", 1),
    "wrong primitive type": lambda r: r.__setitem__("subsets_evaluated", "four"),
    "locator traversal": lambda r: r["probe_cells"][0].__setitem__(
        "comparison_receipt_locator", "../../../etc/passwd"
    ),
    "absolute locator": lambda r: r["probe_cells"][0].__setitem__(
        "comparison_receipt_locator", "/etc/passwd"
    ),
    "duplicate locator": lambda r: r["probe_cells"][1].__setitem__(
        "comparison_receipt_locator", r["probe_cells"][0]["comparison_receipt_locator"]
    ),
}


@pytest.mark.parametrize("name", sorted(_MUTATIONS))
def test_a_resealed_mutation_is_rejected(case_copy: Path, name: str) -> None:
    """Recomputing the self-hash never rescues a contradictory or unbound claim."""
    check.is_true(
        _reject(case_copy, _MUTATIONS[name]),
        f"the loader refused the forged {name} without naming a reason, so an operator is told "
        "the receipt is unusable but not which claim failed",
    )


def test_replacing_a_retained_comparison_receipt_is_rejected(case_copy: Path) -> None:
    """Linked evidence is rebound to its digests, so swapping a retained file is caught."""
    target = case_copy / "qualification_out" / "evidence" / "controls" / "workload_000"
    (target / "comparison_out" / "comparison.json").write_text("{}", encoding="utf-8")
    with pytest.raises(REFUSALS):
        load_and_validate_workload_qualification_receipt(case_copy / RECEIPT_RELATIVE)


def test_a_resealed_duplicate_compiled_probe_closure_is_rejected_first(case_copy: Path) -> None:
    """Reader-side ladder uniqueness rejects before ordinary receipt-to-rung mismatch."""
    receipt_path = case_copy / RECEIPT_RELATIVE
    reseal(receipt_path, _duplicate_compiled_probe_closure)
    with pytest.raises(
        LinkedEvidenceError,
        match="two rungs with one compiled model closure",
    ):
        load_and_validate_workload_qualification_receipt(receipt_path)


def test_replacing_a_retained_generated_configuration_is_rejected(case_copy: Path) -> None:
    """The generated configuration is bound to the digest the comparison recorded for it."""
    target = case_copy / "qualification_out" / "evidence" / "controls" / "workload_000"
    (target / "comparison.json").write_text("{}", encoding="utf-8")
    with pytest.raises(REFUSALS):
        load_and_validate_workload_qualification_receipt(case_copy / RECEIPT_RELATIVE)


def test_same_rung_evidence_from_the_wrong_workload_is_rejected(case_copy: Path) -> None:
    """A genuine same-rung receipt and config cannot be relabeled as another workload's cell."""
    receipt_path = case_copy / RECEIPT_RELATIVE

    def swap(receipt: dict) -> None:
        target = next(
            cell
            for cell in receipt["probe_cells"]
            if cell["workload_index"] == 1
            and cell["group_index"] == 0
            and cell["variant_index"] == 0
        )
        donor = next(
            cell
            for cell in receipt["probe_cells"]
            if cell["workload_index"] == 2
            and cell["group_index"] == 0
            and cell["variant_index"] == 0
        )
        for locator_field, digest_field in (
            ("comparison_config_locator", "comparison_config_raw_sha256"),
            ("comparison_receipt_locator", "comparison_receipt_raw_sha256"),
        ):
            destination = case_copy / "qualification_out" / target[locator_field]
            source = case_copy / "qualification_out" / donor[locator_field]
            destination.write_bytes(source.read_bytes())
            target[digest_field] = raw_digest(destination.read_bytes())
        target["comparison_receipt_sha256"] = donor["comparison_receipt_sha256"]
        target["comparison_status"] = donor["comparison_status"]
        target["outcome"] = donor["outcome"]

    reseal(receipt_path, swap)
    # The donor configuration still records the donor cell's own owned output locator, so the
    # relabelled cell is now refused for naming an evidence location that is not the one it is
    # registered under. That is the same forgery caught one stage earlier than the workload
    # identity checks, which remain the refusal for a forgery that keeps the locator honest.
    with pytest.raises(
        REFUSALS,
        match=r"wrong initial state|wrong workload|does not name the exact owned evidence locator",
    ):
        load_and_validate_workload_qualification_receipt(receipt_path)


def test_same_rung_receipt_identity_from_the_wrong_workload_is_rejected(case_copy: Path) -> None:
    """Receipt hashes bind the workload even when the target generated config remains in place."""
    receipt_path = case_copy / RECEIPT_RELATIVE

    def swap_receipt_only(receipt: dict) -> None:
        target = next(
            cell
            for cell in receipt["probe_cells"]
            if cell["workload_index"] == 1
            and cell["group_index"] == 0
            and cell["variant_index"] == 0
        )
        donor = next(
            cell
            for cell in receipt["probe_cells"]
            if cell["workload_index"] == 2
            and cell["group_index"] == 0
            and cell["variant_index"] == 0
        )
        donor_path = case_copy / "qualification_out" / donor["comparison_receipt_locator"]
        forged = json.loads(donor_path.read_text(encoding="utf-8"))
        forged["inputs"]["configuration_raw_sha256"] = target["comparison_config_raw_sha256"]
        forged["receipt_sha256"] = None
        forged["receipt_sha256"] = compute_self_hash(forged, "receipt_sha256")
        forged_bytes = canonical_json_bytes(forged) + b"\n"
        target_path = case_copy / "qualification_out" / target["comparison_receipt_locator"]
        target_path.write_bytes(forged_bytes)
        target["comparison_receipt_raw_sha256"] = raw_digest(forged_bytes)
        target["comparison_receipt_sha256"] = forged["receipt_sha256"]
        target["comparison_status"] = donor["comparison_status"]
        target["outcome"] = donor["outcome"]

    reseal(receipt_path, swap_receipt_only)
    with pytest.raises(LinkedEvidenceError, match="actions_raw_sha256"):
        load_and_validate_workload_qualification_receipt(receipt_path)


def test_symlinking_a_retained_comparison_receipt_is_rejected(
    case_copy: Path, tmp_path: Path
) -> None:
    """A retained decision-bearing file must be a regular file, not a link to one."""
    decoy = tmp_path / "decoy.json"
    target = (
        case_copy
        / "qualification_out"
        / "evidence"
        / "controls"
        / "workload_000"
        / "comparison_out"
        / "comparison.json"
    )
    decoy.write_bytes(target.read_bytes())
    target.unlink()
    target.symlink_to(decoy)
    with pytest.raises(REFUSALS):
        load_and_validate_workload_qualification_receipt(case_copy / RECEIPT_RELATIVE)


def test_removing_the_retained_raw_configuration_is_rejected(case_copy: Path) -> None:
    """The admitted configuration bytes are part of the evidence, not a convenience copy."""
    (case_copy / "qualification_out" / "qualification.json").unlink()
    with pytest.raises(REFUSALS):
        load_and_validate_workload_qualification_receipt(case_copy / RECEIPT_RELATIVE)


def test_an_unrelated_extra_file_does_not_invalidate_the_receipt(case_copy: Path) -> None:
    """Only registered evidence is bound; an unrelated file in the tree is not a forgery."""
    (case_copy / "qualification_out" / "notes.txt").write_text("scratch", encoding="utf-8")
    load_and_validate_workload_qualification_receipt(case_copy / RECEIPT_RELATIVE)


def test_a_linked_receipt_directory_is_refused(case_copy: Path, tmp_path: Path) -> None:
    """The public entry artifact is read through the same binding as the members it registers.

    The receipt directory is moved outside the owned root and replaced with a link to it, so the
    bytes and every digest stay exactly as published. A loader that opened the aggregate by its
    whole pathname before binding the root would read those bytes and then validate the members
    against a different bound object.
    """
    receipt_directory = case_copy / "qualification_out" / "receipt"
    external = tmp_path / "external_receipt"
    shutil.move(str(receipt_directory), str(external))
    receipt_directory.symlink_to(external, target_is_directory=True)

    with pytest.raises(LinkedEvidenceError):
        load_and_validate_workload_qualification_receipt(case_copy / RECEIPT_RELATIVE)


def test_a_copied_and_renamed_output_root_still_replays(
    published_case: Path, tmp_path: Path
) -> None:
    """Replay is location independent: it binds the current tree, not the recorded strings."""
    receipt_path = relocate(published_case, tmp_path / "elsewhere", "renamed_out")

    document = load_and_validate_workload_qualification_receipt(receipt_path)

    check.equal(
        document["status"],
        "QUALIFIED_FOR_DECLARED_PROBES",
        "an honest campaign copied to a new parent and renamed no longer replays",
    )


def _generated_config(case: Path, record: dict) -> Path:
    """Return the retained generated comparison configuration one record registers."""
    return case / "qualification_out" / record["comparison_config_locator"]


def _retarget_cell_output(case: Path, record: dict, output_dir: str) -> None:
    """Rewrite one cell's recorded output directory and reseal everything that binds it.

    The generated configuration is bound three ways: the aggregate records its raw digest, the
    comparison receipt records the configuration it read, and that receipt has its own self-hash
    which the aggregate also records. All of them are recomputed here so the forgery is internally
    consistent and reaches the output-coherence check rather than stopping at an earlier binding.
    """
    target = _generated_config(case, record)
    generated = json.loads(target.read_text(encoding="utf-8"))
    generated["output_dir"] = output_dir
    target.write_bytes(canonical_json_bytes(generated) + b"\n")
    config_digest = raw_digest(target.read_bytes())
    record["comparison_config_raw_sha256"] = config_digest

    receipt_path = case / "qualification_out" / record["comparison_receipt_locator"]
    comparison = json.loads(receipt_path.read_text(encoding="utf-8"))
    comparison["inputs"]["configuration_raw_sha256"] = config_digest
    comparison["receipt_sha256"] = None
    comparison["receipt_sha256"] = compute_self_hash(comparison, "receipt_sha256")
    receipt_path.write_bytes(canonical_json_bytes(comparison) + b"\n")
    record["comparison_receipt_raw_sha256"] = raw_digest(receipt_path.read_bytes())
    record["comparison_receipt_sha256"] = comparison["receipt_sha256"]


def test_generated_configurations_from_two_historical_roots_are_rejected(case_copy: Path) -> None:
    """One campaign wrote its cells under one output root, so its cells must record one root."""

    def split_roots(receipt: dict) -> None:
        record = receipt["probe_cells"][0]
        locator = record["comparison_receipt_locator"].rsplit("/", 1)[0]
        _retarget_cell_output(case_copy, record, f"/somewhere/else/{locator}")

    assert "share one campaign output root" in _reject(case_copy, split_roots)


def test_a_generated_configuration_naming_another_cell_locator_is_rejected(
    case_copy: Path,
) -> None:
    """A cell's configuration must name its own evidence location, not a sibling's."""

    def borrow_locator(receipt: dict) -> None:
        record = receipt["probe_cells"][0]
        donor = receipt["probe_cells"][1]
        generated = json.loads(_generated_config(case_copy, record).read_text(encoding="utf-8"))
        root = generated["output_dir"].rsplit(
            record["comparison_receipt_locator"].rsplit("/", 1)[0], 1
        )[0]
        borrowed = donor["comparison_receipt_locator"].rsplit("/", 1)[0]
        _retarget_cell_output(case_copy, record, f"{root}{borrowed}")

    # Stripping the cell's own admitted locator is what recovers the historical root, so a
    # configuration carrying a sibling's locator is refused there rather than at the later
    # per-cell comparison: it never yields a root for this cell at all.
    assert "does not name the exact owned evidence locator" in _reject(case_copy, borrow_locator)
