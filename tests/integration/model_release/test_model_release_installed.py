"""Installed CLI and SDK integration matrix for the static Model Change Gate."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_MODEL_RELEASE_FILENAMES = ("model_release.json", "model_release.md")


@dataclass(frozen=True, slots=True)
class _ModelPair:
    """Two admitted source paths and their public Certify MJB identities."""

    baseline: Path
    candidate: Path
    baseline_mjb_sha256: str
    candidate_mjb_sha256: str


@dataclass(frozen=True, slots=True)
class _Discovery:
    """One null-candidate policy result and all of its non-opaque changes."""

    pair: _ModelPair
    receipt: dict[str, Any]
    changes: tuple[dict[str, Any], ...]


def _slider_model(mass: str, *, model_name: str = "release-slider") -> str:
    """Return a compact all-named model whose mass affects a driven slide joint."""
    return f"""
<mujoco model="{model_name}">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="payload_body">
      <joint name="payload_slide" type="slide" axis="1 0 0" damping="0"/>
      <geom name="payload_geom" type="box" size="0.1 0.1 0.1" mass="{mass}"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="payload_motor" joint="payload_slide" gear="1"/>
  </actuator>
</mujoco>
"""


def _hinge_model(joint_range: str) -> str:
    """Return a compact all-named limited-hinge model."""
    return f"""
<mujoco model="release-hinge">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="arm_body">
      <joint name="arm_hinge" type="hinge" axis="0 0 1" limited="true"
             range="{joint_range}"/>
      <geom name="arm_geom" type="box" pos="0.2 0 0" size="0.2 0.05 0.05" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="arm_motor" joint="arm_hinge" gear="1"/>
  </actuator>
</mujoco>
"""


def _two_body_model(alpha_mass: str, zeta_mass: str) -> str:
    """Return two named dynamic branches in deliberately noncanonical source order."""
    return f"""
<mujoco model="release-order">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="zeta_body" pos="0 0 0.3">
      <joint name="zeta_slide" type="slide" axis="1 0 0"/>
      <geom name="zeta_geom" type="box" size="0.08 0.08 0.08" mass="{zeta_mass}"/>
    </body>
    <body name="alpha_body" pos="0 0 -0.3">
      <joint name="alpha_slide" type="slide" axis="1 0 0"/>
      <geom name="alpha_geom" type="box" size="0.08 0.08 0.08" mass="{alpha_mass}"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="zeta_motor" joint="zeta_slide"/>
    <motor name="alpha_motor" joint="alpha_slide"/>
  </actuator>
</mujoco>
"""


def _mesh_model(scale: str) -> str:
    """Return an all-named model with a physical tetrahedral mesh scaled in one axis."""
    return f"""
<mujoco model="release-mesh">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <asset>
    <mesh name="payload_mesh" scale="{scale} 1 1"
          vertex="0 0 0  1 0 0  0 1 0  0 0 1"
          face="0 2 1  0 1 3  0 3 2  1 2 3"/>
  </asset>
  <worldbody>
    <body name="mesh_body">
      <joint name="mesh_slide" type="slide" axis="1 0 0"/>
      <geom name="mesh_geom" type="mesh" mesh="payload_mesh" density="1000"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="mesh_motor" joint="mesh_slide"/>
  </actuator>
</mujoco>
"""


def _retargeted_actuator_model(joint_order: tuple[str, str], target: str) -> str:
    """Return a model whose raw target ID can stay fixed while its named target changes."""
    first, second = joint_order
    return f"""
<mujoco model="release-actuator-target">
  <compiler angle="radian"/>
  <worldbody>
    <body name="root">
      <joint name="{first}" type="hinge" axis="1 0 0"/>
      <joint name="{second}" type="hinge" axis="1 0 0"/>
      <geom name="g" type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="m" joint="{target}"/>
  </actuator>
</mujoco>
"""


def _reordered_joint_axes_model(joint_order: tuple[str, str]) -> str:
    """Keep per-ID axis bytes fixed while swapping which stable name owns each axis."""
    first, second = joint_order
    return f"""
<mujoco model="release-name-identity-mapping">
  <compiler angle="radian"/>
  <worldbody>
    <body name="root">
      <joint name="{first}" type="hinge" axis="1 0 0"/>
      <joint name="{second}" type="hinge" axis="0 1 0"/>
      <geom name="g" type="sphere" size="0.1" mass="1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _run(
    command: str, *arguments: str, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    """Run one installed Metrifid command without relying on the checkout as working directory."""
    return subprocess.run(
        [sys.executable, "-m", "metrifid.cli", command, *arguments],
        cwd=cwd,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=300,
    )


def _write_pair(root: Path, baseline_xml: str, candidate_xml: str) -> tuple[Path, Path]:
    """Write two isolated model roots so source-closure identities remain role-local."""
    baseline_root = root / "baseline"
    candidate_root = root / "candidate"
    baseline_root.mkdir(parents=True)
    candidate_root.mkdir(parents=True)
    baseline = baseline_root / "model.xml"
    candidate = candidate_root / "model.xml"
    baseline.write_text(baseline_xml, encoding="utf-8")
    candidate.write_text(candidate_xml, encoding="utf-8")
    return baseline, candidate


def _json_object(value: object) -> dict[str, Any]:
    """Narrow one decoded test value to a JSON object."""
    assert type(value) is dict
    return value


def _json_array(value: object) -> list[Any]:
    """Narrow one decoded test value to a JSON array."""
    assert type(value) is list
    return value


def _read_json(path: Path) -> dict[str, Any]:
    """Read one emitted JSON object."""
    return _json_object(json.loads(path.read_text(encoding="utf-8")))


def _failure(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    """Decode the final structured operational-failure line."""
    return _json_object(json.loads(result.stderr.strip().splitlines()[-1]))


def _certify_pair(
    root: Path,
    baseline_xml: str,
    candidate_xml: str,
) -> _ModelPair:
    """Learn both exact complete-MJB subjects through the installed public Certify command."""
    baseline, candidate = _write_pair(root, baseline_xml, candidate_xml)
    output = root / "certify-output"
    completed = _run("certify", str(baseline), str(candidate), "--output", str(output), cwd=root)
    assert completed.returncode in {0, 40}, completed.stdout + completed.stderr
    receipt = _read_json(output / "certification.json")
    baseline_receipt = _json_object(receipt["baseline"])
    candidate_receipt = _json_object(receipt["candidate"])
    baseline_artifact = _json_object(baseline_receipt["compiled_artifact"])
    candidate_artifact = _json_object(candidate_receipt["compiled_artifact"])
    return _ModelPair(
        baseline,
        candidate,
        str(baseline_artifact["mjb_sha256"]),
        str(candidate_artifact["mjb_sha256"]),
    )


def _write_policy(
    path: Path,
    pair: _ModelPair,
    rules: Sequence[Mapping[str, object]],
    *,
    candidate_sha256: str | None,
) -> Path:
    """Write one exact strict policy bound to the Certify-learned baseline subject."""
    policy = {
        "schema": "metrifid.model_release_policy",
        "schema_version": 1,
        "baseline_compiled_sha256": pair.baseline_mjb_sha256,
        "candidate_compiled_sha256": candidate_sha256,
        "rules": [dict(rule) for rule in rules],
    }
    path.write_text(
        json.dumps(policy, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path


def _run_review(
    pair: _ModelPair,
    policy: Path,
    output: Path,
    *,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    """Run the installed review-model command against one exact model pair."""
    return _run(
        "review-model",
        str(pair.baseline),
        str(pair.candidate),
        "--policy",
        str(policy),
        "--output",
        str(output),
        cwd=cwd,
    )


def _completed_receipt(
    result: subprocess.CompletedProcess[str],
    output: Path,
    *,
    status: str,
    exit_code: int,
) -> dict[str, Any]:
    """Require one completed status, exact summary linkage, and the two owned files only."""
    assert result.returncode == exit_code, result.stdout + result.stderr
    assert result.stderr == ""
    summary = _json_object(json.loads(result.stdout))
    assert summary["status"] == status
    assert summary["model_release_json"] == "model_release.json"
    assert summary["model_release_markdown"] == "model_release.md"
    assert str(output) not in result.stdout
    assert sorted(item.name for item in output.iterdir()) == sorted(_MODEL_RELEASE_FILENAMES)
    receipt = _read_json(output / "model_release.json")
    assert receipt["status"] == status
    assert receipt["completed_exit_code"] == exit_code
    assert receipt["receipt_sha256"] == summary["receipt_sha256"]
    assert receipt["changes_complete"] is True
    assert receipt["change_count"] == len(_json_array(receipt["changes"]))
    assert receipt["dynamic_behavior_claim"] == "NO_DYNAMIC_BEHAVIOR_CLAIM"
    return receipt


def _assert_no_completed_files(output: Path) -> None:
    """Require a refusal to leave neither completed artifact, whether the directory exists or not."""
    for name in _MODEL_RELEASE_FILENAMES:
        assert not (output / name).exists()


def _change_selector(change: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact selector object from one complete change row."""
    return _json_object(change["selector"])


def _change_key(change: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Return one selector's complete policy match key."""
    selector = _change_selector(change)
    return (
        str(selector["object_type"]),
        str(selector["object_name"]),
        str(selector["field"]),
        str(selector["change_kind"]),
    )


def _rules_for_changes(
    changes: Sequence[Mapping[str, Any]],
    *,
    effects: Mapping[tuple[str, str, str, str], str] | None = None,
) -> list[dict[str, object]]:
    """Turn every discovered non-opaque row into one exact candidate-bound rule."""
    selected_effects = effects or {}
    rules: list[dict[str, object]] = []
    for index, change in enumerate(changes):
        key = _change_key(change)
        rules.append(
            {
                "id": f"derived-{index:04d}",
                "effect": selected_effects.get(key, "ALLOW"),
                "selector": dict(_change_selector(change)),
                "before_sha256": change["before_sha256"],
                "after_sha256": change["after_sha256"],
            }
        )
    return rules


def _discovery(root: Path, baseline_xml: str, candidate_xml: str) -> _Discovery:
    """Run one null-candidate policy and retain every non-opaque change for exact derivation."""
    pair = _certify_pair(root, baseline_xml, candidate_xml)
    assert pair.baseline_mjb_sha256 != pair.candidate_mjb_sha256
    policy = _write_policy(
        root / "discovery-policy.json",
        pair,
        (),
        candidate_sha256=None,
    )
    output = root / "discovery-output"
    receipt = _completed_receipt(
        _run_review(pair, policy, output, cwd=root),
        output,
        status="REVIEW_REQUIRED",
        exit_code=40,
    )
    raw_changes = tuple(_json_object(item) for item in _json_array(receipt["changes"]))
    opaque = tuple(
        change for change in raw_changes if _change_selector(change)["object_type"] == "opaque"
    )
    assert len(opaque) == 1
    assert "candidate_compiled_subject_unbound" in _json_array(
        _json_object(opaque[0]["details"])["reasons"]
    )
    changes = tuple(change for change in raw_changes if change not in opaque)
    assert changes
    return _Discovery(pair, receipt, changes)


def _find_change(
    changes: Sequence[Mapping[str, Any]],
    key: tuple[str, str, str, str],
) -> dict[str, Any]:
    """Return the unique complete change row with one selector key."""
    matches = [dict(change) for change in changes if _change_key(change) == key]
    assert len(matches) == 1, key
    return matches[0]


def test_different_source_text_with_identical_compilation_is_no_change(tmp_path: Path) -> None:
    """Return NO_COMPILED_CHANGE for distinct closures that serialize identically."""
    baseline_xml = _slider_model("1")
    candidate_xml = baseline_xml.replace(
        "<worldbody>", "<!-- source-only comment -->\n  <worldbody>"
    )
    pair = _certify_pair(tmp_path, baseline_xml, candidate_xml)
    assert pair.baseline_mjb_sha256 == pair.candidate_mjb_sha256
    policy = _write_policy(
        tmp_path / "policy.json",
        pair,
        (),
        candidate_sha256=pair.candidate_mjb_sha256,
    )
    output = tmp_path / "review-output"
    receipt = _completed_receipt(
        _run_review(pair, policy, output, cwd=tmp_path),
        output,
        status="NO_COMPILED_CHANGE",
        exit_code=0,
    )
    assert receipt["changes"] == []
    certification = _json_object(receipt["certification_receipt"])
    baseline = _json_object(certification["baseline"])
    candidate = _json_object(certification["candidate"])
    assert baseline["source_closure_sha256"] != candidate["source_closure_sha256"]
    for name in _MODEL_RELEASE_FILENAMES:
        text = (output / name).read_text(encoding="utf-8")
        assert str(tmp_path) not in text
        assert "metrifid-model-release-" not in text
        assert "/private/" not in text
        assert "/tmp/" not in text


def test_mass_change_declares_body_and_complete_derived_closure_and_sdk_negative_completes(
    tmp_path: Path,
) -> None:
    """Discover all mass propagation, allow it exactly, and keep SDK negatives completed."""
    discovery = _discovery(tmp_path, _slider_model("1"), _slider_model("2"))
    keys = {_change_key(change) for change in discovery.changes}
    assert ("body", "payload_body", "mass", "MODIFY") in keys
    public_names = {key[1] for key in keys if key[0] == "compiled_field" and key[3] == "MODIFY"}
    assert {"body_mass", "body_inertia", "body_subtreemass"} <= public_names

    empty_policy = _write_policy(
        tmp_path / "sdk-negative-policy.json",
        discovery.pair,
        (),
        candidate_sha256=discovery.pair.candidate_mjb_sha256,
    )
    from metrifid.model_release import ModelReleaseStatus, review_model_release

    sdk_result = review_model_release(
        str(discovery.pair.baseline),
        str(discovery.pair.candidate),
        str(empty_policy),
        str(tmp_path / "sdk-negative-output"),
    )
    assert sdk_result.status is ModelReleaseStatus.REVIEW_REQUIRED
    assert sdk_result.receipt["completed_exit_code"] == 40
    assert sdk_result.model_release_json.is_file()
    assert sdk_result.model_release_markdown.is_file()

    allow_policy = _write_policy(
        tmp_path / "allow-all-policy.json",
        discovery.pair,
        _rules_for_changes(discovery.changes),
        candidate_sha256=discovery.pair.candidate_mjb_sha256,
    )
    output = tmp_path / "allow-all-output"
    receipt = _completed_receipt(
        _run_review(discovery.pair, allow_policy, output, cwd=tmp_path),
        output,
        status="WITHIN_DECLARED_POLICY",
        exit_code=0,
    )
    changes = tuple(_json_object(item) for item in _json_array(receipt["changes"]))
    assert len(changes) == len(discovery.changes)
    assert all(change["classification"] == "ALLOWED" for change in changes)
    assert all(_change_selector(change)["object_type"] != "opaque" for change in changes)


def test_forbidden_joint_range_is_outside_with_the_stable_first_witness(tmp_path: Path) -> None:
    """Classify the semantic joint-range row as forbidden ahead of derived field rows."""
    discovery = _discovery(tmp_path, _hinge_model("-0.5 0.5"), _hinge_model("-1 1"))
    target = ("joint", "arm_hinge", "range", "MODIFY")
    _find_change(discovery.changes, target)
    rules = _rules_for_changes(discovery.changes, effects={target: "FORBID"})
    policy = _write_policy(
        tmp_path / "forbid-policy.json",
        discovery.pair,
        rules,
        candidate_sha256=discovery.pair.candidate_mjb_sha256,
    )
    output = tmp_path / "forbid-output"
    receipt = _completed_receipt(
        _run_review(discovery.pair, policy, output, cwd=tmp_path),
        output,
        status="OUTSIDE_DECLARED_POLICY",
        exit_code=40,
    )
    witness = _json_object(receipt["first_unexpected_witness"])
    assert _change_key(witness) == target
    assert witness["classification"] == "FORBIDDEN"
    assert _json_object(receipt["classification_counts"])["FORBIDDEN"] == 1


def test_required_change_present_and_missing_follow_outside_precedence(tmp_path: Path) -> None:
    """Complete an exact REQUIRE once and retain a separate missing-required witness."""
    discovery = _discovery(tmp_path, _slider_model("1"), _slider_model("2"))
    target = ("body", "payload_body", "mass", "MODIFY")
    target_change = _find_change(discovery.changes, target)
    assert target_change["before_sha256"] is not None
    assert target_change["after_sha256"] is not None

    present_rules = _rules_for_changes(discovery.changes, effects={target: "REQUIRE"})
    present_policy = _write_policy(
        tmp_path / "require-present.json",
        discovery.pair,
        present_rules,
        candidate_sha256=discovery.pair.candidate_mjb_sha256,
    )
    present_output = tmp_path / "require-present-output"
    present = _completed_receipt(
        _run_review(discovery.pair, present_policy, present_output, cwd=tmp_path),
        present_output,
        status="WITHIN_DECLARED_POLICY",
        exit_code=0,
    )
    required = _find_change(
        tuple(_json_object(item) for item in _json_array(present["changes"])),
        target,
    )
    assert required["classification"] == "REQUIRED"
    assert required["rule_id"] in _json_array(present["satisfied_required_rule_ids"])
    assert present["missing_required_rules"] == []

    missing_rule = {
        "id": "required-parent-change",
        "effect": "REQUIRE",
        "selector": {
            "object_type": "body",
            "object_name": "payload_body",
            "field": "parent",
            "change_kind": "MODIFY",
        },
        "before_sha256": "a" * 64,
        "after_sha256": "b" * 64,
    }
    missing_policy = _write_policy(
        tmp_path / "require-missing.json",
        discovery.pair,
        [*_rules_for_changes(discovery.changes), missing_rule],
        candidate_sha256=discovery.pair.candidate_mjb_sha256,
    )
    missing_output = tmp_path / "require-missing-output"
    missing = _completed_receipt(
        _run_review(discovery.pair, missing_policy, missing_output, cwd=tmp_path),
        missing_output,
        status="OUTSIDE_DECLARED_POLICY",
        exit_code=40,
    )
    assert missing["first_unexpected_witness"] is None
    first_missing = _json_object(missing["first_missing_required_witness"])
    assert first_missing["id"] == "required-parent-change"
    assert len(_json_array(missing["changes"])) == len(discovery.changes)


def test_one_allowed_primary_omission_keeps_complete_rows_and_requires_review(
    tmp_path: Path,
) -> None:
    """Keep allowed derived rows while one deliberately undeclared primary row forces review."""
    discovery = _discovery(tmp_path, _slider_model("1"), _slider_model("2"))
    target = ("body", "payload_body", "mass", "MODIFY")
    rules = [
        rule
        for rule in _rules_for_changes(discovery.changes)
        if _change_key(_json_object(rule)) != target
    ]
    policy = _write_policy(
        tmp_path / "one-undeclared-policy.json",
        discovery.pair,
        rules,
        candidate_sha256=discovery.pair.candidate_mjb_sha256,
    )
    output = tmp_path / "one-undeclared-output"
    receipt = _completed_receipt(
        _run_review(discovery.pair, policy, output, cwd=tmp_path),
        output,
        status="REVIEW_REQUIRED",
        exit_code=40,
    )
    changes = tuple(_json_object(item) for item in _json_array(receipt["changes"]))
    assert len(changes) == len(discovery.changes)
    assert _find_change(changes, target)["classification"] == "UNDECLARED"
    assert any(change["classification"] == "ALLOWED" for change in changes)
    assert _change_key(_json_object(receipt["first_unexpected_witness"])) == target


def test_broken_include_unknown_selector_and_overlap_refuse_without_completed_files(
    tmp_path: Path,
) -> None:
    """Deliver all three invalid inputs through exit 64 with no completed result pair."""
    baseline_xml = _slider_model("1")
    pair = _certify_pair(tmp_path / "subjects", baseline_xml, baseline_xml)

    broken_root = tmp_path / "broken-candidate"
    broken_root.mkdir()
    broken = broken_root / "model.xml"
    broken.write_text(
        '<mujoco model="broken"><include file="missing.xml"/></mujoco>\n',
        encoding="utf-8",
    )
    broken_pair = _ModelPair(
        pair.baseline,
        broken,
        pair.baseline_mjb_sha256,
        pair.candidate_mjb_sha256,
    )
    discovery_policy = _write_policy(
        tmp_path / "broken-policy.json",
        broken_pair,
        (),
        candidate_sha256=None,
    )
    cases: list[tuple[_ModelPair, Path, Path]] = [
        (broken_pair, discovery_policy, tmp_path / "broken-output")
    ]

    unknown = {
        "id": "unknown-selector",
        "effect": "ALLOW",
        "selector": {
            "object_type": "site",
            "object_name": "marker",
            "field": "position",
            "change_kind": "MODIFY",
        },
        "before_sha256": None,
        "after_sha256": None,
    }
    unknown_policy = _write_policy(
        tmp_path / "unknown-policy.json",
        pair,
        (unknown,),
        candidate_sha256=pair.candidate_mjb_sha256,
    )
    cases.append((pair, unknown_policy, tmp_path / "unknown-output"))

    wildcard = {
        "id": "wildcard-mass",
        "effect": "ALLOW",
        "selector": {
            "object_type": "body",
            "object_name": "*",
            "field": "mass",
            "change_kind": "MODIFY",
        },
        "before_sha256": None,
        "after_sha256": None,
    }
    exact = {
        "id": "exact-mass",
        "effect": "FORBID",
        "selector": {
            "object_type": "body",
            "object_name": "payload_body",
            "field": "mass",
            "change_kind": "MODIFY",
        },
        "before_sha256": None,
        "after_sha256": None,
    }
    overlap_policy = _write_policy(
        tmp_path / "overlap-policy.json",
        pair,
        (wildcard, exact),
        candidate_sha256=pair.candidate_mjb_sha256,
    )
    cases.append((pair, overlap_policy, tmp_path / "overlap-output"))

    for selected_pair, policy, output in cases:
        result = _run_review(selected_pair, policy, output, cwd=tmp_path)
        assert result.returncode == 64, result.stdout + result.stderr
        assert result.stdout == ""
        failure = _failure(result)
        assert failure["operation"] == "review-model"
        _assert_no_completed_files(output)


def test_repeated_tied_changes_have_byte_identical_output_and_canonical_first_witness(
    tmp_path: Path,
) -> None:
    """Ignore source/rule order and output location when ordering two undeclared body changes."""
    discovery = _discovery(
        tmp_path,
        _two_body_model("1", "1"),
        _two_body_model("2", "3"),
    )
    policy = _write_policy(
        tmp_path / "undeclared-policy.json",
        discovery.pair,
        (),
        candidate_sha256=discovery.pair.candidate_mjb_sha256,
    )
    receipts: list[dict[str, Any]] = []
    for name in ("first-output", "second-output"):
        output = tmp_path / name
        receipts.append(
            _completed_receipt(
                _run_review(discovery.pair, policy, output, cwd=tmp_path),
                output,
                status="REVIEW_REQUIRED",
                exit_code=40,
            )
        )
    for filename in _MODEL_RELEASE_FILENAMES:
        assert (tmp_path / "first-output" / filename).read_bytes() == (
            tmp_path / "second-output" / filename
        ).read_bytes()
        text = (tmp_path / "first-output" / filename).read_text(encoding="utf-8")
        assert str(tmp_path) not in text
        assert "metrifid-model-release-" not in text
        assert "/private/" not in text
        assert "/tmp/" not in text

    changes = tuple(_json_object(item) for item in _json_array(receipts[0]["changes"]))
    semantic_keys = [
        _change_key(change) for change in changes if change["source"] == "SEMANTIC_OBJECT"
    ]
    assert semantic_keys[:4] == [
        ("body", "alpha_body", "mass", "MODIFY"),
        ("body", "alpha_body", "inertia", "MODIFY"),
        ("body", "zeta_body", "mass", "MODIFY"),
        ("body", "zeta_body", "inertia", "MODIFY"),
    ]
    first = _json_object(receipts[0]["first_unexpected_witness"])
    assert _change_key(first) == ("body", "alpha_body", "mass", "MODIFY")
    assert receipts[0]["change_count"] == len(discovery.changes)


def test_static_all_allowed_can_still_have_directly_observed_dynamic_divergence(
    tmp_path: Path,
) -> None:
    """Keep a static green distinct from a test-side MuJoCo trajectory difference."""
    discovery = _discovery(tmp_path, _slider_model("1"), _slider_model("2"))
    policy = _write_policy(
        tmp_path / "static-allow-policy.json",
        discovery.pair,
        _rules_for_changes(discovery.changes),
        candidate_sha256=discovery.pair.candidate_mjb_sha256,
    )
    output = tmp_path / "static-output"
    receipt = _completed_receipt(
        _run_review(discovery.pair, policy, output, cwd=tmp_path),
        output,
        status="WITHIN_DECLARED_POLICY",
        exit_code=0,
    )
    assert receipt["dynamic_behavior_claim"] == "NO_DYNAMIC_BEHAVIOR_CLAIM"
    limitation_codes = {_json_object(item)["code"] for item in _json_array(receipt["limitations"])}
    assert "STATIC_ONLY_NO_DYNAMIC_EQUIVALENCE" in limitation_codes

    import mujoco

    def final_position(path: Path) -> float:
        """Step one fixed control sequence and return its final slide coordinate."""
        model = mujoco.MjModel.from_xml_path(str(path))
        data = mujoco.MjData(model)
        data.ctrl[0] = 1.0
        for _ in range(100):
            mujoco.mj_step(model, data)
        return float(data.qpos[0])

    baseline_position = final_position(discovery.pair.baseline)
    candidate_position = final_position(discovery.pair.candidate)
    assert abs(baseline_position - candidate_position) > 1e-4


def test_physical_mesh_replacement_is_exposed_and_can_be_exactly_allowed(tmp_path: Path) -> None:
    """Bind a physical mesh edit to compiled geometry rather than source path alone."""
    discovery = _discovery(tmp_path, _mesh_model("1"), _mesh_model("1.25"))
    target = ("mesh", "payload_mesh", "compiled_geometry_sha256", "MODIFY")
    mesh_change = _find_change(discovery.changes, target)
    assert mesh_change["before_sha256"] != mesh_change["after_sha256"]
    policy = _write_policy(
        tmp_path / "mesh-allow-policy.json",
        discovery.pair,
        _rules_for_changes(discovery.changes),
        candidate_sha256=discovery.pair.candidate_mjb_sha256,
    )
    output = tmp_path / "mesh-output"
    receipt = _completed_receipt(
        _run_review(discovery.pair, policy, output, cwd=tmp_path),
        output,
        status="WITHIN_DECLARED_POLICY",
        exit_code=0,
    )
    changes = tuple(_json_object(item) for item in _json_array(receipt["changes"]))
    assert _find_change(changes, target)["classification"] == "ALLOWED"
    assert all(_change_selector(change)["object_type"] != "opaque" for change in changes)


def test_literal_star_in_mujoco_name_fails_closed_as_opaque_review(tmp_path: Path) -> None:
    """Keep a valid MuJoCo name outside wildcard policy syntax without an internal failure."""
    baseline = _slider_model("1").replace("payload_body", "payload*body")
    candidate = _slider_model("2").replace("payload_body", "payload*body")
    discovery = _discovery(tmp_path, baseline, candidate)
    opaque = [
        _json_object(item)
        for item in _json_array(discovery.receipt["changes"])
        if _change_selector(_json_object(item))["object_type"] == "opaque"
    ]
    assert len(opaque) == 1
    reasons = _json_array(_json_object(opaque[0]["details"])["reasons"])
    assert "semantic_name_coverage_incomplete" in reasons


def test_output_path_replaced_during_private_cleanup_never_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recheck the public pathname after private MJB cleanup and preserve attacker bytes."""
    model = _slider_model("1")
    pair = _certify_pair(tmp_path / "subjects", model, model)
    policy = _write_policy(
        tmp_path / "policy.json",
        pair,
        (),
        candidate_sha256=pair.candidate_mjb_sha256,
    )
    output = tmp_path / "review-output"
    displaced = tmp_path / "displaced-output"

    from metrifid.model_release import ModelReleaseOperationError, review_model_release
    from metrifid.model_release import _run as release_run

    original_cleanup = release_run._remove_private_scratch

    def replace_output_after_cleanup(scratch: Path) -> None:
        original_cleanup(scratch)
        output.rename(displaced)
        output.mkdir()
        (output / "model_release.json").write_text("attacker\n", encoding="utf-8")

    monkeypatch.setattr(release_run, "_remove_private_scratch", replace_output_after_cleanup)
    with pytest.raises(ModelReleaseOperationError) as captured:
        review_model_release(
            str(pair.baseline),
            str(pair.candidate),
            str(policy),
            str(output),
        )
    assert captured.value.failure.exit_code == 64
    assert captured.value.failure.reason.code.value == "OUTPUT_PATH_INVALID"
    assert (output / "model_release.json").read_text(encoding="utf-8") == "attacker\n"
    assert sorted(item.name for item in displaced.iterdir()) == sorted(_MODEL_RELEASE_FILENAMES)


def test_actuator_target_identity_uses_typed_names_not_reorderable_numeric_ids(
    tmp_path: Path,
) -> None:
    """Expose a named A-to-B retarget even when both compiled transmission IDs equal zero."""
    baseline = _retargeted_actuator_model(("A", "B"), "A")
    candidate = _retargeted_actuator_model(("B", "A"), "B")
    discovery = _discovery(tmp_path, baseline, candidate)
    target = ("actuator", "m", "targets", "MODIFY")
    target_change = _find_change(discovery.changes, target)
    before = _json_object(target_change["before_value"])
    after = _json_object(target_change["after_value"])
    assert _json_array(before["references"])[0] == {"object_type": "JOINT", "name": "A"}
    assert _json_array(after["references"])[0] == {"object_type": "JOINT", "name": "B"}

    incomplete_rules = [
        rule
        for rule in _rules_for_changes(discovery.changes)
        if _change_key(_json_object(rule)) != target
    ]
    incomplete_policy = _write_policy(
        tmp_path / "incomplete-policy.json",
        discovery.pair,
        incomplete_rules,
        candidate_sha256=discovery.pair.candidate_mjb_sha256,
    )
    incomplete_output = tmp_path / "incomplete-output"
    incomplete = _completed_receipt(
        _run_review(discovery.pair, incomplete_policy, incomplete_output, cwd=tmp_path),
        incomplete_output,
        status="REVIEW_REQUIRED",
        exit_code=40,
    )
    assert (
        _find_change(
            tuple(_json_object(item) for item in _json_array(incomplete["changes"])), target
        )["classification"]
        == "UNDECLARED"
    )

    complete_policy = _write_policy(
        tmp_path / "complete-policy.json",
        discovery.pair,
        _rules_for_changes(discovery.changes),
        candidate_sha256=discovery.pair.candidate_mjb_sha256,
    )
    complete_output = tmp_path / "complete-output"
    complete = _completed_receipt(
        _run_review(discovery.pair, complete_policy, complete_output, cwd=tmp_path),
        complete_output,
        status="REVIEW_REQUIRED",
        exit_code=40,
    )
    complete_opaque = [
        _json_object(item)
        for item in _json_array(complete["changes"])
        if _change_selector(_json_object(item))["object_type"] == "opaque"
    ]
    assert _json_array(_json_object(complete_opaque[0]["details"])["reasons"]) == [
        "compiled_name_identity_mapping_changed"
    ]

    stable_ids = _discovery(
        tmp_path / "stable-ids",
        _retargeted_actuator_model(("A", "B"), "A"),
        _retargeted_actuator_model(("A", "B"), "B"),
    )
    stable_policy = _write_policy(
        tmp_path / "stable-ids-policy.json",
        stable_ids.pair,
        _rules_for_changes(stable_ids.changes),
        candidate_sha256=stable_ids.pair.candidate_mjb_sha256,
    )
    _completed_receipt(
        _run_review(
            stable_ids.pair,
            stable_policy,
            tmp_path / "stable-ids-output",
            cwd=tmp_path,
        ),
        tmp_path / "stable-ids-output",
        status="WITHIN_DECLARED_POLICY",
        exit_code=0,
    )


def test_changed_name_id_mapping_forces_review_for_unprojected_axis_swap(tmp_path: Path) -> None:
    """Never allow stable named semantics to change behind unchanged per-ID public arrays."""
    discovery = _discovery(
        tmp_path,
        _reordered_joint_axes_model(("A", "B")),
        _reordered_joint_axes_model(("B", "A")),
    )
    policy = _write_policy(
        tmp_path / "bound-policy.json",
        discovery.pair,
        _rules_for_changes(discovery.changes),
        candidate_sha256=discovery.pair.candidate_mjb_sha256,
    )
    output = tmp_path / "bound-output"
    receipt = _completed_receipt(
        _run_review(discovery.pair, policy, output, cwd=tmp_path),
        output,
        status="REVIEW_REQUIRED",
        exit_code=40,
    )
    opaque = [
        _json_object(item)
        for item in _json_array(receipt["changes"])
        if _change_selector(_json_object(item))["object_type"] == "opaque"
    ]
    assert len(opaque) == 1
    reasons = _json_array(_json_object(opaque[0]["details"])["reasons"])
    assert reasons == ["compiled_name_identity_mapping_changed"]
