"""Installed-CLI closure for model-release assertions not directly pinned elsewhere.

Each case asserts one outcome at its own status and exit code rather than leaving it implied by
the rest of the suite. Every case drives the installed console entrypoint in a separate process,
so the status, the exit code and the published receipt are all observed the way a user observes
them.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest


def _run(command: str, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run one installed Metrifid command in its own process."""
    return subprocess.run(
        [sys.executable, "-m", "metrifid.cli", command, *arguments],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )


def _json_object(value: object) -> dict[str, Any]:
    """Narrow one decoded value to a JSON object."""
    assert type(value) is dict
    return value


def _slider(mass: str, *, comment: str = "") -> str:
    """Return one fully named single-body model."""
    return f"""
<mujoco model="oracle-closure">{comment}
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


def _write_pair(root: Path, baseline_xml: str, candidate_xml: str) -> tuple[Path, Path]:
    """Write two isolated model roots so each source closure stays role-local."""
    baseline_root = root / "baseline"
    candidate_root = root / "candidate"
    baseline_root.mkdir(parents=True)
    candidate_root.mkdir(parents=True)
    baseline = baseline_root / "model.xml"
    candidate = candidate_root / "model.xml"
    baseline.write_text(baseline_xml, encoding="utf-8")
    candidate.write_text(candidate_xml, encoding="utf-8")
    return baseline, candidate


def _subjects(root: Path, baseline: Path, candidate: Path) -> tuple[str, str]:
    """Learn both exact complete-MJB subjects through the installed Certify command."""
    output = root / "certify-output"
    completed = _run("certify", str(baseline), str(candidate), "--output", str(output))
    assert completed.returncode in {0, 40}, completed.stdout + completed.stderr
    receipt = _json_object(json.loads((output / "certification.json").read_text(encoding="utf-8")))
    return (
        str(_json_object(_json_object(receipt["baseline"])["compiled_artifact"])["mjb_sha256"]),
        str(_json_object(_json_object(receipt["candidate"])["compiled_artifact"])["mjb_sha256"]),
    )


def _policy(
    path: Path,
    subjects: tuple[str, str],
    rules: Sequence[Mapping[str, object]],
    *,
    bind_candidate: bool = True,
) -> Path:
    """Write one policy bound to the exact learned subjects."""
    path.write_text(
        json.dumps(
            {
                "schema": "metrifid.model_release_policy",
                "schema_version": 1,
                "baseline_compiled_sha256": subjects[0],
                "candidate_compiled_sha256": subjects[1] if bind_candidate else None,
                "rules": [dict(rule) for rule in rules],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _review(baseline: Path, candidate: Path, policy: Path, output: Path) -> tuple[int, Path, str]:
    """Run the installed review and return its exit code, output directory and stderr."""
    completed = _run(
        "review-model",
        str(baseline),
        str(candidate),
        "--policy",
        str(policy),
        "--output",
        str(output),
    )
    return completed.returncode, output, completed.stderr


def _receipt(output: Path) -> dict[str, Any]:
    """Read the published Model Change Gate receipt."""
    return _json_object(json.loads((output / "model_release.json").read_text(encoding="utf-8")))


def _refusal(stderr: str) -> dict[str, Any]:
    """Decode the final structured operational-failure line."""
    return _json_object(json.loads(stderr.strip().splitlines()[-1]))


def _selector(object_type: str, object_name: str, field: str, change_kind: str) -> dict[str, str]:
    """Build one exact policy selector."""
    return {
        "object_type": object_type,
        "object_name": object_name,
        "field": field,
        "change_kind": change_kind,
    }


# ---- Both first witnesses are null on a compiled-identical source change ------------------------


def test_no_compiled_change_publishes_both_first_witnesses_as_null(tmp_path: Path) -> None:
    """Pin BOTH_FIRST_WITNESSES_NULL on the installed no-compiled-change journey."""
    baseline, candidate = _write_pair(
        tmp_path, _slider("1.0"), _slider("1.0", comment="\n  <!-- source-only comment -->")
    )
    subjects = _subjects(tmp_path, baseline, candidate)
    assert subjects[0] == subjects[1], "the two sources must compile to one identical artifact"

    code, output, _ = _review(
        baseline, candidate, _policy(tmp_path / "policy.json", subjects, ()), tmp_path / "review"
    )
    assert code == 0
    receipt = _receipt(output)
    assert receipt["status"] == "NO_COMPILED_CHANGE"
    assert receipt["completed_exit_code"] == 0
    assert receipt["changes"] == []
    assert receipt["first_unexpected_witness"] is None
    assert receipt["first_missing_required_witness"] is None
    baseline_role = _json_object(_json_object(receipt["certification_receipt"])["baseline"])
    candidate_role = _json_object(_json_object(receipt["certification_receipt"])["candidate"])
    assert baseline_role["source_closure_sha256"] != candidate_role["source_closure_sha256"]


# ---- A missing REQUIRE outranks an otherwise unchanged compiled artifact ------------------------


def test_a_missing_required_change_outranks_no_compiled_change(tmp_path: Path) -> None:
    """Pin MISSING_REQUIRED_OUTRANKS_NO_COMPILED_CHANGE with an empty observed change list."""
    baseline, candidate = _write_pair(tmp_path, _slider("1.0"), _slider("1.0"))
    subjects = _subjects(tmp_path, baseline, candidate)
    assert subjects[0] == subjects[1], "this case requires an identical compiled artifact"

    required = {
        "id": "require-mass-change",
        "effect": "REQUIRE",
        "selector": _selector("body", "payload_body", "mass", "MODIFY"),
        "before_sha256": "a" * 64,
        "after_sha256": "b" * 64,
    }
    code, output, _ = _review(
        baseline,
        candidate,
        _policy(tmp_path / "policy.json", subjects, (required,)),
        tmp_path / "review",
    )
    assert code == 40
    receipt = _receipt(output)

    # The artifacts are identical, so nothing was observed; the unmet REQUIRE still decides.
    assert receipt["changes"] == []
    assert receipt["status"] == "OUTSIDE_DECLARED_POLICY"
    assert receipt["completed_exit_code"] == 40
    assert receipt["first_unexpected_witness"] is None
    missing = _json_object(receipt["first_missing_required_witness"])
    assert missing["id"] == "require-mass-change"


# ---- Duplicate ids, duplicate selectors and overlap all refuse at exit 64 -----------------------


def _duplicate_id_rules() -> tuple[dict[str, object], ...]:
    """Two rules that share one id and differ in selector."""
    return (
        {
            "id": "same-id",
            "effect": "ALLOW",
            "selector": _selector("body", "payload_body", "mass", "MODIFY"),
            "before_sha256": None,
            "after_sha256": None,
        },
        {
            "id": "same-id",
            "effect": "ALLOW",
            "selector": _selector("body", "payload_body", "inertia", "MODIFY"),
            "before_sha256": None,
            "after_sha256": None,
        },
    )


def _duplicate_selector_rules() -> tuple[dict[str, object], ...]:
    """Two rules that share one exact selector and differ in id."""
    selector = _selector("body", "payload_body", "mass", "MODIFY")
    return (
        {
            "id": "first",
            "effect": "ALLOW",
            "selector": dict(selector),
            "before_sha256": None,
            "after_sha256": None,
        },
        {
            "id": "second",
            "effect": "ALLOW",
            "selector": dict(selector),
            "before_sha256": None,
            "after_sha256": None,
        },
    )


def _overlapping_rules() -> tuple[dict[str, object], ...]:
    """One wildcard rule and one exact rule whose match spaces overlap."""
    return (
        {
            "id": "wildcard",
            "effect": "ALLOW",
            "selector": _selector("body", "*", "mass", "MODIFY"),
            "before_sha256": None,
            "after_sha256": None,
        },
        {
            "id": "exact",
            "effect": "ALLOW",
            "selector": _selector("body", "payload_body", "mass", "MODIFY"),
            "before_sha256": None,
            "after_sha256": None,
        },
    )


@pytest.mark.parametrize(
    ("label", "builder", "reversed_order"),
    [
        pytest.param("duplicate_rule_id", _duplicate_id_rules, False, id="duplicate_rule_id"),
        pytest.param("duplicate_selector", _duplicate_selector_rules, False, id="dup_selector"),
        pytest.param("overlap_declared", _overlapping_rules, False, id="overlap_declared_order"),
        pytest.param("overlap_reversed", _overlapping_rules, True, id="overlap_reversed_order"),
    ],
)
def test_ambiguous_or_duplicate_rules_refuse_before_any_decision(
    tmp_path: Path, label: str, builder: Any, reversed_order: bool
) -> None:
    """Pin every duplicate and overlap refusal at exit 64, including order independence."""
    baseline, candidate = _write_pair(tmp_path, _slider("1.0"), _slider("1.25"))
    subjects = _subjects(tmp_path, baseline, candidate)
    rules = list(builder())
    if reversed_order:
        rules.reverse()

    output = tmp_path / f"review-{label}"
    code, _, stderr = _review(
        baseline, candidate, _policy(tmp_path / f"{label}.json", subjects, rules), output
    )

    assert code == 64, stderr
    failure = _refusal(stderr)
    assert failure["exit_code"] == 64
    assert failure["operation"] == "review-model"
    assert _json_object(failure["reason"])["code"] == "CONFIGURATION_PARSE_FAILED"

    # A refusal publishes no completed pair, whole or partial.
    assert not (output / "model_release.json").exists()
    assert not (output / "model_release.md").exists()


# ---- Outside-policy precedence wins over a concurrent undeclared change -------------------------


def test_outside_policy_precedence_wins_over_a_concurrent_undeclared_change(
    tmp_path: Path,
) -> None:
    """Pin OUTSIDE_PRECEDENCE_WINS when an undeclared change and a missing REQUIRE coexist."""
    baseline, candidate = _write_pair(tmp_path, _slider("1.0"), _slider("1.25"))
    subjects = _subjects(tmp_path, baseline, candidate)

    # Nothing is allowed, so the real mass change is undeclared, which alone would be
    # REVIEW_REQUIRED. The unsatisfiable REQUIRE must outrank it.
    required = {
        "id": "require-absent-change",
        "effect": "REQUIRE",
        "selector": _selector("joint", "payload_slide", "range", "MODIFY"),
        "before_sha256": "c" * 64,
        "after_sha256": "d" * 64,
    }
    code, output, _ = _review(
        baseline,
        candidate,
        _policy(tmp_path / "policy.json", subjects, (required,)),
        tmp_path / "review",
    )
    assert code == 40
    receipt = _receipt(output)

    classifications = {str(_json_object(change)["classification"]) for change in receipt["changes"]}
    assert "UNDECLARED" in classifications, "the actual mass change must be undeclared"
    assert receipt["first_unexpected_witness"] is not None

    missing = _json_object(receipt["first_missing_required_witness"])
    assert missing["id"] == "require-absent-change"

    # Both conditions hold; outside-policy precedence decides the completed status.
    assert receipt["status"] == "OUTSIDE_DECLARED_POLICY"
    assert receipt["completed_exit_code"] == 40
