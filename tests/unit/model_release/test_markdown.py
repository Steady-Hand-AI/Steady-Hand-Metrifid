"""Unit tests for deterministic, complete model-release Markdown rendering."""

from __future__ import annotations

from typing import Any

import pytest

from metrifid.model_release import _markdown as release_markdown


def _change(
    name: str, classification: str = "ALLOWED", rule_id: str | None = "allow"
) -> dict[str, Any]:
    """Return one renderer-focused complete semantic change row."""
    return {
        "selector": {
            "object_type": "body",
            "object_name": name,
            "field": "mass",
            "change_kind": "MODIFY",
        },
        "source": "SEMANTIC_OBJECT",
        "classification": classification,
        "rule_id": rule_id,
        "before_sha256": "1" * 64,
        "after_sha256": "2" * 64,
        "before_value": "before",
        "after_value": "after",
        "details": {},
    }


def _rule(name: str, rule_id: str = "required") -> dict[str, Any]:
    """Return one renderer-focused REQUIRE rule primitive."""
    return {
        "id": rule_id,
        "effect": "REQUIRE",
        "selector": {
            "object_type": "body",
            "object_name": name,
            "field": "mass",
            "change_kind": "MODIFY",
        },
        "before_sha256": "3" * 64,
        "after_sha256": "4" * 64,
    }


def _receipt() -> dict[str, Any]:
    """Return all evidence consumed by the isolated Markdown renderer."""
    hostile = "row|break\n`tick`<script>alert(1)</script>\\tail"
    changes = [_change("alpha"), _change("beta"), _change(hostile)]
    missing = _rule(hostile, "must|change\n`id`<b>")
    return {
        "status": "OUTSIDE_DECLARED_POLICY",
        "completed_exit_code": 40,
        "receipt_sha256": "a" * 64,
        "decision_sha256": "b" * 64,
        "changes_complete": True,
        "change_count": len(changes),
        "dynamic_behavior_claim": "NO_DYNAMIC_BEHAVIOR_CLAIM",
        "static_claim": {
            "claim_kind": "STATIC_COMPILED_MODEL_CHANGE_POLICY_CLASSIFICATION",
            "statement": "This receipt classifies static compiled changes only.",
        },
        "certification_receipt_sha256": "c" * 64,
        "certification_decision_sha256": "d" * 64,
        "certification_receipt": {
            "baseline": {"compiled_artifact": {"mjb_sha256": "e" * 64}},
            "candidate": {"compiled_artifact": {"mjb_sha256": "f" * 64}},
        },
        "policy": {
            "raw_sha256": "0" * 64,
            "semantic_sha256": "1" * 64,
            "baseline_compiled_sha256": "e" * 64,
            "candidate_compiled_sha256": "f" * 64,
            "rule_count": 4,
        },
        "public_field_registry": {
            "schema": "metrifid.mujoco_public_field_registry",
            "schema_version": 1,
            "sha256": "2" * 64,
            "field_count": 675,
        },
        "classification_counts": {
            "ALLOWED": 3,
            "REQUIRED": 0,
            "FORBIDDEN": 0,
            "UNDECLARED": 0,
        },
        "changes": changes,
        "satisfied_required_rule_ids": ["satisfied|rule"],
        "missing_required_rules": [missing],
        "first_unexpected_witness": None,
        "first_missing_required_witness": missing,
        "limitations": [
            {
                "code": "STATIC_ONLY_NO_DYNAMIC_EQUIVALENCE",
                "statement": "No dynamic-equivalence claim is made.",
            },
            {
                "code": "NO_HARDWARE_OR_OPERATIONAL_SAFETY_CLAIM",
                "statement": "No safety claim is made.",
            },
        ],
    }


@pytest.fixture(autouse=True)
def _isolate_renderer_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests focused on rendering; receipt semantics are contract-tested separately."""
    monkeypatch.setattr(release_markdown, "validate_model_release_receipt", lambda _receipt: None)


def test_markdown_is_deterministic_and_prominently_static_only() -> None:
    """Repeat rendering byte-for-byte and state the dynamic and safety non-claims up front."""
    receipt = _receipt()
    first = release_markdown.render_markdown(receipt)
    second = release_markdown.render_markdown(receipt)
    assert first == second
    assert "**STATIC-ONLY DECISION.**" in first
    assert "`NO_DYNAMIC_BEHAVIOR_CLAIM`" in first
    assert "does not establish dynamic equivalence" in first
    assert "hardware safety" in first
    assert first.endswith("\n")


def test_markdown_lists_every_change_without_truncation() -> None:
    """Emit one table row for every receipt change and an exact completeness statement."""
    text = release_markdown.render_markdown(_receipt())
    assert "All `3` of `3` changes are listed" in text
    assert text.count("| SEMANTIC_OBJECT |") == 3
    assert "alpha" in text
    assert "beta" in text
    assert "no truncation" in text.lower()


def test_markdown_escapes_hostile_model_and_rule_names() -> None:
    """Prevent names from forging rows, code spans, HTML, or additional Markdown lines."""
    text = release_markdown.render_markdown(_receipt())
    assert (
        "row\\|break\\n&#96;tick&#96;&lt;script&gt;alert&#40;1&#41;&lt;/script&gt;\\\\tail" in text
    )
    assert "must\\|change\\n&#96;id&#96;&lt;b&gt;" in text
    assert "satisfied\\|rule" in text
    assert "<script>" not in text
    assert "\n`tick`" not in text


def test_markdown_neutralizes_user_controlled_links_and_remote_images() -> None:
    """Prevent a rendered receipt from fetching or presenting an attacker-controlled link."""
    receipt = _receipt()
    attack = "![review](https://attacker.example/pixel)"
    receipt["changes"][0]["selector"]["object_name"] = attack
    receipt["changes"][0]["rule_id"] = attack
    receipt["satisfied_required_rule_ids"] = [attack]
    text = release_markdown.render_markdown(receipt)
    assert attack not in text
    assert "&#33;&#91;review&#93;&#40;https://attacker.example/pixel&#41;" in text
    assert "![" not in text
    assert "](https://" not in text


def test_markdown_renders_empty_change_array_explicitly() -> None:
    """An empty complete array is reported as no compiled change, not silently omitted."""
    receipt = _receipt()
    receipt["changes"] = []
    receipt["change_count"] = 0
    receipt["classification_counts"]["ALLOWED"] = 0
    text = release_markdown.render_markdown(receipt)
    assert "All `0` of `0` changes are listed" in text
    assert "No compiled change was observed." in text
