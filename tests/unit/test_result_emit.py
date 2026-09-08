"""Bounded JSON emission and readable text rendering for one result document."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final

import pytest

from metrifid.json_values import CanonicalValue, canonical_json_bytes, strict_json_loads
from metrifid.result import (
    DEFAULT_JSON_MAX_BYTES,
    build_document,
    build_failure_document,
    encode_result,
    render_text,
)

_RECEIPT_ROOT: Final = Path(__file__).parents[1] / "fixtures" / "result"
_CHANGED_RECEIPT: Final = _RECEIPT_ROOT / "changed_model_release.json"
_EQUAL_RECEIPT: Final = _RECEIPT_ROOT / "equal_model_release.json"
_CERTIFICATION_RECEIPT: Final = _RECEIPT_ROOT / "changed_certification.json"

_RESULT_MEMBERS: Final = frozenset(
    {
        "schema",
        "schema_version",
        "command",
        "observation",
        "compiled_comparison",
        "evaluation",
        "runtime",
        "inputs",
        "findings",
        "finding_count",
        "limitations",
        "artifacts",
        "truncated",
        "problems",
    }
)
_BOUNDARY_SENTENCE: Final = "Not a statement about behaviour, safety, or approval."
_AWKWARD_NAME: Final = 'modèle 🤖 "quoted" \\ back\nnewline'


def _load_receipt(path: Path) -> dict[str, CanonicalValue]:
    """Read one real receipt as strict canonical JSON data."""
    assert path.is_file(), f"missing real receipt fixture: {path}"
    value = strict_json_loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


def _document(
    receipt: dict[str, CanonicalValue],
    *,
    exit_code: int,
    artifacts: dict[str, CanonicalValue] | None = None,
) -> dict[str, CanonicalValue]:
    """Build one complete result document over an already-validated receipt."""
    return build_document(
        command="show",
        exit_code=exit_code,
        observation="recorded",
        receipt=receipt,
        policy_origin="not_recorded",
        reader_version="0.8.0",
        inputs={"baseline": None, "candidate": None, "roots_correspond": None},
        artifacts=artifacts
        if artifacts is not None
        else {
            "output_dir": None,
            "receipt": str(_CHANGED_RECEIPT),
            "receipt_sha256": receipt.get("receipt_sha256"),
            "markdown": None,
            "html": None,
        },
    )


def _changed_document() -> dict[str, CanonicalValue]:
    """Build the result document for the real differing model-release receipt."""
    return _document(_load_receipt(_CHANGED_RECEIPT), exit_code=40)


def _failure_document() -> dict[str, CanonicalValue]:
    """Build one small result document for a command that produced no receipt."""
    return build_failure_document(
        command="diff",
        exit_code=64,
        reason_code="CONFIGURATION_IO_FAILED",
        message="the receipt could not be read",
        reader_version="0.8.0",
        inputs={"baseline": None, "candidate": None, "roots_correspond": None},
        artifacts={"output_dir": None, "receipt": None, "markdown": None, "html": None},
        policy_origin="not_created",
    )


def _awkward_receipt() -> dict[str, CanonicalValue]:
    """Build a hand-made model-release receipt whose object name needs escaping."""
    return {
        "schema": "metrifid.model_release_receipt",
        "status": "REVIEW_REQUIRED",
        "completed_exit_code": 40,
        "receipt_sha256": "a" * 64,
        "policy": {"raw_sha256": "b" * 64, "rule_count": 1},
        "change_count": 1,
        "changes_complete": True,
        "changes": [
            {
                "source": "SEMANTIC_OBJECT",
                "selector": {
                    "object_type": "body",
                    "object_name": _AWKWARD_NAME,
                    "field": "mass",
                    "change_kind": "VALUE_CHANGED",
                },
                "classification": "REVIEW_REQUIRED",
                "rule_id": None,
                "before_value": {"kind": "ieee754_binary64", "bits": "3ff8000000000000"},
                "after_value": {"kind": "ieee754_binary64", "bits": "4000000000000000"},
                "before_sha256": "c" * 64,
                "after_sha256": "d" * 64,
                "details": {},
            }
        ],
        "limitations": [],
        "missing_required_rules": [],
        "certification_receipt": {
            "byte_comparison": {
                "equal": False,
                "differing_byte_count": 3,
                "first_differing_byte_offset": 7,
            },
            "field_report": None,
            "runtime_identity": None,
        },
    }


def _parsed(encoded: bytes) -> dict[str, CanonicalValue]:
    """Decode emitted bytes as one complete JSON result object."""
    decoded = json.loads(encoded.decode("utf-8"))
    assert isinstance(decoded, dict)
    return decoded


def test_the_default_json_ceiling_is_exactly_two_hundred_sixty_two_thousand_bytes() -> None:
    """Pin the frozen default output ceiling to its exact byte value."""
    assert DEFAULT_JSON_MAX_BYTES == 262144


@pytest.mark.parametrize("full", [True, False])
def test_a_small_result_encodes_to_the_canonical_bytes_plus_exactly_one_newline(
    full: bool,
) -> None:
    """Prove the streaming encoder agrees byte for byte with the canonical encoder."""
    document = _failure_document()
    encoded = encode_result(document, full=full)
    assert encoded == canonical_json_bytes(document) + b"\n"


def test_a_hand_made_multibyte_result_also_matches_the_canonical_encoder_byte_for_byte() -> None:
    """Prove the two encoders agree on a document carrying escaped multibyte text."""
    document = _document(_awkward_receipt(), exit_code=40)
    assert encode_result(document, full=True) == canonical_json_bytes(document) + b"\n"


def test_emitted_bytes_are_utf8_json_ending_in_exactly_one_newline_that_round_trips() -> None:
    """Decode the emitted bytes back into the identical document."""
    document = _changed_document()
    encoded = encode_result(document, full=True)
    assert encoded.endswith(b"\n")
    assert encoded.count(b"\n") == 1
    assert not encoded[:-1].endswith(b"\n")
    text = encoded.decode("utf-8", errors="strict")
    assert text.encode("utf-8", errors="strict") == encoded
    assert json.loads(text) == json.loads(canonical_json_bytes(document).decode("utf-8"))
    assert set(_parsed(encoded)) == _RESULT_MEMBERS


def test_the_byte_bound_is_inclusive_of_the_trailing_newline_at_the_exact_length() -> None:
    """Emit in full at exactly the encoded length and summarize one byte below it."""
    document = _changed_document()
    complete = encode_result(document, full=True)
    length = len(complete)

    below = encode_result(document, full=False, max_bytes=length - 1)
    exact = encode_result(document, full=False, max_bytes=length)
    above = encode_result(document, full=False, max_bytes=length + 1)

    assert exact == complete
    assert above == complete
    assert _parsed(exact)["truncated"] is False
    assert _parsed(above)["truncated"] is False
    assert below != complete
    assert _parsed(below)["truncated"] is True
    assert len(below) <= length - 1


def test_an_overflowing_result_emits_a_parsable_summary_rather_than_partial_json() -> None:
    """Replace an oversized document with a complete summary that still parses."""
    document = _changed_document()
    emitted = encode_result(document, full=False, max_bytes=2048)
    summary = _parsed(emitted)

    assert len(emitted) <= 2048
    assert set(summary) == _RESULT_MEMBERS
    assert summary["truncated"] is True
    assert summary["findings"] == []
    assert summary["command"] == document["command"]
    assert summary["observation"] == document["observation"]
    assert summary["compiled_comparison"] == document["compiled_comparison"]
    assert summary["evaluation"] == document["evaluation"]
    limitations = summary["limitations"]
    assert isinstance(limitations, list)
    assert len(limitations) == 1
    assert limitations[0]["category"] == "presentation"
    assert "--full" in limitations[0]["message"]


@pytest.mark.parametrize(
    "max_bytes", [1, 2, 64, 500, 879, 880, 881, 1035, 1036, 1037, 2048, 8192, 18000]
)
def test_every_bounded_emission_parses_as_one_complete_json_document(max_bytes: int) -> None:
    """Prove no bound can make the encoder return a truncated JSON fragment."""
    document = _changed_document()
    emitted = encode_result(document, full=False, max_bytes=max_bytes)
    parsed = _parsed(emitted)
    assert set(parsed) == _RESULT_MEMBERS
    assert emitted.endswith(b"\n")
    assert emitted.count(b"\n") == 1
    assert parsed["truncated"] is True


def test_the_summary_still_reports_the_true_finding_count_after_dropping_the_rows() -> None:
    """Keep the honest finding total even though every finding row was omitted."""
    document = _changed_document()
    assert document["finding_count"] == 12
    findings = document["findings"]
    assert isinstance(findings, list)
    assert len(findings) == 12

    summary = _parsed(encode_result(document, full=False, max_bytes=2048))
    assert summary["findings"] == []
    assert summary["finding_count"] == 12


def test_a_bound_too_small_for_the_path_summary_drops_artifacts_and_emits_anyway() -> None:
    """Make the single further adjustment, then emit even though the bound is passed."""
    document = _changed_document()
    with_paths = encode_result(document, full=False, max_bytes=2048)
    assert _parsed(with_paths)["artifacts"] == document["artifacts"]

    emitted = encode_result(document, full=False, max_bytes=64)
    parsed = _parsed(emitted)

    assert len(emitted) > 64
    assert parsed["artifacts"] == {}
    assert parsed["truncated"] is True
    assert parsed["finding_count"] == 12
    limitations = parsed["limitations"]
    assert isinstance(limitations, list)
    assert len(limitations) == 1
    assert limitations[0]["category"] == "presentation"
    assert "artifact paths" in limitations[0]["message"]
    assert set(parsed) == _RESULT_MEMBERS


def test_a_full_export_ignores_the_byte_bound_and_keeps_every_finding() -> None:
    """Return the complete document under a tiny bound whenever full is requested."""
    document = _changed_document()
    emitted = encode_result(document, full=True, max_bytes=1)
    parsed = _parsed(emitted)

    assert emitted == encode_result(document, full=True)
    assert len(emitted) > 1
    assert parsed["truncated"] is False
    assert parsed["finding_count"] == 12
    findings = parsed["findings"]
    assert isinstance(findings, list)
    assert len(findings) == 12
    assert parsed["limitations"] == document["limitations"]
    assert parsed["artifacts"] == document["artifacts"]


@pytest.mark.parametrize("full", [True, False])
def test_the_encoder_refuses_a_not_a_number_float_instead_of_emitting_a_nan_token(
    full: bool,
) -> None:
    """Reject a nonfinite float rather than writing an unparsable NaN token."""
    document = _failure_document()
    evaluation = document["evaluation"]
    assert isinstance(evaluation, dict)
    document["evaluation"] = {**evaluation, "policy_rule_count": float("nan")}

    assert "NaN" in json.dumps(document)
    with pytest.raises(ValueError, match="not JSON compliant"):
        encode_result(document, full=full)


def test_multibyte_and_escaped_display_text_survives_encoding_and_decoding_unchanged() -> None:
    """Round-trip a name carrying multibyte, quote, backslash and newline characters."""
    document = _document(_awkward_receipt(), exit_code=40)
    findings = document["findings"]
    assert isinstance(findings, list)
    display = findings[0]["display"]
    assert isinstance(display, str)
    assert _AWKWARD_NAME in display

    encoded = encode_result(document, full=True)
    assert encoded.count(b"\n") == 1
    assert "🤖".encode() in encoded
    decoded_findings = _parsed(encoded)["findings"]
    assert isinstance(decoded_findings, list)
    assert decoded_findings[0]["display"] == display
    assert _AWKWARD_NAME in str(decoded_findings[0]["display"])


def test_the_byte_bound_is_measured_in_utf8_bytes_and_not_in_characters() -> None:
    """Bound a multibyte document on its byte length rather than its character count."""
    document = _document(_awkward_receipt(), exit_code=40)
    complete = encode_result(document, full=True)
    byte_length = len(complete)
    character_length = len(complete.decode("utf-8"))
    assert character_length < byte_length

    assert encode_result(document, full=False, max_bytes=byte_length) == complete
    at_character_count = encode_result(document, full=False, max_bytes=character_length)
    assert at_character_count != complete
    assert _parsed(at_character_count)["truncated"] is True


def test_render_text_states_that_every_serialized_byte_matched_for_the_equal_receipt() -> None:
    """Report compiled identity in the byte comparison's own words."""
    document = _document(_load_receipt(_EQUAL_RECEIPT), exit_code=0)
    text = render_text(document)
    lines = text.splitlines()

    assert lines[0] == "Compiled model identical."
    assert lines[1] == "Every serialized byte of the two compiled artifacts is identical."
    assert "serialized byte(s) differ" not in text


def test_render_text_states_the_differing_byte_count_and_the_first_differing_offset() -> None:
    """Report the differing byte total and its first offset from the real receipt."""
    receipt = _load_receipt(_CHANGED_RECEIPT)
    certification = receipt["certification_receipt"]
    assert isinstance(certification, dict)
    comparison = certification["byte_comparison"]
    assert isinstance(comparison, dict)
    assert comparison["differing_byte_count"] == 71
    assert comparison["first_differing_byte_offset"] == 1692

    text = render_text(_document(receipt, exit_code=40))
    lines = text.splitlines()

    assert lines[0] == "Compiled model changed."
    assert lines[1] == "71 serialized byte(s) differ, first at offset 1692."
    assert "identical" not in lines[1]


def test_render_text_for_a_certification_reports_the_same_byte_numbers() -> None:
    """Read compiled identity from a standalone certification exactly as from a release."""
    receipt = _load_receipt(_CERTIFICATION_RECEIPT)
    document = _document(receipt, exit_code=40)
    text = render_text(document)

    assert document["finding_count"] == 9
    assert text.splitlines()[0] == "Compiled model changed."
    assert "71 serialized byte(s) differ, first at offset 1692." in text


def test_render_text_ends_in_a_newline_and_states_the_boundary_without_claiming_approval() -> None:
    """End every rendered view with the boundary sentence and a trailing newline."""
    for document in (
        _document(_load_receipt(_EQUAL_RECEIPT), exit_code=0),
        _changed_document(),
        _failure_document(),
    ):
        text = render_text(document)
        assert text.endswith("\n")
        assert not text.endswith("\n\n")
        assert _BOUNDARY_SENTENCE in text.splitlines()


def test_render_text_lists_only_path_like_artifacts_and_never_the_receipt_digest() -> None:
    """Print the known artifact paths alone, leaving the receipt digest out of the block."""
    receipt = _load_receipt(_CHANGED_RECEIPT)
    digest = receipt["receipt_sha256"]
    assert isinstance(digest, str)
    text = render_text(
        _document(
            receipt,
            exit_code=40,
            artifacts={
                "output_dir": "/runs/a",
                "receipt": "/runs/a/model_release.json",
                "receipt_sha256": digest,
                "markdown": "/runs/a/model_release.md",
                "html": None,
            },
        )
    )
    lines = text.splitlines()

    assert "output_dir /runs/a" in lines
    assert "receipt    /runs/a/model_release.json" in lines
    assert "markdown   /runs/a/model_release.md" in lines
    assert digest not in text
    assert "receipt_sha256" not in text
    assert not any(line.startswith("html") for line in lines)
    assert lines.index("output_dir /runs/a") < lines.index("receipt    /runs/a/model_release.json")
    assert lines.index("receipt    /runs/a/model_release.json") < lines.index(
        "markdown   /runs/a/model_release.md"
    )


def test_render_text_on_a_failure_document_shows_the_problem_code_alone() -> None:
    """Name the reported failure without claiming any compiled comparison happened."""
    document = _failure_document()
    text = render_text(document)
    lines = text.splitlines()

    assert lines[0] == "Compiled comparison not established."
    assert "  CONFIGURATION_IO_FAILED: the receipt could not be read" in lines
    assert "Compiled model identical." not in text
    assert "Compiled model changed." not in text
    assert "serialized byte" not in text
    assert _BOUNDARY_SENTENCE in lines
    assert text.endswith("\n")


def test_a_failure_document_prints_no_compared_block_when_no_role_is_known() -> None:
    """Never print a heading with nothing under it; an empty section states nothing."""
    document = build_failure_document(
        command="diff",
        exit_code=64,
        reason_code="MODEL_ENTRYPOINT_INVALID",
        message="entrypoint_unavailable",
        reader_version="0.7.2",
        inputs={"baseline": None, "candidate": None, "roots_correspond": None},
        artifacts={
            "output_dir": None,
            "receipt": None,
            "receipt_sha256": None,
            "markdown": None,
            "html": None,
        },
        policy_origin="not_created",
    )
    rendered = render_text(document)
    assert "Compared" not in rendered
    assert "MODEL_ENTRYPOINT_INVALID" in rendered
    assert rendered.endswith("\n")


def test_the_bounded_summary_clips_a_long_diagnostic_rather_than_carrying_it_whole() -> None:
    """The contract forbids a raw diagnostic body inside the bounded summary."""
    document = build_failure_document(
        command="show",
        exit_code=64,
        reason_code="CONFIGURATION_PARSE_FAILED",
        message="x" * 5000,
        reader_version="0.7.2",
        inputs={"baseline": None, "candidate": None, "roots_correspond": None},
        artifacts={
            "output_dir": None,
            "receipt": None,
            "receipt_sha256": None,
            "markdown": None,
            "html": None,
        },
        policy_origin="not_applicable",
    )
    emitted = json.loads(encode_result(document, full=False, max_bytes=900))
    assert emitted["truncated"] is True
    carried = emitted["problems"][0]
    assert carried["code"] == "CONFIGURATION_PARSE_FAILED"
    assert len(carried["message"]) < 5000
    assert carried["message"].endswith("...")


def test_the_readable_view_states_the_recorded_evaluation_separately_from_byte_identity() -> None:
    """Compiled identity, the recorded evaluation and this read succeeding are three facts."""
    document = _document(_load_receipt(_CHANGED_RECEIPT), exit_code=40)
    rendered = render_text(document)
    evaluation = document["evaluation"]
    assert isinstance(evaluation, dict)
    assert f"Recorded evaluation: {evaluation['status']}." in rendered
    assert f"exited {evaluation['exit_code']}." in rendered
    identity_line, byte_line, evaluation_line = rendered.splitlines()[:3]
    assert identity_line == "Compiled model changed."
    assert "serialized byte(s) differ" in byte_line
    assert evaluation_line.startswith("Recorded evaluation:")


def test_an_identical_pair_still_reports_the_outcome_its_producer_recorded() -> None:
    """A clean result names its recorded status too, so the line is never a failure marker."""
    rendered = render_text(_document(_load_receipt(_EQUAL_RECEIPT), exit_code=0))
    assert "Recorded evaluation: NO_COMPILED_CHANGE." in rendered
    assert "exited 0." in rendered


def test_a_failure_document_states_no_recorded_evaluation_it_does_not_have() -> None:
    """A command that produced no receipt has no recorded outcome to report."""
    document = build_failure_document(
        command="diff",
        exit_code=64,
        reason_code="MODEL_ENTRYPOINT_INVALID",
        message="entrypoint_unavailable",
        reader_version="0.7.2",
        inputs={"baseline": None, "candidate": None, "roots_correspond": None},
        artifacts={
            "output_dir": None,
            "receipt": None,
            "receipt_sha256": None,
            "markdown": None,
            "html": None,
        },
        policy_origin="not_created",
    )
    assert "Recorded evaluation" not in render_text(document)


@pytest.mark.parametrize(
    ("label", "injected"),
    [
        ("newline", "rule\nCompiled model identical."),
        ("carriage return", "rule\rCompiled model identical."),
        ("line separator", "rule\u2028Compiled model identical."),
        ("next line", "rule\x85Compiled model identical."),
        ("form feed", "rule\x0cCompiled model identical."),
        ("terminal escape", "rule\x1b[2KCompiled model identical."),
    ],
)
def test_receipt_text_can_never_add_a_line_to_the_readable_report(
    label: str, injected: str
) -> None:
    """A receipt is written by its producer, so its text must not forge a line of this report.

    Without this, a rule identity or object name carrying a newline prints a sentence that
    contradicts the comparison rendered directly above it.
    """
    document = _document(_load_receipt(_CHANGED_RECEIPT), exit_code=40)
    limitations = document["limitations"]
    assert isinstance(limitations, list)
    limitations.append({"category": "policy", "message": f"unmet: {injected}", "source": ""})
    lines = render_text(document).splitlines()
    assert not [line for line in lines if line.strip() == "Compiled model identical."], label
    carrier = [line for line in lines if "unmet:" in line]
    assert len(carrier) == 1, label
    assert "Compiled model identical." in carrier[0], "the text is shown, not silently dropped"


def test_an_injected_control_character_is_shown_as_a_visible_escape() -> None:
    """Nothing is removed quietly: the reader still sees exactly what the receipt carried."""
    document = _document(_load_receipt(_EQUAL_RECEIPT), exit_code=0)
    limitations = document["limitations"]
    assert isinstance(limitations, list)
    limitations.append({"category": "policy", "message": "a\nb\u2028c", "source": ""})
    carrier = [line for line in render_text(document).splitlines() if line.strip().startswith("a")]
    assert len(carrier) == 1
    assert carrier[0].strip() == "a\\x0ab\\u2028c"


def test_the_bounded_summary_names_everything_it_actually_omitted() -> None:
    """The summary drops input identities and the producer runtime too, so it must say so."""
    document = _document(_load_receipt(_CHANGED_RECEIPT), exit_code=40)
    summary = json.loads(encode_result(document, full=False, max_bytes=4096))
    assert summary["inputs"] == {}
    assert summary["runtime"]["producer"] is None
    message = summary["limitations"][0]["message"]
    for omitted in ("findings", "limitations", "input identities", "producer runtime"):
        assert omitted in message, omitted
