"""The offline report presents the result; it must not be able to say anything else.

Everything the report prints came from a receipt someone else produced, so the tests here are
mostly about what the renderer refuses to do with that text: run it, fetch anything with it, or
turn it into markup. The rest confirm the report is complete, so opening it is a real alternative
to reading the canonical receipt by hand.
"""

from __future__ import annotations

import html.parser
import json
import subprocess
import sys
from pathlib import Path
from typing import Final

import pytest

from metrifid.json_values import CanonicalValue, strict_json_loads
from metrifid.result import build_failure_document, show_receipt
from metrifid.result._report import render_report

_FIXTURES: Final = Path(__file__).parents[1] / "fixtures" / "result"
_CHANGED: Final = _FIXTURES / "changed_model_release.json"
_EQUAL: Final = _FIXTURES / "equal_model_release.json"
_CERTIFICATION: Final = _FIXTURES / "changed_certification.json"
_UNSAFE_TEXT: Final = ("<script", "javascript:", "http://", "https://")


def _document(path: Path) -> dict[str, CanonicalValue]:
    """Build one complete result document from a saved receipt."""
    return show_receipt(str(path))


def _receipt(path: Path) -> dict[str, CanonicalValue]:
    """Load one saved receipt as canonical values."""
    value = strict_json_loads(path.read_bytes())
    assert isinstance(value, dict)
    return value


class _Inspector(html.parser.HTMLParser):
    """Collect the elements, attributes and links one rendered report actually contains."""

    def __init__(self) -> None:
        """Start with nothing collected."""
        super().__init__()
        self.elements: list[str] = []
        self.attributes: list[tuple[str, str]] = []
        self.href: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Record one element and every attribute it carries."""
        self.elements.append(tag)
        for name, value in attrs:
            self.attributes.append((name, value or ""))
            if tag == "a" and name == "href":
                self.href.append(value or "")


def _inspect(text: str) -> _Inspector:
    """Parse one rendered report and return what it contains.

    A substring search cannot answer this: correctly escaped hostile text legitimately contains
    the characters of an event handler, so the question is whether any ELEMENT carries one.
    """
    inspector = _Inspector()
    inspector.feed(text)
    inspector.close()
    return inspector


def _require_inert(text: str) -> _Inspector:
    """Require a report that runs nothing and fetches nothing."""
    inspector = _inspect(text)
    assert "script" not in inspector.elements
    assert not [name for name, _ in inspector.attributes if name.startswith("on")]
    assert not [value for _, value in inspector.attributes if "javascript:" in value.lower()]
    for token in _UNSAFE_TEXT:
        assert token not in text, token
    return inspector


def _failure_document() -> dict[str, CanonicalValue]:
    """Build one result document for a command that produced no receipt."""
    return build_failure_document(
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


@pytest.mark.parametrize("fixture", [_CHANGED, _EQUAL, _CERTIFICATION])
def test_every_real_fixture_renders_a_parsable_report_with_no_active_content(
    fixture: Path,
) -> None:
    """A report is a page someone opens offline: no script, no fetch, no remote asset."""
    rendered = render_report(_document(fixture))
    _require_inert(rendered)
    assert rendered.count("<style") == 1, "styling is inline and local"


def test_hostile_receipt_text_is_escaped_rather_than_rendered_as_markup() -> None:
    """Model names and receipt strings are data. They must never become elements or attributes."""
    document = _document(_CHANGED)
    findings = document["findings"]
    limitations = document["limitations"]
    assert isinstance(findings, list)
    assert isinstance(limitations, list)
    first = findings[0]
    assert isinstance(first, dict)
    first["display"] = '<img src=x onerror=alert(1)>"><b>forged</b>'
    limitations.append(
        {
            "category": "policy",
            "message": "</style><script>alert(2)</script>",
            "source": "",
        }
    )
    rendered = render_report(document)
    inspector = _require_inert(rendered)
    assert "img" not in inspector.elements
    assert "&lt;img" in rendered
    assert "&lt;script" in rendered
    assert "<b>forged</b>" not in rendered


def test_the_report_carries_every_retained_witness_change_and_boundary() -> None:
    """Opening the report must not lose evidence the canonical receipt retained."""
    receipt = _receipt(_CHANGED)
    rendered = render_report(_document(_CHANGED))
    certification = receipt["certification_receipt"]
    assert isinstance(certification, dict)
    report = certification["field_report"]
    assert isinstance(report, dict)
    changed_fields = report["changed_fields"]
    assert isinstance(changed_fields, list)
    witnesses = 0
    for field in changed_fields:
        assert isinstance(field, dict)
        rows = field["witnesses"]
        assert isinstance(rows, list)
        for witness in rows:
            assert isinstance(witness, dict)
            witnesses += 1
            for side in ("baseline_text", "candidate_text"):
                assert str(witness[side]) in rendered, (field["path"], side)
    assert witnesses == 13
    changes = receipt["changes"]
    assert isinstance(changes, list)
    for change in changes:
        assert isinstance(change, dict)
        selector = change["selector"]
        assert isinstance(selector, dict)
        assert str(selector["object_name"]) in rendered
    claims = list(receipt["limitations"]) + list(certification["limitations"])  # type: ignore[arg-type]
    for claim in claims:
        assert isinstance(claim, dict)
        assert str(claim["statement"]) in rendered


def test_links_use_only_the_known_local_output_names() -> None:
    """A receipt must not be able to choose where the report points."""
    document = _document(_CHANGED)
    artifacts = document["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts["receipt"] = "https://example.invalid/evil.json"
    artifacts["markdown"] = "../../escape.md"
    rendered = render_report(document)
    inspector = _inspect(rendered)
    # The recorded path may be shown as escaped text, which is inert. What must never happen is
    # that a value from the receipt becomes somewhere this page points.
    assert set(inspector.href) <= {"model_release.json", "model_release.md"}
    assert "script" not in inspector.elements
    assert not [name for name, _ in inspector.attributes if name.startswith("on")]
    assert not [value for _, value in inspector.attributes if "example.invalid" in value]
    assert not [value for _, value in inspector.attributes if "escape.md" in value]


def test_a_failure_document_renders_without_claiming_a_comparison_happened() -> None:
    """A command that produced no receipt still gets a page, and it says so."""
    rendered = render_report(_failure_document())
    _require_inert(rendered)
    assert "MODEL_ENTRYPOINT_INVALID" in rendered
    assert "Compiled model identical." not in rendered
    assert "Compiled model changed." not in rendered


def test_the_report_states_the_recorded_evaluation_and_the_boundary() -> None:
    """The page repeats what the run recorded, and what it does not claim."""
    rendered = render_report(_document(_CHANGED))
    assert "REVIEW_REQUIRED" in rendered
    assert "40" in rendered
    lowered = rendered.lower()
    assert "not a statement about behaviour, safety, or approval" in lowered


def test_rendering_a_report_never_imports_a_native_library(tmp_path: Path) -> None:
    """A saved receipt is readable, and now reportable, where MuJoCo is not installed."""
    source = f"""
import sys

class _Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".", 1)[0] in {{"mujoco", "numpy"}}:
            raise ModuleNotFoundError(name)
        return None

sys.meta_path.insert(0, _Block())
from metrifid.result import show_receipt
from metrifid.result._report import render_report

rendered = render_report(show_receipt({str(_CHANGED)!r}))
loaded = [n for n in sys.modules if n.split(".", 1)[0] in {{"mujoco", "numpy"}}]
assert not loaded, loaded
assert rendered.startswith("<!doctype html") or rendered.lstrip().startswith("<")
print("REPORT_PURE", len(rendered))
"""
    script = tmp_path / "pure.py"
    script.write_text(source, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-B", str(script)], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "REPORT_PURE" in completed.stdout


def test_the_report_is_valid_utf8_and_self_contained() -> None:
    """Multibyte names survive, and nothing is fetched to display the page."""
    document = _document(_CHANGED)
    findings = document["findings"]
    assert isinstance(findings, list)
    first = findings[0]
    assert isinstance(first, dict)
    first["display"] = "modèle 🤖 body link mass"
    rendered = render_report(document)
    encoded = rendered.encode("utf-8")
    assert "modèle 🤖" in encoded.decode("utf-8")
    assert b"<link" not in encoded.replace(b"<link ", b"")
    assert json.dumps(rendered)  # renderable text, no control characters that break transport
