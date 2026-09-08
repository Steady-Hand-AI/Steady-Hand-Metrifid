"""Bounded JSON and readable text for one result document.

The default JSON output is one complete document or one honest summary; it is never a partially
packed preview.  The text view renders the same findings, in the same order, from the same members.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Final

from ..json_values import CanonicalValue
from ._contract import NOT_ESTABLISHED
from ._findings import PRESENTATION

__all__ = ["DEFAULT_JSON_MAX_BYTES", "encode_result", "render_text"]

DEFAULT_JSON_MAX_BYTES: Final = 262144
_MAX_SUMMARY_MESSAGE: Final = 200
_OMITTED_MESSAGE: Final = (
    "The complete result exceeded the default output limit, so the findings, limitations, input "
    "identities and producer runtime were omitted from this document. Re-run with --full to write "
    "the complete detail."
)
_PATHS_OMITTED_MESSAGE: Final = (
    "The complete result exceeded the default output limit. Findings, limitations, input "
    "identities, producer runtime and artifact paths were all omitted. Re-run with --full to "
    "write the complete detail."
)
_ENCODER: Final = json.JSONEncoder(
    ensure_ascii=False,
    sort_keys=True,
    separators=(",", ":"),
    allow_nan=False,
)


def _encode_whole(document: Mapping[str, CanonicalValue]) -> bytes:
    """Encode one document with its trailing newline and no byte ceiling.

    Args:
        document: The result document to encode.

    Returns:
        The complete encoded bytes.
    """
    return "".join(_ENCODER.iterencode(document)).encode("utf-8", errors="strict") + b"\n"


def _encode_bounded(document: Mapping[str, CanonicalValue], limit: int) -> bytes | None:
    """Encode one document, or return None as soon as it would pass the byte limit.

    The buffer is discarded rather than emitted when the limit is passed, so no partial JSON can
    reach a caller.

    Args:
        document: The result document to encode.
        limit: The inclusive byte ceiling for the document plus its trailing newline.

    Returns:
        The encoded bytes with a trailing newline, or None when the limit would be passed.
    """
    chunks: list[str] = []
    size = 0
    for chunk in _ENCODER.iterencode(document):
        size += len(chunk.encode("utf-8", errors="strict"))
        if size + 1 > limit:
            return None
        chunks.append(chunk)
    return "".join(chunks).encode("utf-8", errors="strict") + b"\n"


def _summary(
    document: Mapping[str, CanonicalValue], *, keep_paths: bool
) -> dict[str, CanonicalValue]:
    """Build the bounded summary that replaces a result too large to emit in full."""
    artifacts = document.get("artifacts") if keep_paths else {}
    return {
        "schema": document["schema"],
        "schema_version": document["schema_version"],
        "command": document["command"],
        "observation": document["observation"],
        "compiled_comparison": document["compiled_comparison"],
        "evaluation": document["evaluation"],
        "runtime": {"reader": _reader_version(document), "producer": None},
        "inputs": {},
        "findings": [],
        "finding_count": document["finding_count"],
        "limitations": [
            {
                "category": PRESENTATION,
                "message": _OMITTED_MESSAGE if keep_paths else _PATHS_OMITTED_MESSAGE,
                "source": "",
            }
        ],
        "artifacts": artifacts if isinstance(artifacts, dict) else {},
        "truncated": True,
        "problems": _bounded_problems(document.get("problems")),
    }


def _bounded_problems(problems: CanonicalValue) -> list[CanonicalValue]:
    """Keep each problem's reason code, with its diagnostic clipped to a bounded prefix."""
    if not isinstance(problems, list):
        return []
    bounded: list[CanonicalValue] = []
    for problem in problems:
        entry = problem if isinstance(problem, dict) else {}
        message = entry.get("message")
        text = message if isinstance(message, str) else ""
        clipped = text if len(text) <= _MAX_SUMMARY_MESSAGE else text[:_MAX_SUMMARY_MESSAGE] + "..."
        bounded.append({"code": entry.get("code"), "message": clipped})
    return bounded


def _reader_version(document: Mapping[str, CanonicalValue]) -> CanonicalValue:
    """Return the reader version recorded in a document's runtime member."""
    runtime = document.get("runtime")
    return runtime.get("reader") if isinstance(runtime, dict) else None


def encode_result(
    document: Mapping[str, CanonicalValue],
    *,
    full: bool,
    max_bytes: int = DEFAULT_JSON_MAX_BYTES,
) -> bytes:
    """Encode one result document as UTF-8 JSON plus a trailing newline.

    A full export carries every retained finding and is not bounded.  A default emission is
    bounded: if the complete document would pass the limit it is discarded whole and replaced by
    a summary that keeps the verdict, the receipt identity and the finding count.  At most one
    further adjustment is made, dropping the artifact paths, and the final bytes are always
    measured before they are returned.

    Args:
        document: The complete result document.
        full: Whether this is an explicit full export.
        max_bytes: The inclusive ceiling for the default emission.

    Returns:
        The bytes to write, never partial JSON.
    """
    if full:
        return _encode_whole(document)
    encoded = _encode_bounded(document, max_bytes)
    if encoded is not None:
        return encoded
    encoded = _encode_bounded(_summary(document, keep_paths=True), max_bytes)
    if encoded is not None:
        return encoded
    return _encode_whole(_summary(document, keep_paths=False))


_HEADLINE: Final = {
    "identical": "Compiled model identical.",
    "different": "Compiled model changed.",
    NOT_ESTABLISHED: "Compiled comparison not established.",
}
_KIND_HEADING: Final = (
    ("named", "Named changes"),
    ("compiled_field", "Compiled fields"),
    ("residual", "Fail-closed residual"),
)
_ARTIFACT_PATH_ORDER: Final = ("output_dir", "receipt", "markdown", "html")
_CATEGORY_HEADING: Final = (
    ("producer", "Not covered by the producer"),
    ("attribution", "Not attributed"),
    ("policy", "Evaluation conditions"),
    ("presentation", "Omitted from this view"),
    ("claim", "What this does not claim"),
)


def render_text(document: Mapping[str, CanonicalValue]) -> str:
    """Render one result document as the readable terminal view.

    Args:
        document: The complete result document.

    Returns:
        The text to write, ending in a newline.
    """
    lines: list[str] = []
    comparison = document.get("compiled_comparison")
    state = comparison.get("state") if isinstance(comparison, dict) else NOT_ESTABLISHED
    lines.append(_HEADLINE.get(str(state), _HEADLINE[NOT_ESTABLISHED]))
    lines.extend(_byte_lines(comparison if isinstance(comparison, dict) else {}))
    lines.extend(_evaluation_lines(document))
    lines.extend(_problem_lines(document))
    lines.extend(_compared_lines(document))
    lines.extend(_finding_lines(document))
    lines.extend(_limitation_lines(document))
    lines.extend(_footer_lines(document))
    return "\n".join(lines) + "\n"


def _one_line(value: CanonicalValue) -> str:
    """Render one receipt-supplied value confined to a single printable line.

    Text inside a receipt is written by whoever produced it, so it must never be able to add lines
    to this report. A rule identity or object name carrying a newline could otherwise print a
    sentence that contradicts the comparison directly above it. Escape sequences are shown rather
    than dropped, so nothing is silently removed either.
    """
    text = value if isinstance(value, str) else str(value)
    rendered: list[str] = []
    for character in text:
        if character == " " or character.isprintable():
            rendered.append(character)
        elif ord(character) < 0x100:
            rendered.append(f"\\x{ord(character):02x}")
        else:
            rendered.append(f"\\u{ord(character):04x}")
    return "".join(rendered)


def _evaluation_lines(document: Mapping[str, CanonicalValue]) -> list[str]:
    """State the outcome the producing run recorded, whatever this reader's own result was.

    Compiled identity, the recorded evaluation and the success of reading the receipt are three
    different facts. A receipt that records a failed policy evaluation says so here even when the
    artifacts are byte-identical and this command reads it successfully.
    """
    evaluation = document.get("evaluation")
    if not isinstance(evaluation, dict):
        return []
    status = evaluation.get("status")
    if not isinstance(status, str):
        return []
    recorded = evaluation.get("exit_code")
    exited = f" The run that produced it exited {recorded}." if isinstance(recorded, int) else ""
    return [f"Recorded evaluation: {_one_line(status)}.{exited}"]


def _byte_lines(comparison: Mapping[str, CanonicalValue]) -> list[str]:
    """State what the byte comparison actually established, in its own numbers."""
    differing = comparison.get("differing_byte_count")
    offset = comparison.get("first_differing_byte_offset")
    if comparison.get("equal") is True:
        return ["Every serialized byte of the two compiled artifacts is identical."]
    if isinstance(differing, int):
        located = f", first at offset {offset}" if isinstance(offset, int) else ""
        return [f"{differing} serialized byte(s) differ{located}."]
    return []


def _problem_lines(document: Mapping[str, CanonicalValue]) -> list[str]:
    """Render the command failures that prevented a completed comparison."""
    problems = document.get("problems")
    if not isinstance(problems, list) or not problems:
        return []
    lines = [""]
    for problem in problems:
        entry = problem if isinstance(problem, dict) else {}
        lines.append(f"  {_one_line(entry.get('code'))}: {_one_line(entry.get('message'))}")
    return lines


def _compared_lines(document: Mapping[str, CanonicalValue]) -> list[str]:
    """Render each role's live paths when known and its retained identity."""
    inputs = document.get("inputs")
    if not isinstance(inputs, dict):
        return []
    described = [role for role in ("baseline", "candidate") if isinstance(inputs.get(role), dict)]
    if not described:
        return []
    lines = ["", "Compared"]
    for role in described:
        entry = inputs[role]
        if not isinstance(entry, dict):
            continue
        path = entry.get("path") or entry.get("entrypoint") or "not recorded"
        lines.append(f"  {role:<10} {_one_line(path)}")
        root = entry.get("root")
        members = entry.get("member_count")
        detail = []
        if root:
            detail.append(f"root {_one_line(root)}")
        if members is not None:
            detail.append(f"{_one_line(members)} file(s) measured")
        if detail:
            lines.append(f"  {'':<10} {'  '.join(detail)}")
    if any(_member_count(inputs.get(role)) is not None for role in ("baseline", "candidate")):
        lines.append("  every file under each root is measured, not only the entrypoint")
    if inputs.get("roots_correspond") is False:
        lines.append("  the two roots differ: the measured file counts are not a change list")
    return lines


def _member_count(entry: CanonicalValue) -> CanonicalValue:
    """Return one role's admitted member count, or None when the role recorded none."""
    return entry.get("member_count") if isinstance(entry, dict) else None


def _finding_lines(document: Mapping[str, CanonicalValue]) -> list[str]:
    """Render every retained finding, grouped by kind and in receipt order."""
    findings = document.get("findings")
    if not isinstance(findings, list) or not findings:
        return []
    lines: list[str] = []
    for kind, heading in _KIND_HEADING:
        selected = [
            item for item in findings if isinstance(item, dict) and item.get("kind") == kind
        ]
        if not selected:
            continue
        lines.extend(["", f"{heading} ({len(selected)})"])
        lines.extend(f"  {_one_line(item.get('display'))}" for item in selected)
    return lines


def _limitation_lines(document: Mapping[str, CanonicalValue]) -> list[str]:
    """Render every limitation, keeping each category separate."""
    limitations = document.get("limitations")
    if not isinstance(limitations, list) or not limitations:
        return []
    lines: list[str] = []
    for category, heading in _CATEGORY_HEADING:
        selected = [
            item
            for item in limitations
            if isinstance(item, dict) and item.get("category") == category
        ]
        if not selected:
            continue
        lines.extend(["", heading])
        lines.extend(f"  {_one_line(item.get('message'))}" for item in selected)
    return lines


def _footer_lines(document: Mapping[str, CanonicalValue]) -> list[str]:
    """Render the recorded runtime boundary and every known artifact path."""
    runtime = document.get("runtime")
    producer = runtime.get("producer") if isinstance(runtime, dict) else None
    lines = [""]
    if isinstance(producer, dict):
        lines.append(
            f"Static comparison under MuJoCo {_one_line(producer.get('mujoco'))} · "
            f"{_one_line(producer.get('python_implementation'))} "
            f"{_one_line(producer.get('python'))} · "
            f"{_one_line(producer.get('platform_system'))} "
            f"{_one_line(producer.get('platform_machine'))}."
        )
    lines.append("Not a statement about behaviour, safety, or approval.")
    artifacts = document.get("artifacts")
    if isinstance(artifacts, dict):
        known = [
            (name, artifacts.get(name))
            for name in _ARTIFACT_PATH_ORDER
            if isinstance(artifacts.get(name), str)
        ]
        if known:
            lines.append("")
            lines.extend(f"{name:<10} {_one_line(value)}" for name, value in known)
    return lines
