"""One offline HTML presentation of an already-decided result document.

This page decides nothing.  It shows the same facts as the readable text view, in the same order
and from the same document members: every value printed is one the document already carries, an
absent member is named as unrecorded rather than filled in, and every value that came from a
receipt is escaped before it reaches the markup.  There is no script, no remote asset, and no link
whose target a receipt could choose.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from ..json_values import CanonicalValue
from ._contract import NOT_ESTABLISHED, OBSERVATION_NONE
from ._display import value_text
from ._emit import _CATEGORY_HEADING, _HEADLINE, _KIND_HEADING, _one_line

_ABSENT: Final = "not recorded"
_TITLE: Final = "Metrifid comparison report"
# The two file names the paired output writer publishes together. An href is only ever one of
# these exact constants, so no receipt-supplied text can become a link target.
_PUBLISHED_NAMES: Final = frozenset({"model_release.json", "model_release.md"})
_LINKED_ARTIFACTS: Final = frozenset({"receipt", "markdown"})
_ARTIFACT_LABEL: Final = (
    ("output_dir", "Output directory"),
    ("receipt", "Canonical receipt"),
    ("markdown", "Markdown summary"),
    ("html", "This report"),
)
_ROLE_LABEL: Final = (("baseline", "Baseline"), ("candidate", "Candidate"))
_CHANGE_HEADER: Final = (
    '<tr><th scope="col">Change</th><th scope="col">Retained witnesses</th></tr>'
)
_NO_WITNESS: Final = '<span class="muted">none retained in this row</span>'
_ROLE_ROW: Final = (
    ("path", "Live entrypoint path"),
    ("root", "Live model root"),
    ("entrypoint", "Retained entrypoint"),
    ("member_count", "Files measured under the root"),
)
_PRODUCER_ROW: Final = (
    ("metrifid", "Metrifid"),
    ("mujoco", "MuJoCo"),
    ("python", "Python version"),
    ("python_implementation", "Python implementation"),
    ("platform_system", "Platform system"),
    ("platform_machine", "Platform machine"),
    ("runtime_identity_sha256", "Runtime identity sha256"),
)
_STYLE: Final = """
body { margin: 0; padding: 2rem 1rem; background: #ffffff; color: #1b1b1b;
  font-family: ui-sans-serif, system-ui, sans-serif; line-height: 1.5; }
main { max-width: 62rem; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 0.4rem; }
h2 { font-size: 1.1rem; margin: 2rem 0 0.4rem; padding-bottom: 0.2rem;
  border-bottom: 1px solid #d8d8d8; }
h3 { font-size: 0.98rem; margin: 1.1rem 0 0.3rem; }
p { margin: 0.4rem 0; }
table { border-collapse: collapse; width: 100%; margin: 0.4rem 0; }
th, td { text-align: left; vertical-align: top; font-weight: normal;
  padding: 0.25rem 0.7rem 0.25rem 0; border-bottom: 1px solid #ededed; }
th[scope="row"] { width: 17rem; color: #4d4d4d; }
th[scope="col"] { color: #4d4d4d; border-bottom: 1px solid #cfcfcf; }
code { font-family: ui-monospace, monospace; font-size: 0.9em; overflow-wrap: anywhere; }
ul { margin: 0.4rem 0; padding-left: 1.2rem; }
li { margin: 0.55rem 0; }
details { margin: 0.35rem 0; }
summary { color: #4d4d4d; }
.muted, .pointer, .retained { color: #5c5c5c; font-size: 0.9em; }
.verdict { border-left: 6px solid #9a9a9a; padding: 0.2rem 0 0.2rem 0.9rem; }
.verdict.identical { border-left-color: #2f7a45; }
.verdict.different { border-left-color: #9a6b16; }
.verdict.not_established { border-left-color: #6b6b6b; }
.boundary { margin-top: 2rem; padding-top: 0.6rem; border-top: 1px solid #d8d8d8; }
"""


def _escape(value: CanonicalValue) -> str:
    """Escape one receipt-supplied value for placement in HTML text or an attribute.

    The value is first confined to a single printable line by the same rule the text view uses,
    then escaped with quotes so no receipt can close an attribute, an element or the page.

    Args:
        value: One canonical value taken from the result document.

    Returns:
        Escaped markup-safe text for that value.
    """
    return html.escape(_one_line(value), quote=True)


def _optional(value: CanonicalValue) -> str:
    """Escape one optional member, naming it unrecorded when the document carries none."""
    return _ABSENT if value is None else _escape(value)


def _mapping(value: CanonicalValue) -> Mapping[str, CanonicalValue]:
    """Return one canonical object, or an empty mapping when the member is absent."""
    return value if isinstance(value, dict) else {}


def _entries(value: CanonicalValue) -> Sequence[CanonicalValue]:
    """Return one canonical array, or an empty sequence when the member is absent."""
    return value if isinstance(value, list) else ()


def _row(label: CanonicalValue, value: str) -> str:
    """Render one label and value row of a fact table; the value must already be escaped."""
    return f'<tr><th scope="row">{_escape(label)}</th><td>{value}</td></tr>'


def _table(rows: Sequence[str]) -> str:
    """Render one fact table from rows that are already complete markup."""
    return "<table>\n" + "\n".join(rows) + "\n</table>"


def _paragraph(text: str) -> str:
    """Wrap one already-escaped sentence in a paragraph."""
    return f"<p>{text}</p>"


def _headline_section(document: Mapping[str, CanonicalValue]) -> list[str]:
    """State compiled identity and what the byte comparison established, in its own numbers."""
    comparison = _mapping(document.get("compiled_comparison"))
    state = str(comparison.get("state")) if comparison else NOT_ESTABLISHED
    key = state if state in _HEADLINE else NOT_ESTABLISHED
    return [
        f'<section class="verdict {key}">',
        f"<h1>{html.escape(_HEADLINE[key], quote=True)}</h1>",
        _byte_paragraph(comparison),
        "</section>",
    ]


def _byte_paragraph(comparison: Mapping[str, CanonicalValue]) -> str:
    """State what the byte comparison recorded, or that it recorded no counts at all."""
    differing = comparison.get("differing_byte_count")
    offset = comparison.get("first_differing_byte_offset")
    if comparison.get("equal") is True:
        return _paragraph("Every serialized byte of the two compiled artifacts is identical.")
    if isinstance(differing, int):
        located = f", first at offset {_escape(offset)}" if isinstance(offset, int) else ""
        return _paragraph(f"{_escape(differing)} serialized byte(s) differ{located}.")
    return _paragraph("No serialized byte counts are recorded in this document.")


def _evaluation_section(document: Mapping[str, CanonicalValue]) -> list[str]:
    """Report the outcome the producing run recorded, kept apart from this reader's own result."""
    evaluation = _mapping(document.get("evaluation"))
    command = _mapping(document.get("command"))
    rows = [
        _row("Recorded evaluation status", _optional(evaluation.get("status"))),
        _row("Exit recorded by the producing run", _optional(evaluation.get("exit_code"))),
        _row("Observation", _optional(document.get("observation"))),
        _row("Command that produced this view", _command_text(command)),
    ]
    return [
        "<section>",
        "<h2>Recorded evaluation</h2>",
        _paragraph(
            "The status below is the outcome the producing run recorded. It is neither the "
            "compiled identity above nor the fact that reading this receipt succeeded."
        ),
        _table(rows),
        "</section>",
    ]


def _command_text(command: Mapping[str, CanonicalValue]) -> str:
    """Name the command this document came from and the exit code it is returning."""
    name = command.get("name")
    exit_code = command.get("exit_code")
    if name is None and exit_code is None:
        return _ABSENT
    return f"{_optional(name)}, exit {_optional(exit_code)}"


def _problem_section(document: Mapping[str, CanonicalValue]) -> list[str]:
    """Render the command failures that prevented a completed comparison."""
    problems = _entries(document.get("problems"))
    if not problems:
        return []
    parts = ["<section>", "<h2>Problems</h2>"]
    if document.get("observation") == OBSERVATION_NONE:
        parts.append(
            _paragraph(
                "Nothing was observed: this command returned no completed receipt, so no "
                "comparison was performed."
            )
        )
    parts.append("<ul>")
    for problem in problems:
        entry = _mapping(problem)
        code = _optional(entry.get("code"))
        parts.append(f"<li><code>{code}</code> {_optional(entry.get('message'))}</li>")
    parts.extend(["</ul>", "</section>"])
    return parts


def _runtime_section(document: Mapping[str, CanonicalValue]) -> list[str]:
    """Show the producer runtime the receipt recorded beside this reader's own version."""
    runtime = _mapping(document.get("runtime"))
    producer = runtime.get("producer")
    parts = [
        "<section>",
        "<h2>Runtime</h2>",
        _paragraph(
            "The producer runtime is the runtime the comparison ran under. The reader version is "
            "only the code that rendered this page, and covers nothing."
        ),
        "<h3>Producer runtime, recorded in the receipt</h3>",
    ]
    if isinstance(producer, dict):
        parts.append(
            _table([_row(label, _optional(producer.get(name))) for name, label in _PRODUCER_ROW])
        )
    else:
        parts.append(_paragraph("No producer runtime is recorded in this document."))
    parts.append("<h3>Reader</h3>")
    parts.append(_table([_row("Metrifid rendering this page", _optional(runtime.get("reader")))]))
    parts.append("</section>")
    return parts


def _compared_section(document: Mapping[str, CanonicalValue]) -> list[str]:
    """Render each role's live paths when known and the identity the producer retained."""
    inputs = _mapping(document.get("inputs"))
    parts = ["<section>", "<h2>Compared</h2>"]
    described = [(role, label) for role, label in _ROLE_LABEL if isinstance(inputs.get(role), dict)]
    if not described:
        parts.extend(
            [_paragraph("No input identities are recorded in this document."), "</section>"]
        )
        return parts
    for role, label in described:
        entry = _mapping(inputs.get(role))
        parts.append(f"<h3>{html.escape(label, quote=True)}</h3>")
        parts.append(_table([_row(text, _optional(entry.get(name))) for name, text in _ROLE_ROW]))
    parts.extend(_disclosure_paragraphs(inputs, described))
    parts.append("</section>")
    return parts


def _disclosure_paragraphs(
    inputs: Mapping[str, CanonicalValue], described: Sequence[tuple[str, str]]
) -> list[str]:
    """State what the measured file counts do and do not mean."""
    parts: list[str] = []
    if any(_mapping(inputs.get(role)).get("member_count") is not None for role, _ in described):
        parts.append(_paragraph("Every file under each root is measured, not only the entrypoint."))
    if inputs.get("roots_correspond") is False:
        parts.append(
            _paragraph("The two roots differ: the measured file counts are not a change list.")
        )
    parts.append(
        _paragraph(
            "The measured file inventory itself is not reproduced here; it stays in the canonical "
            "receipt."
        )
    )
    return parts


def _grouped(
    items: Sequence[CanonicalValue], member: str, headings: Sequence[tuple[str, str]], other: str
) -> list[tuple[str, list[Mapping[str, CanonicalValue]]]]:
    """Group document rows under their known headings, keeping any unknown row in a final group.

    Args:
        items: The retained rows, in the document's own order.
        member: The row member that carries the group key.
        headings: The known group keys with their headings, in presentation order.
        other: The heading for rows whose key matches no known group.

    Returns:
        One entry per non-empty group, headings first and unmatched rows last.
    """
    known = {key for key, _ in headings}
    keyed = [(_group_key(_mapping(item), member), _mapping(item)) for item in items]
    selected = [
        (heading, [row for row_key, row in keyed if row_key == key]) for key, heading in headings
    ]
    leftover = [row for row_key, row in keyed if row_key not in known]
    if leftover:
        selected.append((other, leftover))
    return [entry for entry in selected if entry[1]]


def _group_key(row: Mapping[str, CanonicalValue], member: str) -> str:
    """Return one row's group key, or the empty string when it carries no textual key."""
    key = row.get(member)
    return key if isinstance(key, str) else ""


def _findings_section(document: Mapping[str, CanonicalValue]) -> list[str]:
    """Render every retained finding, grouped by kind and in the document's own order."""
    findings = _entries(document.get("findings"))
    parts = ["<section>", "<h2>Findings</h2>"]
    recorded = document.get("finding_count")
    if recorded is not None:
        parts.append(
            f'<p class="muted">Findings recorded by the producing run: {_escape(recorded)}.</p>'
        )
    if not findings:
        parts.extend([_paragraph("No findings are retained in this document."), "</section>"])
        return parts
    for heading, rows in _grouped(findings, "kind", _KIND_HEADING, "Other findings"):
        parts.append(f"<h3>{html.escape(heading, quote=True)} ({len(rows)})</h3>")
        parts.append(_table([_CHANGE_HEADER, *(_finding_row(row) for row in rows)]))
    parts.append("</section>")
    return parts


def _finding_row(finding: Mapping[str, CanonicalValue]) -> str:
    """Render one finding row: its display text, where it lives, and the witnesses it retained."""
    pointers = [("receipt row", finding.get("source"))]
    descriptive_source = finding.get("descriptive_source")
    if descriptive_source is not None:
        pointers.append(("descriptive row", descriptive_source))
    located = " · ".join(f"{label} <code>{_escape(value)}</code>" for label, value in pointers)
    witnesses = _witness_details(finding) or _NO_WITNESS
    return (
        "<tr>"
        f"<td>{_paragraph(_optional(finding.get('display')))}"
        f'<p class="pointer">{located}</p></td>'
        f"<td>{witnesses}</td>"
        "</tr>"
    )


def _witness_details(finding: Mapping[str, CanonicalValue]) -> str:
    """Render every witness a finding retained, inside one collapsed element.

    A joined compiled-field row carries its witnesses in the descriptive row attached to it; a
    certification row carries them in its own retained data.  Both are reached here, so no
    retained witness is dropped from the page.

    Args:
        finding: One retained finding.

    Returns:
        The collapsed witness element, or the empty string when the finding retained none.
    """
    descriptive = finding.get("descriptive")
    block = _mapping(descriptive if isinstance(descriptive, dict) else finding.get("data"))
    witnesses = _entries(block.get("witnesses"))
    if not witnesses:
        return ""
    return (
        "<details>"
        f"<summary>{len(witnesses)} retained witness row(s)</summary>"
        f"{_witness_table(witnesses)}"
        '<p class="muted">The index is the producer\'s own array index. It is not resolved to a '
        "named object here.</p>"
        "</details>"
    )


def _witness_table(witnesses: Sequence[CanonicalValue]) -> str:
    """Render the retained witness rows as they were retained, one row each."""
    rows = [
        '<tr><th scope="col">index</th><th scope="col">baseline</th>'
        '<th scope="col">candidate</th></tr>'
    ]
    for witness in witnesses:
        if not isinstance(witness, dict):
            rows.append(f'<tr><td colspan="3">{_escape(value_text(witness))}</td></tr>')
            continue
        rows.append(
            f"<tr><td><code>{_escape(value_text(witness.get('index')))}</code></td>"
            f"<td>{_witness_cell(witness, 'baseline')}</td>"
            f"<td>{_witness_cell(witness, 'candidate')}</td></tr>"
        )
    return _table(rows)


def _witness_cell(witness: Mapping[str, CanonicalValue], side: str) -> str:
    """Render one side of a witness, keeping the producer's own text when the two disagree."""
    decoded = value_text(witness.get(f"{side}_value"))
    retained = witness.get(f"{side}_text")
    if isinstance(retained, str) and retained != decoded:
        return f'{_escape(decoded)} <span class="retained">(retained {_escape(retained)})</span>'
    return _escape(decoded)


def _limitations_section(document: Mapping[str, CanonicalValue]) -> list[str]:
    """Render every limitation, keeping each category separate."""
    limitations = _entries(document.get("limitations"))
    parts = ["<section>", "<h2>Limitations</h2>"]
    if not limitations:
        parts.extend([_paragraph("No limitations are retained in this document."), "</section>"])
        return parts
    for heading, rows in _grouped(limitations, "category", _CATEGORY_HEADING, "Other limitations"):
        parts.append(f"<h3>{html.escape(heading, quote=True)}</h3>")
        parts.append("<ul>")
        parts.extend(_limitation_item(row) for row in rows)
        parts.append("</ul>")
    parts.append("</section>")
    return parts


def _limitation_item(limitation: Mapping[str, CanonicalValue]) -> str:
    """Render one limitation: its message, where it lives, and any detail it recorded."""
    return (
        "<li>"
        f"{_paragraph(_optional(limitation.get('message')))}"
        f'<p class="pointer">receipt row <code>{_optional(limitation.get("source"))}</code></p>'
        f"{_detail_details(limitation.get('detail'))}"
        "</li>"
    )


def _detail_details(detail: CanonicalValue) -> str:
    """Render one limitation's recorded detail inside a collapsed element."""
    if not isinstance(detail, dict) or not detail:
        return ""
    rows = [_row(name, _detail_value(detail[name])) for name in sorted(detail)]
    return f"<details><summary>Recorded detail</summary>{_table(rows)}</details>"


def _detail_value(value: CanonicalValue) -> str:
    """Render one recorded detail value, listing a retained sequence element by element."""
    if isinstance(value, list):
        items = "".join(f"<li>{_escape(value_text(item))}</li>" for item in value)
        return f"<ul>{items}</ul>" if items else _escape(value_text(value))
    return _escape(value_text(value))


def _artifacts_section(document: Mapping[str, CanonicalValue]) -> list[str]:
    """Render every known artifact path, linking only the two published local file names."""
    artifacts = _mapping(document.get("artifacts"))
    rows = [
        _row(label, _artifact_cell(name, artifacts.get(name)))
        for name, label in _ARTIFACT_LABEL
        # This page is rendered before it is published, so its own path is not yet known. Telling
        # the reader that the report they are looking at is "not recorded" would only confuse.
        if not (name == "html" and artifacts.get(name) is None)
    ]
    return [
        "<section>",
        "<h2>Artifacts</h2>",
        _table(rows),
        _paragraph(
            "The complete record — every retained bit, the full measured file inventory and every "
            "omitted member — stays in the canonical receipt, not on this page. A link is offered "
            "only when the file sits beside this report under the name the writer published."
        ),
        "</section>",
    ]


def _artifact_cell(name: str, value: CanonicalValue) -> str:
    """Render one artifact path, as a relative link only when its file name is a published one.

    The href is never a document value: it is one of the two constant names the paired output
    writer publishes, chosen by matching the path's final component against them.

    Args:
        name: The artifacts member being rendered.
        value: The recorded path, or None when the document recorded none.

    Returns:
        The escaped cell markup for that artifact.
    """
    if not isinstance(value, str):
        return _ABSENT
    local = Path(value).name
    if name in _LINKED_ARTIFACTS and local in _PUBLISHED_NAMES:
        published = html.escape(local, quote=True)
        return f'<a href="{published}">{published}</a> <span class="muted">{_escape(value)}</span>'
    return _escape(value)


def _boundary_section(document: Mapping[str, CanonicalValue]) -> list[str]:
    """State the recorded runtime boundary and what the comparison does not claim."""
    producer = _mapping(document.get("runtime")).get("producer")
    if isinstance(producer, dict):
        boundary = (
            f"Static comparison under MuJoCo {_optional(producer.get('mujoco'))} · "
            f"{_optional(producer.get('python_implementation'))} "
            f"{_optional(producer.get('python'))} · "
            f"{_optional(producer.get('platform_system'))} "
            f"{_optional(producer.get('platform_machine'))}."
        )
    else:
        boundary = (
            "No producer runtime is recorded in this document, so nothing here is a static "
            "comparison under a recorded runtime."
        )
    return [
        '<section class="boundary">',
        _paragraph(boundary),
        _paragraph("Not a statement about behaviour, safety, or approval."),
        "</section>",
    ]


def _page(body: str) -> str:
    """Wrap the rendered sections in one standalone offline page with local styles only."""
    policy = "default-src 'none'; style-src 'unsafe-inline'"
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f'<meta http-equiv="Content-Security-Policy" content="{policy}">\n'
        f"<title>{html.escape(_TITLE, quote=True)}</title>\n"
        f"<style>{_STYLE}</style>\n"
        "</head>\n"
        "<body>\n"
        "<main>\n"
        f"{body}\n"
        "</main>\n"
        "</body>\n"
        "</html>\n"
    )


def render_report(document: Mapping[str, CanonicalValue]) -> str:
    """Render one result document as a complete standalone offline HTML page.

    The page presents the same facts as the readable text view, in the same order, from the same
    document members.  Nothing is decided, inferred or ranked here: a member the document does not
    carry is named as unrecorded, and every retained finding, witness and limitation the document
    carries is reachable on the page.

    Args:
        document: The complete result document, from either ``show`` or ``diff``.

    Returns:
        The complete HTML document as a string, ending in a newline.
    """
    parts: list[str] = []
    parts.extend(_headline_section(document))
    parts.extend(_evaluation_section(document))
    parts.extend(_problem_section(document))
    parts.extend(_runtime_section(document))
    parts.extend(_compared_section(document))
    parts.extend(_findings_section(document))
    parts.extend(_limitations_section(document))
    parts.extend(_artifacts_section(document))
    parts.extend(_boundary_section(document))
    return _page("\n".join(parts))


__all__ = ["render_report"]
