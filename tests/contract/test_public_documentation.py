"""The public documents must keep leading with the change the reader actually made.

The documentation was reorganized around two primary routes — a model or asset changed, or the
MuJoCo runtime changed — because a reader arriving with a broken pull request should not have to
choose between seven equally weighted commands first. That structure is easy to erode: a later edit
reorders a section, renames a heading, drops one of the two exact commands, or reintroduces the
stale claim that the current source answers only three questions.

This is a bounded structural contract. It asserts headings, exact commands, required semantic
concepts, and link targets, and it deliberately does not assert whole prose blocks, so the wording
stays free to improve. It reads files and resolves relative targets on disk; it never uses the
network.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import pytest_check as check

_FENCE = re.compile(r"^\s*```")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")
_RELATIVE_LINK = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
_URL = re.compile(r"https://[^\s)>\]]+")

_CERTIFY_COMMAND = "metrifid certify old/robot.xml new/robot.xml --output out/"
_RUNTIME_COMMAND = "metrifid run-runtime-review runtime_review_run.json"
_GALLERY_COMMAND = "python examples/public_cases/run_all.py --output /absolute/absent/output"

_ACCEPTED_COMMANDS = (
    "certify",
    "review-model",
    "compare",
    "qualify-workload",
    "audit-timestep",
    "review-runtime",
    "run-runtime-review",
)

_EVIDENCE_COMMANDS = (
    "review-model",
    "compare",
    "qualify-workload",
    "audit-timestep",
    "review-runtime",
)

_README_HEADINGS = (
    "# Metrifid",
    "## Start with the change you made",
    "### Your model or asset changed",
    "### Your MuJoCo runtime changed",
    "## Follow the evidence",
)

_GETTING_STARTED_HEADINGS = (
    "# Getting started",
    "## Choose your starting point",
    "### A model or asset changed",
    "### The MuJoCo runtime changed",
    "## Continue from the first result",
    "## Run the complaint-backed public cases",
    "## Documentation map",
)

_CAPABILITIES_HEADINGS = (
    "# Capabilities and use cases",
    "## The accepted command surface",
    "## What each command decides",
    "## What Metrifid does not decide",
)

_PUBLIC_CASE_HEADINGS = (
    "# Complaint-backed public cases",
    "## What these cases are",
    "## Run all six cases",
    "## Collision filtering",
    "## Mesh inertia",
    "## Actuator transmission",
    "## Sensor attachment",
    "## External Robosuite Jaco study",
    "## How to interpret the decisions",
    "## Claim boundaries",
)

# The two routes must be reachable before any seven-equal-choice menu. A reader who meets the full
# command table first has already lost the benefit of the reorganization.


_STATUS_TOKEN = re.compile(r"\b(?:NOT_)?REPRODUCED_ON_CURRENT_RUNTIME\b|\breproduction_status\b")
# "reproducible" describes deterministic bytes, not a claim about upstream behavior, so the
# claim words are matched exactly rather than by prefix.
_REPRODUCTION = re.compile(r"\breproduc(?:e|es|ed|ing|tion|tions)\b", re.IGNORECASE)
_NEGATION = re.compile(
    r"\b(?:not|no|never|nor|without|rather than|instead of|neither)\b", re.IGNORECASE
)


def _unnegated_reproduction_claims(flat_text: str) -> list[str]:
    """Return every mention of reproduction that is not denied by a nearby negation.

    The external study reports a status vocabulary that contains the word, so those exact tokens are
    removed before the prose is inspected.
    """
    prose = _STATUS_TOKEN.sub(" ", flat_text)
    found: list[str] = []
    for match in _REPRODUCTION.finditer(prose):
        window = prose[max(0, match.start() - 70) : match.start()]
        if _NEGATION.search(window) is None:
            found.append(prose[max(0, match.start() - 70) : match.end() + 30].strip())
    return found


def _repository_root() -> Path:
    """Return the source checkout that holds the public documents."""
    return Path(__file__).resolve().parents[2]


def _document(name: str) -> Path:
    """Return one public document path under the repository root."""
    return _repository_root() / name


_DOCUMENTS = (
    "README.md",
    "docs/getting_started.md",
    "docs/capabilities_and_use_cases.md",
    "docs/public_cases.md",
)


def _lines_outside_fences(text: str) -> list[tuple[int, str]]:
    """Return numbered lines that are not inside a fenced code block.

    Shell examples in these documents begin with ``#`` comments, which are indistinguishable from
    Markdown headings unless fenced regions are removed first.
    """
    outside: list[tuple[int, str]] = []
    inside = False
    for number, line in enumerate(text.splitlines(), start=1):
        if _FENCE.match(line):
            inside = not inside
            continue
        if not inside:
            outside.append((number, line))
    return outside


def _flattened(text: str) -> str:
    """Return the text with emphasis markers dropped and every whitespace run collapsed.

    Required phrases are asserted as concepts, not as bytes. Prose wraps across lines and picks up
    emphasis markers as it is edited, and neither should break a semantic check.
    """
    return re.sub(r"\s+", " ", text.replace("**", "").replace("*", "").replace("`", ""))


def _headings(path: Path) -> list[str]:
    """Return every Markdown heading in a document, in document order."""
    headings: list[str] = []
    for _, line in _lines_outside_fences(path.read_text(encoding="utf-8")):
        match = _HEADING.match(line)
        if match is not None:
            headings.append(f"{match.group(1)} {match.group(2)}")
    return headings


def _section_body(path: Path, heading: str) -> str:
    """Return the text under one heading, up to the next heading of the same or higher level."""
    text = path.read_text(encoding="utf-8")
    level = len(heading.split(" ", 1)[0])
    lines = text.splitlines()
    start: int | None = None
    for index, line in enumerate(lines):
        match = _HEADING.match(line)
        if match is not None and f"{match.group(1)} {match.group(2)}" == heading:
            start = index + 1
            break
    assert start is not None, f"{path.name} has no heading {heading!r}"
    body: list[str] = []
    for line in lines[start:]:
        match = _HEADING.match(line)
        if match is not None and len(match.group(1)) <= level:
            break
        body.append(line)
    return "\n".join(body)


def _assert_headings_in_order(path: Path, required: tuple[str, ...]) -> None:
    """Require every heading to be present exactly once and in the required relative order."""
    headings = _headings(path)
    positions: list[int] = []
    for heading in required:
        occurrences = [index for index, found in enumerate(headings) if found == heading]
        check.equal(
            len(occurrences),
            1,
            f"{path.name} must contain the heading {heading!r} exactly once, found "
            f"{len(occurrences)}",
        )
        if occurrences:
            positions.append(occurrences[0])
    check.equal(
        positions,
        sorted(positions),
        f"{path.name} headings are out of order: {required}",
    )


@pytest.mark.parametrize("name", _DOCUMENTS)
def test_every_required_public_document_exists(name: str) -> None:
    """A missing document breaks every link that leads a reader to it."""
    assert _document(name).is_file(), f"{name} is missing"


def test_readme_leads_with_the_two_primary_routes() -> None:
    """The README must present both starting points before the detailed command reference."""
    path = _document("README.md")
    _assert_headings_in_order(path, _README_HEADINGS)

    model_route = _section_body(path, "### Your model or asset changed")
    check.is_in(
        _CERTIFY_COMMAND,
        model_route,
        "the model route must carry the exact certify command",
    )
    runtime_route = _section_body(path, "### Your MuJoCo runtime changed")
    check.is_in(
        _RUNTIME_COMMAND,
        runtime_route,
        "the runtime route must carry the exact runtime review command",
    )

    evidence = _section_body(path, "## Follow the evidence")
    for command in _EVIDENCE_COMMANDS:
        check.is_in(command, evidence, f"the evidence section must name {command}")


def test_readme_states_the_required_claim_boundaries() -> None:
    """The three concepts a reader most often gets wrong must be stated somewhere in the README."""
    text = _flattened(_document("README.md").read_text(encoding="utf-8"))
    for phrase in (
        "completed exit 20, 30, or 40 can be a referee decision",
        "does not discover, install, or repair MuJoCo environments",
        "Certify makes no behavioral-equivalence claim",
    ):
        check.is_in(phrase, text, f"the README must state: {phrase}")


def test_no_seven_choice_menu_precedes_the_two_routes() -> None:
    """A full command menu placed first would undo the point of the reorganization."""
    text = _document("README.md").read_text(encoding="utf-8")
    routes_end = text.index("## Follow the evidence")
    preamble = text[:routes_end]
    named = [command for command in _ACCEPTED_COMMANDS if command in preamble]
    check.less(
        len(named),
        len(_ACCEPTED_COMMANDS),
        "the README names all seven commands before both primary routes are presented",
    )


def test_getting_started_routes_the_reader_from_the_first_result() -> None:
    """The guide must offer both starting points and map each follow-up question to one command."""
    path = _document("docs/getting_started.md")
    _assert_headings_in_order(path, _GETTING_STARTED_HEADINGS)

    check.is_in(_CERTIFY_COMMAND, _section_body(path, "### A model or asset changed"))
    check.is_in(_RUNTIME_COMMAND, _section_body(path, "### The MuJoCo runtime changed"))

    continuation = _section_body(path, "## Continue from the first result")
    for concept, command in (
        ("compiled difference", "review-model"),
        ("behavioral consequence", "compare"),
        ("workload detectability", "qualify-workload"),
        ("timestep", "audit-timestep"),
        ("native evidence", "review-runtime"),
    ):
        check.is_in(concept, continuation, f"the continuation map must name {concept!r}")
        check.is_in(command, continuation, f"the continuation map must route to {command}")

    gallery = _section_body(path, "## Run the complaint-backed public cases")
    check.is_in(_GALLERY_COMMAND, gallery, "the guide must carry the exact gallery command")


def test_capabilities_describes_the_accepted_seven_command_surface() -> None:
    """The capabilities document must describe the real surface, not a stale three-question one."""
    path = _document("docs/capabilities_and_use_cases.md")
    _assert_headings_in_order(path, _CAPABILITIES_HEADINGS)

    surface = _section_body(path, "## The accepted command surface")
    for command in _ACCEPTED_COMMANDS:
        check.is_in(command, surface, f"the accepted surface must name {command}")


def test_capabilities_names_the_prior_release_only_as_the_prior_release() -> None:
    """Naming any released version as the current source implementation misleads every reader."""
    path = _document("docs/capabilities_and_use_cases.md")
    text = path.read_text(encoding="utf-8")
    flat = _flattened(text)
    check.is_true(
        re.search(r"answer(?:ing|s)? three different questions", flat) is None,
        "the capabilities document must not claim the product answers only three questions",
    )
    # The document must not present any released version as what the current source implements.
    stale_current = re.compile(
        r"(?:current|this)\s+(?:source|repository|implementation|candidate|release)"
        r"[^.]{0,80}\b\d+\.\d+\.\d+\b"
        r"|\b\d+\.\d+\.\d+\b"
        r"[^.]{0,80}(?:is|as)\s+the\s+current\s+(?:source|implementation|candidate)",
        re.IGNORECASE,
    )
    check.is_true(
        stale_current.search(flat) is None,
        "the capabilities document presents a released version as the current source implementation",
    )


def test_no_public_document_calls_a_completed_decision_a_crash() -> None:
    """A non-green Runtime Review outcome is a decision; describing it as a crash is untrue."""
    forbidden = re.compile(
        r"(?:exit\s+(?:20|30|40)[^.\n]{0,40}(?:is|means)\s+a\s+crash)"
        r"|(?:crash(?:es|ed)?\s+(?:on|with)\s+exit\s+(?:20|30|40))",
        re.IGNORECASE,
    )
    for name in _DOCUMENTS:
        text = _document(name).read_text(encoding="utf-8")
        check.is_true(
            forbidden.search(text) is None,
            f"{name} describes a completed non-green decision as a crash",
        )


def test_public_case_document_has_the_required_structure() -> None:
    """Each case family and the external study need their own named section."""
    path = _document("docs/public_cases.md")
    _assert_headings_in_order(path, _PUBLIC_CASE_HEADINGS)
    check.is_in(
        _GALLERY_COMMAND,
        _section_body(path, "## Run all six cases"),
        "the public case document must carry the exact gallery command",
    )


def test_public_case_document_states_the_analogue_boundary() -> None:
    """Presenting an owned analogue as an upstream reproduction would be a false claim."""
    text = _flattened(_document("docs/public_cases.md").read_text(encoding="utf-8"))
    check.is_in(
        "mechanism analogue",
        text,
        "the public case document must use the phrase 'mechanism analogue'",
    )
    check.is_true(
        "not upstream reproductions" in text or "not copies of the upstream assets" in text,
        "the public case document must state the cases are not upstream reproductions",
    )
    # Stating the boundary once elsewhere does not license an affirmative reproduction claim in
    # another paragraph, so every mention of reproduction must be a denial.
    for unnegated in _unnegated_reproduction_claims(text):
        check.fail(
            f"the public case document claims a reproduction: {unnegated!r}",
        )


def test_public_case_document_links_every_case_origin() -> None:
    """Every case must be traceable to the exact public record its mechanism came from."""
    root = _repository_root()
    gallery = root / "examples" / "public_cases"
    text = _document("docs/public_cases.md").read_text(encoding="utf-8")
    origins = sorted(gallery.rglob("ORIGIN.md"))
    assert origins, "no case origin records were found"
    urls = set()
    for origin in origins:
        for url in _URL.findall(origin.read_text(encoding="utf-8")):
            urls.add(url.rstrip(".,;:"))
    assert urls, "no origin URLs were found in the case records"
    for url in sorted(urls):
        check.is_in(url, text, f"the public case document must link the origin {url}")


def test_every_case_directory_named_in_the_document_resolves() -> None:
    """A link to a case that no longer exists sends a reader nowhere."""
    root = _repository_root()
    path = _document("docs/public_cases.md")
    for target in _RELATIVE_LINK.findall(path.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        resolved = (path.parent / target.split("#", 1)[0]).resolve()
        check.is_true(
            resolved.exists(),
            f"docs/public_cases.md links {target!r}, which does not resolve",
        )
        check.is_true(
            root in resolved.parents or resolved == root,
            f"docs/public_cases.md links {target!r}, which escapes the repository",
        )


@pytest.mark.parametrize("name", _DOCUMENTS)
def test_every_relative_markdown_link_resolves(name: str) -> None:
    """A relative link must reach a real file, or a real anchor inside a real file."""
    root = _repository_root()
    path = _document(name)
    text = path.read_text(encoding="utf-8")
    for target in _RELATIVE_LINK.findall(text):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        if target.startswith("#"):
            check.is_true(
                _anchor_exists(path, target[1:]),
                f"{name} links the anchor {target!r}, which has no matching heading",
            )
            continue
        file_part, _, anchor = target.partition("#")
        resolved = (path.parent / file_part).resolve()
        check.is_true(
            resolved.exists(),
            f"{name} links {target!r}, which does not resolve",
        )
        if anchor and resolved.is_file() and resolved.suffix == ".md":
            check.is_true(
                _anchor_exists(resolved, anchor),
                f"{name} links {target!r}, whose anchor has no matching heading",
            )
        check.is_true(
            root in resolved.parents or resolved == root,
            f"{name} links {target!r}, which escapes the repository",
        )


def _anchor_exists(path: Path, anchor: str) -> bool:
    """Return whether one GitHub-style heading anchor exists in a document."""
    wanted = anchor.strip().lower()
    for heading in _headings(path):
        title = heading.split(" ", 1)[1] if " " in heading else ""
        slug = re.sub(r"[^\w\- ]", "", title.lower()).strip().replace(" ", "-")
        if slug == wanted:
            return True
    return False
