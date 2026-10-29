"""Keep public entry points, provenance, links and claim boundaries discoverable.

These checks cover command examples, supported API names, analogue attribution and link targets.
They leave prose free to improve and inspect local documents without using the network.
"""

from __future__ import annotations

import re
from collections.abc import Callable
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

_CORE_COMMANDS = (
    "certify",
    "review-model",
    "compare",
    "qualify-workload",
    "audit-timestep",
    "review-runtime",
    "run-runtime-review",
)
_ADDITIONAL_ROUTES = ("diff", "show", "repair run", "repair show", "repair replay")
_ACCEPTED_COMMANDS = (*_CORE_COMMANDS, *_ADDITIONAL_ROUTES)

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
    "### Your planar controller misses its target",
    "## Follow the evidence",
)

_GETTING_STARTED_HEADINGS = (
    "# Getting started",
    "## Choose your starting point",
    "### A model or asset changed",
    "### The MuJoCo runtime changed",
    "### A planar pushing controller misses its target",
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
    "## How to interpret the decisions",
    "## Claim boundaries",
)

# The starting routes must be reachable before the full expert command menu.


# "reproducible" describes deterministic bytes, not a claim about upstream behavior, so the
# claim words are matched exactly rather than by prefix.
_REPRODUCTION = re.compile(r"\breproduc(?:e|es|ed|ing|tion|tions)\b", re.IGNORECASE)
_NEGATION = re.compile(
    r"\b(?:not|no|never|nor|without|rather than|instead of|neither)\b", re.IGNORECASE
)


def _unnegated_reproduction_claims(flat_text: str) -> list[str]:
    """Return every mention of reproduction that is not denied by a nearby negation."""
    found: list[str] = []
    for match in _REPRODUCTION.finditer(flat_text):
        window = flat_text[max(0, match.start() - 70) : match.start()]
        if _NEGATION.search(window) is None:
            found.append(flat_text[max(0, match.start() - 70) : match.end() + 30].strip())
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


def test_readme_leads_with_the_three_starting_routes() -> None:
    """The README must present model, runtime and repair routes before the command reference."""
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

    repair_route = _section_body(path, "### Your planar controller misses its target")
    check.is_in("metrifid repair --help", repair_route)
    check.is_in("planar_push_timed15_v1", repair_route)

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


def test_no_full_core_command_menu_precedes_the_starting_routes() -> None:
    """A full command menu placed first would undo the point of the reorganization."""
    text = _document("README.md").read_text(encoding="utf-8")
    routes_end = text.index("## Follow the evidence")
    preamble = text[:routes_end]
    named = [command for command in _CORE_COMMANDS if command in preamble]
    check.less(
        len(named),
        len(_CORE_COMMANDS),
        "the README names all core commands before the starting routes are presented",
    )


def test_getting_started_routes_the_reader_from_the_first_result() -> None:
    """The guide must offer all three routes and map follow-up questions to commands."""
    path = _document("docs/getting_started.md")
    _assert_headings_in_order(path, _GETTING_STARTED_HEADINGS)

    check.is_in(_CERTIFY_COMMAND, _section_body(path, "### A model or asset changed"))
    check.is_in(_RUNTIME_COMMAND, _section_body(path, "### The MuJoCo runtime changed"))
    repair_route = _section_body(path, "### A planar pushing controller misses its target")
    check.is_in("metrifid repair --help", repair_route)
    check.is_in("planar_push_timed15_v1", repair_route)

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


def test_capabilities_describes_core_commands_and_the_separate_repair_routes() -> None:
    """Core expert commands, model conveniences and planar repair must all be discoverable."""
    path = _document("docs/capabilities_and_use_cases.md")
    _assert_headings_in_order(path, _CAPABILITIES_HEADINGS)

    surface = _section_body(path, "## The accepted command surface")
    for command in _CORE_COMMANDS:
        check.is_in(f"`{command}`", surface, f"the core surface must name {command}")
    for command in _ADDITIONAL_ROUTES:
        check.is_in(f"metrifid {command}", surface, f"the accepted surface must name {command}")


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
    """The six owned cases need discoverable mechanism sections and a runnable gallery command."""
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


# ---- The reference and SDK documents must cover the whole public surface ----------------------------
#
# Command examples and APIs must retain every supported route. Focused controls remove a real
# documented command, schema row or function section and reuse the corresponding assertion.

_RUNTIME_REVIEW_FUNCTIONS = (
    "review_runtime_configuration_file",
    "run_runtime_review_configuration_file",
)
# Each completed decision, bound to the exit its own table row must carry.
_RUNTIME_REVIEW_DECISIONS = (
    ("WITHIN_DECLARED_MIGRATION_ENVELOPE", "0"),
    ("INSUFFICIENT_EVIDENCE", "20"),
    ("UNRESOLVED_NEAR_BOUNDARY", "30"),
    ("OUTSIDE_DECLARED_MIGRATION_ENVELOPE", "40"),
)
_TIMESTEP_CLASSIFICATIONS = (
    "WITHIN_DECLARED_TOLERANCE",
    "OUTSIDE_DECLARED_TOLERANCE",
    "INCONCLUSIVE",
    "REFUSED",
)
# Every Runtime Review schema, with the versions its row must name.
_RUNTIME_REVIEW_SCHEMAS = (
    ("metrifid.runtime_review_receipt", ("1", "2")),
    ("metrifid.runtime_review_config", ("1", "2")),
    ("metrifid.runtime_review_run_config", ("1",)),
    ("metrifid.runtime_review_run", ("1", "2")),
    ("metrifid.runtime_review.native_profile_identity", ("1", "2")),
    ("metrifid.runtime_review.integration_state_sentinel", ("1",)),
    ("metrifid.runtime_review_process_command", ("1",)),
)
_STALE_SURFACE_COUNTS = ("all five operations", "five execution entry points")
_TABLE_ROW = re.compile(r"(?m)^\|.*\|\s*$")


def _fenced_blocks_after(text: str, heading: str) -> str:
    """Return the fenced examples within one section, excluding surrounding prose."""
    section = _section_after(text, heading)
    fences = re.finditer(
        r"(?ms)^(?P<fence>```|~~~)[^\n]*\n(?P<body>.*?)^(?P=fence)[ \t]*$", section
    )
    return "\n".join(fence.group("body") for fence in fences)


def _section_after(text: str, heading: str) -> str:
    """Return the text from `heading` up to the next heading of the same or higher level."""
    start = text.find(heading)
    if start < 0:
        return ""
    level = len(heading) - len(heading.lstrip("#"))
    following = re.search(rf"(?m)^#{{1,{level}}} ", text[start + len(heading) :])
    return (
        text[start:]
        if following is None
        else text[start : start + len(heading) + following.start()]
    )


def command_surface_problems(text: str) -> list[str]:
    """Every accepted route must appear in a Commands example, not merely in later prose."""
    block = _fenced_blocks_after(text, "## Commands")
    if not block:
        return ["the Commands section has no fenced command block"]
    return [
        f"the Commands block does not show `metrifid {command}`"
        for command in _ACCEPTED_COMMANDS
        if f"metrifid {command}" not in block
    ]


def runtime_review_decision_problems(text: str) -> list[str]:
    """Each completed decision must be bound to its own exit inside one table row."""
    rows = _TABLE_ROW.findall(text)
    problems: list[str] = []
    for status, exit_code in _RUNTIME_REVIEW_DECISIONS:
        carrying = [row for row in rows if status in row]
        if not carrying:
            problems.append(f"no table row documents the {status} decision")
        elif not any(f"| `{exit_code}` |" in row for row in carrying):
            problems.append(f"the {status} row does not bind exit `{exit_code}`")
    return problems


def timestep_classification_problems(text: str) -> list[str]:
    """The emitted classification strings belong in the Statuses section."""
    section = _section_after(text, "## Statuses")
    if not section:
        return ["the reference has no Statuses section"]
    return [
        f"the Statuses section omits the {classification} timestep classification"
        for classification in _TIMESTEP_CLASSIFICATIONS
        if classification not in section
    ]


def schema_row_problems(text: str) -> list[str]:
    """Each Runtime Review schema needs a row naming it and every version it publishes."""
    rows = _TABLE_ROW.findall(text)
    problems: list[str] = []
    for identifier, versions in _RUNTIME_REVIEW_SCHEMAS:
        carrying = [row for row in rows if f"`{identifier}`" in row]
        if not carrying:
            problems.append(f"no schema row names {identifier}")
            continue
        row = carrying[0]
        problems.extend(
            f"the {identifier} row does not name version `{version}`"
            for version in versions
            if f"`{version}`" not in row
        )
    return problems


def runtime_review_api_problems(text: str, headings: tuple[str, ...]) -> list[str]:
    """Both execution functions must appear across the sections documenting the Runtime Review API.

    The reference documents both under one module heading; the SDK gives each its own section, so the
    requirement is over the union of the named sections rather than over each one.
    """
    problems: list[str] = []
    sections: list[str] = []
    for heading in headings:
        section = _section_after(text, heading)
        if not section:
            problems.append(f"the document has no {heading!r} section")
        else:
            sections.append(section)
    combined = "\n".join(sections)
    problems.extend(
        f"the Runtime Review API sections do not document {function}"
        for function in _RUNTIME_REVIEW_FUNCTIONS
        if function not in combined
    )
    return problems


def stale_surface_count_problems(text: str) -> list[str]:
    """Active prose must not describe the execution surface as five operations."""
    flat = _flattened(text).lower()
    return [
        f"the document still describes the surface as {stale!r}"
        for stale in _STALE_SURFACE_COUNTS
        if stale in flat
    ]


def _reference() -> str:
    return _document("docs/reference.md").read_text(encoding="utf-8")


def _sdk() -> str:
    return _document("docs/sdk.md").read_text(encoding="utf-8")


def test_the_reference_command_block_lists_every_accepted_command() -> None:
    """The Commands examples cover core, convenience and repair routes."""
    assert command_surface_problems(_reference()) == []


def test_the_reference_binds_every_runtime_review_decision_to_its_exit() -> None:
    """A swapped status/exit pairing must not pass."""
    assert runtime_review_decision_problems(_reference()) == []


def test_the_reference_states_the_exact_timestep_classifications() -> None:
    """The emitted values are the long forms, not the constant names."""
    assert timestep_classification_problems(_reference()) == []


def test_the_reference_documents_every_runtime_review_schema_version() -> None:
    """A published document is only discoverable if its schema and versions are listed."""
    assert schema_row_problems(_reference()) == []


_REFERENCE_API_HEADINGS = ("### `metrifid.runtime_review`",)
_SDK_API_HEADINGS = (
    "## `run_runtime_review_configuration_file`",
    "## `review_runtime_configuration_file`",
)


def test_both_documents_expose_runtime_review_and_the_separate_repair_api() -> None:
    """The SDK guide must retain core APIs and direct repair callers to its own contract."""
    assert runtime_review_api_problems(_reference(), _REFERENCE_API_HEADINGS) == []
    assert runtime_review_api_problems(_sdk(), _SDK_API_HEADINGS) == []
    repair = _section_after(_sdk(), "## Planar controller repair")
    for function in (
        "load_request",
        "repair_controller",
        "replay_controller",
        "run_request_file",
        "read_result_file",
        "replay_run_directory",
    ):
        check.is_in(function, repair, f"the separate repair SDK section must name {function}")
        check.is_in(function, _reference(), f"the reference must name {function}")
    check.is_in("repair.md#python-file-workflow", _RELATIVE_LINK.findall(repair))


@pytest.mark.parametrize(
    "name", ["README.md", "docs/getting_started.md", "docs/sdk.md", "docs/reference.md"]
)
def test_no_active_document_counts_the_execution_surface_at_five(name: str) -> None:
    """The core APIs include both Runtime Review routes; a document counting five is stale."""
    assert stale_surface_count_problems(_document(name).read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    ("mutate", "helper", "expected"),
    [
        pytest.param(
            lambda text: text.replace(
                "metrifid run-runtime-review runtime_review_run.json\n", "", 1
            ),
            command_surface_problems,
            "does not show `metrifid run-runtime-review`",
            id="command_removed_from_the_commands_block",
        ),
        pytest.param(
            lambda text: text.replace(
                "metrifid repair replay RUN_DIRECTORY --output NEW_DIRECTORY [--json]\n", "", 1
            ),
            command_surface_problems,
            "does not show `metrifid repair replay`",
            id="repair_replay_removed_from_the_commands_section",
        ),
        pytest.param(
            lambda text: text.replace(
                "| `20` | `review-runtime`, `run-runtime-review` | `INSUFFICIENT_EVIDENCE`",
                "| `30` | `review-runtime`, `run-runtime-review` | `INSUFFICIENT_EVIDENCE`",
                1,
            ),
            runtime_review_decision_problems,
            "INSUFFICIENT_EVIDENCE row does not bind exit `20`",
            id="decision_bound_to_the_wrong_exit",
        ),
        pytest.param(
            lambda text: text.replace("WITHIN_DECLARED_TOLERANCE", "WITHIN", 1),
            timestep_classification_problems,
            "omits the WITHIN_DECLARED_TOLERANCE",
            id="timestep_classification_reduced_to_the_constant_name",
        ),
        pytest.param(
            lambda text: text.replace(
                "| `metrifid.runtime_review_run`, versions `1` and `2` |",
                "| `metrifid.runtime_review_run`, version `1` |",
                1,
            ),
            schema_row_problems,
            "runtime_review_run row does not name version `2`",
            id="schema_row_loses_a_version",
        ),
        pytest.param(
            lambda text: text.replace(
                "| `metrifid.runtime_review_process_command`, version `1` | `command.json` in each retained process directory |\n",
                "",
                1,
            ),
            schema_row_problems,
            "no schema row names metrifid.runtime_review_process_command",
            id="schema_row_removed",
        ),
    ],
)
def test_the_reference_assertions_react_to_a_real_change(
    mutate: Callable[[str], str], helper: Callable[[str], list[str]], expected: str
) -> None:
    """Alter one real row or block, then run the same helper the real document passes."""
    text = _reference()
    assert helper(text) == [], "the real reference must satisfy the helper first"
    mutated = mutate(text)
    assert mutated != text, "the mutation must actually change the document"
    problems = helper(mutated)
    assert any(expected in problem for problem in problems), problems


def test_the_api_assertion_reacts_to_a_removed_section() -> None:
    """Dropping one execution function's SDK section must be detected."""
    text = _sdk()
    assert runtime_review_api_problems(text, _SDK_API_HEADINGS) == []
    heading = "## `review_runtime_configuration_file`"
    section = _section_after(text, heading)
    assert section, "the section must exist before it is removed"
    mutated = text.replace(section, "", 1)
    assert mutated != text
    problems = runtime_review_api_problems(mutated, _SDK_API_HEADINGS)
    assert any("has no" in problem for problem in problems), problems


def test_the_stale_surface_assertion_reacts_to_reverted_prose() -> None:
    """Reintroducing the five-operation wording must be detected."""
    text = _sdk()
    assert stale_surface_count_problems(text) == []
    assert stale_surface_count_problems(text + "\nIt applies to all five operations.\n") != []


# ---- The Intel macOS packaging exception must be stated, and stated as packaging ---------------
#
# Upstream publishes no Darwin x86_64 MuJoCo wheel at or above 3.11, so the project bounds
# dependency resolution on that one platform. Two opposite errors are possible in the public
# documents. The exception can disappear, leaving the old unqualified "no minor ceiling" claim that
# an Intel macOS reader would find untrue. Or it can be overstated into a runtime ceiling, which
# would wrongly tell every reader that Metrifid refuses MuJoCo 3.11 or newer. Both are checked.

_PACKAGING_EXCEPTION_DOCUMENTS = (
    "README.md",
    "docs/canonicalization.md",
    "docs/getting_started.md",
    "docs/reference.md",
    "docs/sdk.md",
    "docs/compiled_certification.md",
    "docs/capabilities_and_use_cases.md",
)

_CEILING_CLAIM = re.compile(r"<\s*3\.11|below 3\.11|at or above 3\.11", re.IGNORECASE)
_PACKAGING_WORDS = (
    "package",
    "dependency",
    "resolution",
    "resolver",
    "install",
    "wheel",
    "metadata",
)
_RUNTIME_WORDS = ("runtime admission", "runtime gate", "admits", "admitted", "capability-based")


def _sentences(flat_text: str) -> list[str]:
    """Split flattened prose into sentences for claim-level checks."""
    return [part.strip() for part in re.split(r"(?<=[.:])\s+", flat_text) if part.strip()]


def intel_packaging_exception_problems(text: str) -> list[str]:
    """Return one problem per way a document misstates the Intel macOS packaging bound."""
    problems: list[str] = []
    sentences = _sentences(_flattened(text))
    claims = [sentence for sentence in sentences if _CEILING_CLAIM.search(sentence)]
    if not claims:
        problems.append("the Intel macOS packaging exception is not stated at all")
        return problems

    located = [
        sentence
        for sentence in claims
        if "x86_64" in sentence and ("Darwin" in sentence or "Intel macOS" in sentence)
    ]
    if not located:
        problems.append("a 3.11 bound is stated without naming the Darwin x86_64 platform")
    elif not any(
        any(word in sentence.lower() for word in _PACKAGING_WORDS) for sentence in located
    ):
        problems.append("the 3.11 bound is not described as a packaging or resolution fact")

    for sentence in claims:
        lowered = sentence.lower()
        if any(word in lowered for word in _RUNTIME_WORDS) and not any(
            word in lowered for word in _PACKAGING_WORDS
        ):
            problems.append(f"the 3.11 bound is described as a runtime ceiling: {sentence[:120]}")
    return problems


def runtime_admission_breadth_problems(text: str) -> list[str]:
    """Keep the core capability-based MuJoCo floor documented alongside narrower repair support."""
    flat = _flattened(text).lower()
    if "3.9" not in flat:
        return ["the document no longer states the stable MuJoCo 3.9 floor"]
    if not any(word in flat for word in ("capability-based", "capability admission", "admitted")):
        return ["the document no longer describes runtime admission as capability-based"]
    return []


@pytest.mark.parametrize("name", _PACKAGING_EXCEPTION_DOCUMENTS)
def test_every_requirements_document_states_the_intel_packaging_exception(name: str) -> None:
    """Each document that describes the MuJoCo requirement must carry the platform exception."""
    text = _document(name).read_text(encoding="utf-8")
    check.equal(intel_packaging_exception_problems(text), [], name)
    check.equal(runtime_admission_breadth_problems(text), [], name)


@pytest.mark.parametrize("name", _PACKAGING_EXCEPTION_DOCUMENTS)
def test_removing_the_intel_packaging_exception_is_detected(name: str) -> None:
    """Reverting a document to an unqualified ceiling claim must fail this contract."""
    text = _document(name).read_text(encoding="utf-8")
    assert intel_packaging_exception_problems(text) == [], name
    mutated = _CEILING_CLAIM.sub("", text)
    assert mutated != text, name
    assert intel_packaging_exception_problems(mutated), name


@pytest.mark.parametrize("name", _PACKAGING_EXCEPTION_DOCUMENTS)
def test_dropping_the_platform_from_the_exception_is_detected(name: str) -> None:
    """A ceiling stated without naming Darwin x86_64 reads as a global bound."""
    text = _document(name).read_text(encoding="utf-8")
    mutated = text.replace("x86_64", "processors")
    assert mutated != text, name
    assert intel_packaging_exception_problems(mutated), name


@pytest.mark.parametrize("name", _PACKAGING_EXCEPTION_DOCUMENTS)
@pytest.mark.parametrize(
    "overstatement",
    [
        "Runtime admission requires MuJoCo below 3.11 and refuses anything newer.",
        "The runtime gate refuses MuJoCo at or above 3.11 on every platform.",
        "Only MuJoCo below 3.11 is admitted.",
    ],
    ids=["admission", "gate", "admitted"],
)
def test_describing_the_exception_as_a_runtime_ceiling_is_detected(
    name: str, overstatement: str
) -> None:
    """Overstating the packaging bound as a runtime rule must fail this contract.

    The packaging bound exists because upstream ships no Intel macOS wheel, not because Metrifid
    refuses newer runtimes. A document that turns it into a runtime rule tells every reader
    something false about what the product admits.
    """
    text = _document(name).read_text(encoding="utf-8")
    assert intel_packaging_exception_problems(text) == [], name
    overstated = f"{text}\n\n{overstatement}\n"
    problems = intel_packaging_exception_problems(overstated)
    assert any("runtime ceiling" in problem for problem in problems), (name, problems)
