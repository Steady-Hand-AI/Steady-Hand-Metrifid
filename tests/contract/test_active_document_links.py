"""No active public surface may advertise the previous release's source tree.

A distribution that links into a previous release's source tree sends a reader to the wrong
documentation, and the `Documentation` and `Changelog` entries in the package metadata reach users
through the package index itself. This is a bounded contract over the active surfaces only: it reads files, resolves
relative Markdown targets on disk, and never touches the network.

Historical prose may still name an older version as released. What is forbidden is an active *link*
into that tree.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Callable
from pathlib import Path
from urllib.parse import unquote

import pytest
import pytest_check as check

# Any pinned-tag link into the repository's own source tree, not just 0.2.1: pinning the previous
# release was the observed defect, but pinning any tag in an active document has the same failure
# mode the next time the version moves.
_PINNED_SOURCE_LINK = re.compile(
    r"https://github\.com/Steady-Hand-AI/Steady-Hand-Metrifid/(?:blob|tree)/(\d+\.\d+\.\d+)/"
)

_RELATIVE_LINK = re.compile(r"\[[^\]]*\]\(([^)#\s]+)(?:#[^)\s]*)?\)")

# CHANGELOG.md is the release history: naming and linking older releases is its job.
_HISTORICAL = {"CHANGELOG.md"}


def _repository_root() -> Path:
    """Return the source checkout that holds the active documents."""
    return Path(__file__).resolve().parents[2]


def _active_documents(root: Path) -> list[Path]:
    """Return every active Markdown document and example whose links users follow."""
    documents = [root / "README.md", root / "CONTRIBUTING.md", root / "SECURITY.md"]
    documents.extend(sorted((root / "docs").glob("*.md")))
    documents.extend(sorted((root / "examples").rglob("*.md")))
    return [path for path in documents if path.is_file() and path.name not in _HISTORICAL]


def test_no_active_document_links_into_a_pinned_source_tree() -> None:
    """No active document may link into a released tag's source tree.

    Ordinary repository documents use relative links. `README.md` is the packaged long description
    and deliberately uses absolute `/blob/main/` and `/tree/main/` URLs instead, for the reason given
    in the packaged-long-description section below. Neither form pins a tag, which is what this test
    forbids.
    """
    root = _repository_root()
    documents = _active_documents(root)
    assert documents, "no active documents were found to scan"
    for path in documents:
        found = sorted(set(_PINNED_SOURCE_LINK.findall(path.read_text(encoding="utf-8"))))
        check.equal(
            found,
            [],
            f"{path.relative_to(root)} links into pinned source tree(s) {found}; "
            "use a relative link, or an absolute /main/ URL in the packaged README",
        )


def test_package_metadata_urls_are_not_pinned_to_a_released_tag() -> None:
    """`project.urls` reaches users through the package index, so it must not pin a tag."""
    root = _repository_root()
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    urls = metadata["project"]["urls"]
    for name, url in sorted(urls.items()):
        check.is_none(
            _PINNED_SOURCE_LINK.search(url),
            f"project.urls.{name} pins a released source tree: {url}",
        )
    for required in ("Documentation", "Changelog"):
        check.is_in(
            required,
            urls,
            f"project.urls is missing the {required} entry users reach from the package index",
        )


def test_every_relative_link_in_an_active_document_resolves_on_disk() -> None:
    """A relative link is only an improvement if it actually points at a file."""
    root = _repository_root()
    for path in _active_documents(root):
        for target in _RELATIVE_LINK.findall(path.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            resolved = (path.parent / target).resolve()
            check.is_true(
                resolved.exists(),
                f"{path.relative_to(root)} links to {target!r}, which does not exist on disk",
            )


# ---- The packaged long description ------------------------------------------------------------------
#
# `project.readme` makes README.md the Core Metadata long description, so a package index renders it on
# the project page. A Markdown renderer preserves a relative href and the index resolves it under the
# project URL rather than the repository, so a link that works on the repository page is broken for a
# reader arriving from the index. Registry files and their core metadata are immutable per version.
#
# This is a source-level contract over one admitted syntax, not a Markdown parser. It reads inline
# `[label](destination)` links with an optional double-quoted title, after backtick- and tilde-fenced
# blocks and inline code are removed. Rather than claiming every other construct is rejected, it proves
# coverage: after the admitted links are consumed, any surviving `](` opener is an unclassified
# navigation-like construct and fails. Reference links and raw HTML navigation carry no `](`, so those
# are named explicitly.

_REPOSITORY_URL = "https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid"
_SOURCE_TREE_URL = re.compile(re.escape(_REPOSITORY_URL) + r"/(blob|tree)/main/(?P<path>[^\s]*)")
# The one admitted destination form. A title, if present, is double-quoted.
# `<` and `>` are excluded from the destination so an angle-bracket form is not misread as a
# literal destination; it falls to the coverage guard instead.
_ADMITTED_LINK = re.compile(r'\[[^\]\[]*\]\(\s*([^)\s<>]+)(?:\s+"[^"]*")?\s*\)')
_FENCED_CODE = re.compile(r"(?ms)^(?P<fence>```|~~~).*?^(?P=fence)")
_INLINE_CODE = re.compile(r"`[^`\n]+`")
# Destination-bearing constructs that contain no "](" and so cannot reach the coverage guard.
_FORBIDDEN_CONSTRUCTS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("reference link definition", re.compile(r"(?m)^\s{0,3}\[[^\]]+\]:\s*\S")),
    ("reference link usage", re.compile(r"\[[^\]]*\]\[[^\]]*\]")),
    ("raw HTML anchor", re.compile(r"(?i)<a[\s>]")),
    ("raw HTML image", re.compile(r"(?i)<img[\s>]")),
)
# Reachability is asserted against parsed destinations, so a URL moved into a code example never counts.
_REQUIRED_DESTINATIONS = (
    f"{_REPOSITORY_URL}/blob/main/docs/getting_started.md",
    f"{_REPOSITORY_URL}/blob/main/docs/reference.md",
    f"{_REPOSITORY_URL}/blob/main/CHANGELOG.md",
    f"{_REPOSITORY_URL}/blob/main/CONTRIBUTING.md",
    f"{_REPOSITORY_URL}/tree/main/examples/certify/",
)


def _navigation_text(markdown: str) -> str:
    """Return the document with fenced blocks and inline code removed."""
    return _INLINE_CODE.sub(" ", _FENCED_CODE.sub(" ", markdown))


def readme_destinations(readme: str) -> tuple[list[str], list[str]]:
    """Return the admitted destinations, and one problem per unclassified construct.

    The second element is the coverage guard: every admitted link is consumed as it is parsed, so a
    surviving `](` is a navigation-like construct this contract does not read and must not ignore.
    """
    prose = _navigation_text(readme)
    problems = [
        f"{form}: this contract reads only inline [label](destination) links, so this form must not "
        f"be used in the packaged README"
        for form, pattern in _FORBIDDEN_CONSTRUCTS
        if pattern.search(prose)
    ]
    destinations: list[str] = []

    def consume(match: re.Match[str]) -> str:
        destinations.append(match.group(1))
        return " "

    residue = _ADMITTED_LINK.sub(consume, prose)
    if "](" in residue:
        problems.append(
            "unclassified inline-link syntax remains after parsing: this contract reads only "
            "[label](destination) with an optional double-quoted title, so a different title form, "
            "a nested label, or an angle-bracket destination must not be used"
        )
    return destinations, problems


def long_description_link_problems(readme: str, root: Path) -> list[str]:
    """Return one message per README link that would not survive package-index rendering."""
    destinations, problems = readme_destinations(readme)
    resolved_root = root.resolve()
    for destination in destinations:
        if destination.startswith(("#", "mailto:")):
            continue
        if not destination.startswith(("http://", "https://")):
            problems.append(
                f"{destination!r}: relative destination. A package index resolves it under the "
                f"project page instead of the repository, so it must be an absolute "
                f"{_REPOSITORY_URL}/blob/main/... or /tree/main/... URL."
            )
            continue
        matched = _SOURCE_TREE_URL.fullmatch(destination.split("#", 1)[0])
        if matched is None:
            continue
        kind = matched.group(1)
        relative = unquote(matched.group("path"))
        target = (root / relative).resolve()
        if not target.is_relative_to(resolved_root):
            problems.append(f"{destination!r}: maps outside the repository root")
        elif not target.exists():
            problems.append(
                f"{destination!r}: names {relative!r}, which does not exist in the checkout"
            )
        elif kind == "blob" and not target.is_file():
            problems.append(
                f"{destination!r}: /blob/main/ must name a file, but {relative!r} is not one"
            )
        elif kind == "tree" and not target.is_dir():
            problems.append(
                f"{destination!r}: /tree/main/ must name a directory, but {relative!r} is not one"
            )
    return problems


def _declared_long_description(root: Path) -> Path:
    """Return the file `project.readme` declares as the packaged long description."""
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return root / metadata["project"]["readme"]


def test_the_declared_long_description_is_the_readme() -> None:
    """The portability rule is only meaningful if the README is what gets packaged."""
    root = _repository_root()
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["readme"] == "README.md"
    assert _declared_long_description(root).is_file()


def test_the_packaged_long_description_carries_only_portable_links() -> None:
    """Every README destination must resolve identically from a repository page and a package index."""
    root = _repository_root()
    readme = _declared_long_description(root).read_text(encoding="utf-8")
    assert long_description_link_problems(readme, root) == []


def test_the_packaged_long_description_still_reaches_the_documentation() -> None:
    """Reachability is judged on parsed destinations, so a URL inside a code example does not count."""
    root = _repository_root()
    readme = _declared_long_description(root).read_text(encoding="utf-8")
    destinations, problems = readme_destinations(readme)
    assert problems == []
    for required in _REQUIRED_DESTINATIONS:
        check.is_in(required, destinations, f"the packaged README no longer links to {required}")


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        pytest.param(
            lambda text: text.replace(
                f"{_REPOSITORY_URL}/blob/main/CHANGELOG.md", "CHANGELOG.md", 1
            ),
            "relative destination",
            id="relative_destination",
        ),
        pytest.param(
            lambda text: text.replace(
                f"{_REPOSITORY_URL}/tree/main/examples/certify/",
                f"{_REPOSITORY_URL}/blob/main/examples/certify/",
                1,
            ),
            "must name a file",
            id="public_target_of_the_wrong_kind",
        ),
        pytest.param(
            lambda text: text.replace(
                f"{_REPOSITORY_URL}/blob/main/CHANGELOG.md",
                f"{_REPOSITORY_URL}/blob/main/../../etc/passwd",
                1,
            ),
            "maps outside the repository root",
            id="target_escaping_the_repository",
        ),
        pytest.param(
            lambda text: text + '\n<a href="docs/reference.md">reference</a>\n',
            "raw HTML anchor",
            id="raw_html_anchor",
        ),
        pytest.param(
            lambda text: text + '\n<img src="docs/diagram.png">\n',
            "raw HTML image",
            id="raw_html_image",
        ),
        pytest.param(
            lambda text: text + "\n[reference]: docs/reference.md\n",
            "reference link definition",
            id="reference_link_definition",
        ),
        pytest.param(
            lambda text: text + "\nSee [x](https://example.com (title)).\n",
            "unclassified inline-link syntax",
            id="parenthesized_title_is_not_silently_admitted",
        ),
        pytest.param(
            lambda text: text + "\nSee [x](<docs/reference.md>).\n",
            "unclassified inline-link syntax",
            id="angle_bracket_destination_is_not_silently_admitted",
        ),
    ],
)
def test_the_long_description_contract_is_mutation_sensitive(
    mutate: Callable[[str], str], expected: str
) -> None:
    """Each way of breaking package-index portability, or of escaping classification, is caught."""
    root = _repository_root()
    readme = _declared_long_description(root).read_text(encoding="utf-8")
    assert long_description_link_problems(readme, root) == [], "the real README must be clean first"
    mutated = mutate(readme)
    assert mutated != readme
    problems = long_description_link_problems(mutated, root)
    assert any(expected in problem for problem in problems), problems


@pytest.mark.parametrize("fence", ["```", "~~~"])
def test_a_link_inside_a_fenced_example_is_not_navigation(fence: str) -> None:
    """Both fence styles are code. A relative link shown as an example is not a broken public link."""
    root = _repository_root()
    readme = _declared_long_description(root).read_text(encoding="utf-8")
    example = f"\n{fence}\n[x](docs/does_not_exist.md)\n{fence}\n"
    assert long_description_link_problems(readme + example, root) == []


def test_a_required_destination_inside_a_code_example_does_not_count() -> None:
    """Reachability must come from parsed navigation, not from the URL appearing anywhere at all."""
    root = _repository_root()
    readme = _declared_long_description(root).read_text(encoding="utf-8")
    required = f"{_REPOSITORY_URL}/blob/main/CHANGELOG.md"
    moved = readme.replace(f"]({required})", "](https://example.com)", 1) + f"\n`{required}`\n"
    destinations, problems = readme_destinations(moved)
    assert problems == []
    assert required in moved, "the URL is still present as text"
    assert required not in destinations, "but it is no longer a parsed destination"
