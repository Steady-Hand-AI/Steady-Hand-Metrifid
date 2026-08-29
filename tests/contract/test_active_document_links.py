"""No active public surface may advertise the previous release's source tree.

A 0.5.0 distribution that links to `blob/0.2.1` sends a reader to the wrong documentation, and the
`Documentation` and `Changelog` entries in the package metadata reach users through the package
index itself. This is a bounded contract over the active surfaces only: it reads files, resolves
relative Markdown targets on disk, and never touches the network.

Historical prose may still name 0.2.1 as a released version. What is forbidden is an active *link*
into that tree.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

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
    """Every in-repository link must be relative rather than pinned to a released tag."""
    root = _repository_root()
    documents = _active_documents(root)
    assert documents, "no active documents were found to scan"
    for path in documents:
        found = sorted(set(_PINNED_SOURCE_LINK.findall(path.read_text(encoding="utf-8"))))
        check.equal(
            found,
            [],
            f"{path.relative_to(root)} links into pinned source tree(s) {found}; "
            "use a relative Markdown link instead",
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
