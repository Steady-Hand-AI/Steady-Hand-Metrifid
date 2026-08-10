"""Distribution-level checks for the public license and contribution policy."""

from __future__ import annotations

import hashlib
from pathlib import Path

_DCO_SHA256 = "dac2b0a921aaf4bcaf484dc082fbea072398bedecf5f1d4dcce7e122bbe5d2d5"


def _root() -> Path:
    """Return the source repository root."""
    return Path(__file__).resolve().parents[2]


def test_apache_license_text_remains_complete() -> None:
    """Keep the complete standard Apache License 2.0 text."""
    text = (_root() / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in text
    assert "Version 2.0, January 2004" in text
    assert "END OF TERMS AND CONDITIONS" in text
    assert "Copyright [yyyy] [name of copyright owner]" in text


def test_notice_identifies_the_copyright_holder() -> None:
    """Publish attribution without replacing or modifying Apache-2.0."""
    text = (_root() / "NOTICE").read_text(encoding="utf-8")
    assert "Copyright 2026 Volodymyr Barylyak" in text
    assert "does not modify the Apache License" in text


def test_dco_is_verbatim_version_1_1() -> None:
    """Carry the exact DCO 1.1 contribution certificate."""
    data = (_root() / "DCO").read_bytes()
    assert hashlib.sha256(data).hexdigest() == _DCO_SHA256
    text = data.decode("utf-8")
    assert "Developer's Certificate of Origin 1.1" in text
    assert "changing it is not allowed" in text


def test_contributor_documents_explain_signoff_and_copyright() -> None:
    """Make the contribution license and sign-off boundary discoverable."""
    contributing = " ".join((_root() / "CONTRIBUTING.md").read_text(encoding="utf-8").split())
    policy = " ".join(
        (_root() / "docs/licensing_and_contributions.md").read_text(encoding="utf-8").split()
    )
    for token in ("Apache License 2.0", "DCO", "git commit -s", "Signed-off-by:"):
        assert token in contributing
    for token in ("not a copyright assignment", "Contributors retain copyright"):
        assert token in contributing
        assert token in policy


def test_readme_links_controlling_license_files() -> None:
    """Point users to the license, attribution, and contribution certificate."""
    readme = (_root() / "README.md").read_text(encoding="utf-8")
    for token in ("LICENSE", "NOTICE", "DCO", "licensing_and_contributions.md"):
        assert token in readme
