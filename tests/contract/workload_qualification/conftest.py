"""Shared fixtures for the workload-qualification contract suites.

One real campaign is published per session and copied per test. Building genuine typed comparison
receipts by hand is not feasible, and a hand-written stand-in would not prove anything about the
binding these suites exist to check, so the tamper and campaign suites start from a receipt the
installed product actually produced.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

from metrifid.json_values import canonical_json_bytes, compute_self_hash
from metrifid.workload_qualification import qualify_configuration_file
from tests._support.workload_qualification import write_case

RECEIPT_RELATIVE = Path("qualification_out") / "receipt" / "workload_qualification.json"


@pytest.fixture(scope="session")
def published_case(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Publish one honest campaign once and return the case directory."""
    root = tmp_path_factory.mktemp("published") / "case"
    qualify_configuration_file(write_case(root))
    return root


@pytest.fixture(scope="session")
def other_published_case(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Publish a second, genuinely different campaign for cross-campaign splice tests."""
    root = tmp_path_factory.mktemp("other") / "case"
    path = write_case(root)
    model = root / "baseline" / "model.xml"
    model.write_text(
        model.read_text(encoding="utf-8").replace('damping="0.02"', 'damping="0.021"'),
        encoding="utf-8",
    )
    qualify_configuration_file(path)
    return root


@pytest.fixture
def other_case(other_published_case: Path, tmp_path: Path) -> Path:
    """Return a private copy of the second campaign."""
    destination = tmp_path / "other"
    shutil.copytree(other_published_case, destination, symlinks=True)
    return destination


@pytest.fixture
def case_copy(published_case: Path, tmp_path: Path) -> Path:
    """Return a private copy of the published campaign for one test to mutate."""
    destination = tmp_path / "case"
    shutil.copytree(published_case, destination, symlinks=True)
    return destination


def reseal(receipt_path: Path, mutate: Callable[[dict], None]) -> None:
    """Apply one mutation and recompute the public self-hash, exactly as an attacker would."""
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    mutate(receipt)
    receipt["receipt_sha256"] = None
    receipt["receipt_sha256"] = compute_self_hash(receipt, "receipt_sha256")
    receipt_path.write_bytes(canonical_json_bytes(receipt) + b"\n")


def relocate(published: Path, destination_parent: Path, output_name: str) -> Path:
    """Copy one published campaign to a new parent, optionally renaming its output root.

    A published receipt records the absolute locations its campaign actually used. Reading one
    somewhere else is an ordinary thing to do — that is how a reader audits evidence someone sent
    them — so replay binds the current tree through descriptors and treats those recorded paths as
    historical coherence metadata. This helper builds exactly that situation.
    """
    destination = destination_parent / "case"
    shutil.copytree(published, destination, symlinks=True)
    if output_name != "qualification_out":
        (destination / "qualification_out").rename(destination / output_name)
    return destination / output_name / "receipt" / "workload_qualification.json"
