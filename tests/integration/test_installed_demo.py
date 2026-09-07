"""The bundled demonstration must work for someone who only ran `pip install metrifid`.

Every assertion here is about that user's experience: run one command from a directory that
contains nothing, from an installation that is not a source checkout, and get both comparison
outcomes, the real retained paths and a clear final line.

The demonstration retains its results through the ordinary allocator, which writes beneath the
running user's home directory. Every invocation here is therefore given a test-owned home, so the
suite never adds to, inspects or removes the runs a real user has kept.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import metrifid

_IDENTICAL_OUTCOME = ("different source, same compiled model", "exit 0 (identical)")
_CHANGED_OUTCOME = ("one changed mass", "exit 40 (different)")


def _run_demo(cwd: Path, home: Path) -> subprocess.CompletedProcess[str]:
    """Run the installed demonstration from one working directory and one test-owned home."""
    environment = dict(os.environ)
    environment["HOME"] = str(home)
    return subprocess.run(
        [sys.executable, "-m", "metrifid.demo"],
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )


def _isolated(tmp_path: Path, name: str) -> tuple[Path, Path]:
    """Return an empty working directory and a private home, both test-owned."""
    workspace = (tmp_path / name).resolve()
    workspace.mkdir(parents=True)
    home = (tmp_path / f"{name}-home").resolve()
    home.mkdir(parents=True)
    return workspace, home


def _retained_receipts(home: Path) -> list[Path]:
    """Return every receipt the demonstration retained beneath its own home."""
    return sorted((home / ".metrifid" / "runs").glob("*/model_release.json"))


def test_metrifid_is_imported_from_an_installed_distribution() -> None:
    """Confirm this suite exercises an installed package rather than a source tree."""
    module_path = Path(metrifid.__file__ or "").resolve()
    assert "site-packages" in module_path.parts


def test_demo_succeeds_from_an_empty_directory(tmp_path: Path) -> None:
    """Run the demo where nothing else exists and require both expected outcomes."""
    workspace, home = _isolated(tmp_path, "empty")
    completed = _run_demo(workspace, home)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    for label, outcome in (_IDENTICAL_OUTCOME, _CHANGED_OUTCOME):
        matching = [
            line
            for line in completed.stdout.splitlines()
            if line.startswith(label) and line.endswith(outcome)
        ]
        assert len(matching) == 1, completed.stdout
    assert completed.stdout.rstrip().endswith("Metrifid demo passed")


def test_demo_prints_the_paths_it_actually_retained(tmp_path: Path) -> None:
    """The printed receipt and report paths must name files that exist and can be read."""
    workspace, home = _isolated(tmp_path, "paths")
    completed = _run_demo(workspace, home)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    printed = [
        line.split(":", 1)[1].strip()
        for line in completed.stdout.splitlines()
        if line.strip().startswith(("receipt", "report"))
    ]
    assert len(printed) == 4, completed.stdout
    for value in printed:
        if value == "not written":
            continue
        assert Path(value).is_file(), value
        assert Path(value).is_relative_to(home), "every retained path belongs to the test home"


def test_the_retained_receipts_are_readable_after_the_demo_exits(tmp_path: Path) -> None:
    """The models are temporary; the evidence is not. Read it back once the demo is gone."""
    workspace, home = _isolated(tmp_path, "readback")
    completed = _run_demo(workspace, home)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    receipts = _retained_receipts(home)
    assert len(receipts) == 2
    outcomes = set()
    for receipt in receipts:
        shown = subprocess.run(
            [sys.executable, "-m", "metrifid.cli", "show", str(receipt), "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert shown.returncode == 0, shown.stderr
        document = json.loads(shown.stdout)
        outcomes.add(document["compiled_comparison"]["state"])
        assert document["observation"] == "recorded"
    assert outcomes == {"identical", "different"}


def test_demo_leaves_the_working_directory_untouched(tmp_path: Path) -> None:
    """Write nothing into the caller's directory; the models live in a temporary tree."""
    workspace, home = _isolated(tmp_path, "clean")
    completed = _run_demo(workspace, home)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(workspace.iterdir()) == []


def test_demo_needs_no_arguments_and_no_repository_files(tmp_path: Path) -> None:
    """Depend on no packaged asset: the demo builds every model it compares."""
    workspace, home = _isolated(tmp_path, "isolated")
    completed = _run_demo(workspace, home)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    # A repository-relative read would surface as a path in the failure text.
    assert "examples/" not in completed.stderr


def test_demo_main_is_importable_and_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Expose `main()` as a normal callable that reports success with an exit code."""
    _, home = _isolated(tmp_path, "inprocess")
    monkeypatch.setenv("HOME", str(home))
    from metrifid.demo import main

    assert callable(main)
    assert main() == 0
    assert len(_retained_receipts(home)) == 2
