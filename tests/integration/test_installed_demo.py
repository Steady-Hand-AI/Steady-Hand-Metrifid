"""The bundled demonstration must work for someone who only ran `pip install metrifid`.

Every assertion here is about that user's experience: run one command from a directory that
contains nothing, from an installation that is not a source checkout, and get both certification
outcomes plus a clear final line.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import metrifid

_EQUIVALENT_STATUS = "CERTIFIED_COMPILED_EQUIVALENCE"
_CHANGED_STATUS = "NOT_CERTIFIED_COMPILED_DIFFERS"


def _run_demo(cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run the installed demonstration module from one working directory."""
    return subprocess.run(
        [sys.executable, "-m", "metrifid.demo"],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=900,
    )


def test_metrifid_is_imported_from_an_installed_distribution() -> None:
    """Confirm this suite exercises an installed package rather than a source tree."""
    module_path = Path(metrifid.__file__ or "").resolve()
    assert "site-packages" in module_path.parts


def test_demo_succeeds_from_an_empty_directory(tmp_path: Path) -> None:
    """Run the demo where nothing else exists and require both expected outcomes."""
    workspace = (tmp_path / "empty").resolve()
    workspace.mkdir()
    completed = _run_demo(workspace)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert _EQUIVALENT_STATUS in completed.stdout
    assert _CHANGED_STATUS in completed.stdout
    assert completed.stdout.rstrip().endswith("Metrifid demo passed")


def test_demo_leaves_the_working_directory_untouched(tmp_path: Path) -> None:
    """Write nothing into the caller's directory; every artifact belongs in a temporary tree."""
    workspace = (tmp_path / "clean").resolve()
    workspace.mkdir()
    completed = _run_demo(workspace)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert list(workspace.iterdir()) == []


def test_demo_needs_no_arguments_and_no_repository_files(tmp_path: Path) -> None:
    """Depend on no packaged asset: the demo builds every model it certifies."""
    workspace = (tmp_path / "isolated").resolve()
    workspace.mkdir()
    completed = _run_demo(workspace)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    # A repository-relative read would surface as a path in the failure text.
    assert "examples/" not in completed.stderr


def test_demo_main_is_importable_and_returns_zero() -> None:
    """Expose `main()` as a normal callable that reports success with an exit code."""
    from metrifid.demo import main

    assert callable(main)
    assert main() == 0
