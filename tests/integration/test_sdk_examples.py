"""The shipped SDK examples must run for a user who copied them out of the repository.

Each script is copied to a fresh temporary directory and executed with the repository deliberately
absent from the working directory, so a script that silently depended on the checkout fails here.
The comparison example is exercised the same way, including its workload preparation step.
"""

from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import metrifid

_SDK_SCRIPTS = (
    "certify_api.py",
    "compare_api.py",
    "audit_api.py",
    "workload_qualification_api.py",
)
_COMPARISON_STATUS = "NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD"


def _repository_root() -> Path:
    """Return the source checkout that holds the example directories."""
    return Path(__file__).resolve().parents[2]


def _examples(name: str) -> Path:
    """Return one example directory, skipping when the checkout is unavailable."""
    directory = _repository_root() / "examples" / name
    if not directory.is_dir():  # pragma: no cover - environment guard
        pytest.skip(f"example directory is unavailable: {directory}")
    return directory


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run one command with the repository absent from the module search path."""
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    return subprocess.run(
        command, cwd=cwd, env=environment, check=False, capture_output=True, text=True, timeout=1800
    )


def test_examples_exercise_an_installed_distribution() -> None:
    """Confirm the examples run against an installed package, not a source tree."""
    assert "site-packages" in Path(metrifid.__file__ or "").resolve().parts


@pytest.mark.parametrize("script", _SDK_SCRIPTS)
def test_each_sdk_example_runs_from_a_copied_directory(tmp_path: Path, script: str) -> None:
    """Copy the SDK examples elsewhere and run each one against the installed wheel."""
    workspace = (tmp_path / "sdk").resolve()
    workspace.mkdir()
    for source in _examples("sdk").glob("*.py"):
        shutil.copy2(source, workspace / source.name)
    completed = _run([sys.executable, script], workspace)
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_runtime_review_execution_sdk_example_exposes_copyable_help(tmp_path: Path) -> None:
    """Keep the native execution example on the installed lazy SDK boundary."""
    workspace = (tmp_path / "runtime-review-sdk").resolve()
    workspace.mkdir()
    source = _examples("sdk") / "runtime_review_run_api.py"
    shutil.copy2(source, workspace / source.name)
    completed = _run([sys.executable, source.name, "--help"], workspace)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "configuration" in completed.stdout


def test_runtime_review_execution_product_example_is_self_contained(tmp_path: Path) -> None:
    """Ship one strict run declaration, manifest, model, and user instructions together."""
    workspace = (tmp_path / "runtime-review-run").resolve()
    shutil.copytree(_examples("runtime_review_run"), workspace)
    assert {path.name for path in workspace.iterdir()} == {
        "README.md",
        "manifest.json",
        "model.xml",
        "runtime_review_run.json",
    }
    run_config = json.loads((workspace / "runtime_review_run.json").read_text(encoding="utf-8"))
    assert run_config["schema"] == "metrifid.runtime_review_run_config"
    assert run_config["manifest"] == "manifest.json"
    assert run_config["fixture_id"] == "smooth_pendulum"
    assert str(run_config["baseline_python"]).startswith("/replace/")
    assert str(run_config["candidate_python"]).startswith("/replace/")
    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    fixtures = manifest["fixtures"]
    assert len(fixtures) == 1
    assert fixtures[0]["xml_path"] == "model.xml"
    assert (workspace / "model.xml").read_text(encoding="utf-8").startswith("<mujoco ")


def test_sdk_examples_import_no_private_metrifid_module() -> None:
    """Keep the examples on the public surface and off the module search path.

    Import statements are read from the parsed syntax tree rather than matched as text, so a
    docstring that merely mentions a private name cannot trip this check.
    """
    for source in sorted(_examples("sdk").glob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        private = [
            name for name in imported if name.startswith(("metrifid._", "metrifid.certify._"))
        ]
        assert private == [], (source.name, private)
        body = source.read_text(encoding="utf-8")
        for mutation in ("sys.path.insert", "sys.path.append", "sys.path.extend", "sys.path ="):
            assert mutation not in body, (source.name, mutation)


def test_comparison_example_prepares_its_workload_and_completes(tmp_path: Path) -> None:
    """Run the complete comparison example from a copied directory, end to end."""
    workspace = (tmp_path / "compare").resolve()
    shutil.copytree(_examples("compare"), workspace)

    prepare = _run([sys.executable, "prepare_workload.py"], workspace)
    assert prepare.returncode == 0, prepare.stdout + prepare.stderr
    assert (workspace / "state.npz").is_file()
    assert (workspace / "actions.npz").is_file()

    console = shutil.which("metrifid")
    command = (
        [console, "compare", "comparison.json"]
        if console
        else [sys.executable, "-m", "metrifid.cli", "compare", "comparison.json"]
    )
    compare = _run(command, workspace)
    assert compare.returncode == 0, compare.stdout + compare.stderr
    assert _COMPARISON_STATUS in compare.stdout
    assert (workspace / "comparison_out" / "comparison.json").is_file()
    assert (workspace / "comparison_out" / "comparison.md").is_file()


def test_workload_preparation_never_overwrites_an_existing_artifact(tmp_path: Path) -> None:
    """Preserve an artifact already on disk: a recorded result may depend on its exact bytes."""
    workspace = (tmp_path / "compare-twice").resolve()
    shutil.copytree(_examples("compare"), workspace)
    assert _run([sys.executable, "prepare_workload.py"], workspace).returncode == 0
    original = (workspace / "state.npz").read_bytes()
    (workspace / "state.npz").write_bytes(original + b"\x00")
    marked = (workspace / "state.npz").read_bytes()

    second = _run([sys.executable, "prepare_workload.py"], workspace)
    assert second.returncode == 0, second.stdout + second.stderr
    assert (workspace / "state.npz").read_bytes() == marked
