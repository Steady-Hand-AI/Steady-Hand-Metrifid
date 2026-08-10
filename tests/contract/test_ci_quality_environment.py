"""The shipped CI workflow must be able to pass on a first push.

Strict MyPy analyses modules that import MuJoCo. If the quality job does not install MuJoCo, that
step fails with ``import-not-found`` and every matrix lane that depends on the job never runs. This
asserts the install line and the job wiring rather than the prose around them.
"""

from __future__ import annotations

import re
from pathlib import Path

_WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/ci.yml"


def _workflow_text() -> str:
    """Return the shipped CI workflow source."""
    assert _WORKFLOW.is_file(), f"missing workflow: {_WORKFLOW}"
    return _WORKFLOW.read_text(encoding="utf-8")


def _quality_job(text: str) -> str:
    """Return only the build-and-quality job body."""
    start = text.index("build_and_quality:")
    following = re.search(r"\n  [a-z_]+:\n", text[start:])
    return text[start : start + following.start()] if following else text[start:]


def test_quality_job_installs_mujoco_before_strict_mypy() -> None:
    """Install MuJoCo in the quality environment so strict MyPy can resolve native imports."""
    job = _quality_job(_workflow_text())
    install_index = job.index("mujoco==3.10.0.*")
    mypy_index = job.index("mypy --strict")
    assert install_index < mypy_index, "MuJoCo must be installed before strict MyPy runs"


def test_quality_job_keeps_the_editable_install_and_every_gate() -> None:
    """Keep the editable quality install and all four quality gates."""
    job = _quality_job(_workflow_text())
    assert "--editable ." in job
    for gate in ("ruff check", "ruff format --check", "mypy --strict", "twine check"):
        assert gate in job, f"missing quality gate: {gate}"


def test_quality_toolchain_stays_constrained() -> None:
    """Install the quality toolchain through the pinned constraints file."""
    assert "-c .github/quality-constraints.txt" in _quality_job(_workflow_text())


def test_workflow_parses_as_yaml_when_a_parser_is_available() -> None:
    """Parse the workflow so a syntax error cannot ship silently."""
    yaml = __import__("importlib").util.find_spec("yaml")
    if yaml is None:
        text = _workflow_text()
        assert text.startswith("name: ci")
        assert "jobs:" in text
        return
    import yaml as yaml_module

    document = yaml_module.safe_load(_workflow_text())
    assert "build_and_quality" in document["jobs"]
