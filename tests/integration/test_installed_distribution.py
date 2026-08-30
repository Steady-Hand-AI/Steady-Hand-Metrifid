"""Normal installed-wheel identity checks executed outside the source tree."""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata as metadata
import importlib.resources as resources
import json
import os
import platform
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import cast

from packaging.requirements import Requirement

import metrifid
from metrifid.json_values import installed_distribution_identity, installed_distribution_sha256

_FROZEN_WORKER_SHA256 = "b00e509a344593806c088c4e49783ed71bacd815466d74bce9e27c931535b4ff"


def test_import_resolves_inside_environment_not_source_checkout() -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises import resolves inside environment not source checkout; release
    evidence must describe the exact package bytes and metadata used for the decision.
    """
    module_path = Path(metrifid.__file__).resolve()
    assert "site-packages" in module_path.parts
    assert "/src/metrifid/" not in module_path.as_posix()
    assert Path.cwd().resolve() not in module_path.parents


def test_normal_installed_distribution_identity_succeeds() -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises normal installed distribution identity succeeds; release evidence
    must describe the exact package bytes and metadata used for the decision.
    """
    identity = installed_distribution_identity()
    digest = installed_distribution_sha256()
    assert identity["distribution_name"] == "metrifid"
    assert identity["distribution_version"] == metrifid.__version__
    assert len(digest) == 64
    assert digest != "0" * 64


def test_installed_runtime_review_resources_are_private_and_exact() -> None:
    """Carry the collector and frozen worker without advertising either as public SDK names."""
    package = resources.files("metrifid.runtime_review")
    worker = package.joinpath("native_evidence_worker.py.txt")
    collector = package.joinpath("_native_profile_identity.py")
    assert worker.is_file()
    assert collector.is_file()
    assert hashlib.sha256(worker.read_bytes()).hexdigest() == _FROZEN_WORKER_SHA256
    import metrifid.runtime_review as runtime_review

    assert "native_evidence_worker" not in runtime_review.__all__
    assert "native_profile_identity" not in runtime_review.__all__


def test_fresh_process_distribution_hash_is_stable(tmp_path: Path) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises fresh process distribution hash is stable; release evidence must
    describe the exact package bytes and metadata used for the decision.
    """
    code = (
        "import json; "
        "from metrifid.json_values import installed_distribution_sha256; "
        "print(json.dumps({'sha256': installed_distribution_sha256()}, sort_keys=True))"
    )
    outputs: list[bytes] = []
    for index in range(2):
        run_dir = tmp_path / str(index)
        run_dir.mkdir()
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            cwd=run_dir,
        )
        outputs.append(result.stdout)
    assert outputs[0] == outputs[1]
    assert json.loads(outputs[0])["sha256"] == installed_distribution_sha256()


def test_pip_launcher_record_is_verified_and_excluded_from_identity() -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises pip launcher record is verified and excluded from identity; release
    evidence must describe the exact package bytes and metadata used for the decision.
    """
    distribution = metadata.distribution("metrifid")
    record_text = distribution.read_text("RECORD")
    assert record_text is not None
    metadata_member = next(
        cast(metadata.PackagePath, member)
        for member in distribution.files or ()
        if str(member).endswith(".dist-info/METADATA")
    )
    install_root = Path(distribution.locate_file(metadata_member)).resolve().parent.parent
    launcher = Path(sysconfig.get_path("scripts")) / "metrifid"
    expected_record_path = os.path.relpath(launcher, start=install_root).replace(os.sep, "/")
    rows = list(csv.reader(record_text.splitlines()))
    assert sum(row[0] == expected_record_path for row in rows) == 1
    assert launcher.is_file()
    identity = installed_distribution_identity()
    members = cast(list[dict[str, object]], identity["members"])
    assert expected_record_path not in {cast(str, member["path"]) for member in members}


def test_installed_metrifid_help_exits_zero(tmp_path: Path) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises installed metrifid help exits zero; release evidence must describe
    the exact package bytes and metadata used for the decision.
    """
    launcher = Path(sysconfig.get_path("scripts")) / "metrifid"
    result = subprocess.run(
        [str(launcher), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "compare" in result.stdout
    assert "run-runtime-review" in result.stdout


def test_installed_runtime_review_execution_help_is_complete(tmp_path: Path) -> None:
    """Describe prepared profiles, twelve cells, and the existing referee from the wheel."""
    launcher = Path(sysconfig.get_path("scripts")) / "metrifid"
    result = subprocess.run(
        [str(launcher), "run-runtime-review", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    normalized = " ".join(result.stdout.split())
    assert "two already-prepared explicit Python profiles" in normalized
    assert "twelve sequential evidence cells" in normalized
    assert "existing Runtime Review evaluator" in normalized


def _active_runtime_requirements() -> list[Requirement]:
    """Return the runtime requirements this environment actually resolves under."""
    parsed = [
        Requirement(requirement)
        for requirement in (metadata.requires("metrifid") or [])
        if "extra ==" not in requirement
    ]
    return [r for r in parsed if r.marker is None or r.marker.evaluate()]


def _is_intel_macos() -> bool:
    """The one platform class whose MuJoCo resolution carries a published ceiling."""
    return (platform.system(), platform.machine()) == ("Darwin", "x86_64")


def test_public_package_and_runtime_dependency_are_installed() -> None:
    """Verify the public package and the MuJoCo requirement this platform actually resolves.

    This runs inside a real installation, so the declared requirement is checked against the
    MuJoCo that is genuinely importable here rather than against a requirement string in the
    abstract. Exactly one MuJoCo requirement may be active, and its ceiling must be present on
    Intel macOS and absent everywhere else.
    """
    from importlib import import_module

    for module_name in (
        "metrifid._model_closure",
        "metrifid._model_admission",
        "metrifid._model_identity",
        "metrifid._model_dependencies",
        "metrifid._model_identity_validation",
    ):
        module = import_module(module_name)
        assert "site-packages" in Path(module.__file__).resolve().parts

    active = _active_runtime_requirements()
    assert {requirement.name for requirement in active} == {"mujoco", "numpy"}

    mujoco_requirements = [r for r in active if r.name == "mujoco"]
    assert len(mujoco_requirements) == 1, [str(r) for r in mujoco_requirements]
    mujoco_specifier = mujoco_requirements[0].specifier
    assert ">=3.9" in {str(s) for s in mujoco_specifier}
    ceilings = {str(s) for s in mujoco_specifier if s.operator in ("<", "<=")}
    assert ceilings == ({"<3.11"} if _is_intel_macos() else set()), (
        platform.system(),
        platform.machine(),
        ceilings,
    )
    # The requirement the environment resolves under must admit the MuJoCo that is installed.
    assert metadata.version("mujoco") in mujoco_specifier, (
        metadata.version("mujoco"),
        str(mujoco_specifier),
    )

    numpy_requirements = [r for r in active if r.name == "numpy"]
    assert [str(r) for r in numpy_requirements] == ["numpy>=1.26"]
    assert metadata.version("numpy") in numpy_requirements[0].specifier
