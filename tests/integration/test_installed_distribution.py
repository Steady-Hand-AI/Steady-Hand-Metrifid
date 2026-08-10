"""Normal installed-wheel identity checks executed outside the source tree."""

from __future__ import annotations

import csv
import importlib.metadata as metadata
import json
import os
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import cast

import metrifid
from metrifid.json_values import installed_distribution_identity, installed_distribution_sha256


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


def test_public_package_and_runtime_dependency_are_installed() -> None:
    """Verify that the public package and its pinned MuJoCo runtime are installed."""
    from importlib import import_module, metadata

    for module_name in (
        "metrifid._model_closure",
        "metrifid._model_admission",
        "metrifid._model_identity",
        "metrifid._model_dependencies",
        "metrifid._model_identity_validation",
    ):
        module = import_module(module_name)
        assert "site-packages" in Path(module.__file__).resolve().parts
    requirements = [
        requirement.split(";")[0].replace(" ", "")
        for requirement in (metadata.requires("metrifid") or [])
        if "extra ==" not in requirement
    ]
    assert "mujoco==3.10.0.*" in requirements
    assert "numpy>=1.26" in requirements
    assert not [value for value in requirements if value.startswith("numpy") and "<" in value]
