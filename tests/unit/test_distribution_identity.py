"""Installed-distribution identity unit tests and hostile manifest cases."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.machinery
import importlib.metadata as metadata
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import cast

import pytest

import metrifid.distribution as distribution_module
from metrifid.distribution import (
    DistributionIdentityError,
    _bound_distribution_identity,
    _imported_package_root,
    installed_distribution_identity,
    installed_distribution_sha256,
)
from metrifid.json_values import CanonicalValue
from metrifid.operational import OperationalReasonCode
from metrifid.version import __version__

_EXACT_CONSOLE_BINDING = "[console_scripts]\nmetrifid = metrifid.cli:main\n"


@dataclass(frozen=True)
class WheelFixture:
    """One synthetic installed wheel rooted at a temporary site directory."""

    root: Path
    package_root: Path
    dist_info: Path
    distribution: metadata.PathDistribution


def _record_hash(payload: bytes) -> str:
    """Compute the canonical record hash value used by distribution identity fixtures.

    Content addressing keeps the mutation boundary explicit for distribution identity.
    """
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode("ascii")
    return encoded.rstrip("=")


def _write_record(
    root: Path,
    dist_info: Path,
    *,
    overrides: dict[str, tuple[str, str]] | None = None,
    duplicate: str | None = None,
) -> None:
    """Write write record data into the isolated test workspace.

    The distribution identity scenario observes real bytes and filesystem effects for
    distribution identity.
    """
    rows: list[list[str]] = []
    override_values = {} if overrides is None else overrides
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative.endswith(".dist-info/RECORD"):
            continue
        payload = path.read_bytes()
        hash_value, size_value = override_values.get(
            relative,
            (f"sha256={_record_hash(payload)}", str(len(payload))),
        )
        rows.append([relative, hash_value, size_value])
        if duplicate == relative:
            rows.append([relative, hash_value, size_value])
    rows.append([dist_info.relative_to(root).as_posix() + "/RECORD", "", ""])
    with (dist_info / "RECORD").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def _make_wheel(
    tmp_path: Path,
    *,
    version: str = __version__,
    include_wheel: bool = True,
    include_init: bool = True,
    include_license: bool = True,
    include_entry_points: bool = False,
    direct_url: object | None = None,
    malformed_direct_url: bool = False,
    editable_marker: bool = False,
    duplicate: str | None = None,
    overrides: dict[str, tuple[str, str]] | None = None,
) -> WheelFixture:
    """Construct the make wheel fixture used by distribution identity scenarios.

    Deterministic setup isolates distribution identity without bypassing the contract boundary
    under assertion.
    """
    root = tmp_path / "site"
    package_root = root / "metrifid"
    package_root.mkdir(parents=True)
    if include_init:
        (package_root / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (package_root / "version.py").write_text(f'__version__ = "{version}"\n', encoding="utf-8")
    (package_root / "py.typed").write_bytes(b"")
    cache = package_root / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"ignored")
    (package_root / "ignored.pyo").write_bytes(b"ignored")

    dist_info = root / f"metrifid-{version}.dist-info"
    dist_info.mkdir(parents=True)
    metadata_text = (
        f"Metadata-Version: 2.4\nName: metrifid\nVersion: {version}\nLicense-File: LICENSE\n"
    )
    (dist_info / "METADATA").write_text(metadata_text, encoding="utf-8")
    if include_wheel:
        (dist_info / "WHEEL").write_text(
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
            encoding="utf-8",
        )
    if include_license:
        license_dir = dist_info / "licenses"
        license_dir.mkdir()
        (license_dir / "LICENSE").write_text("license\n", encoding="utf-8")
    if include_entry_points:
        (dist_info / "entry_points.txt").write_text(
            "[console_scripts]\nmetrifid = metrifid:main\n", encoding="utf-8"
        )
    if malformed_direct_url:
        (dist_info / "direct_url.json").write_text("{", encoding="utf-8")
    elif direct_url is not None:
        (dist_info / "direct_url.json").write_text(json.dumps(direct_url), encoding="utf-8")
    if editable_marker:
        (root / "__editable__.metrifid-0.1.0a2.pth").write_text("editable\n", encoding="utf-8")
    _write_record(root, dist_info, overrides=overrides, duplicate=duplicate)
    return WheelFixture(root, package_root, dist_info, metadata.PathDistribution(dist_info))


def _identity(
    fixture: WheelFixture, loaded: dict[str, object] | None = None
) -> dict[str, CanonicalValue]:
    """Construct the identity fixture used by distribution identity scenarios.

    Deterministic setup isolates distribution identity without bypassing the contract boundary
    under assertion.
    """
    modules = (
        {"metrifid": SimpleNamespace(__file__=str(fixture.package_root / "__init__.py"))}
        if loaded is None
        else loaded
    )
    return _bound_distribution_identity(
        fixture.distribution,
        fixture.package_root.resolve(),
        modules,
    )


def _member_paths(identity: dict[str, CanonicalValue]) -> list[str]:
    """Construct the member paths fixture used by distribution identity scenarios.

    Deterministic setup isolates distribution identity without bypassing the contract boundary
    under assertion.
    """
    members = cast(list[CanonicalValue], identity["members"])
    return [cast(str, cast(dict[str, CanonicalValue], item)["path"]) for item in members]


class _DistributionWrapper:
    """Delegate metadata operations while allowing hostile path overrides."""

    def __init__(
        self,
        base: metadata.PathDistribution,
        redirects: dict[str, Path] | None = None,
    ) -> None:
        """Construct the init fixture used by distribution identity scenarios.

        Deterministic setup isolates DistributionWrapper without bypassing the contract boundary
        under assertion.
        """
        self._base = base
        self._redirects = {} if redirects is None else redirects

    @property
    def version(self) -> str:
        """Construct the version fixture used by distribution identity scenarios.

        Deterministic setup isolates DistributionWrapper without bypassing the contract boundary
        under assertion.
        """
        return self._base.version

    @property
    def metadata(self) -> object:
        """Construct the metadata fixture used by distribution identity scenarios.

        Deterministic setup isolates DistributionWrapper without bypassing the contract boundary
        under assertion.
        """
        return self._base.metadata

    def read_text(self, filename: str) -> str | None:
        """Construct the read text fixture used by distribution identity scenarios.

        Deterministic setup isolates DistributionWrapper without bypassing the contract boundary
        under assertion.
        """
        return self._base.read_text(filename)

    def locate_file(self, path: metadata.PackagePath) -> Path:
        """Construct the locate file fixture used by distribution identity scenarios.

        Deterministic setup isolates DistributionWrapper without bypassing the contract boundary
        under assertion.
        """
        return self._redirects.get(str(path), Path(str(self._base.locate_file(path))))


def _rewrite_record_rows(fixture: WheelFixture, rows: list[list[str]]) -> None:
    """Write rewrite record rows data into the isolated test workspace.

    The distribution identity scenario observes real bytes and filesystem effects for
    distribution identity.
    """
    with (fixture.dist_info / "RECORD").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def _record_rows(fixture: WheelFixture) -> list[list[str]]:
    """Construct the record rows fixture used by distribution identity scenarios.

    Deterministic setup isolates distribution identity without bypassing the contract boundary
    under assertion.
    """
    return list(csv.reader((fixture.dist_info / "RECORD").read_text(encoding="utf-8").splitlines()))


def _make_launcher_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    entry_points_text: str = _EXACT_CONSOLE_BINDING,
    record_paths: tuple[str, ...] | None = None,
) -> tuple[WheelFixture, Path, str]:
    """Construct the make launcher fixture fixture used by distribution identity scenarios.

    Deterministic setup isolates distribution identity without bypassing the contract boundary
    under assertion.
    """
    fixture = _make_wheel(
        tmp_path / "venv" / "lib" / "python3.13",
        include_entry_points=True,
    )
    entry_points = fixture.dist_info / "entry_points.txt"
    entry_points.write_text(entry_points_text, encoding="utf-8")
    _write_record(fixture.root, fixture.dist_info)

    scripts = tmp_path / "venv" / "bin"
    scripts.mkdir(parents=True)
    launcher = scripts / "metrifid"
    launcher.write_bytes(b"#!/usr/bin/env python\nprint('metrifid')\n")
    launcher.chmod(0o755)
    expected_record_path = os.path.relpath(launcher, start=fixture.root).replace(os.sep, "/")

    original_get_path = distribution_module.sysconfig.get_path

    def get_path(name: str, *args: object, **kwargs: object) -> str:
        """Construct the get path fixture used by distribution identity scenarios.

        Deterministic setup isolates make launcher fixture without bypassing the contract
        boundary under assertion.
        """
        if name == "scripts":
            return str(scripts)
        return original_get_path(name, *args, **kwargs)

    monkeypatch.setattr(distribution_module.sysconfig, "get_path", get_path)
    selected_paths = (expected_record_path,) if record_paths is None else record_paths
    rows = _record_rows(fixture)
    self_row = rows.pop()
    payload = launcher.read_bytes()
    for raw_path in selected_paths:
        rows.append([raw_path, f"sha256={_record_hash(payload)}", str(len(payload))])
    rows.append(self_row)
    _rewrite_record_rows(fixture, rows)
    return fixture, launcher, expected_record_path


def test_real_installed_distribution_identity_is_stable() -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises real installed distribution identity is stable; release evidence
    must describe the exact package bytes and metadata used for the decision.
    """
    first = installed_distribution_identity()
    second = installed_distribution_identity()
    assert first == second
    assert installed_distribution_sha256() == installed_distribution_sha256()
    assert first["schema"] == "metrifid.installed_distribution_identity"
    assert first["schema_version"] == 1
    assert first["distribution_name"] == "metrifid"
    assert first["distribution_version"] == __version__
    paths = _member_paths(first)
    assert "metrifid/py.typed" in paths
    assert all("RECORD" not in path for path in paths)


def test_synthetic_wheel_selects_only_frozen_members(tmp_path: Path) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises synthetic wheel selects only frozen members; release evidence must
    describe the exact package bytes and metadata used for the decision.
    """
    fixture = _make_wheel(tmp_path, include_entry_points=True)
    identity = _identity(fixture)
    paths = _member_paths(identity)
    assert paths == sorted(paths)
    assert "metrifid/py.typed" in paths
    assert any(path.endswith("/entry_points.txt") for path in paths)
    assert any(path.endswith("/licenses/LICENSE") for path in paths)
    assert all("__pycache__" not in path for path in paths)
    assert all(not path.endswith((".pyc", ".pyo")) for path in paths)
    assert all(not path.endswith(("/RECORD", "/direct_url.json")) for path in paths)


def test_noneditable_direct_url_is_permitted_but_excluded(tmp_path: Path) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises noneditable direct url is permitted but excluded; release evidence
    must describe the exact package bytes and metadata used for the decision.
    """
    fixture = _make_wheel(tmp_path, direct_url={"url": "file:///wheel.whl"})
    assert all("direct_url.json" not in path for path in _member_paths(_identity(fixture)))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"direct_url": {"dir_info": {"editable": True}}},
        {"editable_marker": True},
    ],
)
def test_editable_install_markers_refuse(tmp_path: Path, kwargs: dict[str, object]) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises editable install markers refuse; release evidence must describe the
    exact package bytes and metadata used for the decision.
    """
    fixture = _make_wheel(tmp_path, **kwargs)
    with pytest.raises(DistributionIdentityError) as raised:
        _identity(fixture)
    assert raised.value.reason_code is OperationalReasonCode.EDITABLE_INSTALL_UNSUPPORTED
    failure = raised.value.to_operational_failure("compare")
    assert failure.tool.execution_identity_state == "UNBOUND"
    assert failure.tool.distribution_sha256 is None


def test_malformed_direct_url_refuses(tmp_path: Path) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises malformed direct url refuses; release evidence must describe the
    exact package bytes and metadata used for the decision.
    """
    fixture = _make_wheel(tmp_path, malformed_direct_url=True)
    with pytest.raises(DistributionIdentityError, match="direct_url") as raised:
        _identity(fixture)
    assert raised.value.reason_code is OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID


def test_imported_root_mismatch_refuses_without_hash(tmp_path: Path) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises imported root mismatch refuses without hash; release evidence must
    describe the exact package bytes and metadata used for the decision.
    """
    fixture = _make_wheel(tmp_path)
    other = tmp_path / "source" / "metrifid"
    other.mkdir(parents=True)
    with pytest.raises(DistributionIdentityError) as raised:
        _bound_distribution_identity(fixture.distribution, other.resolve(), {})
    assert (
        raised.value.reason_code is OperationalReasonCode.EXECUTING_CODE_NOT_INSTALLED_DISTRIBUTION
    )
    failure = raised.value.to_operational_failure("compare")
    assert failure.tool.execution_identity_state == "MISMATCH"
    assert failure.tool.distribution_sha256 is None


def test_loaded_module_mixed_root_refuses(tmp_path: Path) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises loaded module mixed root refuses; release evidence must describe the
    exact package bytes and metadata used for the decision.
    """
    fixture = _make_wheel(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("x = 1\n", encoding="utf-8")
    loaded = {
        "metrifid": SimpleNamespace(__file__=str(fixture.package_root / "__init__.py")),
        "metrifid.foreign": SimpleNamespace(__file__=str(outside)),
    }
    with pytest.raises(DistributionIdentityError) as raised:
        _identity(fixture, loaded)
    assert raised.value.reason_code is OperationalReasonCode.MIXED_METRIFID_MODULE_ROOTS


@pytest.mark.parametrize("module_file", [42, ""])
def test_loaded_module_invalid_file_refuses(tmp_path: Path, module_file: object) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises loaded module invalid file refuses; release evidence must describe
    the exact package bytes and metadata used for the decision.
    """
    fixture = _make_wheel(tmp_path)
    loaded = {"metrifid": SimpleNamespace(__file__=module_file)}
    with pytest.raises(DistributionIdentityError) as raised:
        _identity(fixture, loaded)
    assert raised.value.reason_code is OperationalReasonCode.MIXED_METRIFID_MODULE_ROOTS


def test_loaded_module_without_file_is_ignored(tmp_path: Path) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises loaded module without file is ignored; release evidence must
    describe the exact package bytes and metadata used for the decision.
    """
    fixture = _make_wheel(tmp_path)
    loaded = {
        "unrelated": SimpleNamespace(__file__=str(tmp_path / "outside.py")),
        "metrifid": SimpleNamespace(__file__=None),
    }
    assert _identity(fixture, loaded)["distribution_version"] == __version__


def test_loaded_module_absent_from_manifest_refuses(tmp_path: Path) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises loaded module absent from manifest refuses; release evidence must
    describe the exact package bytes and metadata used for the decision.
    """
    fixture = _make_wheel(tmp_path)
    extra = fixture.package_root / "runtime_only.py"
    extra.write_text("x = 1\n", encoding="utf-8")
    loaded = {
        "metrifid": SimpleNamespace(__file__=str(fixture.package_root / "__init__.py")),
        "metrifid.runtime_only": SimpleNamespace(__file__=str(extra)),
    }
    with pytest.raises(DistributionIdentityError, match="absent") as raised:
        _identity(fixture, loaded)
    assert raised.value.reason_code is OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID


def test_loaded_module_bytes_changed_after_install_refuses(tmp_path: Path) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises loaded module bytes changed after install refuses; release evidence
    must describe the exact package bytes and metadata used for the decision.
    """
    fixture = _make_wheel(tmp_path)
    module_path = fixture.package_root / "version.py"
    original = module_path.read_bytes()
    module_path.write_bytes(b"X" * len(original))
    loaded = {
        "metrifid": SimpleNamespace(__file__=str(fixture.package_root / "__init__.py")),
        "metrifid.version": SimpleNamespace(__file__=str(module_path)),
    }
    with pytest.raises(DistributionIdentityError, match="bytes differ") as raised:
        _identity(fixture, loaded)
    assert raised.value.reason_code is OperationalReasonCode.DISTRIBUTION_MANIFEST_INVALID


@pytest.mark.parametrize(
    "mutation",
    [
        "missing-files",
        "missing-metadata",
        "two-metadata",
        "missing-wheel",
        "missing-init",
        "wrong-version",
        "missing-license",
        "duplicate-path",
    ],
)
def test_invalid_distribution_shapes_refuse(tmp_path: Path, mutation: str) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises invalid distribution shapes refuse; release evidence must describe
    the exact package bytes and metadata used for the decision.
    """
    fixture = _make_wheel(
        tmp_path,
        version="9.9" if mutation == "wrong-version" else __version__,
        include_wheel=mutation != "missing-wheel",
        include_init=mutation != "missing-init",
        include_license=mutation != "missing-license",
        duplicate="metrifid/__init__.py" if mutation == "duplicate-path" else None,
    )
    distribution: metadata.Distribution = fixture.distribution
    if mutation == "missing-files":
        distribution = cast(metadata.Distribution, SimpleNamespace(files=None))
    elif mutation == "missing-metadata":
        rows = [
            row
            for row in csv.reader((fixture.dist_info / "RECORD").read_text().splitlines())
            if not row[0].endswith("/METADATA")
        ]
        with (fixture.dist_info / "RECORD").open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerows(rows)
        distribution = metadata.PathDistribution(fixture.dist_info)
    elif mutation == "two-metadata":
        extra = fixture.root / "other.dist-info" / "METADATA"
        extra.parent.mkdir()
        extra.write_text("Name: other\nVersion: 1\n", encoding="utf-8")
        _write_record(fixture.root, fixture.dist_info)
        distribution = metadata.PathDistribution(fixture.dist_info)
    with pytest.raises((DistributionIdentityError, AttributeError)):
        _bound_distribution_identity(distribution, fixture.package_root.resolve(), {})


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (("", "0"), "lacks a SHA-256"),
        (("md5=AAAA", "0"), "lacks a SHA-256"),
        (("sha256=***", "0"), "invalid SHA-256"),
        (("sha256=AAAA", "not-an-int"), "RECORD is malformed"),
        (("sha256=AAAA", "999"), "size differs"),
    ],
)
def test_bad_record_metadata_refuses(
    tmp_path: Path, override: tuple[str, str], message: str
) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises bad record metadata refuses; release evidence must describe the
    exact package bytes and metadata used for the decision.
    """
    fixture = _make_wheel(
        tmp_path,
        overrides={"metrifid/py.typed": override},
    )
    with pytest.raises((DistributionIdentityError, ValueError, TypeError), match=message):
        _identity(fixture)


def test_missing_nonregular_symlinked_and_unreadable_members_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises missing nonregular symlinked and unreadable members refuse; release
    evidence must describe the exact package bytes and metadata used for the decision.
    """
    missing = _make_wheel(tmp_path / "missing")
    (missing.package_root / "py.typed").unlink()
    with pytest.raises(DistributionIdentityError, match="missing"):
        _identity(missing)

    nonregular = _make_wheel(tmp_path / "directory")
    target = nonregular.package_root / "py.typed"
    target.unlink()
    target.mkdir()
    with pytest.raises(DistributionIdentityError, match="not a regular"):
        _identity(nonregular)

    symlinked = _make_wheel(tmp_path / "symlink")
    target = symlinked.package_root / "py.typed"
    target.unlink()
    target.symlink_to(symlinked.package_root / "__init__.py")
    with pytest.raises(DistributionIdentityError, match="symlinked"):
        _identity(symlinked)

    unreadable = _make_wheel(tmp_path / "unreadable")
    original = Path.read_bytes

    def fail_read(path: Path) -> bytes:
        """Inject the deterministic fail read branch required by this scenario.

        The distribution identity test can assert failure delivery for missing nonregular
        symlinked and unreadable members refuse without depending on incidental runtime errors.
        """
        if path.name == "py.typed":
            raise OSError("denied")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", fail_read)
    with pytest.raises(DistributionIdentityError, match="unreadable"):
        _identity(unreadable)


def test_imported_package_root_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises imported package root validation; release evidence must describe the
    exact package bytes and metadata used for the decision.
    """
    monkeypatch.delitem(sys.modules, "metrifid", raising=False)
    with pytest.raises(DistributionIdentityError, match="not an imported"):
        _imported_package_root()

    module = ModuleType("metrifid")
    monkeypatch.setitem(sys.modules, "metrifid", module)
    with pytest.raises(DistributionIdentityError, match="no package search"):
        _imported_package_root()

    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    spec = importlib.machinery.ModuleSpec("metrifid", loader=None, is_package=True)
    spec.submodule_search_locations = [str(one), str(two)]
    module.__spec__ = spec
    with pytest.raises(DistributionIdentityError, match="exactly one"):
        _imported_package_root()

    spec.submodule_search_locations = [str(one)]
    assert _imported_package_root() == one.resolve()

    file_root = tmp_path / "file-root"
    file_root.write_text("not a directory", encoding="utf-8")
    spec.submodule_search_locations = [str(file_root)]
    with pytest.raises(DistributionIdentityError, match="not a directory"):
        _imported_package_root()

    symlink_root = tmp_path / "symlink-root"
    symlink_root.symlink_to(one, target_is_directory=True)
    spec.submodule_search_locations = [str(symlink_root)]
    with pytest.raises(DistributionIdentityError, match="symlinked"):
        _imported_package_root()
