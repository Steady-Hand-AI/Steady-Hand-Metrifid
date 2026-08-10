"""Installed-distribution identity unit tests and hostile manifest cases."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata as metadata
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import metrifid.distribution as distribution_module
from metrifid.distribution import (
    DistributionIdentityError,
    _bound_distribution_identity,
    _normalize_distribution_path,
    _single_distribution,
    _urlsafe_b64decode,
)
from metrifid.json_values import CanonicalValue
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
    """Encode bytes as the URL-safe digest used by wheel RECORD rows."""
    encoded = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).decode("ascii")
    return encoded.rstrip("=")


def _write_record(
    root: Path,
    dist_info: Path,
    *,
    overrides: dict[str, tuple[str, str]] | None = None,
    duplicate: str | None = None,
) -> None:
    """Write verified member rows to the fixture wheel's RECORD file."""
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
    """Build an unpacked wheel fixture with configurable metadata defects."""
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
    """Construct the identity fixture used by distribution members scenarios.

    Deterministic setup isolates distribution members without bypassing the contract boundary
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
    """Construct the member paths fixture used by distribution members scenarios.

    Deterministic setup isolates distribution members without bypassing the contract boundary
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
        """Construct the init fixture used by distribution members scenarios.

        Deterministic setup isolates DistributionWrapper without bypassing the contract boundary
        under assertion.
        """
        self._base = base
        self._redirects = {} if redirects is None else redirects

    @property
    def version(self) -> str:
        """Construct the version fixture used by distribution members scenarios.

        Deterministic setup isolates DistributionWrapper without bypassing the contract boundary
        under assertion.
        """
        return self._base.version

    @property
    def metadata(self) -> object:
        """Construct the metadata fixture used by distribution members scenarios.

        Deterministic setup isolates DistributionWrapper without bypassing the contract boundary
        under assertion.
        """
        return self._base.metadata

    def read_text(self, filename: str) -> str | None:
        """Construct the read text fixture used by distribution members scenarios.

        Deterministic setup isolates DistributionWrapper without bypassing the contract boundary
        under assertion.
        """
        return self._base.read_text(filename)

    def locate_file(self, path: metadata.PackagePath) -> Path:
        """Construct the locate file fixture used by distribution members scenarios.

        Deterministic setup isolates DistributionWrapper without bypassing the contract boundary
        under assertion.
        """
        return self._redirects.get(str(path), Path(str(self._base.locate_file(path))))


def _rewrite_record_rows(fixture: WheelFixture, rows: list[list[str]]) -> None:
    """Write rewrite record rows data into the isolated test workspace.

    The distribution members scenario observes real bytes and filesystem effects for
    distribution members.
    """
    with (fixture.dist_info / "RECORD").open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)


def _record_rows(fixture: WheelFixture) -> list[list[str]]:
    """Construct the record rows fixture used by distribution members scenarios.

    Deterministic setup isolates distribution members without bypassing the contract boundary
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
    """Construct the make launcher fixture fixture used by distribution members scenarios.

    Deterministic setup isolates distribution members without bypassing the contract boundary
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
        """Construct the get path fixture used by distribution members scenarios.

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


def test_single_distribution_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises single distribution validation; release evidence must describe the
    exact package bytes and metadata used for the decision.
    """
    monkeypatch.setattr(metadata, "distributions", lambda **_: [])
    with pytest.raises(DistributionIdentityError, match="exactly one"):
        _single_distribution()

    marker = cast(metadata.Distribution, object())
    monkeypatch.setattr(metadata, "distributions", lambda **_: [marker, marker])
    with pytest.raises(DistributionIdentityError, match="exactly one"):
        _single_distribution()

    def fail(**_: object) -> list[metadata.Distribution]:
        """Inject the deterministic fail branch required by this scenario.

        The distribution members test can assert failure delivery for single distribution
        validation without depending on incidental runtime errors.
        """
        raise RuntimeError("boom")

    monkeypatch.setattr(metadata, "distributions", fail)
    with pytest.raises(DistributionIdentityError, match="cannot be enumerated"):
        _single_distribution()

    monkeypatch.setattr(metadata, "distributions", lambda **_: [marker])
    assert _single_distribution() is marker


@pytest.mark.parametrize("path", ["", "a\\b", "/absolute", "../escape", "a/../b", "a/./b"])
def test_invalid_distribution_paths_refuse(path: str) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises invalid distribution paths refuse; release evidence must describe
    the exact package bytes and metadata used for the decision.
    """
    with pytest.raises(DistributionIdentityError):
        _normalize_distribution_path(path)


def test_normalization_and_base64_helpers() -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises normalization and base64 helpers; release evidence must describe the
    exact package bytes and metadata used for the decision.
    """
    assert _normalize_distribution_path("metrifid/module.py") == "metrifid/module.py"
    raw = hashlib.sha256(b"payload").digest()
    token = base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")
    assert _urlsafe_b64decode(token) == raw
    with pytest.raises(DistributionIdentityError, match="invalid SHA-256"):
        _urlsafe_b64decode("***")


def test_exact_pip_console_launcher_row_is_verified_but_not_hashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises exact pip console launcher row is verified but not hashed; release
    evidence must describe the exact package bytes and metadata used for the decision.
    """
    fixture, launcher, record_path = _make_launcher_fixture(tmp_path, monkeypatch)
    assert record_path == "../../../bin/metrifid"
    identity = _identity(fixture)
    assert launcher.is_file()
    assert record_path not in _member_paths(identity)
    assert all("../" not in path for path in _member_paths(identity))


@pytest.mark.parametrize(
    "entry_points_text",
    [
        "[console_scripts]\nmetrifid = metrifid:main\n",
        "[console_scripts]\nmetrifid-cli = metrifid.cli:main\n",
        ("[console_scripts]\nmetrifid = metrifid.cli:main\nother = metrifid.cli:main\n"),
    ],
)
def test_external_launcher_requires_the_only_exact_console_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entry_points_text: str,
) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises external launcher requires the only exact console binding; release
    evidence must describe the exact package bytes and metadata used for the decision.
    """
    fixture, _, _ = _make_launcher_fixture(
        tmp_path,
        monkeypatch,
        entry_points_text=entry_points_text,
    )
    with pytest.raises(DistributionIdentityError, match="unauthorized external path"):
        _identity(fixture)


def test_exact_console_binding_requires_one_launcher_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises exact console binding requires one launcher record; release evidence
    must describe the exact package bytes and metadata used for the decision.
    """
    missing, _, expected = _make_launcher_fixture(
        tmp_path / "missing",
        monkeypatch,
        record_paths=(),
    )
    with pytest.raises(DistributionIdentityError, match="missing the expected"):
        _identity(missing)

    duplicate, _, duplicate_expected = _make_launcher_fixture(
        tmp_path / "duplicate",
        monkeypatch,
        record_paths=(expected, expected),
    )
    assert duplicate_expected == expected
    with pytest.raises(DistributionIdentityError, match="duplicate console launcher"):
        _identity(duplicate)


@pytest.mark.parametrize(
    "hostile_path",
    [
        "../../../bin/metrifid-cli",
        "../../../../tmp/metrifid",
        "/absolute/metrifid",
        "..\\..\\..\\bin\\metrifid",
        "../../../bin/../bin/metrifid",
    ],
)
def test_other_external_paths_remain_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hostile_path: str,
) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises other external paths remain refused; release evidence must describe
    the exact package bytes and metadata used for the decision.
    """
    fixture, _, _ = _make_launcher_fixture(
        tmp_path,
        monkeypatch,
        record_paths=(hostile_path,),
    )
    with pytest.raises(DistributionIdentityError):
        _identity(fixture)


def test_launcher_hash_size_and_file_type_are_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises launcher hash size and file type are enforced; release evidence must
    describe the exact package bytes and metadata used for the decision.
    """
    tampered, launcher, _ = _make_launcher_fixture(tmp_path / "tampered", monkeypatch)
    launcher.write_bytes(b"X" * len(launcher.read_bytes()))
    with pytest.raises(DistributionIdentityError, match="launcher bytes differ"):
        _identity(tampered)

    wrong_size, _, expected = _make_launcher_fixture(tmp_path / "size", monkeypatch)
    rows = _record_rows(wrong_size)
    for row in rows:
        if row[0] == expected:
            row[2] = str(int(row[2]) + 1)
    _rewrite_record_rows(wrong_size, rows)
    with pytest.raises(DistributionIdentityError, match="launcher size differs"):
        _identity(wrong_size)

    missing, launcher, _ = _make_launcher_fixture(tmp_path / "missing", monkeypatch)
    launcher.unlink()
    with pytest.raises(DistributionIdentityError, match="launcher is missing"):
        _identity(missing)

    symlinked, launcher, _ = _make_launcher_fixture(tmp_path / "symlink", monkeypatch)
    target = launcher.with_name("target")
    target.write_bytes(launcher.read_bytes())
    launcher.unlink()
    launcher.symlink_to(target)
    with pytest.raises(DistributionIdentityError, match="must not be a symlink"):
        _identity(symlinked)

    nonregular, launcher, _ = _make_launcher_fixture(tmp_path / "directory", monkeypatch)
    launcher.unlink()
    launcher.mkdir()
    with pytest.raises(DistributionIdentityError, match="not a regular file"):
        _identity(nonregular)


def test_launcher_record_hash_metadata_is_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises launcher record hash metadata is enforced; release evidence must
    describe the exact package bytes and metadata used for the decision.
    """
    fixture, _, expected = _make_launcher_fixture(tmp_path, monkeypatch)
    rows = _record_rows(fixture)
    for row in rows:
        if row[0] == expected:
            row[1] = "sha256=***"
    _rewrite_record_rows(fixture, rows)
    with pytest.raises(DistributionIdentityError, match="invalid SHA-256"):
        _identity(fixture)


def test_record_absent_and_malformed_row_refuse(tmp_path: Path) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises record absent and malformed row refuse; release evidence must
    describe the exact package bytes and metadata used for the decision.
    """
    fixture = _make_wheel(tmp_path / "absent")

    class NoRecordDistribution:
        """Represent no record distribution."""

        def read_text(self, filename: str) -> None:
            """Construct the read text fixture used by distribution members scenarios.

            Deterministic setup isolates NoRecordDistribution without bypassing the contract
            boundary under assertion.
            """
            return None

    with pytest.raises(DistributionIdentityError, match="does not expose"):
        _bound_distribution_identity(
            cast(metadata.Distribution, NoRecordDistribution()),
            fixture.package_root.resolve(),
            {},
        )

    malformed = _make_wheel(tmp_path / "malformed")
    _rewrite_record_rows(malformed, [["only", "two"]])
    with pytest.raises(DistributionIdentityError, match="RECORD is malformed"):
        _identity(malformed)


def test_record_invalid_hash_field_negative_size_and_missing_self_refuse(tmp_path: Path) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises record invalid hash field negative size and missing self refuse;
    release evidence must describe the exact package bytes and metadata used for the decision.
    """
    bad_hash = _make_wheel(tmp_path / "hash")
    rows = _record_rows(bad_hash)
    rows[0][1] = "="
    _rewrite_record_rows(bad_hash, rows)
    with pytest.raises(DistributionIdentityError, match="RECORD is malformed"):
        _identity(bad_hash)

    negative = _make_wheel(tmp_path / "negative")
    rows = _record_rows(negative)
    rows[0][2] = "-1"
    _rewrite_record_rows(negative, rows)
    with pytest.raises(DistributionIdentityError, match="RECORD is malformed"):
        _identity(negative)

    no_self = _make_wheel(tmp_path / "self")
    rows = [row for row in _record_rows(no_self) if not row[0].endswith(".dist-info/RECORD")]
    _rewrite_record_rows(no_self, rows)
    with pytest.raises(DistributionIdentityError, match="list itself"):
        _identity(no_self)


def test_install_root_mismatch_refuses(tmp_path: Path) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises install root mismatch refuses; release evidence must describe the
    exact package bytes and metadata used for the decision.
    """
    fixture = _make_wheel(tmp_path)
    outside = tmp_path / "other-site" / "metrifid" / "__init__.py"
    outside.parent.mkdir(parents=True)
    outside.write_bytes((fixture.package_root / "__init__.py").read_bytes())
    wrapped = _DistributionWrapper(
        fixture.distribution,
        {"metrifid/__init__.py": outside},
    )
    with pytest.raises(DistributionIdentityError, match="do not share"):
        _bound_distribution_identity(
            cast(metadata.Distribution, wrapped),
            outside.parent.resolve(),
            {},
        )


def test_dist_info_metadata_path_attacks_refuse(tmp_path: Path) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises dist info metadata path attacks refuse; release evidence must
    describe the exact package bytes and metadata used for the decision.
    """
    missing = _make_wheel(tmp_path / "missing")
    (missing.dist_info / "METADATA").unlink()
    with pytest.raises(DistributionIdentityError, match="METADATA is missing"):
        _identity(missing)

    symlinked = _make_wheel(tmp_path / "symlink")
    metadata_path = symlinked.dist_info / "METADATA"
    original = metadata_path.read_bytes()
    target = symlinked.root / "metadata-target"
    target.write_bytes(original)
    metadata_path.unlink()
    metadata_path.symlink_to(target)
    with pytest.raises(DistributionIdentityError, match="METADATA is missing"):
        _identity(symlinked)

    nonregular = _make_wheel(tmp_path / "directory")
    metadata_path = nonregular.dist_info / "METADATA"
    metadata_path.unlink()
    metadata_path.mkdir()
    with pytest.raises(DistributionIdentityError, match="not a regular"):
        _identity(nonregular)


def test_package_initializer_path_attacks_refuse(tmp_path: Path) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises package initializer path attacks refuse; release evidence must
    describe the exact package bytes and metadata used for the decision.
    """
    missing = _make_wheel(tmp_path / "missing")
    (missing.package_root / "__init__.py").unlink()
    with pytest.raises(DistributionIdentityError, match="initializer is missing"):
        _identity(missing)

    symlinked = _make_wheel(tmp_path / "symlink")
    initializer = symlinked.package_root / "__init__.py"
    target = symlinked.package_root / "init-target.py"
    target.write_bytes(initializer.read_bytes())
    initializer.unlink()
    initializer.symlink_to(target)
    with pytest.raises(DistributionIdentityError, match="initializer is missing"):
        _identity(symlinked)

    nonregular = _make_wheel(tmp_path / "directory")
    initializer = nonregular.package_root / "__init__.py"
    initializer.unlink()
    initializer.mkdir()
    with pytest.raises(DistributionIdentityError, match="not a regular"):
        _identity(nonregular)


def test_loaded_module_path_attacks_refuse(tmp_path: Path) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises loaded module path attacks refuse; release evidence must describe
    the exact package bytes and metadata used for the decision.
    """
    fixture = _make_wheel(tmp_path)

    missing = fixture.package_root / "missing.py"
    with pytest.raises(DistributionIdentityError, match="module file is missing"):
        _identity(fixture, {"metrifid.missing": SimpleNamespace(__file__=str(missing))})

    directory = fixture.package_root / "directory_module.py"
    directory.mkdir()
    with pytest.raises(DistributionIdentityError, match="not regular"):
        _identity(fixture, {"metrifid.directory": SimpleNamespace(__file__=str(directory))})

    link = fixture.package_root / "linked_module.py"
    link.symlink_to(fixture.package_root / "version.py")
    with pytest.raises(DistributionIdentityError, match="module file is missing"):
        _identity(fixture, {"metrifid.link": SimpleNamespace(__file__=str(link))})


def test_nondict_direct_url_is_noneditable_observation(tmp_path: Path) -> None:
    """Keep installed-distribution identity bound to the executing wheel.

    This scenario exercises nondict direct url is noneditable observation; release evidence must
    describe the exact package bytes and metadata used for the decision.
    """
    fixture = _make_wheel(tmp_path, direct_url=[])
    assert _identity(fixture)["distribution_version"] == __version__
