"""Native MuJoCo library identity selection on the admitted Linux and Darwin hosts."""

from __future__ import annotations

import hashlib
import platform
from pathlib import Path
from types import SimpleNamespace

import pytest

from metrifid.compare import _environment as environment

_PAYLOAD = b"native-mujoco-payload"


def _package_root(tmp_path: Path) -> Path:
    """Construct the package root fixture used by compare environment scenarios.

    Deterministic setup isolates compare environment without bypassing the contract boundary
    under assertion.
    """
    root = tmp_path / "mujoco"
    root.mkdir()
    (root / "__init__.py").write_text("", encoding="utf-8")
    return root


def _bind(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    system: str,
) -> None:
    """Construct the bind fixture used by compare environment scenarios.

    Deterministic setup isolates compare environment without bypassing the contract boundary
    under assertion.
    """
    monkeypatch.setattr(
        environment,
        "mujoco",
        SimpleNamespace(__file__=str(root / "__init__.py")),
    )
    monkeypatch.setattr(environment.platform, "system", lambda: system)


def _write(root: Path, name: str, payload: bytes = _PAYLOAD) -> Path:
    """Write write data into the isolated test workspace.

    The compare environment scenario observes real bytes and filesystem effects for compare
    environment.
    """
    path = root / name
    path.write_bytes(payload)
    return path


def test_linux_selects_shared_object(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises linux selects shared object; status, numerical evidence, and
    artifact publication must remain stable for the declared workload.
    """
    root = _package_root(tmp_path)
    _write(root, "libmujoco.so.3.10.0")
    _bind(monkeypatch, root, "Linux")
    assert environment._native_mujoco_sha256() == hashlib.sha256(_PAYLOAD).hexdigest()


def test_darwin_selects_dylib(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises darwin selects dylib; status, numerical evidence, and artifact
    publication must remain stable for the declared workload.
    """
    root = _package_root(tmp_path)
    _write(root, "libmujoco.3.10.0.dylib")
    _bind(monkeypatch, root, "Darwin")
    assert environment._native_mujoco_sha256() == hashlib.sha256(_PAYLOAD).hexdigest()


def test_darwin_ignores_the_linux_pattern(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises darwin ignores the linux pattern; status, numerical evidence, and
    artifact publication must remain stable for the declared workload.
    """
    root = _package_root(tmp_path)
    _write(root, "libmujoco.so.3.10.0")
    _bind(monkeypatch, root, "Darwin")
    with pytest.raises(RuntimeError):
        environment._native_mujoco_sha256()


def test_linux_ignores_the_darwin_pattern(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises linux ignores the darwin pattern; status, numerical evidence, and
    artifact publication must remain stable for the declared workload.
    """
    root = _package_root(tmp_path)
    _write(root, "libmujoco.3.10.0.dylib")
    _bind(monkeypatch, root, "Linux")
    with pytest.raises(RuntimeError):
        environment._native_mujoco_sha256()


@pytest.mark.parametrize("system", ["Linux", "Darwin"])
def test_zero_candidates_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, system: str
) -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises zero candidates refuse; status, numerical evidence, and artifact
    publication must remain stable for the declared workload.
    """
    root = _package_root(tmp_path)
    _bind(monkeypatch, root, system)
    with pytest.raises(RuntimeError):
        environment._native_mujoco_sha256()


@pytest.mark.parametrize(
    ("system", "first", "second"),
    [
        ("Linux", "libmujoco.so.3.10.0", "libmujoco.so.3.10.0.backup"),
        ("Darwin", "libmujoco.3.10.0.dylib", "libmujoco.3.10.0.copy.dylib"),
    ],
)
def test_ambiguous_candidates_refuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    first: str,
    second: str,
) -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises ambiguous candidates refuse; status, numerical evidence, and
    artifact publication must remain stable for the declared workload.
    """
    root = _package_root(tmp_path)
    _write(root, first)
    _write(root, second, b"other-payload")
    _bind(monkeypatch, root, system)
    with pytest.raises(RuntimeError):
        environment._native_mujoco_sha256()


@pytest.mark.parametrize(
    ("system", "name"),
    [("Linux", "libmujoco.so.3.10.0"), ("Darwin", "libmujoco.3.10.0.dylib")],
)
def test_symlinked_library_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, system: str, name: str
) -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises symlinked library is refused; status, numerical evidence, and
    artifact publication must remain stable for the declared workload.
    """
    root = _package_root(tmp_path)
    target = _write(root, "engine.bin")
    (root / name).symlink_to(target)
    _bind(monkeypatch, root, system)
    with pytest.raises(RuntimeError):
        environment._native_mujoco_sha256()


@pytest.mark.parametrize(
    ("system", "real", "link"),
    [
        ("Linux", "libmujoco.so.3.10.0", "libmujoco.so"),
        ("Darwin", "libmujoco.3.10.0.dylib", "libmujoco.dylib"),
    ],
)
def test_symlink_beside_regular_library_does_not_create_ambiguity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    real: str,
    link: str,
) -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises symlink beside regular library does not create ambiguity; status,
    numerical evidence, and artifact publication must remain stable for the declared workload.
    """
    root = _package_root(tmp_path)
    target = _write(root, real)
    (root / link).symlink_to(target)
    _bind(monkeypatch, root, system)
    assert environment._native_mujoco_sha256() == hashlib.sha256(_PAYLOAD).hexdigest()


@pytest.mark.parametrize(
    ("system", "exact", "other"),
    [
        ("Linux", "libmujoco.so.3.10.0", "libmujoco.so.3.9.0"),
        ("Darwin", "libmujoco.3.10.0.dylib", "libmujoco.3.9.0.dylib"),
    ],
)
def test_exact_version_narrowing_is_preserved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    exact: str,
    other: str,
) -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises exact version narrowing is preserved; status, numerical evidence,
    and artifact publication must remain stable for the declared workload.
    """
    root = _package_root(tmp_path)
    _write(root, exact)
    _write(root, other, b"stale-payload")
    _bind(monkeypatch, root, system)
    assert environment._native_mujoco_sha256() == hashlib.sha256(_PAYLOAD).hexdigest()


@pytest.mark.parametrize("system", ["Windows", "Java", "FreeBSD", ""])
def test_unadmitted_platforms_have_no_discovery_pattern(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, system: str
) -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises unadmitted platforms have no discovery pattern; status, numerical
    evidence, and artifact publication must remain stable for the declared workload.
    """
    root = _package_root(tmp_path)
    _write(root, "libmujoco.so.3.10.0")
    _write(root, "libmujoco.3.10.0.dylib")
    _bind(monkeypatch, root, system)
    with pytest.raises(RuntimeError):
        environment._native_mujoco_sha256()


def test_missing_package_path_refuses(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises missing package path refuses; status, numerical evidence, and
    artifact publication must remain stable for the declared workload.
    """
    monkeypatch.setattr(environment, "mujoco", SimpleNamespace())
    monkeypatch.setattr(environment.platform, "system", lambda: "Darwin")
    with pytest.raises(RuntimeError):
        environment._native_mujoco_sha256()


def test_installed_environment_identity_matches_the_running_host() -> None:
    """The real installed runtime must produce a schema-valid identity."""
    identity = environment.build_environment_identity(environment.EngineThreadpoolState.UNKNOWN)
    assert identity.mujoco_version == "3.10.0"
    assert identity.platform == f"{platform.system().lower()}-{platform.machine().lower()}"
    assert identity.libc
    assert identity.platform_release
    assert len(identity.mujoco_native_library_sha256) == 64
