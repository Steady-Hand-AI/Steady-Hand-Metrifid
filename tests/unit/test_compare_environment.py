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
    *,
    native_version: str = "3.12.0",
) -> None:
    """Construct the bind fixture used by compare environment scenarios.

    Deterministic setup isolates compare environment without bypassing the contract boundary
    under assertion.
    """
    runtime = SimpleNamespace(__file__=str(root / "__init__.py"))
    monkeypatch.setattr(environment, "mujoco", runtime)
    monkeypatch.setattr(environment.platform, "system", lambda: system)

    def admit(
        operation: environment.MujocoClaimSurface,
        *,
        runtime_module: object,
    ) -> SimpleNamespace:
        """Return the coherent native identity measured at the admission boundary."""
        assert operation is environment.MujocoClaimSurface.DYNAMIC_REPLAY
        assert runtime_module is runtime
        return SimpleNamespace(native_version_string=native_version)

    monkeypatch.setattr(environment, "admit_mujoco_runtime", admit)


def _write(root: Path, name: str, payload: bytes = _PAYLOAD) -> Path:
    """Write write data into the isolated test workspace.

    The compare environment scenario observes real bytes and filesystem effects for compare
    environment.
    """
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def test_linux_selects_shared_object(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises linux selects shared object; status, numerical evidence, and
    artifact publication must remain stable for the declared workload.
    """
    root = _package_root(tmp_path)
    _write(root, "libmujoco.so.3.12.0")
    _bind(monkeypatch, root, "Linux")
    assert environment._native_mujoco_sha256() == hashlib.sha256(_PAYLOAD).hexdigest()


def test_darwin_selects_dylib(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises darwin selects dylib; status, numerical evidence, and artifact
    publication must remain stable for the declared workload.
    """
    root = _package_root(tmp_path)
    _write(root, "libmujoco.3.12.0.dylib")
    _bind(monkeypatch, root, "Darwin")
    assert environment._native_mujoco_sha256() == hashlib.sha256(_PAYLOAD).hexdigest()


def test_darwin_ignores_the_linux_pattern(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises darwin ignores the linux pattern; status, numerical evidence, and
    artifact publication must remain stable for the declared workload.
    """
    root = _package_root(tmp_path)
    _write(root, "libmujoco.so.3.12.0")
    _bind(monkeypatch, root, "Darwin")
    with pytest.raises(RuntimeError):
        environment._native_mujoco_sha256()


def test_linux_ignores_the_darwin_pattern(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises linux ignores the darwin pattern; status, numerical evidence, and
    artifact publication must remain stable for the declared workload.
    """
    root = _package_root(tmp_path)
    _write(root, "libmujoco.3.12.0.dylib")
    _bind(monkeypatch, root, "Linux")
    with pytest.raises(RuntimeError):
        environment._native_mujoco_sha256()


@pytest.mark.parametrize("system", ["Linux", "Darwin"])
def test_missing_exact_measured_library_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, system: str
) -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises missing exact measured library refusal; status, numerical evidence,
    and artifact publication must remain stable for the declared workload.
    """
    root = _package_root(tmp_path)
    _bind(monkeypatch, root, system)
    with pytest.raises(RuntimeError):
        environment._native_mujoco_sha256()


@pytest.mark.parametrize(
    ("system", "exact"),
    [
        ("Linux", "libmujoco.so.3.12.0"),
        ("Darwin", "libmujoco.3.12.0.dylib"),
    ],
    ids=("linux", "darwin"),
)
def test_ambiguous_duplicate_exact_measured_libraries_refuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    exact: str,
) -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises ambiguous duplicate exact measured libraries refusal; status,
    numerical evidence, and artifact publication must remain stable for the declared workload.
    """
    root = _package_root(tmp_path)
    _write(root, exact)
    _write(root, f"duplicate/{exact}", b"other-payload")
    _bind(monkeypatch, root, system)
    with pytest.raises(RuntimeError):
        environment._native_mujoco_sha256()


@pytest.mark.parametrize(
    ("system", "name"),
    [("Linux", "libmujoco.so.3.12.0"), ("Darwin", "libmujoco.3.12.0.dylib")],
    ids=("linux", "darwin"),
)
def test_symlink_only_exact_measured_library_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, system: str, name: str
) -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises symlink-only exact measured library refusal; status, numerical
    evidence, and artifact publication must remain stable for the declared workload.
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
        ("Linux", "libmujoco.so.3.12.0", "libmujoco.so"),
        ("Darwin", "libmujoco.3.12.0.dylib", "libmujoco.dylib"),
    ],
    ids=("linux", "darwin"),
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
    ("system", "current", "stale"),
    [
        ("Linux", "libmujoco.so.3.12.0", "libmujoco.so.3.10.0"),
        ("Darwin", "libmujoco.3.12.0.dylib", "libmujoco.3.10.0.dylib"),
    ],
    ids=("linux", "darwin"),
)
def test_matching_native_library_is_selected_when_stale_older_bytes_are_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    current: str,
    stale: str,
) -> None:
    """Bind the admitted runtime library rather than a stale package artifact."""
    root = _package_root(tmp_path)
    _write(root, current)
    stale_path = _write(root, stale, b"stale-payload")
    _bind(monkeypatch, root, system, native_version="3.12.0")
    observed = environment._native_mujoco_sha256()
    assert observed == hashlib.sha256(_PAYLOAD).hexdigest()
    assert observed != hashlib.sha256(stale_path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("system", "stale"),
    [
        ("Linux", "libmujoco.so.3.10.0"),
        ("Darwin", "libmujoco.3.10.0.dylib"),
    ],
    ids=("linux", "darwin"),
)
def test_stale_older_native_library_is_never_hashed_for_a_newer_admitted_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    stale: str,
) -> None:
    """Refuse instead of falling back when only stale native bytes are installed."""
    root = _package_root(tmp_path)
    _write(root, stale, b"stale-payload")
    _bind(monkeypatch, root, system, native_version="3.12.0")
    with pytest.raises(RuntimeError, match="matching the admitted runtime"):
        environment._native_mujoco_sha256()


@pytest.mark.parametrize(
    ("system", "near_match"),
    [
        ("Linux", "libmujoco.so.3.12.0.backup"),
        ("Darwin", "libmujoco.3.12.0.copy.dylib"),
    ],
    ids=("linux", "darwin"),
)
def test_similarly_named_file_does_not_replace_the_exact_platform_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    system: str,
    near_match: str,
) -> None:
    """Require the complete platform filename, not a prefix or glob-compatible variant."""
    root = _package_root(tmp_path)
    _write(root, near_match)
    _bind(monkeypatch, root, system)
    with pytest.raises(RuntimeError, match="matching the admitted runtime"):
        environment._native_mujoco_sha256()


@pytest.mark.parametrize("system", ["Windows", "Java", "FreeBSD", ""])
def test_unsupported_platforms_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, system: str
) -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises unadmitted platforms have no discovery pattern; status, numerical
    evidence, and artifact publication must remain stable for the declared workload.
    """
    root = _package_root(tmp_path)
    _write(root, "libmujoco.so.3.12.0")
    _write(root, "libmujoco.3.12.0.dylib")
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
    admission = environment.admit_mujoco_runtime(
        environment.MujocoClaimSurface.DYNAMIC_REPLAY,
        runtime_module=environment.mujoco,
    )
    assert identity.mujoco_version == admission.package_version
    assert admission.native_version_string == environment.mujoco.mj_versionString()
    assert admission.native_version_integer == environment.mujoco.mj_version()
    assert identity.platform == f"{platform.system().lower()}-{platform.machine().lower()}"
    assert identity.libc
    assert identity.platform_release
    assert len(identity.mujoco_native_library_sha256) == 64
