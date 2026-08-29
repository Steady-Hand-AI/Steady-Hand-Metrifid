"""Exact installed runtime and native MuJoCo environment identity for comparison receipts."""

from __future__ import annotations

import hashlib
import importlib.metadata as metadata
import os
import platform
import stat
from pathlib import Path, PurePosixPath
from typing import cast

import mujoco  # type: ignore[import-untyped]
import numpy as np

from .._mujoco_runtime import MujocoClaimSurface, admit_mujoco_runtime
from ..errors import EngineThreadpoolState
from ..json_values import CanonicalValue, canonical_sha256
from ..schemas import EnvironmentIdentity

# Official MuJoCo packaging gives the versioned native engine a platform-specific
# filename. Exactly one regular nonsymlink match for the admitted version must resolve.
_NATIVE_LIBRARY_NAME_TEMPLATES: dict[str, str] = {
    "Linux": "libmujoco.so.{version}",
    "Darwin": "libmujoco.{version}.dylib",
}


def build_environment_identity(
    threadpool_state: EngineThreadpoolState,
) -> EnvironmentIdentity:
    """Measure the exact Python/native engine and bounded host identity."""
    libc_name, libc_version = platform.libc_ver()
    return EnvironmentIdentity(
        mujoco_version=str(mujoco.__version__),
        python_version=platform.python_version(),
        numpy_version=str(np.__version__),
        mujoco_python_distribution_sha256=mujoco_distribution_payload_sha256(),
        mujoco_native_library_sha256=native_mujoco_library_sha256(),
        platform=f"{platform.system().lower()}-{platform.machine().lower()}",
        platform_release=platform.release(),
        libc=f"{libc_name or 'unknown'}-{libc_version or 'unknown'}",
        cpu_identity_sha256=canonical_sha256(_cpu_identity()),
        engine_threadpool_state=threadpool_state,
        environment_sha256=None,
    )


def mujoco_distribution_payload_sha256() -> str:
    """Measure the installed MuJoCo Python distribution payload once, for any consumer."""
    return _distribution_payload_sha256("mujoco", "mujoco")


def native_mujoco_library_sha256() -> str:
    """Measure the single native MuJoCo engine library once, for any consumer."""
    return _native_mujoco_sha256()


def combine_threadpool_states(
    baseline: EngineThreadpoolState,
    candidate: EngineThreadpoolState,
) -> EngineThreadpoolState:
    """Combine role-local threadpool observations using conservative precedence."""
    if EngineThreadpoolState.ACTIVE in {baseline, candidate}:
        return EngineThreadpoolState.ACTIVE
    if EngineThreadpoolState.UNKNOWN in {baseline, candidate}:
        return EngineThreadpoolState.UNKNOWN
    return EngineThreadpoolState.DISABLED


def _distribution_payload_sha256(distribution_name: str, package_name: str) -> str:
    """Hash the installed files of one runtime Python distribution."""
    distribution = metadata.distribution(distribution_name)
    files = distribution.files
    if files is None:
        raise RuntimeError(f"{distribution_name} distribution has no file manifest")
    members: list[CanonicalValue] = []
    for package_path in sorted(files, key=lambda item: PurePosixPath(str(item)).as_posix()):
        normalized = PurePosixPath(str(package_path)).as_posix()
        include = normalized.startswith(f"{package_name}/") or normalized.endswith(
            (".dist-info/METADATA", ".dist-info/WHEEL", ".dist-info/entry_points.txt")
        )
        if ".dist-info/licenses/" in normalized:
            include = True
        if (
            not include
            or "/__pycache__/" in f"/{normalized}/"
            or normalized.endswith((".pyc", ".pyo"))
        ):
            continue
        # locate_file is typed as returning a SimplePath, which Path() does not accept. The same
        # str() normalization already used for `normalized` above yields the identical filesystem
        # path, so the measured member set and the resulting digest are unchanged.
        path = Path(str(distribution.locate_file(package_path)))
        try:
            if path.is_symlink():
                raise OSError("symlinked distribution member")
            mode = path.lstat().st_mode
            if not stat.S_ISREG(mode):
                raise OSError("non-regular distribution member")
            payload = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"unreadable distribution member: {normalized}") from exc
        members.append(
            {
                "path": normalized,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    if not members:
        raise RuntimeError(f"{distribution_name} distribution payload is empty")
    return canonical_sha256(
        {
            "schema": "metrifid.installed_distribution_identity",
            "schema_version": 1,
            "distribution_name": distribution_name,
            "distribution_version": distribution.version,
            "members": members,
        }
    )


def _native_mujoco_sha256() -> str:
    """Hash the native MuJoCo shared library loaded by the Python package."""
    name_template = _NATIVE_LIBRARY_NAME_TEMPLATES.get(platform.system())
    if name_template is None:
        raise RuntimeError("native MuJoCo library discovery is unsupported on this platform")
    package_file = getattr(mujoco, "__file__", None)
    if type(package_file) is not str:
        raise RuntimeError("mujoco package path is unavailable")
    root = Path(package_file).resolve(strict=True).parent
    admission = admit_mujoco_runtime(
        MujocoClaimSurface.DYNAMIC_REPLAY,
        runtime_module=mujoco,
    )
    exact_name = name_template.format(version=admission.native_version_string)
    selected = sorted(
        path for path in root.rglob(exact_name) if path.is_file() and not path.is_symlink()
    )
    if len(selected) != 1:
        raise RuntimeError(
            "exactly one native MuJoCo library matching the admitted runtime must be identifiable"
        )
    return hashlib.sha256(selected[0].read_bytes()).hexdigest()


def _cpu_identity() -> dict[str, CanonicalValue]:
    """Hash a bounded normalized host CPU description."""
    values: dict[str, CanonicalValue] = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file() and not cpuinfo.is_symlink():
        first: dict[str, str] = {}
        for line in cpuinfo.read_text(encoding="utf-8", errors="strict").splitlines():
            if not line:
                break
            if ":" in line:
                key, value = line.split(":", 1)
                key = key.strip()
                if key in {"vendor_id", "cpu family", "model", "model name", "stepping"}:
                    first[key] = value.strip()
        values["first_processor"] = cast(CanonicalValue, first)
    return values


__all__ = [
    "build_environment_identity",
    "combine_threadpool_states",
    "mujoco_distribution_payload_sha256",
    "native_mujoco_library_sha256",
]
