"""Environment admission, guarded compilation, and external-state admission.

The compile guard, the process-wide compile lock and the refusals that depend on implementation
outside this process live here. ``_model_admission`` re-exports them, so every existing import
path keeps working.
"""

from __future__ import annotations

import os
import platform
import re
import sys
import threading
from collections.abc import Callable, Iterable
from typing import SupportsInt, cast

import mujoco  # type: ignore[import-untyped]

from ._model_closure import ModelClosureSnapshot, ModelRole, refuse
from ._model_dependencies import (
    discover_snapshot_dependencies,
    first_complete_root_element,
    read_measured_entrypoint_bytes,
)
from .json_values import CanonicalValue
from .operational import OperationalReasonCode

_MINIMUM_PYTHON: tuple[int, int] = (3, 11)
_SUPPORTED_SYSTEMS: frozenset[str] = frozenset({"Linux", "Darwin"})
_REQUIRED_DIR_FD: tuple[str, ...] = ("open", "stat", "mkdir", "rmdir", "link", "unlink")
_REQUIRED_FOLLOW_SYMLINKS: tuple[str, ...] = ("stat", "link")
_REQUIRED_FD: tuple[str, ...] = ("listdir", "scandir")
_REQUIRED_CALLABLES: tuple[str, ...] = (
    "pread",
    "fstat",
    "fsync",
    "dup",
    "fdopen",
    "read",
    "close",
)
_REQUIRED_FLAGS: tuple[str, ...] = ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
# The engine profile is exact: only the stable 3.10.0 release family is admitted. A binding-only
# ``.postN`` rebuild targets that same native engine and is accepted; prerelease, development,
# 3.10.1 and 3.11+ package versions are not.
_MUJOCO_PACKAGE_PATTERN = re.compile(
    r"\A3\.10\.0(?:\.post(?:0|[1-9][0-9]*))?(?:\+[a-z0-9]+(?:[._-][a-z0-9]+)*)?\Z",
    re.IGNORECASE,
)
_COMPILE_LOCK = threading.RLock()
_CALLBACK_ACCESSORS: tuple[tuple[str, Callable[[], object | None]], ...] = (
    ("mjcb_control", mujoco.get_mjcb_control),
    ("mjcb_sensor", mujoco.get_mjcb_sensor),
    ("mjcb_passive", mujoco.get_mjcb_passive),
    ("mjcb_act_dyn", mujoco.get_mjcb_act_dyn),
    ("mjcb_act_gain", mujoco.get_mjcb_act_gain),
    ("mjcb_act_bias", mujoco.get_mjcb_act_bias),
    ("mjcb_contactfilter", mujoco.get_mjcb_contactfilter),
    ("mjcb_time", mujoco.get_mjcb_time),
    ("mju_user_warning", mujoco.get_mju_user_warning),
    ("mju_user_malloc", mujoco.get_mju_user_malloc),
    ("mju_user_free", mujoco.get_mju_user_free),
)


def _require_minimum_python() -> None:
    """Refuse a Python older than the supported minimum, with no upper bound.

    Only the language level is compared. The interpreter implementation name is never inspected
    here, and no finite set of accepted minors exists, so a future minor is admitted whenever the
    measured POSIX capability surface below is present.
    """
    if (sys.version_info.major, sys.version_info.minor) < _MINIMUM_PYTHON:
        raise refuse(
            OperationalReasonCode.UNSUPPORTED_PYTHON_VERSION,
            "comparison",
            python_version=platform.python_version(),
            minimum_python=f"{_MINIMUM_PYTHON[0]}.{_MINIMUM_PYTHON[1]}",
        )


def _missing_posix_capabilities() -> list[str]:
    """Return every required POSIX capability this interpreter does not provide.

    Architecture is never consulted here. Each entry names the capability class and the specific
    operating-system facility the source relies on for confined, no-follow filesystem work.
    """
    missing: list[str] = []
    missing.extend(
        f"dir_fd:{name}"
        for name in _REQUIRED_DIR_FD
        if getattr(os, name, None) not in os.supports_dir_fd
    )
    missing.extend(
        f"follow_symlinks:{name}"
        for name in _REQUIRED_FOLLOW_SYMLINKS
        if getattr(os, name, None) not in os.supports_follow_symlinks
    )
    missing.extend(
        f"fd:{name}" for name in _REQUIRED_FD if getattr(os, name, None) not in os.supports_fd
    )
    missing.extend(
        f"callable:{name}" for name in _REQUIRED_CALLABLES if not callable(getattr(os, name, None))
    )
    missing.extend(f"flag:{name}" for name in _REQUIRED_FLAGS if not int(getattr(os, name, 0) or 0))
    return missing


def _require_posix_platform() -> None:
    """Refuse a non-POSIX operating system or one missing a required POSIX capability.

    Architecture is never consulted: no allowlist exists here, and the runtime identity receipt is
    the single place that records the machine string as evidence. Native Windows use is
    unsupported because these capabilities are absent; WSL is the documented route.
    """
    system = platform.system()
    if os.name != "posix" or system not in _SUPPORTED_SYSTEMS:
        raise refuse(
            OperationalReasonCode.UNSUPPORTED_PLATFORM,
            "comparison",
            system=system,
            os_name=os.name,
        )
    missing = _missing_posix_capabilities()
    if missing:
        raise refuse(
            OperationalReasonCode.UNSUPPORTED_PLATFORM,
            "comparison",
            system=system,
            os_name=os.name,
            missing_posix_capabilities=cast(CanonicalValue, missing),
        )


def _is_supported_mujoco_package(version: object) -> bool:
    """Report whether one MuJoCo distribution version targets the exact 3.10.0 engine family."""
    return isinstance(version, str) and _MUJOCO_PACKAGE_PATTERN.match(version) is not None


def _require_mujoco_engine() -> None:
    """Refuse a MuJoCo package or loaded native engine outside the exact 3.10.0 profile."""
    package_version = getattr(mujoco, "__version__", None)
    if not _is_supported_mujoco_package(package_version):
        raise refuse(
            OperationalReasonCode.UNSUPPORTED_MUJOCO_VERSION,
            "comparison",
            mujoco_python_version=package_version if isinstance(package_version, str) else None,
        )
    native_string, native_integer = mujoco.mj_versionString(), mujoco.mj_version()
    if native_string != "3.10.0" or native_integer != 3010000:
        raise refuse(
            OperationalReasonCode.MUJOCO_PYTHON_NATIVE_VERSION_MISMATCH,
            "comparison",
            mujoco_python_version=package_version,
            native_version_string=native_string,
            native_version_integer=native_integer,
        )


def require_supported_runtime() -> None:
    """Admit the shared native runtime for every compile or step path.

    Certify, Compare, Audit Timestep and compiled model identity all pass through this one gate.
    Pure artifact writers, canonical JSON helpers and receipt validation never call it.
    """
    _require_minimum_python()
    _require_posix_platform()
    _require_mujoco_engine()


def _compile_reason(role: ModelRole, *, warning: bool) -> OperationalReasonCode:
    """Select the role-specific compile or compile-warning refusal code."""
    if role == "baseline":
        return (
            OperationalReasonCode.BASELINE_MODEL_COMPILE_WARNING
            if warning
            else OperationalReasonCode.BASELINE_MODEL_COMPILE_ERROR
        )
    if role == "candidate":
        return (
            OperationalReasonCode.CANDIDATE_MODEL_COMPILE_WARNING
            if warning
            else OperationalReasonCode.CANDIDATE_MODEL_COMPILE_ERROR
        )
    raise ValueError("model compilation requires a baseline or candidate role")


def _require_mjcf_root(snapshot: ModelClosureSnapshot, role: ModelRole) -> None:
    """Bounded main-root precheck over the measured entrypoint member.

    The entrypoint bytes are read through the same no-follow, size- and hash-verified path used by
    dependency discovery, so a missing, symlinked, non-regular, or mutated entrypoint refuses before
    MuJoCo runs. Only the first complete top-level element is required to be ``mujoco``; trailing
    bytes are ignored here because MuJoCo 3.10.0 tolerates them and remains the final syntax and
    compile authority.
    """
    root = first_complete_root_element(read_measured_entrypoint_bytes(snapshot, role))
    if root is None:
        raise refuse(
            _compile_reason(role, warning=False),
            role,
            issue="no_complete_top_level_element",
            required_root_element="mujoco",
        )
    if root.tag != "mujoco":
        raise refuse(
            _compile_reason(role, warning=False),
            role,
            root_element=root.tag,
            required_root_element="mujoco",
        )


def compile_snapshot_model(snapshot: ModelClosureSnapshot, role: ModelRole) -> mujoco.MjModel:
    """Compile one immutable snapshot while converting warnings and failures to typed refusals."""
    if role not in {"baseline", "candidate"}:
        raise ValueError("model compilation requires a baseline or candidate role")
    warnings: list[str] = []
    with _COMPILE_LOCK:
        active = [name for name, getter in _CALLBACK_ACCESSORS if getter() is not None]
        if active:
            raise refuse(
                OperationalReasonCode.UNSUPPORTED_USER_CALLBACK,
                role,
                active_callbacks=cast(CanonicalValue, active),
            )
        discover_snapshot_dependencies(snapshot, role)
        _require_mjcf_root(snapshot, role)
        entrypoint = snapshot.snapshot_entrypoint
        previous_warning = mujoco.get_mju_user_warning()
        mujoco.set_mju_user_warning(warnings.append)
        try:
            try:
                model = mujoco.MjModel.from_xml_path(str(entrypoint))
            except (ValueError, mujoco.FatalError) as exc:
                raise refuse(
                    _compile_reason(role, warning=False),
                    role,
                    exception_type=type(exc).__name__,
                    message=str(exc),
                ) from exc
        finally:
            mujoco.set_mju_user_warning(previous_warning)
        if warnings:
            raise refuse(
                _compile_reason(role, warning=True),
                role,
                warnings=list(warnings),
            )
        return model


def _active_indices(values: Iterable[SupportsInt]) -> list[int]:
    """Return indices whose MuJoCo integer flag is nonzero."""
    return [index for index, value in enumerate(values) if int(value) >= 0]


def _active_plugin_arrays(model: mujoco.MjModel) -> dict[str, list[int]]:
    """Report compiled plugin-state arrays that contain active entries."""
    result: dict[str, list[int]] = {}
    for name in ("body_plugin", "geom_plugin", "actuator_plugin", "sensor_plugin"):
        values = getattr(model, name, None)
        if values is not None and (indices := _active_indices(values)):
            result[name] = indices
    return result


def _active_history_arrays(model: mujoco.MjModel) -> dict[str, list[int]]:
    """Report compiled history arrays that contain active entries."""
    result: dict[str, list[int]] = {}
    for name in ("actuator_historyadr", "sensor_historyadr"):
        values = getattr(model, name, None)
        if values is not None and (indices := _active_indices(values)):
            result[name] = indices
    return result


def _user_actuator_indices(model: mujoco.MjModel) -> list[int]:
    """Return actuator indices that invoke user gain, bias, or dynamics callbacks."""
    return [
        index
        for index in range(model.nu)
        if int(model.actuator_dyntype[index]) == int(mujoco.mjtDyn.mjDYN_USER)
        or int(model.actuator_gaintype[index]) == int(mujoco.mjtGain.mjGAIN_USER)
        or int(model.actuator_biastype[index]) == int(mujoco.mjtBias.mjBIAS_USER)
    ]


def _sensor_indices(model: mujoco.MjModel, sensor_type: int) -> list[int]:
    """Return compiled sensor indices of one MuJoCo sensor type."""
    return [index for index in range(model.nsensor) if int(model.sensor_type[index]) == sensor_type]


def admit_external_implementation_free_model(model: mujoco.MjModel, role: ModelRole) -> None:
    """Refuse a compiled model whose meaning depends on implementation outside this process.

    This is the admission subset a compile-only operation needs. Plugins, user callbacks and
    plugin sensors name behavior supplied by code this project never observes. Refusals about
    what happens while a model is stepped stay in ``admit_compiled_model``.
    """
    if role not in {"baseline", "candidate"}:
        raise ValueError("model admission requires a baseline or candidate role")
    plugin_arrays = _active_plugin_arrays(model)
    if model.nplugin != 0 or model.npluginstate != 0 or plugin_arrays:
        raise refuse(
            OperationalReasonCode.UNSUPPORTED_PLUGIN_STATE,
            role,
            nplugin=int(model.nplugin),
            npluginstate=int(model.npluginstate),
            active_plugin_references=cast(CanonicalValue, plugin_arrays),
        )
    user_actuators = _user_actuator_indices(model)
    user_sensors = _sensor_indices(model, int(mujoco.mjtSensor.mjSENS_USER))
    if user_actuators or user_sensors:
        raise refuse(
            OperationalReasonCode.UNSUPPORTED_USER_CALLBACK,
            role,
            user_actuator_indices=cast(CanonicalValue, user_actuators),
            user_sensor_indices=cast(CanonicalValue, user_sensors),
        )
    plugin_sensors = _sensor_indices(model, int(mujoco.mjtSensor.mjSENS_PLUGIN))
    if plugin_sensors:
        raise refuse(
            OperationalReasonCode.UNSUPPORTED_PLUGIN_STATE,
            role,
            plugin_sensor_indices=cast(CanonicalValue, plugin_sensors),
        )


def admit_compiled_model(model: mujoco.MjModel, role: ModelRole) -> None:
    """Admit a model for replay: the shared seam plus the stepping-only refusals."""
    admit_external_implementation_free_model(model, role)
    history_arrays = _active_history_arrays(model)
    if model.nhistory != 0 or history_arrays:
        raise refuse(
            OperationalReasonCode.UNSUPPORTED_HISTORY_STATE,
            role,
            nhistory=int(model.nhistory),
            active_history_references=cast(CanonicalValue, history_arrays),
        )
    mocap_indices = _active_indices(model.body_mocapid)
    if model.nmocap != 0 or mocap_indices:
        raise refuse(
            OperationalReasonCode.UNSUPPORTED_MOCAP_STATE,
            role,
            nmocap=int(model.nmocap),
            body_indices=cast(CanonicalValue, mocap_indices),
        )
