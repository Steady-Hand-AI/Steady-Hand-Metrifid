"""Live measurement of the runtime that produced a pair of complete compiled artifacts.

The schema, strict parsing, canonical serialization, and self-hash rules live in
:mod:`metrifid.certify._runtime_schema`, which is pure and importable without MuJoCo or NumPy.
This module keeps only the measurement of the running process, which necessarily imports the
native dependencies, and re-exports the schema names so every existing import path keeps working.
"""

from __future__ import annotations

import platform
import sys

import mujoco  # type: ignore[import-untyped]
import numpy as np

from .certify._runtime_schema import (
    EXECUTION_MODE_NO_MJDATA_EXECUTION,
    RUNTIME_IDENTITY_SCHEMA,
    RUNTIME_IDENTITY_SCHEMA_VERSION,
    CertifyRuntimeIdentity,
)
from .compare._environment import (
    mujoco_distribution_payload_sha256,
    native_mujoco_library_sha256,
)
from .distribution import installed_distribution_sha256
from .version import __version__


def build_certify_runtime_identity(header_words: tuple[int, ...]) -> CertifyRuntimeIdentity:
    """Measure the runtime that produced an artifact carrying these header words."""
    libc_name, libc_version = platform.libc_ver()
    return CertifyRuntimeIdentity(
        schema=RUNTIME_IDENTITY_SCHEMA,
        schema_version=RUNTIME_IDENTITY_SCHEMA_VERSION,
        metrifid_version=__version__,
        metrifid_distribution_sha256=installed_distribution_sha256(),
        mujoco_python_distribution_sha256=mujoco_distribution_payload_sha256(),
        mujoco_native_library_sha256=native_mujoco_library_sha256(),
        mujoco_version=str(mujoco.__version__),
        mujoco_version_string=str(mujoco.mj_versionString()),
        mujoco_version_integer=int(mujoco.mj_version()),
        python_implementation=platform.python_implementation(),
        python_version=platform.python_version(),
        numpy_version=str(np.__version__),
        platform_system=platform.system(),
        platform_machine=platform.machine(),
        platform_release=platform.release(),
        libc=f"{libc_name or 'unknown'}-{libc_version or 'unknown'}",
        byteorder=sys.byteorder,
        mjb_header_words=tuple(int(word) for word in header_words),
        execution_mode=EXECUTION_MODE_NO_MJDATA_EXECUTION,
        runtime_identity_sha256=None,
    ).finalized()


__all__ = [
    "EXECUTION_MODE_NO_MJDATA_EXECUTION",
    "RUNTIME_IDENTITY_SCHEMA",
    "RUNTIME_IDENTITY_SCHEMA_VERSION",
    "CertifyRuntimeIdentity",
    "build_certify_runtime_identity",
]
