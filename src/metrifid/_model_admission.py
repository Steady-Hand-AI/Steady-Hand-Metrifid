"""Compiled-model admission: the stable import surface for compilation and identity.

The implementation lives in two focused modules. This module is the facade the rest of the
package and the existing tests import from, so the split changed no import path.

- :mod:`metrifid._model_compile` - environment admission, the guarded compile, and the
  refusals that depend on implementation supplied outside this process.
- :mod:`metrifid._model_descriptors` - joint and actuator descriptors and the compiled-model
  semantic identity.

Each name below is re-exported with a redundant alias so it is a deliberate part of this
module's surface rather than an incidental import.
"""

from __future__ import annotations

from ._model_compile import admit_compiled_model as admit_compiled_model
from ._model_compile import (
    admit_external_implementation_free_model as admit_external_implementation_free_model,
)
from ._model_compile import compile_snapshot_model as compile_snapshot_model
from ._model_compile import require_supported_runtime as require_supported_runtime
from ._model_descriptors import ActuatorDescriptor as ActuatorDescriptor
from ._model_descriptors import CompiledModelIdentity as CompiledModelIdentity
from ._model_descriptors import JointDescriptor as JointDescriptor
from ._model_descriptors import compile_model_identity as compile_model_identity
from ._mujoco_runtime import MujocoClaimSurface as MujocoClaimSurface
from ._mujoco_runtime import MujocoRuntimeAdmission as MujocoRuntimeAdmission
from ._mujoco_runtime import MujocoSupportTier as MujocoSupportTier
from ._mujoco_runtime import admit_mujoco_runtime as admit_mujoco_runtime

__all__ = [
    "ActuatorDescriptor",
    "CompiledModelIdentity",
    "JointDescriptor",
    "MujocoClaimSurface",
    "MujocoRuntimeAdmission",
    "MujocoSupportTier",
    "admit_compiled_model",
    "admit_external_implementation_free_model",
    "admit_mujoco_runtime",
    "compile_model_identity",
    "compile_snapshot_model",
    "require_supported_runtime",
]
