"""Build and expose strict semantic model-pair identities."""

from __future__ import annotations

from ._model_alignment import (
    _actuator_pairs as _actuator_pairs,
)
from ._model_alignment import (
    _align_joints as _align_joints,
)
from ._model_alignment import (
    _finish_actuator_pairs as _finish_actuator_pairs,
)
from ._model_alignment import (
    _finish_joint_pairs as _finish_joint_pairs,
)
from ._model_alignment import (
    _resolve_endpoint as _resolve_endpoint,
)
from ._model_alignment import (
    _selector_matches as _selector_matches,
)
from ._model_alignment import (
    align_compiled_models as align_compiled_models,
)
from ._model_closure import (
    ModelClosureSnapshot,
)
from ._model_descriptor_types import CompiledModelIdentity
from ._model_identity_types import (
    ModelPairIdentity as ModelPairIdentity,
)
from ._model_identity_types import (
    SemanticAlignment as SemanticAlignment,
)
from ._model_identity_types import (
    _parse_aliases as _parse_aliases,
)


def _finalized_model_pair(
    baseline_snapshot: ModelClosureSnapshot,
    candidate_snapshot: ModelClosureSnapshot,
    baseline_compiled: CompiledModelIdentity,
    candidate_compiled: CompiledModelIdentity,
    alignment: SemanticAlignment,
) -> ModelPairIdentity:
    """Assemble and finalize the complete model-pair identity surface."""
    return ModelPairIdentity(
        ModelPairIdentity._SCHEMA,
        ModelPairIdentity._SCHEMA_VERSION,
        None,
        baseline_snapshot.identity,
        candidate_snapshot.identity,
        baseline_compiled,
        candidate_compiled,
        alignment,
        alignment.summary(),
    ).finalized()
