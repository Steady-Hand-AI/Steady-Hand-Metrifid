"""Build and expose strict semantic model-pair identities."""

from __future__ import annotations

from pathlib import Path

from ._model_admission import admit_compiled_model, require_supported_runtime
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
    create_model_closure_snapshot,
    verify_model_closure_unchanged,
)
from ._model_compile import compile_snapshot_model
from ._model_descriptor_builders import compile_model_identity
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


def build_model_pair_identity(
    baseline_root: Path,
    baseline_entrypoint: str,
    candidate_root: Path,
    candidate_entrypoint: str,
    aliases_json: str | bytes | None = None,
) -> ModelPairIdentity:
    """Snapshot, compile, admit, describe, and align one trusted local model pair."""
    require_supported_runtime()
    with (
        create_model_closure_snapshot(
            baseline_root, baseline_entrypoint, "baseline"
        ) as baseline_snapshot,
        create_model_closure_snapshot(
            candidate_root, candidate_entrypoint, "candidate"
        ) as candidate_snapshot,
    ):
        baseline_model = compile_snapshot_model(baseline_snapshot, "baseline")
        verify_model_closure_unchanged(baseline_snapshot, "baseline")
        candidate_model = compile_snapshot_model(candidate_snapshot, "candidate")
        verify_model_closure_unchanged(candidate_snapshot, "candidate")
        admit_compiled_model(baseline_model, "baseline")
        admit_compiled_model(candidate_model, "candidate")
        baseline_compiled = compile_model_identity(
            baseline_model, baseline_snapshot.identity.sha256(), "baseline"
        )
        candidate_compiled = compile_model_identity(
            candidate_model, candidate_snapshot.identity.sha256(), "candidate"
        )
        aliases, raw_hash, semantic_hash = _parse_aliases(
            aliases_json,
            baseline_snapshot.identity.sha256(),
            candidate_snapshot.identity.sha256(),
        )
        alignment = align_compiled_models(
            baseline_compiled,
            candidate_compiled,
            aliases,
            aliases_raw_sha256=raw_hash,
            aliases_semantic_sha256=semantic_hash,
        )
        return _finalized_model_pair(
            baseline_snapshot,
            candidate_snapshot,
            baseline_compiled,
            candidate_compiled,
            alignment,
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
