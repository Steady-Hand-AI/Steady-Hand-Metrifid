"""Live compiled model-pair lifecycle built from the accepted model admission closure and alignment path."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path

import mujoco  # type: ignore[import-untyped]

from .._model_admission import (
    MujocoClaimSurface,
    MujocoRuntimeAdmission,
    admit_compiled_model,
    compile_model_identity,
    compile_snapshot_model,
    require_supported_runtime,
)
from .._model_closure import (
    ModelClosureSnapshot,
    ModelRole,
    create_model_closure_snapshot,
    verify_model_closure_unchanged,
)
from .._model_descriptor_types import CompiledModelIdentity
from .._model_identity import (
    ModelPairIdentity,
    _finalized_model_pair,
    _parse_aliases,
    align_compiled_models,
)


@dataclass(frozen=True, slots=True)
class LiveModelPair:
    """One completed model admission identity together with the exact compiled models it describes."""

    identity: ModelPairIdentity
    baseline_model: mujoco.MjModel
    candidate_model: mujoco.MjModel
    _baseline_snapshot: ModelClosureSnapshot
    _candidate_snapshot: ModelClosureSnapshot

    def verify_sources_unchanged(self) -> None:
        """Reverify both retained live source closures at a publication boundary."""
        verify_model_closure_unchanged(self._baseline_snapshot, "baseline")
        verify_model_closure_unchanged(self._candidate_snapshot, "candidate")


@contextmanager
def open_live_model_pair(
    *,
    baseline_root: Path,
    baseline_entrypoint: str,
    candidate_root: Path,
    candidate_entrypoint: str,
    aliases_json: str | bytes | None,
) -> Iterator[LiveModelPair]:
    """Keep both closure snapshots and admitted compiled models alive for replay."""
    runtime = require_supported_runtime(MujocoClaimSurface.DYNAMIC_REPLAY)
    with ExitStack() as stack:
        baseline_snapshot = stack.enter_context(
            create_model_closure_snapshot(baseline_root, baseline_entrypoint, "baseline")
        )
        candidate_snapshot = stack.enter_context(
            create_model_closure_snapshot(candidate_root, candidate_entrypoint, "candidate")
        )
        baseline_model, baseline_compiled = _admitted_model(baseline_snapshot, "baseline", runtime)
        candidate_model, candidate_compiled = _admitted_model(
            candidate_snapshot, "candidate", runtime
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
        identity = _finalized_model_pair(
            baseline_snapshot,
            candidate_snapshot,
            baseline_compiled,
            candidate_compiled,
            alignment,
        )
        try:
            yield LiveModelPair(
                identity,
                baseline_model,
                candidate_model,
                baseline_snapshot,
                candidate_snapshot,
            )
        finally:
            verify_model_closure_unchanged(baseline_snapshot, "baseline")
            verify_model_closure_unchanged(candidate_snapshot, "candidate")


@contextmanager
def open_snapshot_model_pair(snapshot: ModelClosureSnapshot) -> Iterator[LiveModelPair]:
    """Compile both audit roles once from one campaign-owned immutable source snapshot."""
    runtime = require_supported_runtime(MujocoClaimSurface.DYNAMIC_REPLAY)
    baseline_model, baseline_compiled = _admitted_model(snapshot, "baseline", runtime)
    candidate_model, candidate_compiled = _admitted_model(snapshot, "candidate", runtime)
    aliases, raw_hash, semantic_hash = _parse_aliases(
        None,
        snapshot.identity.sha256(),
        snapshot.identity.sha256(),
    )
    alignment = align_compiled_models(
        baseline_compiled,
        candidate_compiled,
        aliases,
        aliases_raw_sha256=raw_hash,
        aliases_semantic_sha256=semantic_hash,
    )
    identity = _finalized_model_pair(
        snapshot,
        snapshot,
        baseline_compiled,
        candidate_compiled,
        alignment,
    )
    yield LiveModelPair(identity, baseline_model, candidate_model, snapshot, snapshot)


def _admitted_model(
    snapshot: ModelClosureSnapshot,
    role: ModelRole,
    runtime: MujocoRuntimeAdmission,
) -> tuple[mujoco.MjModel, CompiledModelIdentity]:
    """Compile, immutability-check, admit, and describe one snapshot model."""
    model = compile_snapshot_model(snapshot, role)
    verify_model_closure_unchanged(snapshot, role)
    admit_compiled_model(model, role, runtime)
    compiled = compile_model_identity(model, snapshot.identity.sha256(), role)
    return model, compiled


__all__ = ["LiveModelPair", "open_live_model_pair"]
