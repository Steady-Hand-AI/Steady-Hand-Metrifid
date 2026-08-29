"""Test-only adapter onto the live compiled model-pair lifecycle.

The product has one authority for admitting a model-pair identity from two trusted local roots:
``metrifid.compare._model_pair.open_live_model_pair``. It keeps both closure snapshots and both
admitted compiled models alive for the caller and reverifies the sources on exit. Tests that need
only the resulting identity drive that authority through this adapter rather than a second shipped
entry point that duplicates its admission, compile, and alignment path.
"""

from __future__ import annotations

from pathlib import Path

from metrifid._model_identity import ModelPairIdentity
from metrifid.compare._model_pair import open_live_model_pair


def build_model_pair_identity(
    baseline_root: Path,
    baseline_entrypoint: str,
    candidate_root: Path,
    candidate_entrypoint: str,
    aliases_json: str | bytes | None = None,
) -> ModelPairIdentity:
    """Return the identity the live model-pair lifecycle admits for one trusted local pair."""
    with open_live_model_pair(
        baseline_root=baseline_root,
        baseline_entrypoint=baseline_entrypoint,
        candidate_root=candidate_root,
        candidate_entrypoint=candidate_entrypoint,
        aliases_json=aliases_json,
    ) as pair:
        return pair.identity
