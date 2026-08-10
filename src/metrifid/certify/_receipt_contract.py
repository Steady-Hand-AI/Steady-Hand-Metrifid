"""Frozen claim text and member registries for certification receipts."""

from __future__ import annotations

from collections.abc import Mapping

RECEIPT_SCHEMA = "metrifid.compiled_equivalence_receipt"
RECEIPT_SCHEMA_VERSION = 1

_HASH_FIELD = "receipt_sha256"

# The frozen root member set. A receipt with an unknown or missing root member is refused.
_ROOT_MEMBERS = (
    "schema",
    "schema_version",
    "status",
    "completed_exit_code",
    "tool",
    "runtime_identity",
    "baseline",
    "candidate",
    "byte_comparison",
    "field_report",
    "artifact_claim",
    "behavior_implication",
    "limitations",
    "decision_sha256",
    "receipt_sha256",
)

_ROLE_MEMBERS = (
    "role",
    "source_closure",
    "source_closure_sha256",
    "source_closure_total_bytes",
    "compiled_artifact",
)

# The decision is fixed by these members alone. The descriptive field report, the conditional
# behavior implication and the prose limitations are deliberately outside it.
_DECISION_MEMBERS = (
    "schema",
    "schema_version",
    "status",
    "completed_exit_code",
    "tool",
    "runtime_identity",
    "baseline",
    "candidate",
    "byte_comparison",
)

CERTIFIED_ARTIFACT_CLAIM = (
    "The measured baseline and candidate source closures, compiled under the recorded runtime "
    "identity, produced byte-identical complete MJB artifacts."
)

NOT_CERTIFIED_ARTIFACT_CLAIM = (
    "The measured baseline and candidate source closures, compiled under the recorded runtime "
    "identity, did not produce byte-identical complete MJB artifacts."
)

ARTIFACT_CLAIM_EXCLUSIONS = (
    "source text equality",
    "license equality",
    "visual intent",
    "task suitability",
    "hardware safety",
    "cross-version portability",
)

BEHAVIOR_IMPLICATION = (
    "If the certified bytes are loaded under the recorded runtime, neither mjModel is modified "
    "afterward, complete mjData initial state and every external input/state/code surface are "
    "identical, and execution is deterministic, then the compiled model cannot be the source of "
    "divergence."
)

BEHAVIOR_IMPLICATION_PREMISES = (
    "The certified bytes are loaded under the recorded runtime identity.",
    "Neither mjModel is modified after certification.",
    "Complete mjData initial state is identical.",
    "Every external input, state and code surface is identical.",
    "Execution is deterministic.",
)

NOT_CERTIFIED_GUIDANCE = (
    "The behavioral significance of this difference is unknown. This command makes no "
    "workload-bounded statement. Run metrifid compare for a workload-bounded decision."
)

REQUIRED_LIMITATIONS = (
    "EXACT_RECORDED_RUNTIME_ONLY",
    "POST_CERTIFICATION_MJMODEL_MUTATION_OUTSIDE_CLAIM",
    "EXTERNAL_CODE_AND_INPUT_EQUIVALENCE_NOT_ESTABLISHED",
    "NO_SOURCE_TEXT_LICENSE_OR_VISUAL_INTENT_CLAIM",
    "NO_CROSS_MUJOCO_VERSION_CLAIM",
)

_LIMITATION_STATEMENTS: Mapping[str, str] = {
    "EXACT_RECORDED_RUNTIME_ONLY": (
        "The statement holds only for the exact recorded runtime identity. No other MuJoCo "
        "build, Python build, platform or architecture is covered."
    ),
    "POST_CERTIFICATION_MJMODEL_MUTATION_OUTSIDE_CLAIM": (
        "Any mutation of mjModel after compilation is outside the claim. Byte-identical "
        "artifacts say nothing about a model a caller edits afterward."
    ),
    "EXTERNAL_CODE_AND_INPUT_EQUIVALENCE_NOT_ESTABLISHED": (
        "Equivalence of external code, inputs, controllers and state is not established here "
        "and is not implied by identical compiled artifacts."
    ),
    "NO_SOURCE_TEXT_LICENSE_OR_VISUAL_INTENT_CLAIM": (
        "No claim is made about source text, licensing or visual intent. Different sources can "
        "compile to identical artifacts."
    ),
    "NO_CROSS_MUJOCO_VERSION_CLAIM": (
        "No claim is made across MuJoCo versions. A different engine version may serialize the "
        "same model differently."
    ),
}
