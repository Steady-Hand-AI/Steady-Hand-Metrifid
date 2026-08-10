"""Candidate rows, the recommendation policy, the aggregate receipt and its Markdown.

Nothing here executes a comparison. It turns completed candidate results into the published
audit surface: one row per candidate, the recommendation, the self-hashed aggregate and the
human-readable report.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Final, cast

from ._audit_config import AuditConfig, _rational_primitive
from ._workload import WorkloadArtifacts
from .json_values import Binary64, CanonicalValue, ExactRational, compute_self_hash
from .operational import OperationalReasonCode, OperationalToolObservation

AUDIT_SCHEMA: Final = "metrifid.timestep_audit"
AUDIT_SCHEMA_VERSION: Final = 1
RECOMMENDATION_POLICY: Final = "largest_within_tolerance_completed_prefix"

WITHIN: Final = "WITHIN_DECLARED_TOLERANCE"
OUTSIDE: Final = "OUTSIDE_DECLARED_TOLERANCE"
INCONCLUSIVE: Final = "INCONCLUSIVE"
REFUSED: Final = "REFUSED"

_STATUS_CLASSIFICATION: Final[dict[str, str]] = {
    "NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD": WITHIN,
    "MATERIAL_BEHAVIOR_CHANGE": OUTSIDE,
    "COVERAGE_INSUFFICIENT": INCONCLUSIVE,
    "NONDETERMINISTIC_REPLAY": INCONCLUSIVE,
}

CLAIM_BOUNDARY: Final = (
    "This audit reports behaviour under one declared open-loop workload, the monitored "
    "hinge/slide coordinates, and the declared tolerances only. The reference timestep is "
    "the comparison reference; it is not asserted to be physically correct or a ground "
    "truth. The audit does not establish physical correctness, hardware safety, "
    "closed-loop policy quality, global model equivalence, contact/sensor/reward "
    "equivalence, or causality."
)


def candidate_token(value: ExactRational) -> str:
    """Return the frozen artifact token for one normalized candidate timestep."""
    return f"dt_{value.numerator}_over_{value.denominator}"


def _candidate_row(
    token: str,
    step_dt: ExactRational,
    steps: int | None,
    reference_steps: int,
    *,
    classification: str,
    comparison_status: str | None = None,
    operational_reason: str | None = None,
    receipt: object | None = None,
    failure_sha256: str | None = None,
    comparison_json: str | None = None,
    comparison_markdown: str | None = None,
    operational_failure_json: str | None = None,
) -> dict[str, CanonicalValue]:
    """Build one candidate row with exactly the frozen field set."""
    factor: CanonicalValue = None
    if steps is not None:
        factor = _rational_primitive(ExactRational(reference_steps, steps))
    ratio: CanonicalValue = None
    witness: CanonicalValue = None
    crossing: CanonicalValue = None
    receipt_sha: CanonicalValue = None
    if receipt is not None:
        primitive = cast("dict[str, CanonicalValue]", receipt)
        receipt_sha = primitive.get("receipt_sha256")
        ratio, witness = _worst_witness(primitive)
        crossing = primitive.get("first_crossing")
    return {
        "token": token,
        "step_dt": _rational_primitive(step_dt),
        "steps_per_control_interval": steps,
        "reference_to_candidate_step_count_factor": factor,
        "classification": classification,
        "comparison_status": comparison_status,
        "operational_reason": operational_reason,
        "maximum_tolerance_ratio": ratio,
        "worst_witness": witness,
        "first_crossing": crossing,
        "receipt_sha256": receipt_sha,
        "failure_sha256": failure_sha256,
        "comparison_json": comparison_json,
        "comparison_markdown": comparison_markdown,
        "operational_failure_json": operational_failure_json,
    }


def _worst_witness(receipt: Mapping[str, CanonicalValue]) -> tuple[CanonicalValue, CanonicalValue]:
    """Return the maximum dimensionless ratio and its physical-unit witness."""
    metrics = receipt.get("metrics")
    if not isinstance(metrics, Mapping):
        return None, None
    joints = metrics.get("joints")
    if not isinstance(joints, Sequence) or isinstance(joints, (str, bytes)):
        return None, None
    best: tuple[float, CanonicalValue, CanonicalValue] | None = None
    for joint in joints:
        if not isinstance(joint, Mapping):
            continue
        name = joint.get("canonical_name")
        entries = joint.get("metrics")
        if not isinstance(entries, Mapping):
            continue
        for metric_name in sorted(entries):
            candidate = _witness_candidate(name, metric_name, entries[metric_name])
            if candidate is not None and (best is None or candidate[0] > best[0]):
                best = candidate
    if best is None:
        return None, None
    return best[1], best[2]


def _witness_candidate(
    joint_name: CanonicalValue, metric_name: str, value: object
) -> tuple[float, CanonicalValue, CanonicalValue] | None:
    """Return one valid ratio candidate and its physical-unit witness."""
    if not isinstance(value, Mapping):
        return None
    raw = value.get("maximum_ratio")
    try:
        ratio = Binary64.from_primitive(raw).to_float()
    except (TypeError, ValueError):
        return None
    return (
        ratio,
        cast(CanonicalValue, raw),
        {
            "joint": joint_name,
            "metric": metric_name,
            "maximum_error": value.get("maximum_error"),
            "tolerance": value.get("tolerance"),
            "worst_boundary_index": value.get("worst_boundary_index"),
            "worst_time": value.get("worst_time"),
        },
    )


def _is_skippable_grid_refusal(row: Mapping[str, CanonicalValue]) -> bool:
    """Exactly one refusal may be skipped: a nonintegral candidate control grid.

    That refusal is a statement about the declared schedule, not about the model's behaviour.
    The candidate timestep does not divide `control_dt`, so the candidate was never executed
    and nothing about the surrounding evidence is in doubt. Every other operational failure
    produced no trustworthy comparison evidence and must block the prefix.

    The predicate is deliberately exact and defensive. A malformed or future `REFUSED` row that
    does not meet all three conditions blocks the prefix, so a regression in the classification
    mapping cannot silently recreate false acceptance here.
    """
    return (
        row.get("classification") == REFUSED
        and row.get("operational_reason") == OperationalReasonCode.CONTROL_GRID_NONINTEGRAL.value
        and row.get("steps_per_control_interval") is None
    )


def _recommendation(rows: Sequence[Mapping[str, CanonicalValue]]) -> dict[str, CanonicalValue]:
    """Largest candidate in the ascending completed prefix that stays within tolerance.

    Only an exact nonintegral candidate-grid refusal is skipped; it never breaks the prefix.
    The first OUTSIDE, INCONCLUSIVE, or otherwise refused row stops the scan permanently, so a
    later within-tolerance candidate is reported but never recommended across missing evidence.
    """
    chosen: Mapping[str, CanonicalValue] | None = None
    blocked = False
    for row in rows:
        classification = row["classification"]
        if classification == WITHIN:
            chosen = row
            continue
        if _is_skippable_grid_refusal(row):
            continue
        blocked = True
        break
    return {
        "policy": RECOMMENDATION_POLICY,
        "step_dt": chosen["step_dt"] if chosen is not None else None,
        "candidate_token": chosen["token"] if chosen is not None else None,
        "blocked_by_prior_non_within": blocked,
    }


def _decimal(value: Mapping[str, CanonicalValue] | None) -> str:
    """Render an exact-rational evidence object for the human audit report."""
    if not isinstance(value, Mapping):
        return "—"
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    if isinstance(numerator, int) and isinstance(denominator, int) and denominator:
        return f"{format(numerator / denominator, '.12g')} ({numerator}/{denominator})"
    try:
        return format(Binary64.from_primitive(value).to_float(), ".12g")
    except (TypeError, ValueError):
        return "—"


def _escape(value: CanonicalValue) -> str:
    """Escape one user- or model-controlled string for table and inline placement.

    This is the single escaping authority for the report. A cell separator, a carriage
    return, or a newline inside a model-controlled name would otherwise forge table
    structure or split a row.
    """
    if not isinstance(value, str):
        return "—"
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\r", "\\r").replace("\n", "\\n")


def _worst_witness_text(value: CanonicalValue) -> str:
    """Render the preserved worst witness without any raw Python primitive."""
    if not isinstance(value, Mapping):
        return "—"
    return (
        f"`{_escape(value.get('joint'))}` `{_escape(value.get('metric'))}` "
        f"{_decimal(cast('Mapping[str, CanonicalValue]', value.get('maximum_error')))} "
        f"vs {_decimal(cast('Mapping[str, CanonicalValue]', value.get('tolerance')))}"
    )


def _first_crossing_text(value: CanonicalValue) -> str:
    """Render the preserved first crossing: joint, metric, boundary, time, error, tolerance."""
    if not isinstance(value, Mapping):
        return "—"
    boundary = value.get("boundary_index")
    return (
        f"`{_escape(value.get('joint_name'))}` `{_escape(value.get('metric'))}` "
        f"boundary {boundary if isinstance(boundary, int) else '—'} "
        f"at {_decimal(cast('Mapping[str, CanonicalValue]', value.get('time')))} s, error "
        f"{_decimal(cast('Mapping[str, CanonicalValue]', value.get('error')))} vs tolerance "
        f"{_decimal(cast('Mapping[str, CanonicalValue]', value.get('tolerance')))}"
    )


def _render_markdown(aggregate: Mapping[str, CanonicalValue]) -> str:
    """Readable audit report. Never emits raw Python primitive representations.

    The rendering is a pure function of the aggregate: the same aggregate always produces
    byte-identical UTF-8 with LF line endings.
    """
    reference = cast("Mapping[str, CanonicalValue]", aggregate["reference"])
    inputs = cast("Mapping[str, CanonicalValue]", aggregate["inputs"])
    tool = cast("Mapping[str, CanonicalValue]", aggregate["tool"])
    rows = cast("Sequence[Mapping[str, CanonicalValue]]", aggregate["candidates"])
    rec = cast("Mapping[str, CanonicalValue]", aggregate["recommendation"])
    lines = _audit_header_lines(reference, inputs)
    lines.extend(_candidate_table_lines(rows))
    lines.extend(_recommendation_markdown_lines(rec))
    lines.extend(_audit_footer_lines(aggregate, tool, inputs))
    return "\n".join(lines)


def _audit_header_lines(
    reference: Mapping[str, CanonicalValue], inputs: Mapping[str, CanonicalValue]
) -> list[str]:
    """Render the audit title and reference summary table."""
    return [
        "# metrifid timestep fidelity audit",
        "",
        "## Reference",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Reference timestep | {_decimal(cast('Mapping[str, CanonicalValue]', reference['step_dt']))} s |",
        f"| Reference steps per control interval | {reference['steps_per_control_interval']} |",
        f"| Control interval | {_decimal(cast('Mapping[str, CanonicalValue]', inputs['control_dt']))} s |",
        f"| Repeats | {inputs['repeats']} |",
        f"| Workload kind | `{_escape(inputs['workload_kind'])}` |",
        f"| Workload label | {_escape(inputs['workload_label'])} |",
    ]


def _candidate_table_lines(rows: Sequence[Mapping[str, CanonicalValue]]) -> list[str]:
    """Render the candidate outcome table with physical-unit witnesses."""
    lines = [
        "",
        "## Candidates",
        "",
        "| Candidate timestep | Steps/interval | Step-count factor | Classification "
        "| Operational reason | Maximum tolerance ratio | Worst witness | First crossing |",
        "|---|---:|---:|---|---|---:|---|---|",
    ]
    for row in rows:
        steps = row["steps_per_control_interval"]
        reason = row.get("operational_reason")
        lines.append(
            f"| {_decimal(cast('Mapping[str, CanonicalValue]', row['step_dt']))} s "
            f"(`{_escape(row['token'])}`) "
            f"| {steps if steps is not None else '—'} "
            f"| {_decimal(cast('Mapping[str, CanonicalValue]', row['reference_to_candidate_step_count_factor']))} "
            f"| {_escape(row['classification'])} "
            f"| {_escape(reason) if reason is not None else '—'} "
            f"| {_decimal(cast('Mapping[str, CanonicalValue]', row['maximum_tolerance_ratio']))} "
            f"| {_worst_witness_text(row.get('worst_witness'))} "
            f"| {_first_crossing_text(row.get('first_crossing'))} |"
        )
    return lines


def _recommendation_markdown_lines(rec: Mapping[str, CanonicalValue]) -> list[str]:
    """Render the prefix-policy recommendation and its blocking state."""
    lines = ["", "## Recommendation", ""]
    if rec["step_dt"] is None:
        lines.append(
            "No candidate is recommended: the ascending completed prefix does not contain "
            "a within-tolerance result."
        )
    else:
        lines.append(
            f"Largest candidate supported by an unbroken within-tolerance completed prefix: "
            f"**{_decimal(cast('Mapping[str, CanonicalValue]', rec['step_dt']))} s** "
            f"(`{_escape(rec['candidate_token'])}`)."
        )
    blocked = "yes" if rec["blocked_by_prior_non_within"] else "no"
    lines.extend(
        [
            "",
            f"A non-within result prevented recommendation of larger completed candidates: {blocked}.",
            "",
            f"Policy: `{_escape(rec['policy'])}`. Monotonicity assumed: no.",
        ]
    )
    return lines


def _audit_footer_lines(
    aggregate: Mapping[str, CanonicalValue],
    tool: Mapping[str, CanonicalValue],
    inputs: Mapping[str, CanonicalValue],
) -> list[str]:
    """Render the claim boundary and published identity digests."""
    return [
        "",
        "## Claim boundary",
        "",
        _escape(aggregate["claim_boundary"]),
        "",
        "## Identities",
        "",
        f"- Tool: `{_escape(tool['version'])}` / `{_escape(tool['distribution_sha256'])}`",
        f"- Configuration: `{_escape(inputs['configuration_raw_sha256'])}`",
        f"- Audit: `{_escape(aggregate['audit_sha256'])}`",
        "",
    ]


def _aggregate(
    config: AuditConfig,
    raw: bytes,
    tool: OperationalToolObservation,
    reference_dt: ExactRational,
    reference_steps: int,
    rows: Sequence[dict[str, CanonicalValue]],
    source_tree_sha256: str,
    workload: WorkloadArtifacts,
) -> dict[str, CanonicalValue]:
    """Assemble and self-hash the deterministic aggregate surface."""
    aggregate: dict[str, CanonicalValue] = {
        "schema": AUDIT_SCHEMA,
        "schema_version": AUDIT_SCHEMA_VERSION,
        "tool": {
            "version": tool.version,
            "execution_identity_state": tool.execution_identity_state,
            "distribution_sha256": tool.distribution_sha256,
        },
        "inputs": _aggregate_inputs(config, raw, source_tree_sha256, workload),
        "reference": {
            "step_dt": _rational_primitive(reference_dt),
            "steps_per_control_interval": reference_steps,
        },
        "candidates": cast(CanonicalValue, list(rows)),
        "recommendation": cast(CanonicalValue, _recommendation(rows)),
        "monotonicity_assumed": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "audit_sha256": None,
    }
    aggregate["audit_sha256"] = compute_self_hash(aggregate, "audit_sha256")
    return aggregate


def _aggregate_inputs(
    config: AuditConfig,
    raw: bytes,
    source_tree_sha256: str,
    workload: WorkloadArtifacts,
) -> dict[str, CanonicalValue]:
    """Build aggregate identities only from the frozen campaign model and workload."""
    return {
        "configuration_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "source_model_tree_sha256": source_tree_sha256,
        "entrypoint": config.entrypoint,
        "initial_state_raw_sha256": workload.state.raw_file_sha256,
        "actions_raw_sha256": workload.actions.raw_file_sha256,
        "initial_state_semantic_sha256": workload.state.semantic_sha256,
        "actions_semantic_sha256": workload.actions.semantic_sha256,
        "control_dt": _rational_primitive(config.control_dt),
        "repeats": config.repeats,
        "workload_kind": config.workload_kind,
        "workload_label": config.workload_label,
    }
