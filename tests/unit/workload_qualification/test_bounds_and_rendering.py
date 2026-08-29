"""Schema-v1 bounds, planned campaign size, and Markdown structural safety."""

from __future__ import annotations

import json

import pytest
import pytest_check as check

from metrifid._json_admission import CONFIG_JSON_LIMITS, RECEIPT_JSON_LIMITS
from metrifid.json_values import canonical_json_bytes
from metrifid.workload_qualification._config import (
    MAX_PROBE_GROUPS,
    MAX_VARIANTS,
    MAX_WORKLOADS,
    REQUIRED_BUDGET,
)
from metrifid.workload_qualification._evidence import planned_comparisons
from metrifid.workload_qualification._safe import markdown_block, markdown_code, markdown_label
from tests._support.workload_qualification import config, probe_group

MAX_PLANNED = 2064


def test_the_planned_comparison_formula_matches_the_documented_maximum() -> None:
    """Planned = workloads + workloads x sum(variants per group); schema v1 bounds it at 2064."""
    check.equal(
        MAX_WORKLOADS + MAX_WORKLOADS * (MAX_PROBE_GROUPS * MAX_VARIANTS),
        MAX_PLANNED,
        f"the schema-v1 cardinality bounds no longer plan the documented maximum of "
        f"{MAX_PLANNED} comparisons, so every receipt-sizing argument built on it is stale",
    )


def test_planned_comparisons_counts_controls_and_every_rung() -> None:
    """Two groups of two rungs over three workloads plans three controls and twelve cells."""
    groups = [
        probe_group("first", ("0.1", "0.2"), "0.1"),
        probe_group("second", ("0.1", "0.2"), "0.1"),
    ]
    qualification = config(groups, ("w1", "w2", "w3"))
    check.equal(
        planned_comparisons(qualification),
        3 + 3 * 4,
        "the planned campaign size does not cover one zero-change control per workload plus "
        "one comparison for every workload-by-rung cell",
    )


def test_a_maximum_schema_v1_configuration_fits_the_reused_json_limits() -> None:
    """The reused bounded-JSON limits cover schema-v1 maximum cardinality with room to spare.

    This is why no qualification-specific limit is introduced: the established configuration and
    receipt limits already admit the largest document schema version 1 can describe.
    """
    variants = [
        {
            "magnitude": f"0.{index + 1:03d}",
            "candidate": {
                "model_root": f"probes/group/rung_{index}",
                "entrypoint": "model.xml",
                "declared_step_dt": "0.001",
            },
        }
        for index in range(MAX_VARIANTS)
    ]
    document = {
        "schema_version": 1,
        "baseline": {
            "model_root": "baseline",
            "entrypoint": "model.xml",
            "declared_step_dt": "0.001",
        },
        "probe_groups": [
            {
                "probe_id": f"group_{index:02d}",
                "parameter": "hinge.damping",
                "direction": "increase",
                "magnitude_semantics": "absolute increase in native units",
                "required_detection_magnitude": "0.001",
                "variants": variants,
            }
            for index in range(MAX_PROBE_GROUPS)
        ],
        "workloads": [
            {
                "workload_id": f"workload_{index:02d}",
                "initial_state": f"workloads/w{index}/state.npz",
                "actions": f"workloads/w{index}/actions.npz",
                "control_dt": "0.01",
            }
            for index in range(MAX_WORKLOADS)
        ],
        "repeats": 5,
        "joint_tolerances": {
            f"joint_{index:02d}": {
                "joint_type": "hinge",
                "angle_rad": "0.000001",
                "angular_velocity_rad_s": "0.0001",
            }
            for index in range(64)
        },
        "aliases": None,
        "budget": REQUIRED_BUDGET,
        "output_dir": "qualification_out",
    }
    payload = canonical_json_bytes(document)
    check.less(
        len(payload),
        CONFIG_JSON_LIMITS.max_bytes,
        "the largest configuration schema version 1 can describe is too large for the reused "
        "configuration byte limit, so that limit no longer admits a legal document",
    )
    check.less(
        _nodes(document),
        CONFIG_JSON_LIMITS.max_nodes,
        "the largest configuration schema version 1 can describe carries more JSON nodes than "
        "the reused configuration limit admits",
    )
    check.less(
        _depth(document),
        CONFIG_JSON_LIMITS.max_depth,
        "the largest configuration schema version 1 can describe nests deeper than the reused "
        "configuration limit admits",
    )
    # The aggregate receipt echoes this configuration plus one record per planned comparison.
    check.less(
        MAX_PLANNED * 12 + _nodes(document),
        RECEIPT_JSON_LIMITS.max_nodes,
        "the aggregate receipt for a maximum campaign carries more JSON nodes than the reused "
        "receipt limit admits, so a legal campaign could not publish its receipt",
    )


def _nodes(value: object) -> int:
    """Count every value and every object member name, as the admission limits do."""
    if isinstance(value, dict):
        return 1 + sum(1 + _nodes(item) for item in value.values())
    if isinstance(value, list):
        return 1 + sum(_nodes(item) for item in value)
    return 1


def _depth(value: object, level: int = 0) -> int:
    """Return the maximum nesting depth, with the root at depth zero."""
    if isinstance(value, dict):
        return max((_depth(item, level + 1) for item in value.values()), default=level)
    if isinstance(value, list):
        return max((_depth(item, level + 1) for item in value), default=level)
    return level


HOSTILE_LABELS = (
    "pipe|injection",
    "back`tick",
    "back\\slash",
    "carriage\rreturn",
    "line\nfeed",
    "<script>alert(1)</script>",
    "/absolute/looking/path",
    "naïve — Ünicode",
    "```fence",
    "# heading",
)


@pytest.mark.parametrize("label", HOSTILE_LABELS)
def test_a_hostile_label_cannot_restructure_the_report(label: str) -> None:
    """A rendered label never opens a cell, heading, fence, or raw HTML block."""
    rendered = markdown_label(label)
    check.is_not_in(
        "\n",
        rendered,
        "a rendered label keeps a real line feed, so it can end its table row and start a "
        "new report block",
    )
    check.is_not_in(
        "\r",
        rendered,
        "a rendered label keeps a real carriage return, so it can end its table row and start "
        "a new report block",
    )
    for structural in ("|", "`", "<", ">"):
        check.is_not_in(
            structural,
            rendered.replace(f"\\{structural}", ""),
            f"a rendered label leaves {structural!r} unescaped, so it can open a table cell, "
            f"a code span, or a raw HTML block in the report",
        )
    check.is_false(
        rendered.startswith("#"),
        "a rendered label still begins with '#', so it can promote itself to a report heading",
    )
    check.is_not_in(
        "```",
        markdown_block(label),
        "a label placed in a fenced block can still close the fence and escape the block",
    )
    coded = markdown_code(label)
    check.is_true(
        coded.startswith("`"),
        "the inline code span around a label does not open with a backtick",
    )
    check.is_true(
        coded.endswith("`"),
        "the inline code span around a label does not close with a backtick",
    )


@pytest.mark.parametrize("label", HOSTILE_LABELS)
def test_canonical_json_preserves_the_exact_admitted_string(label: str) -> None:
    """Escaping governs display only; the JSON surface carries the admitted bytes."""
    payload = canonical_json_bytes({"label": label})
    check.equal(
        json.loads(payload)["label"],
        label,
        "the canonical JSON surface no longer carries the admitted label exactly, so display "
        "escaping has leaked into the stored value",
    )
    check.is_not_in(
        b"\r",
        payload,
        "the canonical JSON payload carries a raw carriage return instead of an escape",
    )
    check.equal(
        payload.count(b"\n"),
        0,
        "the canonical JSON payload carries a raw line feed instead of an escape",
    )
