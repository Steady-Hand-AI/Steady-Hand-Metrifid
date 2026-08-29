"""Cross-process determinism of one installed Model Change Gate review.

Every ordering the product publishes is built from Python containers whose iteration order a
different hash seed is allowed to change. These regressions run the installed command in separate
interpreter processes under explicitly different ``PYTHONHASHSEED`` values and require the whole
published result, down to the canonical receipt bytes, to be identical.

The claim is deliberately narrow: this is cross-process determinism within one exact profile, on
one platform, one interpreter and one MuJoCo build. Nothing here claims byte identity across
operating systems, architectures or MuJoCo versions.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

# Two explicit, deliberately unrelated seeds. Zero is excluded because it disables hash
# randomization entirely and would not exercise the property under test.
_HASH_SEEDS = ("1", "8675309")

_BASELINE_MASS = "1.0"
_CANDIDATE_MASS = "1.25"

# A rule-free policy that leaves the candidate unbound is fail-closed for a differing
# candidate, so this scenario completes as REVIEW_REQUIRED with exit 40 every time.
_EXPECTED_STATUS = "REVIEW_REQUIRED"
_EXPECTED_EXIT = 40

# Fields that must agree exactly across seeds. Ordering fields are listed explicitly rather than
# relying on the whole-receipt comparison alone, so a reordering failure names itself.
_ORDER_BEARING_FIELDS = (
    "status",
    "completed_exit_code",
    "changes",
    "first_unexpected_witness",
    "first_missing_required_witness",
    "decision_sha256",
    "receipt_sha256",
)


def _multi_body_model(payload_mass: str) -> str:
    """Return a model with enough named objects for an ordering difference to be visible.

    The bodies, joints, geoms and actuators are authored in deliberately noncanonical name order
    so that any reliance on insertion or hash order shows up as a reordered change list.
    """
    bodies = []
    for name in ("zeta", "alpha", "omega", "beta", "gamma", "delta"):
        bodies.append(
            f"""
    <body name="{name}_body" pos="0 0 0.{len(name)}">
      <joint name="{name}_slide" type="slide" axis="1 0 0" damping="0"/>
      <geom name="{name}_geom" type="box" size="0.05 0.05 0.05" mass="{payload_mass}"/>
    </body>"""
        )
    motors = "\n".join(
        f'    <motor name="{name}_motor" joint="{name}_slide" gear="1"/>'
        for name in ("zeta", "alpha", "omega", "beta", "gamma", "delta")
    )
    return f"""
<mujoco model="release-determinism">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>{"".join(bodies)}
  </worldbody>
  <actuator>
{motors}
  </actuator>
</mujoco>
"""


def _run_installed(
    command: str, *arguments: str, hash_seed: str | None = None
) -> subprocess.CompletedProcess[str]:
    """Run one installed Metrifid command in a separate process under an explicit hash seed."""
    environment = os.environ.copy()
    if hash_seed is not None:
        environment["PYTHONHASHSEED"] = hash_seed
    return subprocess.run(
        [sys.executable, "-m", "metrifid.cli", command, *arguments],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=600,
    )


def _json_object(value: object) -> dict[str, Any]:
    """Narrow one decoded value to a JSON object."""
    assert type(value) is dict
    return value


def _write_pair(root: Path) -> tuple[Path, Path]:
    """Write two isolated model roots for one deterministic review."""
    baseline_root = root / "baseline"
    candidate_root = root / "candidate"
    baseline_root.mkdir(parents=True)
    candidate_root.mkdir(parents=True)
    baseline = baseline_root / "model.xml"
    candidate = candidate_root / "model.xml"
    baseline.write_text(_multi_body_model(_BASELINE_MASS), encoding="utf-8")
    candidate.write_text(_multi_body_model(_CANDIDATE_MASS), encoding="utf-8")
    return baseline, candidate


def _learn_baseline_subject(root: Path, baseline: Path, candidate: Path) -> str:
    """Learn the exact baseline complete-MJB subject through the installed Certify command."""
    output = root / "certify-output"
    completed = _run_installed("certify", str(baseline), str(candidate), "--output", str(output))
    assert completed.returncode in {0, 40}, completed.stdout + completed.stderr
    receipt = _json_object(json.loads((output / "certification.json").read_text(encoding="utf-8")))
    role = _json_object(receipt["baseline"])
    return str(_json_object(role["compiled_artifact"])["mjb_sha256"])


def _write_policy(path: Path, baseline_compiled_sha256: str) -> Path:
    """Write one rule-free discovery policy bound to the exact baseline subject."""
    path.write_text(
        json.dumps(
            {
                "schema": "metrifid.model_release_policy",
                "schema_version": 1,
                "baseline_compiled_sha256": baseline_compiled_sha256,
                "candidate_compiled_sha256": None,
                "rules": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="module")
def reviews_under_each_seed(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run one identical review once per hash seed and return every published result."""
    root = tmp_path_factory.mktemp("determinism")
    baseline, candidate = _write_pair(root)
    policy = _write_policy(root / "policy.json", _learn_baseline_subject(root, baseline, candidate))

    results: dict[str, Any] = {}
    for seed in _HASH_SEEDS:
        output = root / f"review-{seed}"
        completed = _run_installed(
            "review-model",
            str(baseline),
            str(candidate),
            "--policy",
            str(policy),
            "--output",
            str(output),
            hash_seed=seed,
        )
        # The expected outcome is pinned exactly. A disjunction over exit codes would still pass
        # if a hash seed moved the completed status, which is the property under test.
        assert completed.returncode == _EXPECTED_EXIT, completed.stdout + completed.stderr
        results[seed] = {
            "exit": completed.returncode,
            "stdout": completed.stdout,
            "json_bytes": (output / "model_release.json").read_bytes(),
            "markdown_bytes": (output / "model_release.md").read_bytes(),
        }
    return results


def test_two_hash_seeds_publish_identical_canonical_receipt_bytes(
    reviews_under_each_seed: dict[str, Any],
) -> None:
    """Require byte-identical published evidence from two explicitly different hash seeds."""
    first, second = (reviews_under_each_seed[seed] for seed in _HASH_SEEDS)
    assert first["exit"] == second["exit"]
    assert first["json_bytes"] == second["json_bytes"]
    assert first["markdown_bytes"] == second["markdown_bytes"]


def test_two_hash_seeds_agree_on_every_ordered_decision_field(
    reviews_under_each_seed: dict[str, Any],
) -> None:
    """Require the status, ordered change records and both first witnesses to agree exactly."""
    receipts = [
        _json_object(json.loads(reviews_under_each_seed[seed]["json_bytes"].decode("utf-8")))
        for seed in _HASH_SEEDS
    ]
    first, second = receipts
    assert first["status"] == _EXPECTED_STATUS
    assert first["completed_exit_code"] == _EXPECTED_EXIT
    for field in _ORDER_BEARING_FIELDS:
        assert first[field] == second[field], f"{field} disagreed across hash seeds"

    # The change list must be a real ordering, not a single row that cannot show reordering.
    changes = first["changes"]
    assert type(changes) is list
    assert len(changes) > 1
    assert [json.dumps(change, sort_keys=True) for change in changes] == [
        json.dumps(change, sort_keys=True) for change in second["changes"]
    ]


def test_two_hash_seeds_agree_on_the_linked_certification_identities(
    reviews_under_each_seed: dict[str, Any],
) -> None:
    """Require the embedded Certify receipt and the policy semantic identity to agree exactly."""
    receipts = [
        _json_object(json.loads(reviews_under_each_seed[seed]["json_bytes"].decode("utf-8")))
        for seed in _HASH_SEEDS
    ]
    first, second = receipts
    for field in ("certification_receipt_sha256", "certification_decision_sha256"):
        assert first[field] == second[field]
    assert (
        _json_object(first["policy"])["semantic_sha256"]
        == _json_object(second["policy"])["semantic_sha256"]
    )
    assert _json_object(first["certification_receipt"]) == _json_object(
        second["certification_receipt"]
    )


def test_each_seed_actually_changed_string_hashing_in_the_child_process() -> None:
    """Prove the seeds this module uses really do change hashing in a child interpreter.

    Echoing the environment variable back would only prove the variable was set. What matters is
    that the two seeds produce genuinely different string hashing, because that is the mechanism
    a container-ordering bug would ride on. ``hash`` of an object is derived from its address and
    is not seed-dependent, so a string is used here.
    """
    program = "import os;print(os.environ['PYTHONHASHSEED']);print(hash('metrifid'))"
    echoed: list[str] = []
    hashes: list[str] = []
    for seed in _HASH_SEEDS:
        completed = subprocess.run(
            [sys.executable, "-c", program],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        lines = completed.stdout.strip().splitlines()
        echoed.append(lines[0])
        hashes.append(lines[1])

    assert echoed == list(_HASH_SEEDS)
    assert len(set(hashes)) == len(_HASH_SEEDS), (
        f"the two seeds produced the same string hash {hashes}; they do not exercise ordering"
    )

    # The same seed must reproduce its own hash, or the two seeds differing would prove nothing.
    repeated = subprocess.run(
        [sys.executable, "-c", program],
        env={**os.environ, "PYTHONHASHSEED": _HASH_SEEDS[0]},
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    assert repeated.stdout.strip().splitlines()[1] == hashes[0]


# ------------------------------------------------------------------------------------------------
# Tie ordering: two undeclared changes that sort against each other, under permuted policy and
# source declarations and different hash seeds. The three invariants asserted here are
# TWO_UNDECLARED_CHANGES_PRESENT,
# POLICY_AND_SOURCE_DECLARATION_PERMUTATIONS_PRESERVE_ORDER and
# FRESH_PROCESS_AND_HASH_SEED_PRESERVE_RECEIPT_BYTES. They are asserted together, against the
# completed status and exit code, so no one of them can be satisfied by a different scenario.
# ------------------------------------------------------------------------------------------------

_TIE_EXPECTED_STATUS = "REVIEW_REQUIRED"
_TIE_EXPECTED_EXIT = 40


def _tied_pair_model(alpha_mass: str, zeta_mass: str, *, alpha_first: bool) -> str:
    """Return two independently named branches, authored in either source declaration order."""
    alpha = f"""
    <body name="alpha_body" pos="0 0 -0.3">
      <joint name="alpha_slide" type="slide" axis="1 0 0"/>
      <geom name="alpha_geom" type="box" size="0.08 0.08 0.08" mass="{alpha_mass}"/>
    </body>"""
    zeta = f"""
    <body name="zeta_body" pos="0 0 0.3">
      <joint name="zeta_slide" type="slide" axis="1 0 0"/>
      <geom name="zeta_geom" type="box" size="0.08 0.08 0.08" mass="{zeta_mass}"/>
    </body>"""
    bodies = (alpha + zeta) if alpha_first else (zeta + alpha)
    return f"""
<mujoco model="release-tie-ordering">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>{bodies}
  </worldbody>
  <actuator>
    <motor name="zeta_motor" joint="zeta_slide"/>
    <motor name="alpha_motor" joint="alpha_slide"/>
  </actuator>
</mujoco>
"""


def _write_tied_pair(root: Path, *, alpha_first: bool) -> tuple[Path, Path]:
    """Write one tied baseline/candidate pair in the requested source declaration order."""
    baseline_root = root / "baseline"
    candidate_root = root / "candidate"
    baseline_root.mkdir(parents=True)
    candidate_root.mkdir(parents=True)
    baseline = baseline_root / "model.xml"
    candidate = candidate_root / "model.xml"
    baseline.write_text(_tied_pair_model("1", "1", alpha_first=alpha_first), encoding="utf-8")
    candidate.write_text(_tied_pair_model("2", "3", alpha_first=alpha_first), encoding="utf-8")
    return baseline, candidate


def _learn_subjects(root: Path, baseline: Path, candidate: Path) -> tuple[str, str]:
    """Learn both exact complete-MJB subjects through the installed Certify command."""
    output = root / "certify-output"
    completed = _run_installed("certify", str(baseline), str(candidate), "--output", str(output))
    assert completed.returncode in {0, 40}, completed.stdout + completed.stderr
    receipt = _json_object(json.loads((output / "certification.json").read_text(encoding="utf-8")))
    return (
        str(_json_object(_json_object(receipt["baseline"])["compiled_artifact"])["mjb_sha256"]),
        str(_json_object(_json_object(receipt["candidate"])["compiled_artifact"])["mjb_sha256"]),
    )


def _zeta_allow_rules(root: Path, baseline: Path, candidate: Path, subjects: tuple[str, str]):
    """Discover every change, then return exact ALLOW rules for the zeta branch alone.

    Declaring one branch and leaving the other undeclared is what keeps the completed status at
    REVIEW_REQUIRED while still giving the policy a non-empty rule list to permute.
    """
    discovery_policy = root / "discovery-policy.json"
    discovery_policy.write_text(
        json.dumps(
            {
                "schema": "metrifid.model_release_policy",
                "schema_version": 1,
                "baseline_compiled_sha256": subjects[0],
                "candidate_compiled_sha256": subjects[1],
                "rules": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    output = root / "discovery"
    completed = _run_installed(
        "review-model",
        str(baseline),
        str(candidate),
        "--policy",
        str(discovery_policy),
        "--output",
        str(output),
    )
    assert completed.returncode == _TIE_EXPECTED_EXIT, completed.stdout + completed.stderr
    receipt = _json_object(json.loads((output / "model_release.json").read_text(encoding="utf-8")))
    rules = []
    for change in receipt["changes"]:
        entry = _json_object(change)
        selector = _json_object(entry["selector"])
        if selector.get("object_type") == "opaque":
            continue
        if str(selector.get("object_name", "")) != "zeta_body":
            continue
        rules.append(
            {
                "id": f"allow-{len(rules) + 1:04d}",
                "effect": "ALLOW",
                "selector": selector,
                "before_sha256": entry["before_sha256"],
                "after_sha256": entry["after_sha256"],
            }
        )
    assert rules, "the discovery pass produced no declarable zeta_body changes"
    return rules, receipt


def _write_permuted_policy(path: Path, subjects: tuple[str, str], rules: list, *, reverse: bool):
    """Write the same rule set in a chosen declaration order."""
    ordered = list(reversed(rules)) if reverse else list(rules)
    path.write_text(
        json.dumps(
            {
                "schema": "metrifid.model_release_policy",
                "schema_version": 1,
                "baseline_compiled_sha256": subjects[0],
                "candidate_compiled_sha256": subjects[1],
                "rules": ordered,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture(scope="module")
def tie_ordering_runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the tied review three ways: permuted declarations, and a pure fresh-process repeat.

    ``declared_first`` and ``declared_last`` permute both the source declaration order and the
    policy rule order. Permuting the source changes the source bytes, so those two runs are
    compared on decision order, not on receipt bytes. ``repeat`` re-runs exactly the inputs of
    ``declared_first`` in a fresh process under the other hash seed and into a different output
    directory, which is where byte identity is required.
    """
    root = tmp_path_factory.mktemp("tie-ordering")
    runs: dict[str, Any] = {}
    reference_rules: list | None = None
    first_inputs: tuple[Path, Path, Path] | None = None
    first_subjects: tuple[str, str] | None = None

    for label, alpha_first, reverse_rules, seed in (
        ("declared_first", True, False, _HASH_SEEDS[0]),
        ("declared_last", False, True, _HASH_SEEDS[1]),
    ):
        workspace = root / label
        workspace.mkdir()
        baseline, candidate = _write_tied_pair(workspace, alpha_first=alpha_first)
        subjects = _learn_subjects(workspace, baseline, candidate)
        rules, discovery_receipt = _zeta_allow_rules(workspace, baseline, candidate, subjects)
        if reference_rules is None:
            reference_rules = rules
        policy = _write_permuted_policy(
            workspace / "policy.json", subjects, reference_rules, reverse=reverse_rules
        )
        output = workspace / "review"
        completed = _run_installed(
            "review-model",
            str(baseline),
            str(candidate),
            "--policy",
            str(policy),
            "--output",
            str(output),
            hash_seed=seed,
        )
        assert completed.returncode == _TIE_EXPECTED_EXIT, completed.stdout + completed.stderr
        if first_inputs is None:
            first_inputs = (baseline, candidate, policy)
            first_subjects = subjects
        runs[label] = {
            "exit": completed.returncode,
            "seed": seed,
            "alpha_first": alpha_first,
            "rules_reversed": reverse_rules,
            "declared_rule_count": len(reference_rules),
            "policy_bytes": policy.read_bytes(),
            "json_bytes": (output / "model_release.json").read_bytes(),
            "markdown_bytes": (output / "model_release.md").read_bytes(),
            "discovery_change_count": len(discovery_receipt["changes"]),
        }

    assert first_inputs is not None
    assert reference_rules is not None
    assert first_subjects is not None
    baseline, candidate, policy = first_inputs

    # Same subjects, same source bytes, same rule set, reversed declaration order. This isolates
    # the policy half of the permutation from the source half, which necessarily recompiles.
    permuted_policy = _write_permuted_policy(
        root / "permuted-policy.json",
        first_subjects,
        reference_rules,
        reverse=True,
    )
    permuted_output = root / "policy-permuted-review"
    completed = _run_installed(
        "review-model",
        str(baseline),
        str(candidate),
        "--policy",
        str(permuted_policy),
        "--output",
        str(permuted_output),
        hash_seed=_HASH_SEEDS[1],
    )
    assert completed.returncode == _TIE_EXPECTED_EXIT, completed.stdout + completed.stderr
    runs["policy_permuted"] = {
        "exit": completed.returncode,
        "policy_bytes": permuted_policy.read_bytes(),
        "json_bytes": (permuted_output / "model_release.json").read_bytes(),
        "markdown_bytes": (permuted_output / "model_release.md").read_bytes(),
    }

    repeat_output = root / "repeat-review"
    completed = _run_installed(
        "review-model",
        str(baseline),
        str(candidate),
        "--policy",
        str(policy),
        "--output",
        str(repeat_output),
        hash_seed=_HASH_SEEDS[1],
    )
    assert completed.returncode == _TIE_EXPECTED_EXIT, completed.stdout + completed.stderr
    runs["repeat"] = {
        "exit": completed.returncode,
        "seed": _HASH_SEEDS[1],
        "json_bytes": (repeat_output / "model_release.json").read_bytes(),
        "markdown_bytes": (repeat_output / "model_release.md").read_bytes(),
    }
    return runs


def _ordered_decision_keys(receipt_bytes: bytes) -> list[tuple[str, ...]]:
    """Return the complete ordered change list as selector/classification tuples."""
    receipt = _json_object(json.loads(receipt_bytes.decode("utf-8")))
    keys: list[tuple[str, ...]] = []
    for change in receipt["changes"]:
        entry = _json_object(change)
        selector = _json_object(entry["selector"])
        keys.append(
            (
                str(selector.get("object_type")),
                str(selector.get("object_name")),
                str(selector.get("field")),
                str(selector.get("change_kind")),
                str(entry["classification"]),
                str(entry["source"]),
            )
        )
    return keys


def test_tied_changes_preserve_decision_order_under_policy_and_source_permutation(
    tie_ordering_runs: dict[str, Any],
) -> None:
    """Pin that permuting the policy and the source declarations does not move the decision."""
    first, second = tie_ordering_runs["declared_first"], tie_ordering_runs["declared_last"]

    # Both permutation axes named by the oracle case really were exercised.
    assert first["alpha_first"] != second["alpha_first"], (
        "source declaration order was not permuted"
    )
    assert first["rules_reversed"] != second["rules_reversed"], "policy rule order was not permuted"
    assert first["policy_bytes"] != second["policy_bytes"], "the two policy files were identical"
    assert first["declared_rule_count"] > 1, "a single rule cannot demonstrate order independence"
    assert first["exit"] == second["exit"] == _TIE_EXPECTED_EXIT

    # The complete ordered change list and both first witnesses are unmoved. Receipt bytes are
    # deliberately not compared here: permuting the source declaration recompiles the model, so
    # the two runs have genuinely different subjects and the receipt says so honestly.
    assert _ordered_decision_keys(first["json_bytes"]) == _ordered_decision_keys(
        second["json_bytes"]
    )
    left = _json_object(json.loads(first["json_bytes"].decode("utf-8")))
    right = _json_object(json.loads(second["json_bytes"].decode("utf-8")))
    assert left["status"] == right["status"] == _TIE_EXPECTED_STATUS
    assert (
        _json_object(left["first_unexpected_witness"])["selector"]
        == _json_object(right["first_unexpected_witness"])["selector"]
    )
    assert left["first_missing_required_witness"] == right["first_missing_required_witness"]


def test_tied_changes_keep_identical_receipt_bytes_in_a_fresh_process_and_seed(
    tie_ordering_runs: dict[str, Any],
) -> None:
    """Pin byte identity for the same inputs in a fresh process, other seed, other output."""
    first, repeat = tie_ordering_runs["declared_first"], tie_ordering_runs["repeat"]
    assert first["seed"] != repeat["seed"], "the interpreter hash seed was not varied"
    assert first["exit"] == repeat["exit"] == _TIE_EXPECTED_EXIT
    assert first["json_bytes"] == repeat["json_bytes"]
    assert first["markdown_bytes"] == repeat["markdown_bytes"]


def test_permuting_only_the_policy_rule_order_preserves_the_whole_decision(
    tie_ordering_runs: dict[str, Any],
) -> None:
    """Pin that the declared rule order is canonicalized, and that the raw file is still named.

    This run holds the source bytes and both compiled subjects fixed and reverses only the order
    the rules are written in. The canonicalized policy semantics, the ordered change list, both
    witnesses, the status and the exit code are all unmoved. The receipt is not byte-identical,
    and must not be: ``policy.raw_sha256`` identifies the exact admitted policy file, and those
    two files genuinely differ. Recording that honestly is the correct behavior, so it is pinned
    here rather than papered over.
    """
    first, permuted = tie_ordering_runs["declared_first"], tie_ordering_runs["policy_permuted"]
    assert first["policy_bytes"] != permuted["policy_bytes"], "the rule order was not permuted"
    assert first["exit"] == permuted["exit"] == _TIE_EXPECTED_EXIT

    left = _json_object(json.loads(first["json_bytes"].decode("utf-8")))
    right = _json_object(json.loads(permuted["json_bytes"].decode("utf-8")))
    left_policy = _json_object(left["policy"])
    right_policy = _json_object(right["policy"])

    # The declaration order is canonicalized away everywhere it could affect a decision.
    assert left_policy["semantic_sha256"] == right_policy["semantic_sha256"]
    assert left_policy["rules"] == right_policy["rules"]
    assert _ordered_decision_keys(first["json_bytes"]) == _ordered_decision_keys(
        permuted["json_bytes"]
    )
    assert left["status"] == right["status"] == _TIE_EXPECTED_STATUS
    assert left["first_unexpected_witness"] == right["first_unexpected_witness"]
    assert left["first_missing_required_witness"] == right["first_missing_required_witness"]
    assert left["certification_receipt"] == right["certification_receipt"]

    # The exact admitted policy file is still identified, so these three differ by design.
    assert left_policy["raw_sha256"] != right_policy["raw_sha256"]
    assert left["decision_sha256"] != right["decision_sha256"]
    assert left["receipt_sha256"] != right["receipt_sha256"]


def test_tied_undeclared_changes_keep_the_canonical_first_witness(
    tie_ordering_runs: dict[str, Any],
) -> None:
    """Require both branches to survive, one declared and one undeclared, with the earlier one."""
    receipt = _json_object(
        json.loads(tie_ordering_runs["declared_first"]["json_bytes"].decode("utf-8"))
    )
    assert receipt["status"] == _TIE_EXPECTED_STATUS
    assert receipt["completed_exit_code"] == _TIE_EXPECTED_EXIT

    classifications: dict[str, set[str]] = {}
    for change in receipt["changes"]:
        entry = _json_object(change)
        name = str(_json_object(entry["selector"]).get("object_name", ""))
        classifications.setdefault(name, set()).add(str(entry["classification"]))

    assert "UNDECLARED" in classifications["alpha_body"]
    assert classifications["zeta_body"] == {"ALLOWED"}

    # TWO_UNDECLARED_CHANGES_PRESENT: the undeclared branch contributes more than one change, so
    # the ordering between tied undeclared rows is a real ordering rather than a single row.
    undeclared = [
        change
        for change in receipt["changes"]
        if str(_json_object(change)["classification"]) == "UNDECLARED"
    ]
    assert len(undeclared) >= 2, f"expected at least two undeclared changes, got {len(undeclared)}"
    assert all(
        str(_json_object(_json_object(change)["selector"]).get("object_name")) != "zeta_body"
        for change in undeclared
    )

    witness = _json_object(receipt["first_unexpected_witness"])
    selector = _json_object(witness["selector"])
    assert selector["object_type"] == "body"
    assert selector["object_name"] == "alpha_body"
    assert receipt["first_missing_required_witness"] is None
