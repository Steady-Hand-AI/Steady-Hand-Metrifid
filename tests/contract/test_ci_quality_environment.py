"""Contract for the frozen CI workflow and the release evidence helper.

The workflow is frozen for this release, so its raw-byte SHA-256 is the contract: any byte that
changes fails here before anything can normalize or overlook it. A digest constrains the file that
is admitted, not how a hosted runner behaves.

The remaining workflow assertions are diagnostic. They name individual workflow properties and
decide nothing.

The evidence helper is checked semantically, because the values it judges are produced at runtime
and cannot be frozen.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

_REPOSITORY = Path(__file__).resolve().parents[2]
_WORKFLOW = _REPOSITORY / ".github/workflows/ci.yml"
_QUALITY_CONSTRAINTS = _REPOSITORY / ".github/quality-constraints.txt"
_VALIDATOR = _REPOSITORY / ".github/scripts/validate_ci_evidence.py"
_ACTION = _REPOSITORY / "action.yml"

# The expected digest of the frozen release workflow.
_FROZEN_WORKFLOW_SHA256 = "a202183f36d2feda738eaf83323e37e6cfa99c73505c8109b4046977e3cf39a5"

_ALL_JOBS = (
    "build_and_quality",
    "matrix_lane",
    "minimum_dependency_lane",
    "retained_compatibility_lane",
    "sdist_install_lane",
    "distribution_equivalence",
    "expanded_full_diagnostics",
    "test_composite_action",
    "release_matrix_summary",
)
_FULL_LANES = ("linux_x64_py312_full", "linux_x64_py311_numpy_min")
_SMOKE_LANES = ("linux_x64_py313", "linux_x64_py314", "macos_arm64_py314", "macos_x64_py312")
# The jobs whose steps can execute the complete installed-wheel suite, and so the jobs that
# must provision the build frontend those tests shell out to.
_COMPLETE_SUITE_OWNERS = ("matrix_lane", "minimum_dependency_lane", "expanded_full_diagnostics")
_PUBLIC_COMMANDS = (
    "certify",
    "review-model",
    "compare",
    "audit-timestep",
    "qualify-workload",
    "review-runtime",
    "run-runtime-review",
)
# A label-driven activity type can let a skipped required check report success.
_LABEL_PATHS = (
    "labeled",
    "github.event.label",
    "ci-expanded-full",
    "pull_request.draft",
    "DIAGNOSTIC_LABEL",
)


class FrozenWorkflowError(AssertionError):
    """The shipped workflow does not match the frozen bytes."""


def _workflow_bytes() -> bytes:
    """Return the shipped CI workflow exactly as it is stored."""
    assert _WORKFLOW.is_file(), f"missing workflow: {_WORKFLOW}"
    return _WORKFLOW.read_bytes()


def _workflow_text() -> str:
    """Return the shipped CI workflow source, for the diagnostic assertions below."""
    return _workflow_bytes().decode("utf-8")


def check_frozen_workflow(workflow: bytes) -> str:
    """Require the workflow to match the frozen bytes, and return its digest.

    Nothing is normalized first. Line endings, whitespace, comments and quoting are all part of the
    frozen bytes, so a one-byte difference is a difference.
    """
    observed = hashlib.sha256(workflow).hexdigest()
    if observed != _FROZEN_WORKFLOW_SHA256:
        raise FrozenWorkflowError(
            f"the CI workflow does not match the frozen bytes: expected "
            f"{_FROZEN_WORKFLOW_SHA256}, observed {observed}"
        )
    return observed


def _job(text: str, name: str) -> str:
    """Return one top-level job body, for the diagnostic assertions below.

    This reads the frozen file only to make a failure legible. It is not an authority: the digest
    above already decided whether these bytes are acceptable.
    """
    start = text.index(f"\n  {name}:\n")
    following = re.search(r"\n  [a-z_]+:\n", text[start + 1 :])
    if following is None:
        return text[start:]
    return text[start : start + 1 + following.start()]


@functools.lru_cache(maxsize=1)
def _load_validator() -> ModuleType:
    """Import the decision helper straight from the path CI invokes."""
    assert _VALIDATOR.is_file(), f"missing decision helper: {_VALIDATOR}"
    specification = importlib.util.spec_from_file_location("validate_ci_evidence", _VALIDATOR)
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules["validate_ci_evidence"] = module
    specification.loader.exec_module(module)
    return module


# ---- The frozen-workflow authority -----------------------------------------------------------------


_TYPES_LINE = "    types: [opened, synchronize, reopened]\n"


def _relocate_pull_request_types(text: str) -> str:
    """Move the activity list from `pull_request` to `push`, leaving exactly one occurrence."""
    assert text.count(_TYPES_LINE) == 1, "expected one activity list to relocate"
    moved = text.replace(_TYPES_LINE, "", 1).replace(
        "  push:\n    branches: [main]\n", "  push:\n" + _TYPES_LINE + "    branches: [main]\n", 1
    )
    assert moved.count(_TYPES_LINE) == 1, "the relocation must not duplicate the activity list"
    assert "  push:\n" + _TYPES_LINE in moved, "the activity list must now sit under push"
    return moved


def test_the_shipped_workflow_matches_the_frozen_bytes() -> None:
    """The whole workflow contract: the shipped bytes hash to the frozen digest."""
    assert check_frozen_workflow(_workflow_bytes()) == _FROZEN_WORKFLOW_SHA256


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        pytest.param(
            "one_character",
            lambda t: t.replace("runs-on: ubuntu-24.04", "runs-on: ubuntu-24.05", 1),
            id="one_character",
        ),
        pytest.param(
            "comment_only",
            lambda t: t.replace("name: ci\n", "name: ci\n# harmless\n", 1),
            id="comment_only",
        ),
        pytest.param(
            "flow_style_step",
            lambda t: t.replace(
                "      - name: Record sdist lane evidence\n",
                "      - {name: Hidden, run: python -m pytest tests}\n"
                "      - name: Record sdist lane evidence\n",
                1,
            ),
            id="flow_style_step",
        ),
        pytest.param(
            "no_target_pytest",
            lambda t: t.replace(
                "      - name: Record sdist lane evidence\n",
                "      - name: Hidden\n        run: python -m pytest -q\n"
                "      - name: Record sdist lane evidence\n",
                1,
            ),
            id="no_target_pytest",
        ),
        pytest.param(
            "repository_root_pytest",
            lambda t: t.replace(
                "      - name: Record sdist lane evidence\n",
                "      - name: Hidden\n        run: python -m pytest -q .\n"
                "      - name: Record sdist lane evidence\n",
                1,
            ),
            id="repository_root_pytest",
        ),
        pytest.param(
            "relocated_pull_request_types",
            _relocate_pull_request_types,
            id="relocated_pull_request_types",
        ),
        pytest.param(
            "inactive_head_comparison",
            lambda t: t.replace(
                '            test "$head_sha" = "$PR_HEAD_SHA"',
                '            # test "$head_sha" = "$PR_HEAD_SHA"',
                1,
            ),
            id="inactive_head_comparison",
        ),
        pytest.param(
            "commented_output_assertion",
            lambda t: t.replace(
                '          if [ "${{ steps.test_64.outputs.status }}" != "refused" ]; then exit 1; fi',
                '          # if [ "${{ steps.test_64.outputs.status }}" != "refused" ]; then exit 1; fi',
                1,
            ),
            id="commented_output_assertion",
        ),
        pytest.param(
            "noop_output_assertion",
            lambda t: t.replace(
                '"${{ steps.test_64.outputs.status }}" != "refused" ]; then exit 1; fi',
                '"${{ steps.test_64.outputs.status }}" != "refused" ]; then :; fi',
                1,
            ),
            id="noop_output_assertion",
        ),
        pytest.param(
            "canary_propagation_removed",
            lambda t: t.replace(
                '          echo "BASH_ENV=$RUNNER_TEMP/metrifid_canary.sh" >> "$GITHUB_ENV"\n',
                "",
                1,
            ),
            id="canary_propagation_removed",
        ),
        pytest.param(
            "rebound_receipt_value",
            lambda t: t.replace(
                '              "checkout_tree": os.environ["CHECKOUT_TREE"],',
                '              "checkout_tree": "0" * 40,  # os.environ["CHECKOUT_TREE"]',
                1,
            ),
            id="rebound_receipt_value",
        ),
    ],
)
def test_any_workflow_byte_change_fails_the_freeze(name: str, mutate: Any) -> None:
    """Every changed byte fails the freeze."""
    original = _workflow_text()
    mutated = mutate(original)
    assert mutated != original, name
    with pytest.raises(FrozenWorkflowError) as raised:
        check_frozen_workflow(mutated.encode("utf-8"))
    assert _FROZEN_WORKFLOW_SHA256 in str(raised.value)
    assert "observed" in str(raised.value)


# ---- Diagnostic assertions about the frozen workflow -------------------------------------------------
#
# The digest above is the decision authority. These assertions decide nothing; they name individual
# workflow properties so that a mismatch is legible.


def test_the_default_matrix_is_one_complete_lane_and_four_boundary_smokes() -> None:
    """One complete lane and four boundary smokes, each pinned by coordinate."""
    matrix = _job(_workflow_text(), "matrix_lane")
    for lane in (*_FULL_LANES[:1], *_SMOKE_LANES):
        assert f"- id: {lane}\n" in matrix, lane
    assert matrix.count("tier: full") == 1
    assert matrix.count("tier: smoke") == 4
    for runner in ("ubuntu-24.04", "macos-15", "macos-15-intel"):
        assert runner in matrix, runner


def test_exactly_two_complete_suites_run_in_normal_ci() -> None:
    """Two complete suites is the budget; a third would re-prove one of them."""
    text = _workflow_text()
    owners = [name for name in _ALL_JOBS if "full.xml tests" in _job(text, name)]
    assert owners == ["matrix_lane", "minimum_dependency_lane", "expanded_full_diagnostics"], owners


def test_a_smoke_lane_never_runs_the_complete_suite() -> None:
    """The tier gate, not a comment, is what keeps a smoke lane cheap."""
    matrix = _job(_workflow_text(), "matrix_lane")
    complete = matrix.index("Complete installed-wheel test suite")
    focused = matrix.index("Focused semantic suites for this boundary")
    assert "if: matrix.tier == 'full'" in matrix[complete - 200 : complete + 200]
    assert "if: matrix.tier == 'smoke'" in matrix[focused - 200 : focused + 200]


def test_the_second_fixed_seed_full_rerun_is_gone() -> None:
    """Rerunning the same suite at the same seed on the same runtime proves nothing new."""
    text = _workflow_text()
    assert "randomized.xml" not in text
    assert "Fixed randomized installed-wheel suite" not in text


def test_the_shared_focused_lists_are_exact_and_have_one_authority() -> None:
    """Both focused lists are defined once, at workflow level, with their exact entries."""
    text = _workflow_text()
    for name in ("FOCUSED_SEMANTIC_TESTS", "SDIST_FOCUSED_TESTS"):
        assert len(re.findall(rf"^\s*{name}\s*:", text, re.M)) == 1, name
        assert re.search(rf"(?:export\s+)?{name}=", text) is None, name
    for entry in (
        "tests/unit/runtime_compatibility",
        "tests/contract/runtime_compatibility",
        "tests/integration/runtime_compatibility",
        "tests/unit/test_operational_registry.py",
        "tests/unit/test_compare_environment.py",
        "tests/contract/test_certify_admission.py",
        "tests/contract/test_model_identity_contract.py",
        "tests/contract/model_release/test_receipt.py",
        "tests/contract/test_public_api.py",
        "tests/integration/test_installed_distribution.py",
    ):
        assert entry in text, entry
    for entry in (
        "tests/contract/test_runtime_review_public_surface.py",
        "tests/integration/test_examples.py",
        "tests/integration/test_sdk_examples.py",
        "tests/integration/test_operation_labels_and_config_admission.py",
    ):
        assert entry in text, entry


def test_the_minimum_dependency_lane_installs_the_declared_floor_before_the_wheel() -> None:
    """The floor must be resolved first, then the wheel with --no-deps, or it is not a floor."""
    job = _job(_workflow_text(), "minimum_dependency_lane")
    floor = job.index('pip install "numpy==1.26.4" "mujoco==3.9.0"')
    wheel = job.index('pip install --no-deps "$(ls release-artifacts/dist/*.whl)"')
    assert floor < wheel, "the minimum profile must be installed before the wheel"
    assert "printf '%s\\n' 'full' > lane-evidence/tier.txt" in job


def test_every_default_lane_binds_to_the_original_artifact_manifest() -> None:
    """A lane that does not check the original digest could be testing a parity rebuild."""
    text = _workflow_text()
    for name in (
        "matrix_lane",
        "minimum_dependency_lane",
        "retained_compatibility_lane",
        "sdist_install_lane",
    ):
        job = _job(text, name)
        assert "original_manifest.json" in job, name
        assert "installed_artifact.json" in job, name


def test_the_parity_rebuild_is_named_as_evidence_and_never_installed() -> None:
    """The wheel rebuilt from the sdist must not masquerade as the publication candidate."""
    job = _job(_workflow_text(), "distribution_equivalence")
    assert "parity_from_sdist" in job
    assert '"role": "parity_evidence_only"' in job
    installs = re.findall(r"pip install ([^\n]*)", job)
    assert installs, "expected at least the build-backend install"
    for arguments in installs:
        assert arguments.strip() == "-c .github/quality-constraints.txt build", arguments


def test_no_lane_reintroduces_a_runner_local_mujoco_resolver_override() -> None:
    """The published metadata, not a runner workaround, must select an installable MuJoCo.

    A constraints file, a binary-only pin or an exact MuJoCo pin around the candidate install
    would make the Intel lane green while the published metadata stayed broken for every user.
    That is the exact failure this release corrects, so the mechanism may not come back under
    any spelling.
    """
    text = _workflow_text()
    for token in (
        "PIP_CONSTRAINT",
        "PIP_ONLY_BINARY",
        "intel-constraint",
        "intel_mujoco_pin",
        "INTEL_MUJOCO_PIN",
        "MUJOCO_PATH",
    ):
        assert token not in text, token
    assert "mujoco" not in _QUALITY_CONSTRAINTS.read_text(encoding="utf-8")
    # The two lanes that deliberately install an exact floor or an exact retained profile are
    # named; every other job must leave the resolver alone. The pattern matches a version pin,
    # not any Python comparison that happens to mention MuJoCo: the required lane legitimately
    # compares its install report against the imported ``mujoco.__version__``.
    pin = re.compile(r"(?<![\w.])mujoco\s*==\s*[\"']?[0-9]", re.IGNORECASE)
    for name in ("build_and_quality", "matrix_lane", "sdist_install_lane"):
        job = _job(text, name)
        assert pin.search(job) is None, (name, pin.search(job))
    # Controls: the pattern must ignore the legitimate comparison and still catch a real pin.
    assert pin.search("assert reported_mujoco == mujoco.__version__") is None
    assert pin.search('printf "mujoco==3.10.0"') is not None
    assert pin.search("pip install mujoco==3.9.0") is not None


def test_the_required_intel_lane_installs_the_candidate_wheel_ordinarily() -> None:
    """The Intel lane must prove the ordinary user install, and record pip's own resolve.

    ``--no-deps`` or a preinstalled MuJoCo would skip the dependency resolution this lane exists
    to observe, so the candidate install must carry its dependencies and emit the install report
    the release summary binds to.
    """
    job = _job(_workflow_text(), "matrix_lane")
    installs = re.findall(r"pip install ((?:[^\n]*\\\n)*[^\n]*)", job)
    candidate = [line for line in installs if "release-artifacts/dist/*.whl" in line]
    assert len(candidate) == 1, candidate
    command = candidate[0]
    assert "--report lane-evidence/install_report.json" in command, command
    for forbidden in ("--no-deps", "--only-binary", "-c ", "--constraint", "--index-url"):
        assert forbidden not in command, (forbidden, command)
    # The report is written into the directory the lane uploads as its evidence.
    assert 'pathlib.Path("lane-evidence").mkdir' in job


def test_the_intel_lane_is_the_only_required_lane_carrying_a_packaging_ceiling() -> None:
    """Exactly one required lane declares the Darwin x86_64 packaging bound."""
    validator = _load_validator()
    bounded = {
        lane: want.mujoco_wheel_ceiling
        for lane, want in validator.LANE_EXPECTATIONS.items()
        if want.mujoco_wheel_ceiling is not None
    }
    assert bounded == {"macos_x64_py312": (3, 11)}
    want = validator.LANE_EXPECTATIONS["macos_x64_py312"]
    assert (want.platform_system, want.platform_machine) == ("Darwin", "x86_64")
    # The bound must not be expressed as a frozen exact version: that would stop the lane from
    # observing which release the resolver actually selects.
    assert want.mujoco_version is None


def test_runner_architecture_is_measured_rather_than_assumed() -> None:
    """A runner label is a promise; platform.machine() is the observation."""
    matrix = _job(_workflow_text(), "matrix_lane")
    assert 'machine == os.environ["EXPECT_MACHINE"]' in matrix
    assert matrix.count("expect_machine:") == 5


def test_retained_exact_profiles_keep_their_own_boundary() -> None:
    """The retained runtimes are focused lanes, not complete suites."""
    job = _job(_workflow_text(), "retained_compatibility_lane")
    assert 'mujoco: "3.10.0"' in job
    assert 'mujoco: "3.11.0"' in job
    assert "retained-evidence/focused.xml" in job
    assert "full.xml tests" not in job
    assert "validate_mujoco_compatibility.py" in job


def test_the_sdist_lane_proves_packaging_not_behaviour() -> None:
    """The direct-sdist lane runs the focused public-surface list, not the whole suite."""
    job = _job(_workflow_text(), "sdist_install_lane")
    assert "sdist-evidence/focused.xml" in job
    assert "full.xml tests" not in job
    assert 'pip install "$(ls release-artifacts/dist/*.tar.gz)"' in job


def test_every_lane_exercises_all_seven_command_surfaces() -> None:
    """A lane must invoke each command and leave a marker proving it, not just print a list."""
    text = _workflow_text()
    for name in (
        "matrix_lane",
        "minimum_dependency_lane",
        "retained_compatibility_lane",
        "sdist_install_lane",
    ):
        job = _job(text, name)
        assert "printf '### metrifid %s --help\\n'" in job, name
        loop = job[job.index("for command in") : job.index("; do", job.index("for command in"))]
        for command in _PUBLIC_COMMANDS:
            assert command in loop, (name, command)


def test_expanded_diagnostics_cannot_run_on_an_ordinary_push() -> None:
    """The broad matrix is manual-only; any other route reintroduces the duplication it removed."""
    job = _job(_workflow_text(), "expanded_full_diagnostics")
    assert "if: github.event_name == 'workflow_dispatch' && inputs.expanded_full_matrix" in job
    assert "continue-on-error: true" in job
    assert "fail-fast: false" in job
    assert "expanded_full_diagnostics" not in _load_validator().REQUIRED_UPSTREAM


def test_label_trigger_and_summary_skip_paths_are_absent() -> None:
    """A skipped required check reports success, so no job may be conditionally absent."""
    text = _workflow_text()
    for path in _LABEL_PATHS:
        assert path not in text, path
    assert (
        re.search(r"^    if: always\(\)$", _job(text, "release_matrix_summary"), re.M) is not None
    )
    for name in _ALL_JOBS:
        if name in ("release_matrix_summary", "expanded_full_diagnostics"):
            continue
        assert re.search(r"^    if:", _job(text, name), re.M) is None, name


def test_pull_request_activity_types_are_explicit() -> None:
    """Exactly the three activity types that mean there is new code to judge."""
    assert "    types: [opened, synchronize, reopened]\n" in _workflow_text()


def test_the_workflow_never_uses_a_privileged_trigger_or_secret() -> None:
    """A diagnostic opt-in must not become a privilege escalation."""
    text = _workflow_text()
    assert "pull_request_target" not in text
    assert "secrets." not in text
    assert "permissions:\n  contents: read\n" in text
    assert "contents: write" not in text


def test_the_summary_needs_every_required_job_including_the_build() -> None:
    """Without the build there is no original artifact and no review subject to bind anything to."""
    job = _job(_workflow_text(), "release_matrix_summary")
    for name in _load_validator().REQUIRED_UPSTREAM:
        assert f"- {name}" in job, name
    assert "always()" in job


def test_the_summary_delegates_the_decision_to_the_single_helper() -> None:
    """Evidence logic lives in one importable module, not in embedded workflow Python."""
    job = _job(_workflow_text(), "release_matrix_summary")
    assert "python .github/scripts/validate_ci_evidence.py" in job
    assert "<<'PY'" not in job, "the summary must not carry a second decision authority"
    assert "PASS_METRIFID_RELEASE_MATRIX" not in job
    for argument in ("--event-name", "--event-sha", "--pull-request-head-sha"):
        assert argument in job, argument


def test_no_blocking_step_swallows_its_own_failure() -> None:
    """`|| true` and an unaccounted continue-on-error are how a red run reports green."""
    text = _workflow_text()
    assert "|| true" not in text
    tolerated = {
        "expanded_full_diagnostics": 1,
        "release_matrix_summary": 4,
        "test_composite_action": 4,
    }
    total = 0
    for name, expected in tolerated.items():
        found = _job(text, name).count("continue-on-error: true")
        assert found == expected, (name, found, expected)
        total += expected
    assert text.count("continue-on-error: true") == total


def test_every_tolerated_failure_in_the_action_test_is_actually_asserted() -> None:
    """A negative test must check the outcome it provoked, not merely survive it."""
    job = _job(_workflow_text(), "test_composite_action")
    provoked = re.findall(
        r"id: (test_\w+)\n        uses: \./\n        continue-on-error: true", job
    )
    assert len(provoked) == 4, provoked
    for step in provoked:
        assert f'steps.{step}.outcome }}}}" != "failure"' in job, step


# ---- The reviewed release surfaces are locked by their bytes ----------------------------------
#
# `action.yml` and the CI workflow are held by exact SHA-256, not interpreted. This file neither
# parses GitHub Actions YAML nor evaluates Bash: that would approximate two evaluators, and an
# approximation is where a weakened step hides. Changing either surface requires a digest update,
# review of the new bytes, and a hosted CI run; what the action does when it runs is proved by the
# `test_composite_action` job on GitHub.

_FROZEN_ACTION_SHA256 = "c80cad44468604fc9d3154254d9d57f87839ae681b311d72e5010331fdac38a9"

_ADVERSARIAL_STRICT = '$(touch "$RUNNER_TEMP/metrifid-strict-expanded")'


class FrozenActionError(AssertionError):
    """The shipped composite action does not match the frozen bytes."""


def _action_bytes() -> bytes:
    """Return the shipped composite action exactly as it is stored."""
    assert _ACTION.is_file(), f"missing action: {_ACTION}"
    return _ACTION.read_bytes()


def check_frozen_action(action: bytes) -> str:
    """Require the action to match the frozen bytes, and return its digest."""
    observed = hashlib.sha256(action).hexdigest()
    if observed != _FROZEN_ACTION_SHA256:
        raise FrozenActionError(
            f"the composite action does not match the frozen bytes: expected "
            f"{_FROZEN_ACTION_SHA256}, observed {observed}"
        )
    return observed


def test_the_shipped_action_matches_the_frozen_bytes() -> None:
    """The reviewed action is the one that ships."""
    assert check_frozen_action(_action_bytes()) == _FROZEN_ACTION_SHA256


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        pytest.param(
            "direct_interpolation",
            lambda t: t.replace('if [ "$STRICT" !=', 'if [ "${{ inputs.strict }}" !=', 1),
            id="direct_interpolation",
        ),
        pytest.param(
            "weakened_strict_guard",
            lambda t: t.replace(
                'if [ "$STRICT" != "true" ] && [ "$STRICT" != "false" ]; then',
                'if [ "$STRICT" = "nonsense" ]; then',
                1,
            ),
            id="weakened_strict_guard",
        ),
        pytest.param(
            "bare_platform_global_mktemp",
            lambda t: t.replace(
                'OUTPUT_DIR=$(mktemp -d "$RUNNER_TEMP/metrifid-certify-XXXXXX")',
                "OUTPUT_DIR=$(mktemp -d)",
                1,
            ),
            id="bare_platform_global_mktemp",
        ),
        pytest.param(
            "runner_temp_guard_disabled",
            lambda t: t.replace(
                'if [ -z "${RUNNER_TEMP:-}" ] || [ ! -d "$RUNNER_TEMP" ]',
                'if [ -n "${RUNNER_TEMP:-}" ] && [ ! -d "$RUNNER_TEMP" ]',
                1,
            ),
            id="runner_temp_guard_disabled",
        ),
    ],
)
def test_any_action_byte_change_fails_the_freeze(name: str, mutate: Any) -> None:
    """Reviewed bytes changed. The diagnosis is the digest, not a reading of the change."""
    original = _action_bytes().decode("utf-8")
    mutated = mutate(original)
    assert mutated != original, name
    with pytest.raises(FrozenActionError) as raised:
        check_frozen_action(mutated.encode("utf-8"))
    assert _FROZEN_ACTION_SHA256 in str(raised.value)
    assert "observed" in str(raised.value)


# The only behavioural claim here; both controls run their own temporary scripts.


def test_the_adversarial_payload_executes_under_direct_interpolation(tmp_path: Path) -> None:
    """The fixture only means anything if the historical vulnerable form would really run it.

    A payload that failed to parse would leave the sentinel absent for the wrong reason.
    """
    sentinel = tmp_path / "metrifid-strict-expanded"
    program = tmp_path / "vulnerable.sh"
    program.write_text(f'STRICT="{_ADVERSARIAL_STRICT}"\n', encoding="utf-8")
    environment = {**os.environ, "RUNNER_TEMP": str(tmp_path)}
    parsed = subprocess.run(["bash", "-n", str(program)], capture_output=True, text=True)
    assert parsed.returncode == 0, parsed.stderr
    ran = subprocess.run(
        ["bash", str(program)], capture_output=True, text=True, cwd=tmp_path, env=environment
    )
    assert ran.returncode == 0, ran.stderr
    assert sentinel.exists(), "the payload must execute under direct interpolation"


def test_the_adversarial_payload_is_inert_through_the_env_guard(tmp_path: Path) -> None:
    """The same payload, handled the way the action handles it, expands nothing."""
    sentinel = tmp_path / "metrifid-strict-expanded"
    program = tmp_path / "guard.sh"
    program.write_text(
        'if [ "$STRICT" != "true" ] && [ "$STRICT" != "false" ]; then\n  exit 1\nfi\n',
        encoding="utf-8",
    )
    environment = {**os.environ, "RUNNER_TEMP": str(tmp_path), "STRICT": _ADVERSARIAL_STRICT}
    ran = subprocess.run(
        ["bash", str(program)], capture_output=True, text=True, cwd=tmp_path, env=environment
    )
    assert ran.returncode == 1, "the value must be refused"
    assert not sentinel.exists(), "the value must not be expanded"


def test_multiline_shell_fails_fast_and_on_pipe_failure() -> None:
    """A lane that pipes test output must not pass because `tee` succeeded."""
    text = _workflow_text()
    assert "defaults:\n  run:\n    # " in text
    assert "shell: bash" in text


def test_every_external_action_is_pinned_to_a_full_commit_sha() -> None:
    """A moving tag is a supply-chain hole in a job that touches release artifacts."""
    for reference in re.findall(r"uses: ([^\s]+)", _workflow_text()):
        if reference.startswith("./"):
            continue
        _, _, version = reference.partition("@")
        assert re.fullmatch(r"[0-9a-f]{40}", version), reference


def test_the_workflow_carries_no_placeholder_or_commented_alternative() -> None:
    """Half-finished infrastructure is indistinguishable from working infrastructure."""
    text = _workflow_text()
    for marker in ("TODO", "FIXME", "XXX", "placeholder"):
        assert marker not in text, marker


def test_every_runtime_identity_records_the_complete_structural_field_set() -> None:
    """Every lane records the same fields, including the diagnostic job."""
    text = _workflow_text()
    validator = _load_validator()
    for name in (
        "build_and_quality",
        "matrix_lane",
        "minimum_dependency_lane",
        "retained_compatibility_lane",
        "sdist_install_lane",
        "expanded_full_diagnostics",
    ):
        job = _job(text, name)
        for field_name in validator.REQUIRED_IDENTITY_FIELDS:
            assert f'"{field_name}"' in job, (name, field_name)
        assert re.search(r"^\s+import numpy$", job, re.M) is not None, name


def test_the_build_job_records_the_exact_review_subject() -> None:
    """A pull request builds a merge commit, so the head a reviewer reads is fetched by ref."""
    job = _job(_workflow_text(), "build_and_quality")
    assert "persist-credentials: false" in job
    assert 'git fetch --no-tags --depth=1 origin "refs/pull/${PR_NUMBER}/head"' in job
    assert 'test "$head_sha" = "$PR_HEAD_SHA"' in job
    assert 'test "$checkout_sha" = "$EVENT_SHA"' in job
    assert "checkout_tree=\"$(git rev-parse 'HEAD^{tree}')\"" in job
    assert "head_tree=\"$(git rev-parse 'FETCH_HEAD^{tree}')\"" in job
    assert "quality-evidence/checkout_identity.json" in job
    for field_name in _load_validator().CHECKOUT_IDENTITY_FIELDS:
        assert f'"{field_name}"' in job, field_name


# ---- Deterministic test tooling ---------------------------------------------------------------------


def test_ci_pins_the_exact_test_tools_every_pytest_lane_installs() -> None:
    """A lane that runs `-n auto` without xdist installed fails for the wrong reason."""
    constraints = _QUALITY_CONSTRAINTS.read_text(encoding="utf-8")
    assert "pytest-check==2.9.1" in constraints
    assert "pytest-xdist==3.8.0" in constraints
    text = _workflow_text()
    for name in (
        "build_and_quality",
        "matrix_lane",
        "minimum_dependency_lane",
        "retained_compatibility_lane",
        "sdist_install_lane",
    ):
        job = _job(text, name)
        assert "pytest-check" in job, name
        assert "pytest-xdist" in job, name
        assert "-c .github/quality-constraints.txt" in job, name


# ---- The build frontend every complete installed-wheel suite needs ---------------------------------
#
# The committed distribution contract shells out to `python -m build`. A job that runs the complete
# suite without the constrained build frontend fails those nine tests for a provisioning reason.
# PyYAML is not installed in the lanes that run this suite, so the workflow is read as indented text.
# The reads below are anchored to step boundaries, to property indentation, and to the installer
# string, so a comment or a nested `run:` line is never mistaken for an active install.

_COMPLETE_SUITE_MARKER = "full.xml tests"
_CONSTRAINED_INSTALL = "python -m pip install -c .github/quality-constraints.txt"
_FULL_TIER = "matrix.tier == 'full'"


def _steps(job: str) -> list[str]:
    """Return each step of one job body."""
    return re.split(r"\n(?=      - )", job)[1:]


def _constrained_packages(step: str) -> set[str]:
    """Return every package this step installs through the constraints file.

    Shell comments go first, so a commented-out install never counts as provisioning. Continuations
    are then joined the way the shell joins them: the backslash and the newline are removed and the
    next line's own indentation is what separates the arguments.
    """
    script = "\n".join(line.split("#", 1)[0] for line in step.splitlines())
    packages: set[str] = set()
    for command in script.replace("\\\n", "").split("\n"):
        if _CONSTRAINED_INSTALL not in command:
            continue
        packages.update(
            argument
            for argument in command.split(_CONSTRAINED_INSTALL, 1)[1].split()
            if not argument.startswith("-")
        )
    return packages


def _step_property(step: str, name: str) -> str | None:
    """Return one active property of a step, or None when the step does not set it.

    Only a line at the step's own property indentation counts, so a value inside a `run:` block or
    behind a `#` is never read as the property itself.
    """
    for line in step.splitlines():
        if line.startswith(f"        {name}:") and not line[8:9].isspace():
            return line.split(":", 1)[1].strip()
    return None


def _build_provisioning_steps(text: str, owner: str) -> list[str]:
    """Return the steps of `owner` that install constrained build before its complete suite."""
    steps = _steps(_job(text, owner))
    suite = next(index for index, step in enumerate(steps) if _COMPLETE_SUITE_MARKER in step)
    return [step for step in steps[:suite] if "build" in _constrained_packages(step)]


def test_the_constraints_file_pins_the_build_frontend_exactly() -> None:
    """Provisioning is only deterministic if the frontend itself is pinned to one version."""
    constraints = _QUALITY_CONSTRAINTS.read_text(encoding="utf-8")
    assert re.search(r"^build==[0-9]+(\.[0-9]+)+$", constraints, flags=re.MULTILINE), constraints


def test_every_complete_suite_owner_provisions_the_constrained_build_frontend() -> None:
    """Each job that can run the complete suite installs constrained build before pytest."""
    text = _workflow_text()
    owners = tuple(name for name in _ALL_JOBS if _COMPLETE_SUITE_MARKER in _job(text, name))
    assert owners == _COMPLETE_SUITE_OWNERS, owners
    for owner in owners:
        assert len(_build_provisioning_steps(text, owner)) == 1, owner


def test_the_matrix_build_provision_is_full_tier_only() -> None:
    """Only the complete row acquires the frontend; the four smoke rows stay cheap."""
    steps = _steps(_job(_workflow_text(), "matrix_lane"))
    provisioning = [step for step in steps if "build" in _constrained_packages(step)]
    assert len(provisioning) == 1, provisioning
    assert _step_property(provisioning[0], "if") == _FULL_TIER
    suite = next(step for step in steps if _COMPLETE_SUITE_MARKER in step)
    assert _step_property(suite, "if") == _FULL_TIER


def test_no_smoke_or_focused_only_lane_acquires_the_build_frontend() -> None:
    """A lane that never runs the distribution tests has no reason to install a build frontend."""
    text = _workflow_text()
    for name in ("retained_compatibility_lane", "sdist_install_lane"):
        job = _job(text, name)
        assert all("build" not in _constrained_packages(step) for step in _steps(job)), name


# ---- The Intel macOS lane runs the real composite action --------------------------------------
#
# A bare `mktemp -d` allocates under the platform-global temporary root, which macOS reaches through
# a symlink, and Metrifid refuses an output path with a symlinked component. Nothing but executing
# the real action on that runner catches it: the Linux action job cannot, and the macOS package
# lanes never invoked the action at all. These assertions require that proof to stay in place after
# a legitimate digest update; the digest itself remains the authority on the bytes.

_MACOS_ACTION_LANE = "macos_x64_py312"
_MACOS_ACTION_GUARD = "matrix.id == 'macos_x64_py312'"
_MACOS_ACTION_ID = "macos_action"
_MACOS_ACTION_STEP = "      - name: Real composite action on Intel macOS\n"
_MACOS_LANE_EVIDENCE = "name: lane-${{ matrix.id }}"
_MACOS_EXIT_CODE_CHECK = (
    'if [ "${{ steps.macos_action.outputs.exit_code }}" != "0" ]; then exit 1; fi'
)
_MACOS_STATUS_CHECK = (
    'if [ "${{ steps.macos_action.outputs.status }}"'
    ' != "certified_compiled_equivalence" ]; then exit 1; fi'
)
_MACOS_ACTION_INPUTS = (
    "baseline_mjcf: examples/certify/equivalent/baseline.xml",
    "candidate_mjcf: examples/certify/equivalent/candidate.xml",
    'python_version: "3.12"',
)


def _active_lines(step: str) -> list[str]:
    """Return the step's lines with comments and indentation removed.

    Raw substring membership cannot tell an assertion from a commented-out assertion, and a disabled
    check that still reads as present is exactly how a proof survives review while proving nothing.
    Comments go first, the same way `_constrained_packages` reads an install line.
    """
    return [code for line in step.splitlines() if (code := line.split("#", 1)[0].strip())]


def _local_action_problems(steps: list[str]) -> list[str]:
    """Return why the lane's `uses: ./` step is not the reviewed one."""
    local = [step for step in steps if _step_property(step, "uses") == "./"]
    if len(local) != 1:
        return [f"expected exactly one local-action step, found {len(local)}"]
    step = local[0]
    active = _active_lines(step)
    problems: list[str] = []
    if _step_property(step, "id") != _MACOS_ACTION_ID:
        problems.append(f"the local-action step is not id {_MACOS_ACTION_ID}")
    if _step_property(step, "if") != _MACOS_ACTION_GUARD:
        problems.append("the local-action step is not guarded to the Intel macOS row")
    problems.extend(
        f"the local-action step does not actively pass {wanted}"
        for wanted in _MACOS_ACTION_INPUTS
        if wanted not in active
    )
    return problems


def _output_check_problems(steps: list[str]) -> list[str]:
    """Return why the lane does not actively require both of the action's published outputs."""
    checks = [step for step in steps if f"steps.{_MACOS_ACTION_ID}.outputs" in step]
    if len(checks) != 1:
        return [f"expected exactly one output check, found {len(checks)}"]
    step = checks[0]
    active = _active_lines(step)
    problems: list[str] = []
    if _step_property(step, "if") != _MACOS_ACTION_GUARD:
        problems.append("the output check is not guarded to the Intel macOS row")
    if _MACOS_EXIT_CODE_CHECK not in active:
        problems.append("the output check does not actively require exit_code 0")
    if _MACOS_STATUS_CHECK not in active:
        problems.append("the output check does not actively require the certified status")
    if _step_property(step, "continue-on-error") is not None:
        problems.append("the output check tolerates its own failure")
    return problems


def _ordering_problems(steps: list[str]) -> list[str]:
    """Return why the action proof does not sit after the lane's evidence upload.

    Installing the local action rewrites this runner's environment, so it must not run before the
    wheel the lane verified or the evidence the release summary reads.
    """
    where: dict[str, int] = {}
    for index, step in enumerate(steps):
        if _MACOS_LANE_EVIDENCE in step:
            where["upload"] = index
        if _step_property(step, "id") == _MACOS_ACTION_ID:
            where["action"] = index
        if f"steps.{_MACOS_ACTION_ID}.outputs" in step:
            where["check"] = index
    if set(where) != {"upload", "action", "check"}:
        return [f"the lane is missing one of upload/action/check: {sorted(where)}"]
    if not where["upload"] < where["action"] < where["check"]:
        return [f"the action proof is not placed after the evidence upload: {where}"]
    return []


def _macos_action_problems(text: str) -> list[str]:
    """Return every reason the Intel macOS action proof is not intact in `text`."""
    steps = _steps(_job(text, "matrix_lane"))
    return _local_action_problems(steps) + _output_check_problems(steps) + _ordering_problems(steps)


def _comment_out(text: str, assertion: str) -> str:
    """Disable one assertion the way a person disables it: by commenting it out, not deleting it."""
    line = f"          {assertion}\n"
    assert text.count(line) == 1, assertion
    return text.replace(line, f"          # {assertion}\n", 1)


def _drop_macos_action_step(text: str) -> str:
    """Remove the whole `uses: ./` step, leaving the guarded output check behind."""
    start = text.index(_MACOS_ACTION_STEP)
    end = text.index("      - name: Verify the Intel macOS action outputs\n", start)
    return text[:start] + text[end:]


def _misguard_macos_action_step(text: str) -> str:
    """Point the guard at a row the matrix never produces, so the action never runs."""
    return text.replace(
        f"        if: {_MACOS_ACTION_GUARD}\n        uses: ./\n",
        "        if: matrix.id == 'no_such_lane'\n        uses: ./\n",
        1,
    )


def _bypass_local_action(text: str) -> str:
    """Replace the action with a schema-valid shell step that cannot exercise it.

    The whole `uses:`/`with:` block goes, so the result is an ordinary `run:` step rather than a
    step carrying both keys, which GitHub would reject before this boundary is ever reached.
    """
    start = text.index("        uses: ./\n")
    tail = '          python_version: "3.12"\n'
    end = text.index(tail, start) + len(tail)
    return text[:start] + "        run: echo 'certification skipped'\n" + text[end:]


def _comment_out_macos_status_check(text: str) -> str:
    """Stop requiring the action's status output."""
    return _comment_out(text, _MACOS_STATUS_CHECK)


def _comment_out_macos_exit_code_check(text: str) -> str:
    """Stop requiring the action's exit-code output."""
    return _comment_out(text, _MACOS_EXIT_CODE_CHECK)


def test_the_intel_macos_lane_runs_the_real_composite_action() -> None:
    """The action's macOS boundary is proved by running the action, not by asserting about it."""
    assert _macos_action_problems(_workflow_text()) == []


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        pytest.param("removed", _drop_macos_action_step, id="removed"),
        pytest.param("misguarded", _misguard_macos_action_step, id="misguarded"),
        pytest.param("bypassed", _bypass_local_action, id="bypassed"),
        pytest.param("status_commented", _comment_out_macos_status_check, id="status_commented"),
        pytest.param(
            "exit_code_commented", _comment_out_macos_exit_code_check, id="exit_code_commented"
        ),
    ],
)
def test_each_way_of_defeating_the_macos_action_proof_is_rejected(name: str, mutate: Any) -> None:
    """A digest update must not be able to carry a silently disabled proof with it."""
    text = _workflow_text()
    mutated = mutate(text)
    assert mutated != text, name
    assert _macos_action_problems(mutated), name


def test_only_the_action_job_and_the_intel_macos_lane_install_the_local_action() -> None:
    """`uses: ./` reinstalls Metrifid into a runner, so its blast radius stays known and small."""
    text = _workflow_text()
    owners = {name for name in _ALL_JOBS if "uses: ./" in _job(text, name)}
    assert owners == {"matrix_lane", "test_composite_action"}, owners
    # The lane the guard names is one of the declared smoke rows, so this adds no matrix entry.
    assert _MACOS_ACTION_LANE in _SMOKE_LANES
    assert _MACOS_ACTION_GUARD == f"matrix.id == '{_MACOS_ACTION_LANE}'"


@pytest.mark.parametrize("owner", _COMPLETE_SUITE_OWNERS)
def test_disabling_one_owners_provision_fails_the_contract(owner: str) -> None:
    """Each owner is proved on its own, and a commented-out install never counts as provisioning."""
    text = _workflow_text()
    job = _job(text, owner)
    provisioning = _build_provisioning_steps(text, owner)
    assert len(provisioning) == 1, owner
    disabled = provisioning[0].replace(_CONSTRAINED_INSTALL, f"# {_CONSTRAINED_INSTALL}", 1)
    assert disabled != provisioning[0]
    mutated = text.replace(job, job.replace(provisioning[0], disabled, 1), 1)
    assert _build_provisioning_steps(mutated, owner) == []
    for other in (name for name in _COMPLETE_SUITE_OWNERS if name != owner):
        assert _build_provisioning_steps(mutated, other), other


def test_every_exact_ci_pin_satisfies_its_development_extra_lower_bound() -> None:
    """A CI pin below the published floor would test something the project does not support."""
    pyproject = (_REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
    extra = pyproject[pyproject.index("dev = [") : pyproject.index("]", pyproject.index("dev = ["))]
    floors = dict(re.findall(r'"([A-Za-z0-9_.-]+)>=([0-9.]+)"', extra))
    pins = dict(
        re.findall(
            r"^([A-Za-z0-9_.-]+)==([0-9.]+)$",
            _QUALITY_CONSTRAINTS.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
    )
    checked = 0
    for name, floor in floors.items():
        if name not in pins:
            continue
        checked += 1
        assert tuple(int(p) for p in pins[name].split(".")) >= tuple(
            int(p) for p in floor.split(".")
        ), (name, pins[name], floor)
    assert checked >= 8, checked


def test_quality_constraints_do_not_freeze_a_mujoco_release() -> None:
    """The resolver-latest lanes must be free to pick up a newer supported MuJoCo."""
    assert "mujoco" not in _QUALITY_CONSTRAINTS.read_text(encoding="utf-8")


def test_latest_authority_installers_leave_mujoco_unpinned() -> None:
    """Quality and sdist lanes resolve the newest stable MuJoCo."""
    text = _workflow_text()
    for name in ("build_and_quality", "sdist_install_lane"):
        assert re.search(r"mujoco\s*==", _job(text, name), flags=re.IGNORECASE) is None, name


def test_quality_job_installs_mujoco_before_strict_mypy() -> None:
    """Strict MyPy analyses modules that import MuJoCo, so the runtime must exist first."""
    job = _job(_workflow_text(), "build_and_quality")
    assert job.index("\n            mujoco\n") < job.index("mypy --strict")


def test_strict_mypy_covers_the_decision_helper() -> None:
    """The helper decides releases; it is held to the same standard as the product."""
    job = _job(_workflow_text(), "build_and_quality")
    assert "mypy --strict src/metrifid .github/scripts/validate_ci_evidence.py" in job


def test_the_build_authority_publishes_one_original_manifest() -> None:
    """One build, one manifest, with the identity every consumer verifies against."""
    job = _job(_workflow_text(), "build_and_quality")
    assert "twine check --strict dist/*" in job
    assert '"filename": path.name' in job
    assert '"size_bytes": len(data)' in job
    assert '"sha256": hashlib.sha256(data).hexdigest()' in job
    assert "original_manifest.json" in job


# ---- The decision helper, driven directly ------------------------------------------------------------


_EVENT_SHA = "1" * 40
_HEAD_SHA = "2" * 40
# The synthetic candidate is a real wheel name carrying the real release version, because the
# summary now binds the install report to the accepted artifact by file name and by the version
# that file name encodes. A placeholder could not exercise either check.
_CANDIDATE_VERSION = "0.7.2"
_CANDIDATE_WHEEL = f"metrifid-{_CANDIDATE_VERSION}-py3-none-any.whl"
_CANDIDATE_SDIST = f"metrifid-{_CANDIDATE_VERSION}.tar.gz"
_DIGEST = "a" * 64
_SDIST_DIGEST = "b" * 64
_HELP = "\n".join(f"### metrifid {command} --help" for command in _PUBLIC_COMMANDS)
_FULL_CASES = ("tests.unit.test_a::test_one", "tests.unit.test_b::test_two")
_FOCUSED_CASES = ("tests.unit.test_a::test_one",)


@pytest.fixture(scope="module")
def validator() -> ModuleType:
    """The decision helper, imported from the exact path CI runs."""
    return _load_validator()


def _write(path: Path, payload: object) -> None:
    """Write one synthetic evidence document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        path.write_text(payload, encoding="utf-8")
        return
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _junit(
    cases: tuple[str, ...], *, failing: str | None = None, erroring: str | None = None
) -> str:
    """Render a minimal JUnit document with optional failure or error."""
    parts = ['<?xml version="1.0" encoding="utf-8"?>', '<testsuite name="synthetic">']
    for case in cases:
        classname, _, name = case.partition("::")
        if case == failing:
            parts.append(f'<testcase classname="{classname}" name="{name}"><failure/></testcase>')
        elif case == erroring:
            parts.append(f'<testcase classname="{classname}" name="{name}"><error/></testcase>')
        else:
            parts.append(f'<testcase classname="{classname}" name="{name}"/>')
    parts.append("</testsuite>")
    return "\n".join(parts)


def _native_integer(version: str) -> int:
    """Encode a MuJoCo triplet the way the runtime does.

    Computed here from the published formula rather than asked of the helper under test, so a
    mistake in the helper cannot make its own fixture agree with it.
    """
    major, minor, patch = (int(part) for part in version.split("."))
    return major * 1_000_000 + minor * 1_000 + patch


def _resolver_version(want: Any) -> str:
    """The MuJoCo release a resolver lane would land on, derived from the lane's own bound.

    A lane with a packaging ceiling must synthesize a release below it; deriving the value from
    the ceiling keeps the fixture correct if the ceiling ever moves, where a literal would not.
    """
    ceiling = want.mujoco_wheel_ceiling
    if ceiling is None:
        return "3.11.0"
    return f"{ceiling[0]}.{ceiling[1] - 1}.0"


def _install_report(want: Any, *, mujoco_version: str | None = None) -> dict[str, Any]:
    """Build a pip install report that matches the lane it claims to describe."""
    version = mujoco_version or _resolver_version(want)
    return {
        "version": "1",
        "pip_version": "26.2.1",
        "environment": {
            "platform_system": want.platform_system,
            "platform_machine": want.platform_machine,
            "python_version": want.python_major_minor,
        },
        "install": [
            {
                "is_direct": True,
                "requested": True,
                "download_info": {
                    "url": f"file:///release-artifacts/dist/{_CANDIDATE_WHEEL}",
                    "archive_info": {"hashes": {"sha256": _DIGEST}},
                },
                "metadata": {"name": "metrifid", "version": _CANDIDATE_VERSION},
            },
            {
                "is_direct": False,
                "requested": False,
                "download_info": {
                    "url": (
                        f"https://files.pythonhosted.org/packages/mujoco-{version}"
                        f"-cp312-cp312-macosx_10_16_x86_64.whl"
                    )
                },
                "metadata": {"name": "mujoco", "version": version},
            },
            {
                "is_direct": False,
                "requested": False,
                "download_info": {"url": "https://files.pythonhosted.org/packages/numpy-2.3.5.whl"},
                "metadata": {"name": "numpy", "version": "2.3.5"},
            },
        ],
    }


def _identity(want: Any) -> dict[str, Any]:
    """Build a runtime identity document that actually matches its lane's expectation."""
    mujoco_version = want.mujoco_version or _resolver_version(want)
    return {
        "package_version": mujoco_version,
        "package_base_version": mujoco_version,
        "native_version_string": mujoco_version,
        "native_version_integer": _native_integer(mujoco_version),
        "numpy_version": want.numpy_version or "2.3.5",
        "platform_machine": want.platform_machine,
        "platform_system": want.platform_system,
        "python_major_minor": want.python_major_minor,
        "python_version": f"{want.python_major_minor}.0 (synthetic)",
        "support_tier": "ADMITTED_CAPABILITY_COMPATIBLE_PROFILE",
    }


def test_the_synthetic_native_encoding_matches_the_installed_runtime() -> None:
    """Check the fixture's encoding against the real library, not against the helper.

    The runtime packs each component into three decimal digits, so 3.11.0 encodes as 3011000.
    Evidence that encodes it any other way is not lane-correct.
    """
    import mujoco

    assert _native_integer(mujoco.mj_versionString()) == mujoco.mj_version()
    assert _native_integer("3.11.0") == 3011000


def _green_upstream() -> dict[str, str]:
    """Every required job succeeded."""
    return dict.fromkeys(_load_validator().REQUIRED_UPSTREAM, "success")


def _build_green_tree(root: Path) -> None:
    """Write a complete, self-consistent, lane-correct evidence tree."""
    validator = _load_validator()
    original = root / "original-artifacts"
    _write(
        original / "original_manifest.json",
        {
            "wheel": {"filename": _CANDIDATE_WHEEL, "size_bytes": 10, "sha256": _DIGEST},
            "sdist": {"filename": _CANDIDATE_SDIST, "size_bytes": 10, "sha256": _SDIST_DIGEST},
        },
    )
    _write(
        original / "quality-evidence" / "runtime_identity.json",
        _identity(validator.BUILD_EXPECTATION),
    )
    _write(
        original / "quality-evidence" / "checkout_identity.json",
        {
            "schema_version": 1,
            "event_name": "push",
            "event_sha": _EVENT_SHA,
            "checkout_sha": _EVENT_SHA,
            "checkout_tree": "3" * 40,
            "pull_request_head_sha": None,
            "pull_request_head_tree": None,
        },
    )
    for lane, want in validator.LANE_EXPECTATIONS.items():
        directory = root / f"lane-{lane}"
        _write(directory / "lane_id.txt", lane + "\n")
        _write(directory / "tier.txt", want.tier + "\n")
        _write(directory / "installed_artifact.json", {"kind": "wheel", "sha256": _DIGEST})
        _write(directory / "runtime_identity.json", _identity(want))
        if want.mujoco_wheel_ceiling is not None:
            _write(directory / "install_report.json", _install_report(want))
        _write(directory / "cli_help.log", _HELP)
        if want.tier == "full":
            _write(directory / "full.xml", _junit(_FULL_CASES))
        else:
            _write(directory / "focused.xml", _junit(_FOCUSED_CASES))
    for role, want in validator.RETAINED_EXPECTATIONS.items():
        directory = root / f"retained-{role}"
        _write(directory / "installed_artifact.json", {"kind": "wheel", "sha256": _DIGEST})
        _write(directory / "runtime_identity.json", _identity(want))
        _write(directory / "cli_help.log", _HELP)
        _write(directory / "focused.xml", _junit(_FOCUSED_CASES))
        exact_tuple = {"package_version": want.mujoco_version}
        encoded = json.dumps(
            exact_tuple, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        _write(
            directory / "compatibility" / "compatibility_validation.json",
            {
                "passed": True,
                "profile_role": role,
                "expected_mujoco_package_version": want.mujoco_version,
                "retained_exact_profile_validation": {
                    "validation_tier": "VALIDATED_EXACT_PROFILE",
                    "exact_profile_tuple": exact_tuple,
                    "exact_profile_tuple_sha256": hashlib.sha256(encoded).hexdigest(),
                },
            },
        )
    directory = root / "sdist-install-evidence"
    _write(directory / "installed_artifact.json", {"kind": "sdist", "sha256": _SDIST_DIGEST})
    _write(directory / "runtime_identity.json", _identity(validator.SDIST_EXPECTATION))
    _write(directory / "cli_help.log", _HELP)
    _write(directory / "focused.xml", _junit(_FOCUSED_CASES))


@pytest.fixture
def green_tree(tmp_path: Path) -> Path:
    """A complete lane-correct evidence tree the helper must accept."""
    root = tmp_path / "evidence"
    _build_green_tree(root)
    return root


def _validate(validator: ModuleType, root: Path, **overrides: Any) -> dict[str, Any]:
    """Run the helper with this file's default trusted inputs."""
    arguments: dict[str, Any] = {
        "upstream": _green_upstream(),
        "event_name": "push",
        "event_sha": _EVENT_SHA,
        "pull_request_head_sha": "",
    }
    arguments.update(overrides)
    result = validator.validate(
        root,
        arguments["upstream"],
        arguments["event_name"],
        arguments["event_sha"],
        arguments["pull_request_head_sha"],
    )
    assert isinstance(result, dict)
    return result


def test_the_helper_accepts_a_complete_green_evidence_tree(
    validator: ModuleType, green_tree: Path
) -> None:
    """A helper that has never accepted anything cannot be trusted to reject correctly."""
    summary = _validate(validator, green_tree)
    assert summary["original_wheel_sha256"] == _DIGEST
    assert summary["full_lanes"] == list(_FULL_LANES)
    assert sorted(summary["smoke_lanes"]) == sorted(_SMOKE_LANES)
    assert summary["complete_suite_test_count"] == len(_FULL_CASES)
    assert summary["accepted_subject"]["checkout_sha"] == _EVENT_SHA


def test_the_helper_ignores_optional_diagnostic_artifacts(
    validator: ModuleType, green_tree: Path
) -> None:
    """Diagnostics are never required, and never allowed to satisfy a required lane."""
    (green_tree / "diagnostic-macos_x64_py311").mkdir()
    summary = _validate(validator, green_tree)
    assert summary["ignored_diagnostic_artifacts"] == ["diagnostic-macos_x64_py311"]


def test_the_helper_rejects_a_missing_lane(validator: ModuleType, green_tree: Path) -> None:
    """A lane that never uploaded evidence has not proved its boundary."""
    shutil.rmtree(green_tree / "lane-macos_arm64_py314")
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


@pytest.mark.parametrize("result", ["skipped", "failure", "cancelled"])
def test_the_helper_rejects_a_required_job_that_did_not_succeed(
    validator: ModuleType, green_tree: Path, result: str
) -> None:
    """A conditionally skipped job reports success externally, so the result is judged here."""
    upstream = _green_upstream()
    upstream["minimum_dependency_lane"] = result
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree, upstream=upstream)


def test_the_helper_rejects_a_lane_that_installed_another_wheel(
    validator: ModuleType, green_tree: Path
) -> None:
    """Installing a parity rebuild instead of the original is exactly what this catches."""
    _write(
        green_tree / "lane-linux_x64_py313" / "installed_artifact.json",
        {"kind": "wheel", "sha256": "c" * 64},
    )
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


@pytest.mark.parametrize("kind", ["failing", "erroring"])
def test_the_helper_rejects_a_failed_or_errored_junit_case(
    validator: ModuleType, green_tree: Path, kind: str
) -> None:
    """A green job result cannot outvote a red test in the report it uploaded."""
    _write(
        green_tree / "lane-linux_x64_py312_full" / "full.xml",
        _junit(_FULL_CASES, **{kind: _FULL_CASES[0]}),
    )
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


def test_the_helper_rejects_a_smoke_lane_mislabelled_as_full(
    validator: ModuleType, green_tree: Path
) -> None:
    """A smoke lane claiming a complete suite is claiming a boundary it did not pay for."""
    _write(green_tree / "lane-linux_x64_py314" / "tier.txt", "full\n")
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


def test_the_helper_rejects_a_smoke_lane_that_uploaded_a_complete_report(
    validator: ModuleType, green_tree: Path
) -> None:
    """The tier and the uploaded report must agree in both directions."""
    _write(green_tree / "lane-linux_x64_py314" / "full.xml", _junit(_FULL_CASES))
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


def test_the_helper_rejects_a_missing_required_identity_field(
    validator: ModuleType, green_tree: Path
) -> None:
    """A lane that never recorded its runtime cannot show which boundary it proved."""
    identity = _identity(validator.LANE_EXPECTATIONS["macos_x64_py312"])
    del identity["platform_machine"]
    _write(green_tree / "lane-macos_x64_py312" / "runtime_identity.json", identity)
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


@pytest.mark.parametrize(
    ("lane", "field_name", "wrong"),
    [
        ("macos_arm64_py314", "platform_system", "Linux"),
        ("macos_arm64_py314", "platform_machine", "x86_64"),
        ("linux_x64_py313", "python_major_minor", "3.12"),
        ("linux_x64_py311_numpy_min", "numpy_version", "2.3.5"),
        ("linux_x64_py311_numpy_min", "package_version", "3.11.0"),
        ("macos_x64_py312", "package_version", "3.11.0"),
    ],
)
def test_the_helper_rejects_an_identity_that_is_not_its_declared_lane(
    validator: ModuleType, green_tree: Path, lane: str, field_name: str, wrong: str
) -> None:
    """A well-formed document describing another runtime does not prove this lane's boundary."""
    identity = _identity(validator.LANE_EXPECTATIONS[lane])
    identity[field_name] = wrong
    _write(green_tree / f"lane-{lane}" / "runtime_identity.json", identity)
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


def test_the_helper_accepts_the_intel_lane_install_report(
    validator: ModuleType, green_tree: Path
) -> None:
    """The green tree already ships a well-formed report; it must be accepted as evidence."""
    _validate(validator, green_tree)
    report = json.loads(
        (green_tree / "lane-macos_x64_py312" / "install_report.json").read_text(encoding="utf-8")
    )
    mujoco = next(e for e in report["install"] if e["metadata"]["name"] == "mujoco")
    assert mujoco["download_info"]["url"].endswith(".whl")
    assert mujoco["metadata"]["version"] == "3.10.0"


def test_the_helper_rejects_a_missing_intel_install_report(
    validator: ModuleType, green_tree: Path
) -> None:
    """Evidence that was never produced cannot stand in for the resolve it claims to describe."""
    (green_tree / "lane-macos_x64_py312" / "install_report.json").unlink()
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        ({"version": "2"}, "an unknown report schema version"),
        ({"version": None}, "a missing report schema version"),
        ({"environment": {}}, "an environment with no platform association"),
        ({"install": []}, "an empty install list"),
        ({"install": None}, "a missing install list"),
    ],
    ids=["schema-version", "no-schema-version", "empty-environment", "no-installs", "no-list"],
)
def test_the_helper_rejects_malformed_intel_install_evidence(
    validator: ModuleType, green_tree: Path, mutation: dict[str, Any], reason: str
) -> None:
    """A report that is present but not well formed proves nothing about the resolve."""
    want = validator.LANE_EXPECTATIONS["macos_x64_py312"]
    report = _install_report(want)
    report.update(mutation)
    _write(green_tree / "lane-macos_x64_py312" / "install_report.json", report)
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


def test_the_helper_rejects_a_mujoco_source_archive_in_the_intel_report(
    validator: ModuleType, green_tree: Path
) -> None:
    """A source archive is the exact defect the platform bound exists to prevent."""
    want = validator.LANE_EXPECTATIONS["macos_x64_py312"]
    report = _install_report(want)
    entry = next(e for e in report["install"] if e["metadata"]["name"] == "mujoco")
    version = entry["metadata"]["version"]
    entry["download_info"]["url"] = (
        f"https://files.pythonhosted.org/packages/mujoco-{version}.tar.gz"
    )
    _write(green_tree / "lane-macos_x64_py312" / "install_report.json", report)
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


@pytest.mark.parametrize("version", ["3.11.0", "3.12.0", "4.0.0"])
def test_the_helper_rejects_a_mujoco_at_or_above_the_intel_ceiling(
    validator: ModuleType, green_tree: Path, version: str
) -> None:
    """Upstream publishes no Intel macOS wheel at or above the ceiling."""
    want = validator.LANE_EXPECTATIONS["macos_x64_py312"]
    _write(
        green_tree / "lane-macos_x64_py312" / "install_report.json",
        _install_report(want, mujoco_version=version),
    )
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


def test_the_helper_rejects_a_mujoco_below_the_project_floor_in_the_intel_report(
    validator: ModuleType, green_tree: Path
) -> None:
    """The ceiling does not license dropping under the declared floor."""
    want = validator.LANE_EXPECTATIONS["macos_x64_py312"]
    _write(
        green_tree / "lane-macos_x64_py312" / "install_report.json",
        _install_report(want, mujoco_version="3.8.0"),
    )
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


@pytest.mark.parametrize(
    ("field_name", "wrong"),
    [("platform_system", "Linux"), ("platform_machine", "arm64")],
)
def test_the_helper_rejects_an_intel_report_from_another_platform(
    validator: ModuleType, green_tree: Path, field_name: str, wrong: str
) -> None:
    """A resolve performed somewhere else does not prove the Intel macOS boundary."""
    want = validator.LANE_EXPECTATIONS["macos_x64_py312"]
    report = _install_report(want)
    report["environment"][field_name] = wrong
    _write(green_tree / "lane-macos_x64_py312" / "install_report.json", report)
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


def test_the_helper_rejects_an_intel_report_that_skipped_dependency_resolution(
    validator: ModuleType, green_tree: Path
) -> None:
    """A --no-deps install records only the requested wheel and resolves nothing."""
    want = validator.LANE_EXPECTATIONS["macos_x64_py312"]
    report = _install_report(want)
    report["install"] = [e for e in report["install"] if e["metadata"]["name"] == "metrifid"]
    _write(green_tree / "lane-macos_x64_py312" / "install_report.json", report)
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


@pytest.mark.parametrize("field_name", ["is_direct", "requested"])
def test_the_helper_rejects_an_intel_report_whose_candidate_was_not_directly_installed(
    validator: ModuleType, green_tree: Path, field_name: str
) -> None:
    """The evidence must come from installing the candidate wheel itself."""
    want = validator.LANE_EXPECTATIONS["macos_x64_py312"]
    report = _install_report(want)
    entry = next(e for e in report["install"] if e["metadata"]["name"] == "metrifid")
    entry[field_name] = False
    _write(green_tree / "lane-macos_x64_py312" / "install_report.json", report)
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


def test_the_helper_rejects_an_intel_report_whose_candidate_was_built_from_source(
    validator: ModuleType, green_tree: Path
) -> None:
    """An artifact that is not the ordinary wheel installation is not this lane's evidence."""
    want = validator.LANE_EXPECTATIONS["macos_x64_py312"]
    report = _install_report(want)
    entry = next(e for e in report["install"] if e["metadata"]["name"] == "metrifid")
    entry["download_info"]["url"] = f"file:///release-artifacts/dist/{_CANDIDATE_SDIST}"
    _write(green_tree / "lane-macos_x64_py312" / "install_report.json", report)
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


def test_the_helper_rejects_a_within_range_report_runtime_mismatch(
    validator: ModuleType, green_tree: Path
) -> None:
    """Two MuJoCo releases can each satisfy the bound and still not be one installation.

    A report naming 3.9.0 while the lane imported 3.10.0 passes every range check yet describes a
    different installation from the one the lane measured. Range agreement is not identity.
    """
    want = validator.LANE_EXPECTATIONS["macos_x64_py312"]
    identity = json.loads(
        (green_tree / "lane-macos_x64_py312" / "runtime_identity.json").read_text(encoding="utf-8")
    )
    assert identity["package_version"] == "3.10.0"
    _write(
        green_tree / "lane-macos_x64_py312" / "install_report.json",
        _install_report(want, mujoco_version="3.9.0"),
    )
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


def test_the_helper_accepts_a_report_that_names_the_imported_runtime(
    validator: ModuleType, green_tree: Path
) -> None:
    """The control for the test above: agreement on the same in-range release is accepted."""
    want = validator.LANE_EXPECTATIONS["macos_x64_py312"]
    identity = _identity(want)
    identity["package_version"] = "3.9.0"
    identity["package_base_version"] = "3.9.0"
    identity["native_version_string"] = "3.9.0"
    identity["native_version_integer"] = _native_integer("3.9.0")
    _write(green_tree / "lane-macos_x64_py312" / "runtime_identity.json", identity)
    _write(
        green_tree / "lane-macos_x64_py312" / "install_report.json",
        _install_report(want, mujoco_version="3.9.0"),
    )
    assert _validate(validator, green_tree)["original_wheel_sha256"] == _DIGEST


def _candidate_entry(report: dict[str, Any]) -> dict[str, Any]:
    """The install-report entry for the directly requested candidate wheel."""
    entry = next(e for e in report["install"] if e["metadata"]["name"] == "metrifid")
    assert isinstance(entry, dict)
    return entry


def test_the_helper_rejects_a_different_candidate_wheel_filename(
    validator: ModuleType, green_tree: Path
) -> None:
    """The report must name the accepted candidate, not some other wheel."""
    want = validator.LANE_EXPECTATIONS["macos_x64_py312"]
    report = _install_report(want)
    _candidate_entry(report)["download_info"]["url"] = (
        "https://example.invalid/packages/metrifid-0.7.2-py3-none-manylinux1_x86_64.whl"
    )
    _write(green_tree / "lane-macos_x64_py312" / "install_report.json", report)
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


def test_the_helper_rejects_a_percent_encoded_lookalike_candidate(
    validator: ModuleType, green_tree: Path
) -> None:
    """The basename is parsed and decoded, so an encoded path cannot forge the accepted name."""
    want = validator.LANE_EXPECTATIONS["macos_x64_py312"]
    report = _install_report(want)
    _candidate_entry(report)["download_info"]["url"] = (
        f"https://example.invalid/{_CANDIDATE_WHEEL}/other-1.0-py3-none-any.whl"
    )
    _write(green_tree / "lane-macos_x64_py312" / "install_report.json", report)
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


@pytest.mark.parametrize("version", ["9.9.9", "0.7.0", "0.7.2.dev0"])
def test_the_helper_rejects_a_candidate_metadata_version_the_filename_does_not_encode(
    validator: ModuleType, green_tree: Path, version: str
) -> None:
    """The reported version must be the one the accepted wheel filename encodes."""
    want = validator.LANE_EXPECTATIONS["macos_x64_py312"]
    report = _install_report(want)
    _candidate_entry(report)["metadata"]["version"] = version
    _write(green_tree / "lane-macos_x64_py312" / "install_report.json", report)
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (lambda entry: entry["download_info"].pop("archive_info"), "no archive_info at all"),
        (lambda entry: entry["download_info"].__setitem__("archive_info", {}), "no hashes"),
        (
            lambda entry: entry["download_info"]["archive_info"].__setitem__("hashes", {}),
            "no sha256",
        ),
        (
            lambda entry: entry["download_info"]["archive_info"]["hashes"].__setitem__(
                "sha256", "c" * 64
            ),
            "an unrelated sha256",
        ),
    ],
    ids=["missing-archive", "missing-hashes", "missing-sha256", "wrong-sha256"],
)
def test_the_helper_rejects_a_candidate_archive_digest_that_is_not_the_accepted_one(
    validator: ModuleType, green_tree: Path, mutate: Callable[[dict[str, Any]], object], reason: str
) -> None:
    """The candidate archive digest must equal the accepted original wheel digest."""
    want = validator.LANE_EXPECTATIONS["macos_x64_py312"]
    report = _install_report(want)
    mutate(_candidate_entry(report))
    _write(green_tree / "lane-macos_x64_py312" / "install_report.json", report)
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


def test_the_helper_rejects_a_second_directly_requested_distribution(
    validator: ModuleType, green_tree: Path
) -> None:
    """Exactly one entry may be the directly requested candidate wheel."""
    want = validator.LANE_EXPECTATIONS["macos_x64_py312"]
    report = _install_report(want)
    extra = next(e for e in report["install"] if e["metadata"]["name"] == "numpy")
    extra["is_direct"] = True
    extra["requested"] = True
    _write(green_tree / "lane-macos_x64_py312" / "install_report.json", report)
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


def test_the_helper_rejects_a_build_identity_from_the_wrong_interpreter(
    validator: ModuleType, green_tree: Path
) -> None:
    """The build authority is pinned to Python 3.11 on Linux x86_64."""
    identity = _identity(validator.BUILD_EXPECTATION)
    identity["python_major_minor"] = "3.12"
    _write(
        green_tree / "original-artifacts" / "quality-evidence" / "runtime_identity.json", identity
    )
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


def test_the_helper_accepts_a_well_formed_pull_request_subject(
    validator: ModuleType, green_tree: Path
) -> None:
    """On a pull request both the merge checkout and the fetched head must be recorded."""
    _write(
        green_tree / "original-artifacts" / "quality-evidence" / "checkout_identity.json",
        {
            "schema_version": 1,
            "event_name": "pull_request",
            "event_sha": _EVENT_SHA,
            "checkout_sha": _EVENT_SHA,
            "checkout_tree": "3" * 40,
            "pull_request_head_sha": _HEAD_SHA,
            "pull_request_head_tree": "4" * 40,
        },
    )
    summary = _validate(
        validator, green_tree, event_name="pull_request", pull_request_head_sha=_HEAD_SHA
    )
    assert summary["accepted_subject"]["pull_request_head_sha"] == _HEAD_SHA
    assert (
        summary["accepted_subject"]["checkout_tree"]
        != summary["accepted_subject"]["pull_request_head_tree"]
    )


@pytest.mark.parametrize(
    ("mutation", "event", "head"),
    [
        pytest.param("drop_key", "push", "", id="drop_key"),
        pytest.param("extra_key", "push", "", id="extra_key"),
        pytest.param("bad_schema", "push", "", id="bad_schema"),
        pytest.param("malformed_sha", "push", "", id="malformed_sha"),
        pytest.param("checkout_is_not_event", "push", "", id="checkout_is_not_event"),
        pytest.param("event_mismatch", "push", "", id="event_mismatch"),
        pytest.param("null_head_on_pull_request", "pull_request", _HEAD_SHA, id="null_pr_head"),
        pytest.param(
            "head_does_not_match_trusted_input", "pull_request", _HEAD_SHA, id="pr_head_mismatch"
        ),
        pytest.param("head_present_on_push", "push", "", id="pr_head_on_push"),
    ],
)
def test_the_helper_rejects_a_bad_review_subject(
    validator: ModuleType, green_tree: Path, mutation: str, event: str, head: str
) -> None:
    """A receipt that cannot bind this run's subject is not evidence about this run."""
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "event_name": event,
        "event_sha": _EVENT_SHA,
        "checkout_sha": _EVENT_SHA,
        "checkout_tree": "3" * 40,
        "pull_request_head_sha": _HEAD_SHA if event == "pull_request" else None,
        "pull_request_head_tree": "4" * 40 if event == "pull_request" else None,
    }
    if mutation == "drop_key":
        del receipt["checkout_tree"]
    elif mutation == "extra_key":
        receipt["reviewer_note"] = "extra"
    elif mutation == "bad_schema":
        receipt["schema_version"] = 2
    elif mutation == "malformed_sha":
        receipt["checkout_sha"] = "not-an-object-id"
        receipt["event_sha"] = "not-an-object-id"
    elif mutation == "checkout_is_not_event":
        receipt["checkout_sha"] = "5" * 40
    elif mutation == "event_mismatch":
        receipt["event_name"] = "workflow_dispatch"
    elif mutation == "null_head_on_pull_request":
        receipt["pull_request_head_sha"] = None
        receipt["pull_request_head_tree"] = None
    elif mutation == "head_does_not_match_trusted_input":
        receipt["pull_request_head_sha"] = "6" * 40
    elif mutation == "head_present_on_push":
        receipt["pull_request_head_sha"] = _HEAD_SHA
        receipt["pull_request_head_tree"] = "4" * 40
    _write(
        green_tree / "original-artifacts" / "quality-evidence" / "checkout_identity.json", receipt
    )
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree, event_name=event, pull_request_head_sha=head)


def test_the_helper_compares_complete_inventory_only_between_the_complete_lanes(
    validator: ModuleType, green_tree: Path
) -> None:
    """Two complete suites must agree on which tests exist; a smoke lane is not compared."""
    _write(
        green_tree / "lane-linux_x64_py311_numpy_min" / "full.xml",
        _junit((*_FULL_CASES, "tests.unit.test_c::test_three")),
    )
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


def test_the_helper_exits_nonzero_when_it_rejects(validator: ModuleType, green_tree: Path) -> None:
    """The workflow depends on the process exit code, not on the message."""
    shutil.rmtree(green_tree / "lane-linux_x64_py313")
    assert validator.main(_argv(green_tree)) == 1


def test_the_helper_exits_zero_on_green_evidence(validator: ModuleType, green_tree: Path) -> None:
    """The accepting path must also be exercised through the real entry point."""
    assert validator.main(_argv(green_tree)) == 0


def _argv(root: Path) -> list[str]:
    """The exact command line the summary job builds."""
    return [
        "--evidence-root",
        str(root),
        "--upstream-results",
        json.dumps(_green_upstream()),
        "--event-name",
        "push",
        "--event-sha",
        _EVENT_SHA,
        "--pull-request-head-sha",
        "",
    ]


# ---- Mandatory red probes for incoherent runtime and receipt identity ----------------------------


@pytest.mark.parametrize(
    ("field_name", "wrong"),
    [
        pytest.param("package_version", "banana", id="mujoco_malformed"),
        pytest.param("package_version", "9.9.9", id="mujoco_incoherent_with_base"),
        pytest.param("package_version", "3.8.0", id="mujoco_below_floor"),
        pytest.param("numpy_version", "banana", id="numpy_malformed"),
        pytest.param("python_version", "9.9.0", id="python_banner_mismatch"),
        pytest.param("native_version_integer", 0, id="native_integer_zero"),
        pytest.param("native_version_integer", 999999, id="native_integer_wrong"),
        pytest.param("native_version_integer", 3110, id="native_integer_old_bad_encoding"),
        pytest.param("native_version_integer", True, id="native_integer_boolean"),
        pytest.param("support_tier", "SOMETHING_ELSE", id="wrong_support_tier"),
    ],
)
def test_the_helper_rejects_incoherent_runtime_identity(
    validator: ModuleType, green_tree: Path, field_name: str, wrong: Any
) -> None:
    """A field that is malformed, below floor, or contradicts its siblings is not evidence."""
    lane = "linux_x64_py313"
    identity = _identity(validator.LANE_EXPECTATIONS[lane])
    identity[field_name] = wrong
    _write(green_tree / f"lane-{lane}" / "runtime_identity.json", identity)
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


@pytest.mark.parametrize(
    "wrong",
    [
        pytest.param(True, id="boolean_true"),
        pytest.param(1.0, id="float_one"),
        pytest.param("1", id="string_one"),
        pytest.param(None, id="null"),
        pytest.param(2, id="unsupported_version"),
    ],
)
def test_the_helper_requires_an_exact_integer_schema_version(
    validator: ModuleType, green_tree: Path, wrong: Any
) -> None:
    """`True == 1` in Python, so a lookalike schema version must fail on its type."""
    receipt = json.loads(
        (
            green_tree / "original-artifacts" / "quality-evidence" / "checkout_identity.json"
        ).read_text()
    )
    receipt["schema_version"] = wrong
    _write(
        green_tree / "original-artifacts" / "quality-evidence" / "checkout_identity.json", receipt
    )
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


# ---- Field-specific runtime grammars ---------------------------------------------------------------

_MUJOCO_PACKAGE = "is not a canonical stable MuJoCo package version"
_MUJOCO_EXACT = "is not an exact MAJOR.MINOR.PATCH MuJoCo version"
_NUMPY_RELEASE = "is not a canonical stable NumPy release"
_PYTHON_BANNER = "is not a Python version banner with a canonical triplet"


@pytest.mark.parametrize(
    ("field_name", "wrong", "refusal"),
    [
        pytest.param("package_version", "03.11.0", _MUJOCO_PACKAGE, id="mujoco_leading_zero"),
        pytest.param(
            "package_version",
            "3.1000.0",
            "has a component of 1000, which the native encoding cannot represent",
            id="mujoco_component_too_large",
        ),
        pytest.param(
            "package_version",
            "3.11.0+foo..bar",
            _MUJOCO_PACKAGE,
            id="mujoco_empty_local_identifier",
        ),
        pytest.param(
            "package_version", "3.11.0+", _MUJOCO_PACKAGE, id="mujoco_trailing_local_separator"
        ),
        pytest.param("package_version", "3.11.0rc1", _MUJOCO_PACKAGE, id="mujoco_prerelease"),
        pytest.param(
            "package_base_version", "3.11.0.post1", _MUJOCO_EXACT, id="suffixed_base_string"
        ),
        pytest.param(
            "native_version_string", "3.11.0+local", _MUJOCO_EXACT, id="suffixed_native_string"
        ),
        pytest.param("numpy_version", "01.026.004", _NUMPY_RELEASE, id="numpy_leading_zeros"),
        pytest.param(
            "numpy_version", "1.26.4+foo..bar", _NUMPY_RELEASE, id="numpy_malformed_local"
        ),
        pytest.param("numpy_version", "1.27.0rc1", _NUMPY_RELEASE, id="numpy_prerelease"),
        pytest.param(
            "numpy_version",
            "1.25.0",
            "resolver-latest NumPy 1.25.0 is below the declared floor 1.26",
            id="numpy_below_floor",
        ),
        pytest.param("python_version", "3.13.13banana", _PYTHON_BANNER, id="python_trailing_text"),
        pytest.param("python_version", "03.13.13", _PYTHON_BANNER, id="python_leading_zero"),
        pytest.param(
            "python_version",
            None,
            "python_version must be a string, got NoneType",
            id="python_null",
        ),
        pytest.param("python_version", "", "python_version must not be empty", id="python_empty"),
        pytest.param(
            "python_version",
            "3.12.13 (main)",
            "python_version reports 3.12 but python_major_minor says 3.13",
            id="python_lane_mismatch",
        ),
        # `$` also matches before a terminal newline, so an exact field needs a full-string match.
        pytest.param("package_version", "3.11.0\n", _MUJOCO_PACKAGE, id="mujoco_trailing_newline"),
        pytest.param(
            "package_version", "3.11.0\r", _MUJOCO_PACKAGE, id="mujoco_trailing_carriage_return"
        ),
        pytest.param("package_version", "3.11.0 ", _MUJOCO_PACKAGE, id="mujoco_trailing_space"),
        pytest.param("numpy_version", "2.3.5\n", _NUMPY_RELEASE, id="numpy_trailing_newline"),
        pytest.param(
            "numpy_version", "2.3.5\r", _NUMPY_RELEASE, id="numpy_trailing_carriage_return"
        ),
        pytest.param("numpy_version", "2.3.5 ", _NUMPY_RELEASE, id="numpy_trailing_space"),
        # Only one ASCII space introduces a real `sys.version` continuation.
        pytest.param(
            "python_version", "3.13.13!garbage", _PYTHON_BANNER, id="python_punctuation_bang"
        ),
        pytest.param(
            "python_version", "3.13.13_garbage", _PYTHON_BANNER, id="python_punctuation_underscore"
        ),
        pytest.param(
            "python_version", "3.13.13/garbage", _PYTHON_BANNER, id="python_punctuation_slash"
        ),
        pytest.param(
            "python_version", "3.13.13+garbage", _PYTHON_BANNER, id="python_punctuation_plus"
        ),
        pytest.param("python_version", "3.13.13.1", _PYTHON_BANNER, id="python_fourth_component"),
        pytest.param("python_version", "3.13.13\n", _PYTHON_BANNER, id="python_trailing_newline"),
        pytest.param(
            "python_version", "3.13.13\tgarbage", _PYTHON_BANNER, id="python_tab_separator"
        ),
        # A continuation is opened by one ASCII space and is nonempty; its first character is
        # neither Unicode whitespace nor an ASCII control or DEL.
        pytest.param(
            "python_version", "3.13.13 ", _PYTHON_BANNER, id="python_space_without_continuation"
        ),
        pytest.param("python_version", "3.13.13 \x00", _PYTHON_BANNER, id="python_nul_after_space"),
        pytest.param("python_version", "3.13.13 \t", _PYTHON_BANNER, id="python_tab_after_space"),
        pytest.param(
            "python_version", "3.13.13 \n", _PYTHON_BANNER, id="python_newline_after_space"
        ),
        pytest.param(
            "python_version",
            "3.13.13 \rgarbage",
            _PYTHON_BANNER,
            id="python_carriage_return_after_space",
        ),
        pytest.param(
            "python_version",
            "3.13.13 \u0085garbage",
            _PYTHON_BANNER,
            id="python_next_line_after_space",
        ),
        pytest.param(
            "python_version",
            "3.13.13 \u00a0garbage",
            _PYTHON_BANNER,
            id="python_no_break_space_after_space",
        ),
        pytest.param(
            "python_version",
            "3.13.13 \u2003garbage",
            _PYTHON_BANNER,
            id="python_em_space_after_space",
        ),
        pytest.param(
            "python_version",
            "3.13.13 \u2028garbage",
            _PYTHON_BANNER,
            id="python_line_separator_after_space",
        ),
    ],
)
def test_the_helper_rejects_a_noncanonical_runtime_field(
    validator: ModuleType, green_tree: Path, field_name: str, wrong: Any, refusal: str
) -> None:
    """Each field has its own admitted syntax; a value outside it is not an admitted runtime.

    The expected refusal is written out per case, so a value cannot pass this test by being refused
    for some other reason.
    """
    lane = "linux_x64_py313"
    identity = _identity(validator.LANE_EXPECTATIONS[lane])
    identity[field_name] = wrong
    _write(green_tree / f"lane-{lane}" / "runtime_identity.json", identity)
    with pytest.raises(validator.EvidenceError) as raised:
        _validate(validator, green_tree)
    assert refusal in str(raised.value)


@pytest.mark.parametrize(
    "package_version",
    [
        pytest.param("3.11.0+vendor-1", id="hyphen_local"),
        pytest.param("3.11.0+vendor_1", id="underscore_local"),
        pytest.param("3.11.0.post1", id="post_release"),
    ],
)
def test_the_helper_admits_a_product_admitted_package_suffix(
    validator: ModuleType, green_tree: Path, package_version: str
) -> None:
    """A local or post suffix is legitimate on the package field while base and native stay exact."""
    lane = "linux_x64_py313"
    identity = _identity(validator.LANE_EXPECTATIONS[lane])
    identity["package_version"] = package_version
    identity["package_base_version"] = "3.11.0"
    identity["native_version_string"] = "3.11.0"
    identity["native_version_integer"] = 3011000
    _write(green_tree / f"lane-{lane}" / "runtime_identity.json", identity)
    assert _validate(validator, green_tree)["original_wheel_sha256"] == _DIGEST


def test_the_helper_accepts_a_real_python_version_banner(
    validator: ModuleType, green_tree: Path
) -> None:
    """The real `sys.version` shape must pass, or the grammar is simply too strict.

    A continuation is free-form after its first character, so it may run on across lines. Its
    first character only has to be neither Unicode whitespace nor an ASCII control or DEL, so a
    non-ASCII interpreter banner opens one just as well as an ASCII one.
    """
    lane = "linux_x64_py313"
    for banner in (
        "3.13.13 (main, Jun  3 2026, 09:12:44) \n[Clang 17.0.0 ]",
        "3.13.13 éclair",
    ):
        identity = _identity(validator.LANE_EXPECTATIONS[lane])
        identity["python_version"] = banner
        _write(green_tree / f"lane-{lane}" / "runtime_identity.json", identity)
        assert _validate(validator, green_tree)["original_wheel_sha256"] == _DIGEST, banner


def test_the_helper_rejects_a_resolver_runtime_below_the_declared_floor(
    validator: ModuleType, green_tree: Path
) -> None:
    """A resolver-latest lane is unpinned, not unbounded."""
    lane = "linux_x64_py313"
    identity = _identity(validator.LANE_EXPECTATIONS[lane])
    identity |= {
        "package_version": "3.8.0",
        "package_base_version": "3.8.0",
        "native_version_string": "3.8.0",
        "native_version_integer": 3008000,
    }
    _write(green_tree / f"lane-{lane}" / "runtime_identity.json", identity)
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree)


# ---- Only supported workflow events are admitted ----------------------------------------------------


@pytest.mark.parametrize(
    "event",
    [
        pytest.param("schedule", id="schedule"),
        pytest.param("merge_group", id="merge_group"),
        pytest.param("", id="empty"),
        pytest.param("release", id="release"),
    ],
)
def test_the_helper_rejects_an_unsupported_event(
    validator: ModuleType, green_tree: Path, event: str
) -> None:
    """A consistent receipt for an event this contract never reasoned about is still not evidence."""
    receipt = json.loads(
        (
            green_tree / "original-artifacts" / "quality-evidence" / "checkout_identity.json"
        ).read_text()
    )
    receipt["event_name"] = event
    _write(
        green_tree / "original-artifacts" / "quality-evidence" / "checkout_identity.json", receipt
    )
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree, event_name=event)


@pytest.mark.parametrize(
    "event",
    [pytest.param("push", id="push"), pytest.param("workflow_dispatch", id="workflow_dispatch")],
)
def test_the_helper_admits_each_supported_non_pull_request_event(
    validator: ModuleType, green_tree: Path, event: str
) -> None:
    """Push and manual dispatch keep their null pull-request semantics."""
    receipt = json.loads(
        (
            green_tree / "original-artifacts" / "quality-evidence" / "checkout_identity.json"
        ).read_text()
    )
    receipt["event_name"] = event
    _write(
        green_tree / "original-artifacts" / "quality-evidence" / "checkout_identity.json", receipt
    )
    summary = _validate(validator, green_tree, event_name=event)
    assert summary["accepted_subject"]["event_name"] == event
    assert summary["accepted_subject"]["pull_request_head_sha"] is None


def test_the_helper_rejects_a_missing_trusted_event(
    validator: ModuleType, green_tree: Path
) -> None:
    """A run that reports no event at all is refused before any event-specific receipt logic."""
    with pytest.raises(validator.EvidenceError):
        _validate(validator, green_tree, event_name=None)


def test_the_helper_accepts_an_exact_python_version_triplet(
    validator: ModuleType, green_tree: Path
) -> None:
    """A banner with no continuation at all is the triplet by itself."""
    lane = "linux_x64_py313"
    identity = _identity(validator.LANE_EXPECTATIONS[lane])
    identity["python_version"] = "3.13.13"
    _write(green_tree / f"lane-{lane}" / "runtime_identity.json", identity)
    assert _validate(validator, green_tree)["original_wheel_sha256"] == _DIGEST
