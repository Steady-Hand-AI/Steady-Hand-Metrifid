"""Behavioral repair contracts with counted fake steps, never native physics."""

from __future__ import annotations

import copy
import hashlib
import math
import struct
from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace

import pytest

from metrifid.repair import (
    RepairController,
    RepairLimits,
    RepairRequest,
    RepairTask,
    _engine,
    _profile,
    repair_controller,
    replay_controller,
)
from metrifid.repair._engine import _runtime as inspect_runtime
from metrifid.repair._physical_types import Measurement, Snapshot


def request(*, parameters: tuple[float, ...] = (0.5,) * 15, **limits: int | float) -> RepairRequest:
    """Build a supported explicit request without historical fixtures."""
    return RepairRequest(
        RepairTask(-0.06, (0.08, 0.06), math.pi / 2),
        RepairController(parameters),
        RepairLimits(**limits),
    )


def measured(value: float, finer: bool = False, *, qualified: bool | None = None) -> Measurement:
    """Return complete controlled hold data with the declared step identity."""
    dt = 0.001 if finer else 0.002
    return Measurement(
        "COMPLETE",
        value == 0 if qualified is None else qualified,
        "COMPLETE",
        (0.005 * (1 + value), 0.01, 0.001, 0.01),
        completed_steps=round(4 / dt),
        expected_steps=round(4 / dt),
        dt_s=dt,
        hold_samples=round(0.2 / dt) + 1,
        expected_hold_samples=round(0.2 / dt) + 1,
    )


def snapshot(scene_hash: str, *, offset: float = 0.0) -> Snapshot:
    """Create a deterministic fixed-shape state for nonphysical admission checks."""
    shapes = {"qpos": (9,), "qvel": (8,), "act": (0,), "mocap_pos": (0, 3), "mocap_quat": (0, 4)}
    arrays = tuple((name, shape, (offset,) * math.prod(shape)) for name, shape in shapes.items())
    payload = b"".join(
        struct.pack("<" + "d" * len(values), *values) for _, _, values in sorted(arrays)
    )
    return Snapshot(
        arrays, (offset, 0.0, 0.01), 0.0, scene_hash, hashlib.sha256(payload).hexdigest()
    )


class FakeBackend:
    """Generated input identities and counted deterministic evaluator outcomes."""

    def __init__(self, supplied: RepairRequest, budget: _engine._Budget, harness: Harness) -> None:
        """Count setup through the exact guard and step seam used by physics."""
        self.profile, self.nominal_xml, self.finer_xml, self.identities = (
            _profile.generate_expected(supplied)
        )
        self.snapshot = snapshot(str(self.identities["nominal_scene_sha256"]))
        self.initialization = harness.initialization
        self.harness = harness
        for _ in range(_profile.setup_steps()):
            budget.step(self._operation)

    def _operation(self) -> None:
        self.harness.invoked += 1
        self.harness.on_step(self.harness.invoked)

    def evaluate(
        self,
        parameters: tuple[float, ...],
        *,
        finer: bool,
        check: Callable[[], None],
        step: Callable[[Callable[[], None]], None],
    ) -> Measurement:
        """Run only no-op integration calls, preserving actual fake-call counters."""
        check()
        self.harness.calls.append((parameters, finer))
        result = self.harness.provider(parameters, finer, len(self.harness.calls))
        for _ in range(result.completed_steps):
            step(self._operation)
        return result


class Harness:
    """Own one test's fake model, outcomes, steps and clock."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Replace physical creation and runtime discovery, leaving engine logic live."""
        self.calls: list[tuple[tuple[float, ...], bool]] = []
        self.created: list[RepairRequest] = []
        self.invoked = 0
        self.clock = 0.0
        self.provider: Callable[[tuple[float, ...], bool, int], Measurement] = (
            lambda _parameters, finer, _index: measured(0.2, finer)
        )
        self.on_step: Callable[[int], None] = lambda _index: None
        self.initialization = Measurement("COMPLETE", False, "SETTLED", completed_steps=200)
        monkeypatch.setattr(_engine, "_runtime", lambda: {"scope": "controlled software fixture"})
        monkeypatch.setattr(_engine, "_create_backend", self.create)
        monkeypatch.setattr(_engine.time, "process_time", lambda: self.clock)

    def create(self, supplied: RepairRequest, budget: _engine._Budget) -> FakeBackend:
        """Record actual admitted inputs reaching model creation."""
        self.created.append(supplied)
        return FakeBackend(supplied, budget, self)


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> Harness:
    """Provide isolated controlled runtime seams."""
    return Harness(monkeypatch)


def test_exact_eight_term_objective_and_unknown_measurements() -> None:
    """Every hold metric at both timesteps can determine the objective."""
    limits = _profile.HOLD_LIMITS
    for side in range(2):
        for index in range(4):
            maxima = list(limits)
            maxima[index] *= 1.25
            pair = [
                replace(measured(0.0, bool(case)), hold_maxima=tuple(limits)) for case in range(2)
            ]
            pair[side] = replace(pair[side], hold_maxima=tuple(maxima))
            assert _engine._objective(*pair) == pytest.approx(0.25)
    assert _engine._objective(measured(0.0), replace(measured(0.0, True), hold_maxima=None)) is None
    assert (
        _engine._objective(
            replace(measured(0.0), hold_maxima=(math.nan, 0.0, 0.0, 0.0)), measured(0.0, True)
        )
        is None
    )


def test_supplied_start_reaches_evaluator_and_best_is_retained(harness: Harness) -> None:
    """Finite exhaustion retains actual unqualified starting measurements."""
    supplied = request(parameters=(0.4,) * 15, max_attempts=2)
    result = repair_controller(supplied)
    assert harness.created == [supplied]
    assert harness.calls[0][0] == supplied.controller.parameters_normalized
    assert result.status == "NO_QUALIFIED_POLICY_FOUND"
    assert result.reason == "ATTEMPT_LIMIT"
    assert result.selected_parameters == supplied.controller.parameters_normalized
    assert result.selected_screening.objective == pytest.approx(0.2)
    assert not result.qualified
    assert result.confirmation is None


def test_coordinate_order_preferred_sign_and_strict_tolerance(harness: Harness) -> None:
    """Negative accepted movement sets the next sweep's preferred sign."""
    harness.provider = lambda p, finer, _: measured(0.5 if p[0] < 0.5 else 1.0, finer)
    result = repair_controller(request(max_attempts=32, max_integration_steps=1_000_000))
    nominal = [point for point, finer in harness.calls if not finer]
    assert [point[0] for point in nominal[1:3]] == [0.51, 0.49]
    assert nominal[3][0:2] == (0.49, 0.51)
    assert nominal[31][0] == 0.48
    assert result.selected_parameters[0] == 0.49
    assert result.costs.attempts == 32


def test_tiny_improvement_is_not_accepted_and_stagnant_sweep_halves(harness: Harness) -> None:
    """The comparator retains its 1e-10 tolerance and unchanged halving order."""
    harness.provider = lambda p, finer, _: measured(0.2 if p == (0.5,) * 15 else 0.2 - 5e-11, finer)
    result = repair_controller(request(max_attempts=34, max_integration_steps=1_000_000))
    nominal = [point for point, finer in harness.calls if not finer]
    assert nominal[31][0] == 0.505
    assert nominal[32][0] == 0.495
    assert result.selected_parameters == (0.5,) * 15


def test_changed_boundary_candidate_is_confirmed_and_replayable(harness: Harness) -> None:
    """Generated and selected controllers admit closed-box boundary coordinates."""
    supplied = request(parameters=(0.99,) + (0.5,) * 14)
    harness.provider = lambda p, finer, _: measured(0.0 if p[0] == 1.0 else 0.2, finer)
    result = repair_controller(supplied)
    assert (result.status, result.exit_code) == ("REPAIRED", 0)
    assert result.selected_parameters[0] == 1.0
    assert result.confirmation_complete
    assert result.qualified
    assert result.costs.attempts == 2
    assert result.costs.integration_steps == 18_200
    replay = replay_controller(result)
    assert (replay.status, replay.exit_code) == ("QUALIFIED", 0)
    assert replay.costs.attempts == 0
    assert replay.costs.integration_steps == 6_200


def test_unchanged_zero_requires_complete_confirmation(harness: Harness) -> None:
    """An unchanged initial zero gets its own confirmation and already status."""
    harness.provider = lambda _p, finer, _i: measured(0.0, finer)
    result = repair_controller(request())
    assert (result.status, result.exit_code) == ("ALREADY_QUALIFIED", 0)
    assert result.original_parameters == result.selected_parameters
    assert len(harness.calls) == 4
    assert result.costs.integration_steps == 12_200
    assert result.costs.attempts == 1


def test_first_zero_false_confirmation_terminates(harness: Harness) -> None:
    """Complete physical nonqualification cannot continue searching from zero."""
    harness.provider = lambda _p, finer, index: measured(0.0 if index <= 2 else 0.2, finer)
    result = repair_controller(request())
    assert (result.status, result.reason) == (
        "NO_QUALIFIED_POLICY_FOUND",
        "CONFIRMATION_DID_NOT_QUALIFY",
    )
    assert result.confirmation_complete
    assert not result.qualified
    assert len(harness.calls) == 4
    assert result.costs.attempts == 1


def test_zero_scalar_without_qualified_confirmation_is_not_success(harness: Harness) -> None:
    """A zero objective does not replace the physical qualification predicate."""
    harness.provider = lambda _p, finer, _i: measured(0.0, finer, qualified=False)
    result = repair_controller(request())
    assert result.exit_code == 10
    assert not result.qualified
    assert result.confirmation.objective == 0.0


def test_interrupted_confirmation_is_incomplete(harness: Harness) -> None:
    """A missing peer cannot turn screening zero into success."""

    def provider(_p: tuple[float, ...], finer: bool, index: int) -> Measurement:
        if index == 4:
            return Measurement(
                "INCOMPLETE", False, "WARNING_REQUIRES_ADJUDICATION", completed_steps=7
            )
        return measured(0.0, finer)

    harness.provider = provider
    result = repair_controller(request())
    assert (result.status, result.exit_code) == ("INCOMPLETE", 20)
    assert not result.confirmation_complete
    assert not result.qualified
    assert result.costs.integration_steps == 8_207


def test_known_invalid_initial_policy_and_initial_state(harness: Harness) -> None:
    """Known invalid initialization and known invalid policy have distinct reasons."""
    harness.initialization = Measurement("HARD_REJECTION", False, "SETTLED_TILT")
    initial = repair_controller(request())
    assert initial.reason == "INITIAL_STATE_NOT_ADMISSIBLE"
    assert initial.exit_code == 10
    assert not harness.calls
    harness.initialization = Measurement("COMPLETE", False, "SETTLED")
    harness.provider = lambda _p, _finer, _i: Measurement(
        "HARD_REJECTION", False, "TILT", completed_steps=7
    )
    invalid = repair_controller(request())
    assert invalid.reason == "INITIAL_POLICY_NOT_SEARCHABLE"
    assert invalid.exit_code == 10
    assert invalid.selected_parameters is None
    assert invalid.costs.integration_steps == 207
    assert len(harness.calls) == 1


def test_later_known_hard_rejection_is_charged_then_search_continues(harness: Harness) -> None:
    """A measured hard-rejected nominal prefix can skip finer without hiding cost."""

    def provider(p: tuple[float, ...], finer: bool, _index: int) -> Measurement:
        if p[0] > 0.5:
            return Measurement("HARD_REJECTION", False, "TILT", completed_steps=7)
        return measured(0.0 if p[0] < 0.5 else 0.2, finer)

    harness.provider = provider
    result = repair_controller(request())
    assert result.status == "REPAIRED"
    assert result.costs.attempts == 3
    assert result.costs.integration_steps == 18_207
    rejected = result.screening_measurements[1]
    assert rejected.finer is None
    assert rejected.known_completed_steps == 7


def test_unknown_later_receipt_stops_without_later_recovery(harness: Harness) -> None:
    """An unknown prefix is neither an ordinary rejection nor a finite penalty."""
    harness.provider = lambda p, finer, _: (
        measured(0.2, finer)
        if p == (0.5,) * 15
        else Measurement("INCOMPLETE", False, "NONFINITE_STATE", completed_steps=7)
    )
    result = repair_controller(request())
    assert result.status == "INCOMPLETE"
    assert result.reason == "NONFINITE_STATE"
    assert result.costs.attempts == 2
    assert result.costs.integration_steps == 6_207
    assert len(harness.calls) == 3
    assert result.selected_screening.objective > 0


def test_evaluator_exception_keeps_partial_cost_and_unknown_native_call(harness: Harness) -> None:
    """A failed native call cannot manufacture a complete zero-work receipt."""

    def fail(index: int) -> None:
        if index == 6_208:
            raise RuntimeError("controlled native call failure")

    harness.on_step = fail
    result = repair_controller(request())
    assert result.status == "ERROR"
    assert result.exit_code == 70
    assert result.costs.integration_steps is None
    assert result.costs.known_completed_steps == 6_207
    assert result.costs.unknown_step_calls == 1
    assert result.selected_screening is result.initial_screening
    assert "RuntimeError" in result.diagnostic


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, SystemExit])
def test_process_control_exceptions_propagate(
    harness: Harness, interruption: type[BaseException]
) -> None:
    """Process-control exceptions are never fabricated completed results."""

    def stop(_index: int) -> None:
        raise interruption()

    harness.on_step = stop
    with pytest.raises(interruption):
        repair_controller(request())


def test_out_of_box_attempt_costs_no_physics(harness: Harness) -> None:
    """Out-of-domain generated proposals consume attempts without native calls."""
    result = repair_controller(request(parameters=(0.995,) + (0.5,) * 14, max_attempts=3))
    assert result.costs.attempts == 3
    assert result.costs.integration_steps == 12_200
    assert result.screening_measurements[1].classification == "OUT_OF_BOX"
    assert result.screening_measurements[1].known_completed_steps == 0
    assert len(harness.calls) == 4


def test_minimum_step_budget_reserves_confirmation_before_next_screen(harness: Harness) -> None:
    """A refused physical pair is not called and consumes no speculative cost."""
    result = repair_controller(request(max_integration_steps=12_200))
    assert result.status == "NO_QUALIFIED_POLICY_FOUND"
    assert result.reason == "INTEGRATION_STEP_RESERVE"
    assert result.costs.integration_steps == 6_200
    assert result.costs.attempts == 1
    assert len(harness.calls) == 2


def test_clock_starts_before_runtime_discovery(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lazy import/runtime time is charged before any setup can begin."""

    def runtime() -> dict[str, str]:
        harness.clock = 91.0
        return {"scope": "controlled costly runtime discovery"}

    monkeypatch.setattr(_engine, "_runtime", runtime)
    result = repair_controller(request(max_cpu_seconds=120))
    assert result.status == "INCOMPLETE"
    assert result.reason == "CPU_LIMIT"
    assert not harness.created
    assert result.costs.integration_steps == 0
    assert result.costs.cpu_seconds == 91.0


def test_confirmation_uses_shared_full_clock_and_replay_gets_fresh_cap(harness: Harness) -> None:
    """Confirmation may use reserved CPU; independent replay has no search cutoff."""

    def provider(_p: tuple[float, ...], finer: bool, index: int) -> Measurement:
        harness.clock = 89.0 if index <= 2 else 100.0
        return measured(0.0, finer)

    harness.provider = provider
    result = repair_controller(request(max_cpu_seconds=120))
    assert result.status == "ALREADY_QUALIFIED"
    assert result.costs.cpu_seconds == 100.0
    harness.provider = lambda _p, finer, _index: measured(0.0, finer)
    harness.on_step = lambda _index: setattr(harness, "clock", 200.0)
    replay = replay_controller(result)
    assert replay.status == "QUALIFIED"
    assert replay.costs.cpu_seconds == 100.0
    assert replay.costs.attempts == 0
    assert replay.costs.integration_steps == 6_200


def test_completion_after_full_cpu_cap_is_not_success(harness: Harness) -> None:
    """The final completed step still faces the full-clock guard."""
    harness.provider = lambda _p, finer, _i: measured(0.0, finer)
    harness.on_step = lambda index: setattr(harness, "clock", 120.0 if index == 12_200 else 0.0)
    result = repair_controller(request(max_cpu_seconds=120))
    assert result.status == "INCOMPLETE"
    assert not result.qualified
    assert result.costs.integration_steps == 12_200


def test_runtime_and_manually_tampered_request_refuse_before_physics(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neither unsupported native versions nor constructed objects bypass admission."""

    def refuse() -> dict[str, str]:
        raise _engine._RuntimeRefused("controlled unsupported runtime")

    monkeypatch.setattr(_engine, "_runtime", refuse)
    result = repair_controller(request())
    assert result.status == "REFUSED"
    assert result.reason == "UNSUPPORTED_RUNTIME"
    invalid = request()
    object.__setattr__(invalid.controller, "parameters_normalized", (0.0,) * 15)
    refused = repair_controller(invalid)
    assert refused.status == "REFUSED"
    assert refused.reason == "INVALID_REQUEST"
    assert not harness.created


def qualified_record(harness: Harness):
    """Obtain a retained record through the real repair boundary."""
    harness.provider = lambda _p, finer, _i: measured(0.0, finer)
    return repair_controller(request())


def test_replay_freshly_evaluates_without_trusting_qualification(harness: Harness) -> None:
    """Saved success is not replay's success, and the input result is unchanged."""
    source = qualified_record(harness)
    before = copy.deepcopy(source)
    harness.provider = lambda _p, finer, _i: measured(0.3, finer)
    replay = replay_controller(source)
    assert replay.status == "NOT_QUALIFIED"
    assert replay.exit_code == 10
    assert replay.measurement.objective == pytest.approx(0.3)
    assert source == before
    assert replay.costs.integration_steps == 6_200


@pytest.mark.parametrize("field", ["profile", "nominal_xml", "finer_xml", "timestep_identities"])
def test_inconsistent_profile_or_generated_scene_refuses_before_setup(
    harness: Harness, field: str
) -> None:
    """Recomputed caller identities cannot admit arbitrary retained scene content."""
    source = qualified_record(harness)
    if field == "profile":
        source.profile["family"] = "unsupported"
    elif field == "nominal_xml":
        source.nominal_xml += " "
        source.timestep_identities["nominal_scene_sha256"] = hashlib.sha256(
            source.nominal_xml.encode()
        ).hexdigest()
    elif field == "finer_xml":
        source.finer_xml += " "
        source.timestep_identities["finer_scene_sha256"] = hashlib.sha256(
            source.finer_xml.encode()
        ).hexdigest()
    elif field == "timestep_identities":
        source.timestep_identities["finer_dt_s"] = 0.003
    created = len(harness.created)
    replay = replay_controller(source)
    assert replay.status == "REFUSED"
    assert len(harness.created) == created


def test_recomputed_reset_hash_cannot_replace_independent_settle(harness: Harness) -> None:
    """Well-formed self-consistent physical arrays must match independent settling."""
    source = qualified_record(harness)
    source.reset = snapshot(source.reset.nominal_scene_sha256, offset=0.001)
    calls = len(harness.calls)
    replay = replay_controller(source)
    assert replay.status == "REFUSED"
    assert replay.reason == "RESET_MISMATCH"
    assert replay.costs.integration_steps == 200
    assert len(harness.calls) == calls


def test_replay_refuses_bad_or_contradictory_selected_bindings(harness: Harness) -> None:
    """Invalid vectors and disagreement between retained bindings are input refusals."""
    source = qualified_record(harness)
    for vector in (None, (-0.01,) + (0.5,) * 14, (0.6,) * 15):
        changed = copy.deepcopy(source)
        changed.selected_parameters = vector
        replay = replay_controller(changed)
        assert replay.status == "REFUSED"
        assert replay.costs.integration_steps == 0


def test_consistent_valid_controller_substitution_is_evaluated(harness: Harness) -> None:
    """Unsigned consistent controller replacement is freshly evaluated, not authenticated."""
    source = qualified_record(harness)
    changed = (0.6,) * 15
    source.selected_parameters = changed
    source.selected_screening = replace(source.selected_screening, parameters_normalized=changed)
    source.confirmation = replace(source.confirmation, parameters_normalized=changed)
    replay = replay_controller(source)
    assert replay.status == "QUALIFIED"
    assert harness.calls[-1][0] == changed


def test_consistent_supported_task_change_is_readmitted(harness: Harness) -> None:
    """A consistent changed task is replayed as that valid supplied task."""
    source = qualified_record(harness)
    source.request = replace(source.request, task=RepairTask(-0.08, (0.07, 0.05), math.pi / 2))
    source.profile, source.nominal_xml, source.finer_xml, source.timestep_identities = (
        _profile.generate_expected(source.request)
    )
    source.reset = snapshot(source.timestep_identities["nominal_scene_sha256"])
    replay = replay_controller(source)
    assert replay.status == "QUALIFIED"
    assert harness.created[-1].task == source.request.task


def test_finalization_past_cpu_cap_cannot_retain_success(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Post-evaluation bookkeeping still faces the invocation's final guard."""
    harness.provider = lambda _p, finer, _i: measured(0.0, finer)
    original = _engine._classify_pair

    def delayed(pair):
        original(pair)
        if len(harness.calls) == 4:
            harness.clock = 120.0

    monkeypatch.setattr(_engine, "_classify_pair", delayed)
    result = repair_controller(request(max_cpu_seconds=120))
    assert result.status == "INCOMPLETE"
    assert not result.qualified
    assert result.costs.integration_steps == 12_200


def test_partial_setup_cpu_interruption_keeps_actual_steps(harness: Harness) -> None:
    """The setup loop shares the invocation cutoff and preserves completed work."""
    harness.on_step = lambda index: setattr(harness, "clock", 90.0 if index == 73 else 0.0)
    result = repair_controller(request(max_cpu_seconds=120))
    assert result.status == "INCOMPLETE"
    assert result.reason == "CPU_LIMIT"
    assert result.costs.integration_steps == 73
    assert result.costs.attempts == 0
    assert not harness.calls


def _runtime_modules(monkeypatch: pytest.MonkeyPatch, *, mujoco: str, numpy: str = "2.5.4") -> None:
    monkeypatch.setattr(_engine, "_runtime", inspect_runtime)
    monkeypatch.setattr(_engine.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(_engine.platform, "system", lambda: "Linux")
    monkeypatch.setattr(_engine.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(_engine.platform, "python_version", lambda: "3.13.5")
    monkeypatch.setattr(_engine.sys, "version_info", (3, 13))
    original_import = _engine.importlib.import_module

    def import_native(name: str, package: str | None = None):
        if name in ("mujoco", "numpy"):
            return SimpleNamespace(__version__=mujoco if name == "mujoco" else numpy)
        return original_import(name, package)

    monkeypatch.setattr(_engine.importlib, "import_module", import_native)


@pytest.mark.parametrize("mujoco", ["3.16.0", "3.17.0"])
def test_exact_supported_runtimes_reach_repair_and_fresh_replay(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, mujoco: str
) -> None:
    """Real admission retains each detected version through counted fake execution."""
    _runtime_modules(monkeypatch, mujoco=mujoco)
    harness.provider = lambda _parameters, finer, _index: measured(0.0, finer)
    repaired = repair_controller(request())
    assert repaired.status == "ALREADY_QUALIFIED"
    assert repaired.runtime == {
        "implementation": "CPython",
        "python": "3.13.5",
        "system": "Linux",
        "machine": "x86_64",
        "mujoco": mujoco,
        "numpy": "2.5.4",
    }
    assert repaired.costs.known_completed_steps == 12200
    replayed = replay_controller(repaired)
    assert replayed.status == "QUALIFIED"
    assert replayed.runtime == repaired.runtime
    assert replayed.costs.known_completed_steps == 6200
    assert len(harness.created) == 2
    assert harness.invoked == 18400


@pytest.mark.parametrize(
    ("mujoco", "numpy"),
    [
        ("3.15.0", "2.5.4"),
        ("3.16.1", "2.5.4"),
        ("3.17.1", "2.5.4"),
        ("3.17.0rc1", "2.5.4"),
        ("3.17.0+vendor", "2.5.4"),
        ("3.18.0", "2.5.4"),
        ("3.17.0", "2.5.5"),
    ],
)
def test_unlisted_runtime_versions_refuse_before_setup(
    harness: Harness, monkeypatch: pytest.MonkeyPatch, mujoco: str, numpy: str
) -> None:
    """Exact version support does not silently admit nearby or future versions."""
    _runtime_modules(monkeypatch, mujoco=mujoco, numpy=numpy)
    result = repair_controller(request())
    assert result.status == "REFUSED"
    assert result.reason == "UNSUPPORTED_RUNTIME"
    assert not harness.created
    assert harness.invoked == 0


@pytest.mark.parametrize("failure", [ValueError, RuntimeError, KeyboardInterrupt])
def test_native_import_failure_refuses_and_process_interrupt_propagates(
    harness: Harness,
    monkeypatch: pytest.MonkeyPatch,
    failure: type[BaseException],
) -> None:
    """Binary runtime import failures refuse before any physical setup."""
    monkeypatch.setattr(_engine, "_runtime", inspect_runtime)
    monkeypatch.setattr(_engine.platform, "python_implementation", lambda: "CPython")
    monkeypatch.setattr(_engine.platform, "system", lambda: "Linux")
    monkeypatch.setattr(_engine.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(_engine.platform, "python_version", lambda: "3.13.0")
    monkeypatch.setattr(_engine.sys, "version_info", (3, 13))
    original_import = _engine.importlib.import_module

    def import_native(name: str, package: str | None = None):
        if name in ("mujoco", "numpy"):
            raise failure("controlled unavailable native runtime")
        return original_import(name, package)

    monkeypatch.setattr(_engine.importlib, "import_module", import_native)
    if failure is KeyboardInterrupt:
        with pytest.raises(KeyboardInterrupt):
            repair_controller(request())
    else:
        result = repair_controller(request())
        assert result.status == "REFUSED"
        assert result.reason == "UNSUPPORTED_RUNTIME"
        assert failure.__name__ in result.diagnostic
    assert not harness.created


@pytest.mark.parametrize(
    "changes",
    [
        {"completed_steps": 1999},
        {"expected_steps": 1999},
        {"hold_samples": 100},
        {"expected_hold_samples": 100},
        {"dt_s": 0.001},
        {"hard_failures": ("TILT",)},
    ],
)
def test_finite_scalar_requires_explicit_complete_timestep_receipts(changes) -> None:
    """A complete flag alone cannot replace actual step, hold and timestep evidence."""
    nominal = replace(measured(0.0), **changes)
    assert _engine._objective(nominal, measured(0.0, True)) is None


def test_first_exact_zero_confirms_even_below_improvement_tolerance(harness: Harness) -> None:
    """The first-zero confirmation rule also applies to a tiny positive incumbent."""
    harness.provider = lambda p, finer, _index: measured(5e-11 if p == (0.5,) * 15 else 0.0, finer)
    result = repair_controller(request())
    expected = (0.51,) + (0.5,) * 14
    assert result.status == "REPAIRED"
    assert result.costs.attempts == 2
    assert len(harness.calls) == 6
    assert 0 < result.initial_screening.objective < 1e-10
    assert result.selected_parameters == expected
    assert result.confirmation.parameters_normalized == expected
    assert all(point == expected for point, _finer in harness.calls[2:])
    assert result.qualified
