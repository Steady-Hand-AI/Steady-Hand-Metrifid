"""Exact comparison preexecution step and trace-budget tests without a MuJoCo runtime."""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
import types
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from metrifid import Binary64, ExactRational
from metrifid._model_closure import AlignedActuator, AlignedJoint
from metrifid._timegrid import TimeGrid
from metrifid.compare import _budget
from metrifid.compare._budget import (
    MAX_TOTAL_INTERNAL_STEPS,
    MAX_TRACE_FLOAT64_BYTES,
    evaluate_preexecution_budgets,
)
from metrifid.errors import (
    EngineThreadpoolState,
    ReasonCode,
    ReasonRecord,
)
from metrifid.schemas import (
    AlignmentSummary,
    EnvironmentIdentity,
    ModelClosureIdentity,
    ModelClosureMember,
    TargetReference,
)


def _grid(
    *,
    control_intervals: int,
    baseline_substeps: int,
    candidate_substeps: int,
) -> TimeGrid:
    """Construct the grid fixture used by compare budget scenarios.

    Deterministic setup isolates compare budget without bypassing the contract boundary under
    assertion.
    """
    control = ExactRational(1, 1)
    baseline_step = ExactRational(1, baseline_substeps)
    candidate_step = ExactRational(1, candidate_substeps)
    return TimeGrid(
        baseline_step_dt=baseline_step,
        candidate_step_dt=candidate_step,
        control_dt=control,
        baseline_compiled_timestep=Binary64.from_float(float(Fraction(1, baseline_substeps))),
        candidate_compiled_timestep=Binary64.from_float(float(Fraction(1, candidate_substeps))),
        baseline_substeps_per_control=baseline_substeps,
        candidate_substeps_per_control=candidate_substeps,
        control_intervals=control_intervals,
        boundary_count=control_intervals + 1,
        horizon=control.multiplied_by_int(control_intervals),
    )


def _hinge_joints(count: int) -> tuple[AlignedJoint, ...]:
    """Construct the hinge joints fixture used by compare budget scenarios.

    Deterministic setup isolates compare budget without bypassing the contract boundary under
    assertion.
    """
    return tuple(
        AlignedJoint(
            f"joint_{index:03d}",
            "HINGE",
            (index, 1),
            (index, 1),
            (index, 1),
            (index, 1),
        )
        for index in range(count)
    )


def _actuator(*, activation_width: int) -> AlignedActuator:
    """Construct the actuator fixture used by compare budget scenarios.

    Deterministic setup isolates compare budget without bypassing the contract boundary under
    assertion.
    """
    family = "NONE" if activation_width == 0 else "INTEGRATOR"
    address = None if activation_width == 0 else 0
    return AlignedActuator(
        "motor",
        "JOINT",
        (TargetReference("JOINT", "joint_000"),),
        family,
        activation_width,
        0,
        0,
        address,
        address,
    )


def _reason_by_code(reasons: Sequence[ReasonRecord], code: ReasonCode) -> ReasonRecord:
    """Construct the reason by code fixture used by compare budget scenarios.

    Deterministic setup isolates compare budget without bypassing the contract boundary under
    assertion.
    """
    return next(reason for reason in reasons if reason.code is code)


@dataclass(slots=True)
class _SpyBackend:
    """Represent spy backend."""

    new_data_calls: int = 0
    forward_calls: int = 0
    step_calls: int = 0

    def new_data(self) -> SimpleNamespace:
        """Construct the new data fixture used by compare budget scenarios.

        Deterministic setup isolates SpyBackend without bypassing the contract boundary under
        assertion.
        """
        self.new_data_calls += 1
        return SimpleNamespace(threadpool=0)

    def forward(self, _data: object) -> None:
        """Construct the forward fixture used by compare budget scenarios.

        Deterministic setup isolates SpyBackend without bypassing the contract boundary under
        assertion.
        """
        self.forward_calls += 1
        raise AssertionError("forward must not run in the preexecution branch")

    def step(self, _data: object) -> None:
        """Construct the step fixture used by compare budget scenarios.

        Deterministic setup isolates SpyBackend without bypassing the contract boundary under
        assertion.
        """
        self.step_calls += 1
        raise AssertionError("step must not run in the preexecution branch")


@dataclass(frozen=True, slots=True)
class _WorkloadMember:
    """Represent workload member."""

    raw_file_sha256: str
    semantic_sha256: str


@dataclass(frozen=True, slots=True)
class _ActionsMember(_WorkloadMember):
    """Represent actions member."""

    metadata: SimpleNamespace


def _digest(label: str) -> str:
    """Compute the canonical digest value used by compare budget fixtures.

    Content addressing keeps the mutation boundary explicit for compare budget.
    """
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _closure(label: str) -> ModelClosureIdentity:
    """Construct the closure fixture used by compare budget scenarios.

    Deterministic setup isolates compare budget without bypassing the contract boundary under
    assertion.
    """
    member = ModelClosureMember("model.xml", len(label), _digest(f"member:{label}"))
    return ModelClosureIdentity("model.xml", 1, (member,))


def _environment(threadpool: EngineThreadpoolState) -> EnvironmentIdentity:
    """Construct the environment fixture used by compare budget scenarios.

    Deterministic setup isolates compare budget without bypassing the contract boundary under
    assertion.
    """
    return EnvironmentIdentity(
        mujoco_version="3.10.0",
        python_version="3.13.5",
        numpy_version="2.3.5",
        mujoco_python_distribution_sha256=_digest("mujoco-python"),
        mujoco_native_library_sha256=_digest("mujoco-native"),
        platform="linux-x86_64",
        platform_release="constructed",
        libc="glibc-constructed",
        cpu_identity_sha256=_digest("cpu"),
        engine_threadpool_state=threadpool,
        environment_sha256=None,
    )


def _write_config(
    root: Path,
    *,
    output_name: str,
    step_token: str,
    control_token: str,
) -> Path:
    """Write write config data into the isolated test workspace.

    The compare budget scenario observes real bytes and filesystem effects for compare budget.
    """
    value = {
        "schema_version": 1,
        "baseline": {
            "model_root": "baseline",
            "entrypoint": "model.xml",
            "declared_step_dt": step_token,
        },
        "candidate": {
            "model_root": "candidate",
            "entrypoint": "model.xml",
            "declared_step_dt": step_token,
        },
        "initial_state": "state.npz",
        "actions": "actions.npz",
        "control_dt": control_token,
        "repeats": 2,
        "joint_tolerances": {
            "joint_000": {
                "joint_type": "hinge",
                "angle_rad": "0.001",
                "angular_velocity_rad_s": "0.01",
            }
        },
        "aliases": None,
        "output_dir": output_name,
    }
    path = root / "comparison.json"
    path.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")
    return path


def _isolate_modules(monkeypatch: pytest.MonkeyPatch, names: Sequence[str]) -> None:
    """Remove each module and its parent-package attribute, restoring both exactly at teardown.

    Two things have to be undone, not one. ``importlib.import_module`` installs the module in
    ``sys.modules`` *and* sets it as an attribute on its parent package, so restoring
    ``sys.modules`` alone leaves the parent attribute pointing at the stub-loaded module for the
    rest of the session. Both are recorded here through ``monkeypatch``.

    Each pair is deliberately ``setitem``/``setattr`` followed by ``delitem``/``delattr``. The
    first call records the prior value *or its absence*; the second performs the removal. A bare
    ``delitem(..., raising=False)`` records nothing when the key was already absent, so a module
    created later by the re-import would survive teardown. Nothing here removes an entry
    without first recording what was there.
    """
    for name in names:
        parent_name, _, attribute = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            monkeypatch.setattr(parent, attribute, getattr(parent, attribute, None), raising=False)
            monkeypatch.delattr(parent, attribute, raising=False)
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name))
        monkeypatch.delitem(sys.modules, name, raising=False)


def _load_orchestrator(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Construct the load orchestrator fixture used by compare budget scenarios.

    Deterministic setup isolates compare budget without bypassing the contract boundary under
    assertion.
    """
    module_names = (
        "metrifid.compare._orchestrator",
        "metrifid._workload",
        "metrifid.compare._environment",
        "metrifid.compare._model_pair",
        "metrifid.compare._mujoco_backend",
    )
    _isolate_modules(monkeypatch, module_names)

    workload = types.ModuleType("metrifid._workload")
    workload.load_workload_artifacts = lambda *_args, **_kwargs: None
    environment = types.ModuleType("metrifid.compare._environment")
    environment.build_environment_identity = lambda state: _environment(state)

    def combine_threadpool_states(
        left: EngineThreadpoolState, right: EngineThreadpoolState
    ) -> EngineThreadpoolState:
        """Construct the combine threadpool states fixture used by compare budget scenarios.

        Deterministic setup isolates load orchestrator without bypassing the contract boundary
        under assertion.
        """
        if EngineThreadpoolState.ACTIVE in {left, right}:
            return EngineThreadpoolState.ACTIVE
        if EngineThreadpoolState.UNKNOWN in {left, right}:
            return EngineThreadpoolState.UNKNOWN
        return EngineThreadpoolState.DISABLED

    environment.combine_threadpool_states = combine_threadpool_states
    model_pair = types.ModuleType("metrifid.compare._model_pair")
    model_pair.open_live_model_pair = lambda **_kwargs: None
    backend = types.ModuleType("metrifid.compare._mujoco_backend")
    backend.MuJoCoBackend = object
    backend.observed_threadpool_state = lambda _data: EngineThreadpoolState.DISABLED

    monkeypatch.setitem(sys.modules, workload.__name__, workload)
    monkeypatch.setitem(sys.modules, environment.__name__, environment)
    monkeypatch.setitem(sys.modules, model_pair.__name__, model_pair)
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    return importlib.import_module("metrifid.compare._orchestrator")


def _configure_orchestrator(
    monkeypatch: pytest.MonkeyPatch,
    orchestrator: Any,
    *,
    joints: tuple[AlignedJoint, ...],
    actuators: tuple[AlignedActuator, ...],
    control_intervals: int,
    timestep: float,
) -> tuple[_SpyBackend, _SpyBackend]:
    """Construct the configure orchestrator fixture used by compare budget scenarios.

    Deterministic setup isolates compare budget without bypassing the contract boundary under
    assertion.
    """
    pair, baseline_model, candidate_model, workload, backends = _orchestrator_fixture_values(
        joints, actuators, control_intervals, timestep
    )
    monkeypatch.setattr(orchestrator, "installed_distribution_sha256", lambda: _digest("tool"))
    monkeypatch.setattr(
        orchestrator,
        "open_live_model_pair",
        lambda **_kwargs: _open_pair(pair, baseline_model, candidate_model),
    )
    monkeypatch.setattr(orchestrator, "load_workload_artifacts", lambda *_args: workload)
    monkeypatch.setattr(
        orchestrator,
        "MuJoCoBackend",
        lambda model: backends[cast(str, model.role)],
    )
    monkeypatch.setattr(
        orchestrator,
        "observed_threadpool_state",
        lambda _data: EngineThreadpoolState.DISABLED,
    )
    monkeypatch.setattr(
        orchestrator, "build_environment_identity", lambda state: _environment(state)
    )
    monkeypatch.setattr(
        orchestrator,
        "combine_threadpool_states",
        lambda _left, _right: EngineThreadpoolState.DISABLED,
    )
    return backends["baseline"], backends["candidate"]


@contextmanager
def _open_pair(
    pair: SimpleNamespace, baseline_model: SimpleNamespace, candidate_model: SimpleNamespace
) -> Iterator[SimpleNamespace]:
    """Yield one configured live model pair fixture."""
    yield SimpleNamespace(
        identity=pair,
        baseline_model=baseline_model,
        candidate_model=candidate_model,
    )


def _orchestrator_fixture_values(
    joints: tuple[AlignedJoint, ...],
    actuators: tuple[AlignedActuator, ...],
    control_intervals: int,
    timestep: float,
) -> tuple[
    SimpleNamespace, SimpleNamespace, SimpleNamespace, SimpleNamespace, dict[str, _SpyBackend]
]:
    """Build model-pair, workload, and backend fixtures for orchestration tests."""
    alignment = SimpleNamespace(
        joints=joints,
        actuators=actuators,
        aliases_raw_sha256=None,
        aliases_semantic_sha256=None,
    )
    summary = AlignmentSummary(
        None,
        tuple(item.canonical_name for item in joints),
        tuple(item.canonical_name for item in actuators),
        (),
    ).finalized()
    pair = SimpleNamespace(
        baseline_closure=_closure("baseline"),
        candidate_closure=_closure("candidate"),
        alignment=alignment,
        alignment_summary=summary,
    )
    baseline_model = SimpleNamespace(role="baseline", opt=SimpleNamespace(timestep=timestep))
    candidate_model = SimpleNamespace(role="candidate", opt=SimpleNamespace(timestep=timestep))
    workload = SimpleNamespace(
        state=_WorkloadMember(_digest("state-raw"), _digest("state-semantic")),
        actions=_ActionsMember(
            _digest("actions-raw"),
            _digest("actions-semantic"),
            SimpleNamespace(control_intervals=control_intervals),
        ),
    )
    baseline_backend = _SpyBackend()
    candidate_backend = _SpyBackend()
    backends = {"baseline": baseline_backend, "candidate": candidate_backend}
    return pair, baseline_model, candidate_model, workload, backends


def test_budget_module_exports_only_frozen_internal_surface() -> None:
    """Expose the budget constants and evaluator used by orchestration."""
    assert callable(_budget.evaluate_preexecution_budgets)
    assert MAX_TOTAL_INTERNAL_STEPS == 10_000_000
    assert MAX_TRACE_FLOAT64_BYTES == 268_435_456


def test_representative_request_is_admitted_with_exact_7680_step_count() -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises representative request is admitted with exact 7680 step count;
    status, numerical evidence, and artifact publication must remain stable for the declared
    workload.
    """
    grid = _grid(control_intervals=256, baseline_substeps=5, candidate_substeps=5)
    reasons = evaluate_preexecution_budgets(
        grid,
        3,
        _hinge_joints(1),
        (_actuator(activation_width=0),),
    )
    assert 256 * 3 * (5 + 5) == 7_680
    assert reasons == ()


def test_internal_step_limit_is_inclusive_and_first_constructible_overage_refuses() -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises internal step limit is inclusive and first constructible overage
    refuses; status, numerical evidence, and artifact publication must remain stable for the
    declared workload.
    """
    exact = _grid(
        control_intervals=1,
        baseline_substeps=2_500_000,
        candidate_substeps=2_500_000,
    )
    assert evaluate_preexecution_budgets(exact, 2, _hinge_joints(1), ()) == ()

    over = _grid(
        control_intervals=1,
        baseline_substeps=2_500_000,
        candidate_substeps=2_500_001,
    )
    reasons = evaluate_preexecution_budgets(over, 2, _hinge_joints(1), ())
    assert tuple(reason.code for reason in reasons) == (ReasonCode.INTERNAL_STEP_BUDGET_EXCEEDED,)
    evidence = reasons[0].to_primitive()["evidence"]
    assert evidence == {
        "baseline_substeps_per_control": 2_500_000,
        "candidate_substeps_per_control": 2_500_001,
        "control_intervals": 1,
        "maximum_total_internal_steps": 10_000_000,
        "repeats": 2,
        "requested_total_internal_steps": 10_000_002,
    }


def test_enormous_admitted_substep_integer_refuses_without_iteration_or_float(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises enormous admitted substep integer refuses without iteration or
    float; status, numerical evidence, and artifact publication must remain stable for the
    declared workload.
    """
    enormous = 10**300
    grid = _grid(
        control_intervals=1,
        baseline_substeps=enormous,
        candidate_substeps=enormous,
    )

    def forbidden(*_args: object, **_kwargs: object) -> object:
        """Construct the forbidden fixture used by compare budget scenarios.

        Deterministic setup isolates enormous admitted substep integer refuses without iteration
        or float without bypassing the contract boundary under assertion.
        """
        raise AssertionError("budget evaluation must not iterate or convert through float")

    monkeypatch.setattr(_budget, "range", forbidden, raising=False)
    monkeypatch.setattr(_budget, "float", forbidden, raising=False)
    monkeypatch.setattr(TimeGrid, "iter_canonical_boundaries", forbidden)
    monkeypatch.setattr(TimeGrid, "iter_role_boundary_time_bits", forbidden)

    reasons = evaluate_preexecution_budgets(grid, 2, _hinge_joints(1), ())
    reason = _reason_by_code(reasons, ReasonCode.INTERNAL_STEP_BUDGET_EXCEEDED)
    evidence = reason.to_primitive()["evidence"]
    assert evidence["requested_total_internal_steps"] == 4 * enormous
    assert isinstance(evidence["requested_total_internal_steps"], int)


def test_trace_memory_limit_is_inclusive_and_smallest_layout_overage_refuses() -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises trace memory limit is inclusive and smallest layout overage refuses;
    status, numerical evidence, and artifact publication must remain stable for the declared
    workload.
    """
    grid = _grid(control_intervals=65_535, baseline_substeps=1, candidate_substeps=1)
    exact_joints = _hinge_joints(64)
    assert grid.boundary_count * 2 * 2 * 8 * (64 + 64) == MAX_TRACE_FLOAT64_BYTES
    assert (
        evaluate_preexecution_budgets(
            grid,
            2,
            exact_joints,
            (_actuator(activation_width=0),),
        )
        == ()
    )

    reasons = evaluate_preexecution_budgets(
        grid,
        2,
        exact_joints,
        (_actuator(activation_width=1),),
    )
    assert tuple(reason.code for reason in reasons) == (ReasonCode.TRACE_MEMORY_BUDGET_EXCEEDED,)
    evidence = reasons[0].to_primitive()["evidence"]
    assert evidence == {
        "boundary_count": 65_536,
        "canonical_activation_width": 1,
        "canonical_qpos_width": 64,
        "canonical_qvel_width": 64,
        "float64_bytes_per_value": 8,
        "maximum_trace_float64_bytes": 268_435_456,
        "repeats": 2,
        "role_count": 2,
        "requested_trace_float64_bytes": 270_532_608,
    }


def test_both_reasons_are_unique_canonical_and_exact() -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises both reasons are unique canonical and exact; status, numerical
    evidence, and artifact publication must remain stable for the declared workload.
    """
    grid = _grid(control_intervals=65_535, baseline_substeps=100, candidate_substeps=100)
    reasons = evaluate_preexecution_budgets(
        grid,
        2,
        _hinge_joints(64),
        (_actuator(activation_width=1),),
    )
    assert tuple(reason.code for reason in reasons) == (
        ReasonCode.INTERNAL_STEP_BUDGET_EXCEEDED,
        ReasonCode.TRACE_MEMORY_BUDGET_EXCEEDED,
    )
    assert len(reasons) == len({reason.code for reason in reasons}) == 2
    step_evidence = reasons[0].to_primitive()["evidence"]
    trace_evidence = reasons[1].to_primitive()["evidence"]
    assert step_evidence["requested_total_internal_steps"] == 26_214_000
    assert step_evidence["maximum_total_internal_steps"] == 10_000_000
    assert trace_evidence["requested_trace_float64_bytes"] == 270_532_608
    assert trace_evidence["maximum_trace_float64_bytes"] == 268_435_456


def test_role_local_canonical_width_mismatch_is_an_internal_invariant() -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises role local canonical width mismatch is an internal invariant;
    status, numerical evidence, and artifact publication must remain stable for the declared
    workload.
    """
    broken = object.__new__(AlignedJoint)
    object.__setattr__(broken, "canonical_name", "broken")
    object.__setattr__(broken, "joint_type", "HINGE")
    object.__setattr__(broken, "baseline_qpos", (0, 1))
    object.__setattr__(broken, "candidate_qpos", (0, 2))
    object.__setattr__(broken, "baseline_qvel", (0, 1))
    object.__setattr__(broken, "candidate_qvel", (0, 1))
    with pytest.raises(ValueError, match="qpos widths are incompatible"):
        evaluate_preexecution_budgets(
            _grid(control_intervals=1, baseline_substeps=1, candidate_substeps=1),
            2,
            (broken,),
            (),
        )
