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

from metrifid import Binary64, ComparisonReceipt, ComparisonStatus, ExactRational
from metrifid._model_closure import AlignedActuator, AlignedJoint
from metrifid._timegrid import TimeGrid
from metrifid.errors import (
    EngineThreadpoolState,
    ReasonCode,
    ReasonRecord,
    status_exit_code,
)
from metrifid.operational import OperationalReasonCode
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
    """Build an exact control grid with role-specific substep counts."""
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
    """Build aligned single-width hinge joints for trace-budget calculations."""
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
    """Build an aligned joint actuator with the selected activation width."""
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
    """Select one decision reason by its stable reason code."""
    return next(reason for reason in reasons if reason.code is code)


@dataclass(slots=True)
class _SpyBackend:
    """Represent spy backend."""

    new_data_calls: int = 0
    forward_calls: int = 0
    step_calls: int = 0

    def new_data(self) -> SimpleNamespace:
        """Record allocation and return data with threadpool disabled."""
        self.new_data_calls += 1
        return SimpleNamespace(threadpool=0)

    def forward(self, _data: object) -> None:
        """Reject any forward call reached after a preexecution refusal."""
        self.forward_calls += 1
        raise AssertionError("forward must not run in the preexecution branch")

    def step(self, _data: object) -> None:
        """Reject any simulation step reached after a preexecution refusal."""
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
    """Hash a fixture label into deterministic identity evidence."""
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _closure(label: str) -> ModelClosureIdentity:
    """Build a one-member model closure with deterministic content identity."""
    member = ModelClosureMember("model.xml", len(label), _digest(f"member:{label}"))
    return ModelClosureIdentity("model.xml", 1, (member,))


def _environment(threadpool: EngineThreadpoolState) -> EnvironmentIdentity:
    """Build a finalized runtime environment with the selected threadpool state."""
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
    """Write a comparison configuration with caller-selected timing tokens."""
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
    """Remove modules and parent attributes while preserving exact teardown state."""
    for name in names:
        parent_name, _, attribute = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            monkeypatch.setattr(parent, attribute, getattr(parent, attribute, None), raising=False)
            monkeypatch.delattr(parent, attribute, raising=False)
        monkeypatch.setitem(sys.modules, name, sys.modules.get(name))
        monkeypatch.delitem(sys.modules, name, raising=False)


def _load_orchestrator(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Import the orchestrator against controlled workload and backend modules."""
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
        """Combine role observations using active-then-unknown precedence."""
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
    """Bind deterministic pair, workload, backend, and environment collaborators."""
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


def test_preexecution_orchestrator_refuses_before_replay_and_publishes_exact_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publish exact bounded evidence without replay when preexecution budgets fail."""
    orchestrator = _load_orchestrator(monkeypatch)
    joints = _hinge_joints(64)
    actuators = (_actuator(activation_width=1),)
    baseline_backend, candidate_backend = _configure_orchestrator(
        monkeypatch,
        orchestrator,
        joints=joints,
        actuators=actuators,
        control_intervals=65_535,
        timestep=0.0001,
    )
    replay_calls: list[str] = []

    def forbidden_replay(**kwargs: object) -> object:
        """Record and reject replay attempted after a preexecution refusal."""
        replay_calls.append(cast(str, kwargs["role"]))
        raise AssertionError("run_role_repeats must not be called")

    monkeypatch.setattr(orchestrator, "run_role_repeats", forbidden_replay)
    config_path = _write_config(
        tmp_path,
        output_name="result",
        step_token="0.0001",
        control_token="0.01",
    )
    result = orchestrator.compare_configuration_file(config_path)

    assert replay_calls == []
    assert baseline_backend.new_data_calls == candidate_backend.new_data_calls == 1
    assert baseline_backend.forward_calls == candidate_backend.forward_calls == 0
    assert baseline_backend.step_calls == candidate_backend.step_calls == 0
    assert result.comparison_json.is_file()
    assert result.comparison_markdown.is_file()
    _assert_preexecution_result(result)


def test_compare_keeps_sources_bound_until_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep both model-source bindings open and reverify them before publication."""
    orchestrator = _load_orchestrator(monkeypatch)
    _configure_orchestrator(
        monkeypatch,
        orchestrator,
        joints=_hinge_joints(64),
        actuators=(_actuator(activation_width=1),),
        control_intervals=65_535,
        timestep=0.0001,
    )
    original_open = orchestrator.open_live_model_pair
    real_publish = orchestrator.publish_results
    events: list[str] = []

    @contextmanager
    def retained_pair(**kwargs: object) -> Iterator[SimpleNamespace]:
        """Expose a verifier only while the model-pair context remains open."""
        events.append("opened")
        with original_open(**kwargs) as live:

            def verify_sources_unchanged() -> None:
                """Record both source checks while the binding is retained."""
                assert "closed" not in events
                events.extend(("verify:baseline", "verify:candidate"))

            live.verify_sources_unchanged = verify_sources_unchanged
            yield live
        events.append("closed")

    def observe_publication(output: object, *, json_bytes: bytes, markdown_text: str) -> Any:
        """Require publication to occur before the retained pair is released."""
        assert events == ["opened", "verify:baseline", "verify:candidate"]
        events.append("publish")
        return real_publish(output, json_bytes=json_bytes, markdown_text=markdown_text)

    monkeypatch.setattr(orchestrator, "open_live_model_pair", retained_pair)
    monkeypatch.setattr(orchestrator, "publish_results", observe_publication)
    config_path = _write_config(
        tmp_path,
        output_name="result",
        step_token="0.0001",
        control_token="0.01",
    )
    orchestrator.compare_configuration_file(config_path)

    assert events == [
        "opened",
        "verify:baseline",
        "verify:candidate",
        "publish",
        "verify:baseline",
        "verify:candidate",
        "closed",
    ]


def test_compare_refuses_output_inside_model_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse a canonical output below either declared model root before admission."""
    orchestrator = _load_orchestrator(monkeypatch)
    monkeypatch.setattr(orchestrator, "installed_distribution_sha256", lambda: _digest("tool"))
    baseline_root = tmp_path / "baseline"
    candidate_root = tmp_path / "candidate"
    baseline_root.mkdir()
    candidate_root.mkdir()
    output_alias = tmp_path / "baseline-alias"
    output_alias.symlink_to(baseline_root, target_is_directory=True)
    config_path = _write_config(
        tmp_path,
        output_name="baseline-alias/results",
        step_token="0.005",
        control_token="0.01",
    )

    with pytest.raises(orchestrator.ComparisonOperationError) as caught:
        orchestrator.compare_configuration_file(config_path)

    failure = caught.value.failure
    assert failure.reason.code is OperationalReasonCode.OUTPUT_PATH_INVALID
    assert failure.reason.evidence["issue"] == "output_inside_model_root"
    assert not (baseline_root / "results").exists()


def _assert_preexecution_result(result: Any) -> None:
    """Assert the exact published preexecution budget receipt and Markdown evidence."""
    primitive = json.loads(result.comparison_json.read_text(encoding="utf-8"))
    receipt = ComparisonReceipt.from_primitive(primitive)
    assert receipt == result.receipt
    assert receipt.status is ComparisonStatus.COVERAGE_INSUFFICIENT
    assert int(status_exit_code(receipt.status)) == 20
    assert receipt.first_crossing is None
    assert tuple(reason.code for reason in receipt.reasons) == (
        ReasonCode.INTERNAL_STEP_BUDGET_EXCEEDED,
        ReasonCode.TRACE_MEMORY_BUDGET_EXCEEDED,
    )
    _assert_preexecution_repeatability(receipt)
    _assert_preexecution_numerical_and_metrics(receipt)
    markdown = result.comparison_markdown.read_text(encoding="utf-8")
    assert "- Internal steps requested: `26214000`; maximum: `10000000`." in markdown
    assert "- Trace float64 bytes requested: `270532608`; maximum: `268435456`." in markdown


def _assert_preexecution_repeatability(receipt: ComparisonReceipt) -> None:
    """Assert both role summaries show replay was skipped before allocation."""
    assert receipt.repeatability.to_primitive() == {
        "schema": "metrifid.repeatability_evidence",
        "schema_version": 1,
        "baseline": {
            "captured_boundary_counts": [],
            "complete_repeats": 0,
            "evaluated": False,
            "reason": "preexecution_budget_exceeded",
            "repeat_count": 2,
            "signatures": [],
            "stable": None,
        },
        "candidate": {
            "captured_boundary_counts": [],
            "complete_repeats": 0,
            "evaluated": False,
            "reason": "preexecution_budget_exceeded",
            "repeat_count": 2,
            "signatures": [],
            "stable": None,
        },
    }


def _assert_preexecution_numerical_and_metrics(receipt: ComparisonReceipt) -> None:
    """Assert bounded numerical and metric evidence for a preexecution refusal."""
    expected_numerical_role = {
        "captured_boundary_count": 0,
        "complete": False,
        "error_logs": [],
        "evaluated": False,
        "expected_boundary_count": 65_536,
        "first_warning": None,
        "initial_state_preserved": None,
        "invalid_boundary_index": None,
        "invalid_kind": None,
        "reason": "preexecution_budget_exceeded",
    }
    assert receipt.numerical_evidence.to_primitive() == {
        "schema": "metrifid.numerical_evidence",
        "schema_version": 1,
        "baseline": expected_numerical_role,
        "candidate": expected_numerical_role,
    }
    assert receipt.metrics.to_primitive() == {
        "schema": "metrifid.metric_evidence",
        "schema_version": 1,
        "evaluated": False,
        "joints": [],
        "reason": "preexecution_budget_exceeded",
    }


def test_below_budget_orchestrator_still_enters_existing_replay_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enter both role replays when requested work remains within budgets."""
    orchestrator = _load_orchestrator(monkeypatch)
    joints = _hinge_joints(1)
    actuators = (_actuator(activation_width=0),)
    _configure_orchestrator(
        monkeypatch,
        orchestrator,
        joints=joints,
        actuators=actuators,
        control_intervals=2,
        timestep=0.005,
    )
    calls: list[str] = []

    def replay(**kwargs: object) -> SimpleNamespace:
        """Record each replay role and return its stable result token."""
        role = cast(str, kwargs["role"])
        calls.append(role)
        return SimpleNamespace(role=role)

    evaluation = _stable_replay_evaluation()
    monkeypatch.setattr(orchestrator, "run_role_repeats", replay)
    monkeypatch.setattr(orchestrator, "evaluate_role_pair", lambda **_kwargs: evaluation)
    config_path = _write_config(
        tmp_path,
        output_name="result",
        step_token="0.005",
        control_token="0.01",
    )
    result = orchestrator.compare_configuration_file(config_path)
    assert calls == ["baseline", "candidate"]
    assert result.receipt.status is ComparisonStatus.NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD
    assert result.receipt.reasons == ()


def _stable_replay_evaluation() -> SimpleNamespace:
    """Build successful repeatability, numerical, and empty metric replay evidence."""
    return SimpleNamespace(
        reasons=(),
        repeatability={
            "schema": "metrifid.repeatability_evidence",
            "schema_version": 1,
            "baseline": {
                "repeat_count": 2,
                "stable": True,
                "signatures": [_digest("baseline"), _digest("baseline")],
                "complete_repeats": 2,
                "captured_boundary_counts": [3, 3],
            },
            "candidate": {
                "repeat_count": 2,
                "stable": True,
                "signatures": [_digest("candidate"), _digest("candidate")],
                "complete_repeats": 2,
                "captured_boundary_counts": [3, 3],
            },
        },
        numerical_evidence={
            "schema": "metrifid.numerical_evidence",
            "schema_version": 1,
            "baseline": {"complete": True},
            "candidate": {"complete": True},
        },
        metrics={
            "schema": "metrifid.metric_evidence",
            "schema_version": 1,
            "evaluated": True,
            "joints": [],
        },
        first_crossing=None,
    )


def test_isolation_restores_a_pre_existing_module_object_identity() -> None:
    """Restore the exact module object that existed before dependency isolation."""
    import metrifid.compare._orchestrator as orchestrator

    original = sys.modules["metrifid.compare._orchestrator"]
    assert original is orchestrator
    with pytest.MonkeyPatch.context() as patch:
        _isolate_modules(patch, ("metrifid.compare._orchestrator",))
        assert "metrifid.compare._orchestrator" not in sys.modules
        sys.modules["metrifid.compare._orchestrator"] = types.ModuleType("stub")
    assert sys.modules["metrifid.compare._orchestrator"] is original


def test_isolation_restores_a_pre_existing_parent_package_attribute_identity() -> None:
    """Restore the exact parent-package attribute replaced during isolation."""
    import metrifid.compare as package
    import metrifid.compare._orchestrator  # noqa: F401  (ensures the attribute exists)

    original = package._orchestrator
    with pytest.MonkeyPatch.context() as patch:
        _isolate_modules(patch, ("metrifid.compare._orchestrator",))
        assert not hasattr(package, "_orchestrator")
        package._orchestrator = types.ModuleType("stub")
    assert package._orchestrator is original


def test_isolation_leaves_an_originally_absent_module_absent() -> None:
    """Remove a transient module that was absent before isolation."""
    name = "metrifid.compare._absent_for_this_test"
    assert name not in sys.modules
    with pytest.MonkeyPatch.context() as patch:
        _isolate_modules(patch, (name,))
        sys.modules[name] = types.ModuleType(name)
    assert name not in sys.modules, "an originally absent module must not survive teardown"


def test_isolation_leaves_an_originally_absent_parent_attribute_absent() -> None:
    """Remove a transient parent attribute that was originally absent."""
    import metrifid.compare as package

    assert not hasattr(package, "_absent_for_this_test")
    with pytest.MonkeyPatch.context() as patch:
        _isolate_modules(patch, ("metrifid.compare._absent_for_this_test",))
        package._absent_for_this_test = types.ModuleType("stub")
    assert not hasattr(package, "_absent_for_this_test")


def test_a_real_import_after_teardown_is_bound_to_the_real_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Resolve the real dependency after all stubbed import state is restored."""
    import metrifid.compare as package
    import metrifid.compare._model_pair as real_model_pair

    _ABSENT = object()
    attribute_before = getattr(package, "_orchestrator", _ABSENT)
    module_before = sys.modules.get("metrifid.compare._orchestrator")

    stubbed = _load_orchestrator(monkeypatch)
    assert sys.modules["metrifid.compare._model_pair"] is not real_model_pair
    monkeypatch.undo()

    assert sys.modules["metrifid.compare._model_pair"] is real_model_pair
    assert sys.modules.get("metrifid.compare._orchestrator") is module_before
    assert getattr(package, "_orchestrator", _ABSENT) is attribute_before
    assert getattr(package, "_orchestrator", _ABSENT) is not stubbed

    # A real import after teardown must reach the real dependency, never the stub.
    from metrifid.compare._orchestrator import open_live_model_pair

    assert open_live_model_pair is real_model_pair.open_live_model_pair
    assert sys.modules["metrifid.compare._orchestrator"] is not stubbed
