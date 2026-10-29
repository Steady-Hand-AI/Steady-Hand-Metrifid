"""Bounded cyclic coordinate repair and independently admitted replay."""

from __future__ import annotations

import importlib
import math
import platform
import sys
import time
from collections.abc import Callable
from typing import TYPE_CHECKING

from . import _profile
from ._physical_types import Measurement, snapshot_well_formed, validate_snapshot
from ._request import (
    RepairInputError,
    RepairLimits,
    RepairRequest,
    validate_request,
    validate_selected_parameters,
)
from ._types import EngineResult, ExecutionCosts, PairedMeasurement, ReplayResult

if TYPE_CHECKING:
    from ._physics import Backend

_SUPPORTED_MUJOCO_VERSIONS = ("3.16.0", "3.17.0")


class _Interrupted(Exception):
    """A cooperative limit or unknown measurement prevents completion."""


class _SearchEnded(Exception):
    """A finite bound was reached between fully accounted evaluations."""


class _RuntimeRefused(Exception):
    """The installed runtime is outside the supported execution lane."""


class _Budget:
    """One invocation clock and counters shared by setup and all evaluations."""

    def __init__(self, limits: RepairLimits, started: float, *, search: bool) -> None:
        self.limits = limits
        self.started = started
        self.search = search
        self.attempts = 0
        self.completed = 0
        self.unknown = 0

    def check(self) -> None:
        """Refuse further work at the applicable cooperative CPU ceiling."""
        fraction = 0.75 if self.search else 1.0
        if time.process_time() - self.started >= self.limits.max_cpu_seconds * fraction:
            raise _Interrupted("CPU_LIMIT")
        if self.unknown:
            raise _Interrupted("UNKNOWN_INTEGRATION_WORK")

    def step(self, operation: Callable[[], None]) -> None:
        """Count each completed integration call without inferring from time."""
        self.check()
        if self.completed >= self.limits.max_integration_steps:
            raise _Interrupted("INTEGRATION_STEP_LIMIT_DURING_EVALUATION")
        try:
            operation()
        except Exception:
            self.unknown += 1
            raise
        self.completed += 1
        self.check()

    def costs(self, *, search_fraction: float | None) -> ExecutionCosts:
        """Capture actual work, retaining an unknown total after a failed call."""
        return ExecutionCosts(
            attempts=self.attempts,
            integration_steps=self.completed if not self.unknown else None,
            known_completed_steps=self.completed,
            unknown_step_calls=self.unknown,
            cpu_seconds=max(0.0, time.process_time() - self.started),
            limits=self.limits,
            search_cpu_fraction=search_fraction,
        )


def _runtime() -> dict[str, str]:
    """Admit the supported runtime before any physical model is created."""
    versions = {
        "implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
    }
    python_version = sys.version_info[:2]
    if (
        versions["implementation"] != "CPython"
        or python_version != (3, 13)
        or versions["system"] != "Linux"
        or versions["machine"] != "x86_64"
    ):
        raise _RuntimeRefused("Requires Linux x86_64 and CPython 3.13")
    for name, supported in (("mujoco", _SUPPORTED_MUJOCO_VERSIONS), ("numpy", ("2.5.4",))):
        expected = " or ".join(supported)
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            raise _RuntimeRefused(
                f"Required runtime {name} {expected} is unavailable ({type(exc).__name__})"
            ) from exc
        actual = str(getattr(module, "__version__", "unknown"))
        versions[name] = actual
        if actual not in supported:
            raise _RuntimeRefused(f"Requires {name} {expected}; found {actual}")
    return versions


def _create_backend(request: RepairRequest, budget: _Budget) -> Backend:
    """Load native evaluation only after request and runtime admission."""
    from ._physics import create_backend

    return create_backend(request, step=budget.step, check=budget.check)


def _complete_measurement(measured: Measurement, dt: float) -> bool:
    expected = round(_profile.EPISODE_SECONDS / dt)
    hold = round(_profile.HOLD_SECONDS / dt) + 1
    counts = (
        measured.completed_steps,
        measured.expected_steps,
        measured.hold_samples,
        measured.expected_hold_samples,
    )
    return (
        measured.classification == "COMPLETE"
        and all(type(value) is int for value in counts)
        and measured.completed_steps == measured.expected_steps == expected
        and measured.hold_samples == measured.expected_hold_samples == hold
        and type(measured.dt_s) is float
        and measured.dt_s == dt
        and not measured.hard_failures
    )


def _objective(nominal: Measurement, finer: Measurement) -> float | None:
    """Return the exact worst of the eight measured normalized hold violations."""
    limits = _profile.HOLD_LIMITS
    values: list[float] = []
    for measurement, dt in zip(
        (nominal, finer), (_profile.NOMINAL_DT_S, _profile.FINER_DT_S), strict=True
    ):
        maxima = measurement.hold_maxima
        if not _complete_measurement(measurement, dt) or maxima is None or len(maxima) != 4:
            return None
        for maximum, limit in zip(maxima, limits, strict=True):
            if type(maximum) not in (int, float) or not math.isfinite(maximum) or maximum < 0:
                return None
            values.append(max(0.0, maximum / limit - 1.0))
    return max(values)


def _classify_pair(pair: PairedMeasurement) -> None:
    nominal, finer = pair.nominal, pair.finer
    obtained = [measurement for measurement in (nominal, finer) if measurement is not None]
    unknown = next((m for m in obtained if m.classification == "INCOMPLETE"), None)
    if unknown is not None:
        pair.reason = unknown.reason
    elif any(m.classification == "HARD_REJECTION" for m in obtained):
        pair.classification = "HARD_REJECTION"
        pair.reason = "PHYSICAL_HARD_REJECTION"
        pair.complete = True
    elif nominal is not None and finer is not None:
        pair.objective = _objective(nominal, finer)
        if pair.objective is not None:
            pair.classification = "COMPLETE"
            pair.reason = "COMPLETE_FINITE_HARD_VALID_PAIR"
            pair.complete = True
            pair.qualified = pair.objective == 0.0 and nominal.qualified and finer.qualified
        else:
            pair.reason = "INCOMPLETE_OR_INVALID_PAIRED_MEASUREMENT"
    else:
        pair.reason = "MISSING_TIMESTEP_MEASUREMENT"


def _evaluate_pair(backend: Backend, pair: PairedMeasurement, budget: _Budget) -> None:
    before_steps, before_unknown = budget.completed, budget.unknown
    try:
        pair.nominal = backend.evaluate(
            pair.parameters_normalized, finer=False, check=budget.check, step=budget.step
        )
        budget.check()
        if pair.nominal.classification == "COMPLETE":
            pair.finer = backend.evaluate(
                pair.parameters_normalized, finer=True, check=budget.check, step=budget.step
            )
            budget.check()
        _classify_pair(pair)
    except _Interrupted as exc:
        _retain_partial(pair, exc)
        pair.reason = str(exc)
        raise
    except Exception as exc:
        _retain_partial(pair, exc)
        pair.reason = f"EVALUATOR_EXCEPTION: {type(exc).__name__}"
        raise
    finally:
        pair.known_completed_steps = budget.completed - before_steps
        pair.unknown_step_calls = budget.unknown - before_unknown


def _retain_partial(pair: PairedMeasurement, exc: Exception) -> None:
    measured = getattr(exc, "measurement", None)
    if isinstance(measured, Measurement):
        if pair.nominal is None:
            pair.nominal = measured
        elif pair.finer is None:
            pair.finer = measured


def _pair_steps() -> int:
    return _profile.pair_steps()


def _screen(
    result: EngineResult, backend: Backend, budget: _Budget, parameters: tuple[float, ...]
) -> PairedMeasurement:
    budget.check()
    if budget.attempts >= budget.limits.max_attempts:
        raise _SearchEnded("ATTEMPT_LIMIT")
    in_box = all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in parameters)
    if in_box and budget.completed + 2 * _pair_steps() > budget.limits.max_integration_steps:
        raise _SearchEnded("INTEGRATION_STEP_RESERVE")
    budget.attempts += 1
    pair = PairedMeasurement(parameters)
    result.screening_measurements.append(pair)
    if result.initial_screening is None:
        result.initial_screening = pair
    if not in_box:
        pair.classification = "OUT_OF_BOX"
        pair.reason = "OUT_OF_BOX"
        pair.complete = True
        return pair
    _evaluate_pair(backend, pair, budget)
    if pair.classification == "INCOMPLETE":
        raise _Interrupted(pair.reason)
    return pair


def _select(result: EngineResult, pair: PairedMeasurement) -> None:
    result.selected_parameters = pair.parameters_normalized
    result.selected_screening = pair


def _confirm(result: EngineResult, backend: Backend, budget: _Budget) -> None:
    parameters = result.selected_parameters
    assert parameters is not None
    budget.search = False
    budget.check()
    if budget.completed + _pair_steps() > budget.limits.max_integration_steps:
        raise _Interrupted("CONFIRMATION_STEP_LIMIT")
    pair = PairedMeasurement(parameters)
    result.confirmation = pair
    _evaluate_pair(backend, pair, budget)
    result.confirmation_complete = pair.complete
    if pair.classification == "INCOMPLETE":
        raise _Interrupted(pair.reason)
    if pair.qualified:
        result.qualified = True
        result.status = (
            "ALREADY_QUALIFIED" if parameters == result.original_parameters else "REPAIRED"
        )
        result.exit_code = 0
        result.reason = "COMPLETE_PAIRED_CONFIRMATION"
    else:
        result.status = "NO_QUALIFIED_POLICY_FOUND"
        result.exit_code = 10
        result.reason = "CONFIRMATION_DID_NOT_QUALIFY"


def _poll_coordinate(
    result: EngineResult,
    backend: Backend,
    budget: _Budget,
    coordinate: int,
    step: float,
    preferred: list[int],
) -> bool:
    best = result.selected_screening
    assert best is not None
    assert best.objective is not None
    for sign in (preferred[coordinate], -preferred[coordinate]):
        trial = list(best.parameters_normalized)
        trial[coordinate] += sign * step
        measured = _screen(result, backend, budget, tuple(trial))
        if measured.objective is not None and (
            measured.objective == 0.0 or measured.objective < best.objective - 1e-10
        ):
            _select(result, measured)
            preferred[coordinate] = sign
            return True
    return False


def _search(result: EngineResult, backend: Backend, budget: _Budget) -> None:
    assert result.original_parameters is not None
    first = _screen(result, backend, budget, result.original_parameters)
    if first.classification == "HARD_REJECTION":
        raise _SearchEnded("INITIAL_POLICY_NOT_SEARCHABLE")
    _select(result, first)
    if first.objective == 0.0:
        _confirm(result, backend, budget)
        return
    step, preferred = 0.01, [1] * 15
    for _ in range(100):
        moved = False
        for coordinate in range(15):
            accepted = _poll_coordinate(result, backend, budget, coordinate, step, preferred)
            moved = moved or accepted
            if result.selected_screening is not None and result.selected_screening.objective == 0.0:
                _confirm(result, backend, budget)
                return
        if not moved:
            step *= 0.5
            if step < 0.00125:
                raise _SearchEnded("STEP_FLOOR_REACHED")
    raise _SearchEnded("SWEEP_LIMIT")


def _retain_backend(result: EngineResult, backend: Backend) -> None:
    result.profile = backend.profile
    result.nominal_xml = backend.nominal_xml
    result.finer_xml = backend.finer_xml
    result.timestep_identities = backend.identities
    result.reset = backend.snapshot
    result.initialization = backend.initialization


def _initialization(backend: Backend) -> None:
    if backend.initialization.classification == "HARD_REJECTION":
        raise _SearchEnded("INITIAL_STATE_NOT_ADMISSIBLE")
    if backend.initialization.classification != "COMPLETE":
        raise _Interrupted(backend.initialization.reason)


def _diagnostic(exc: Exception) -> str:
    return f"{type(exc).__name__}: {str(exc)[:240]}"


def _finish(
    result: EngineResult | ReplayResult,
    budget: _Budget | None,
    started: float,
    *,
    search_fraction: float | None,
) -> None:
    if budget is None:
        result.costs.cpu_seconds = max(0.0, time.process_time() - started)
        return
    if result.exit_code in (0, 10):
        try:
            budget.check()
        except _Interrupted as exc:
            result.status, result.exit_code, result.reason = "INCOMPLETE", 20, str(exc)
            result.qualified = False
    result.costs = budget.costs(search_fraction=search_fraction)


def repair_controller(request: RepairRequest) -> EngineResult:
    """Repair one admitted controller with bounded cyclic coordinate polling.

    Native execution requires Linux x86_64, CPython 3.13, MuJoCo 3.16.0 or
    3.17.0 and NumPy 2.5.4. One process-CPU clock includes native imports, setup and every
    evaluation. Search stops at 75% of the cap; confirmation uses the full cap.
    Expected input/runtime refusals and execution outcomes are returned as data.
    KeyboardInterrupt and SystemExit propagate to the caller.
    """
    started = time.process_time()
    result = EngineResult("REFUSED", 64, "INVALID_REQUEST")
    budget: _Budget | None = None
    try:
        admitted = validate_request(request)
        result.request = admitted
        result.original_parameters = admitted.controller.parameters_normalized
        budget = _Budget(admitted.limits, started, search=True)
        result.runtime = _runtime()
        budget.check()
        backend = _create_backend(admitted, budget)
        _retain_backend(result, backend)
        budget.check()
        _initialization(backend)
        _search(result, backend, budget)
    except RepairInputError as exc:
        result.reason, result.diagnostic = exc.reason, _diagnostic(exc)
    except _RuntimeRefused as exc:
        result.reason, result.diagnostic = "UNSUPPORTED_RUNTIME", str(exc)
    except _SearchEnded as exc:
        result.status, result.exit_code, result.reason = "NO_QUALIFIED_POLICY_FOUND", 10, str(exc)
    except _Interrupted as exc:
        result.status, result.exit_code, result.reason = "INCOMPLETE", 20, str(exc)
    except Exception as exc:
        result.status, result.exit_code, result.reason = "ERROR", 70, "EVALUATOR_EXCEPTION"
        result.diagnostic = _diagnostic(exc)
    finally:
        _finish(result, budget, started, search_fraction=0.75)
    return result


def _replay_inputs(source: EngineResult) -> tuple[RepairRequest, tuple[float, ...]]:
    if not isinstance(source, EngineResult):
        raise RepairInputError("Expected an EngineResult", reason="INVALID_REQUEST")
    request = validate_request(source.request)
    parameters = validate_selected_parameters(source.selected_parameters)
    if source.original_parameters != request.controller.parameters_normalized:
        raise RepairInputError(
            "Original controller binding differs from request", reason="INVALID_REQUEST"
        )
    for measured in (source.selected_screening, source.confirmation):
        if measured is not None and measured.parameters_normalized != parameters:
            raise RepairInputError(
                "Selected controller bindings disagree", reason="INVALID_REQUEST"
            )
    profile, nominal, finer, identities = _profile.generate_expected(request)
    if (
        source.profile != profile
        or source.nominal_xml != nominal
        or source.finer_xml != finer
        or source.timestep_identities != identities
    ):
        raise RepairInputError(
            "Unsupported or inconsistent retained profile/scenes", reason="INVALID_REQUEST"
        )
    scene_hash = identities["nominal_scene_sha256"]
    assert isinstance(scene_hash, str)
    if not snapshot_well_formed(source.reset, scene_hash):
        raise RepairInputError("Invalid retained physical reset", reason="INVALID_REQUEST")
    return request, parameters


def _admit_replay(source: EngineResult) -> tuple[RepairRequest, tuple[float, ...]]:
    try:
        return _replay_inputs(source)
    except RepairInputError:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise RepairInputError("Malformed retained repair data") from exc


def _replay_evaluate(
    result: ReplayResult, source: EngineResult, request: RepairRequest, budget: _Budget
) -> None:
    backend = _create_backend(request, budget)
    budget.check()
    if backend.initialization.classification != "COMPLETE":
        if backend.initialization.classification == "INCOMPLETE":
            raise _Interrupted(backend.initialization.reason)
        result.reason = "RESET_MISMATCH"
        return
    if not validate_snapshot(source.reset, backend.snapshot):
        result.reason = "RESET_MISMATCH"
        return
    if budget.completed + _pair_steps() > budget.limits.max_integration_steps:
        raise _Interrupted("REPLAY_STEP_LIMIT")
    assert result.selected_parameters is not None
    pair = PairedMeasurement(result.selected_parameters)
    result.measurement = pair
    _evaluate_pair(backend, pair, budget)
    if pair.classification == "INCOMPLETE":
        raise _Interrupted(pair.reason)
    result.qualified = pair.qualified
    result.status = "QUALIFIED" if pair.qualified else "NOT_QUALIFIED"
    result.exit_code = 0 if pair.qualified else 10
    result.reason = "COMPLETE_PAIRED_EVALUATION" if pair.qualified else pair.reason


def replay_controller(source: EngineResult) -> ReplayResult:
    """Re-admit retained inputs and independently evaluate the selected vector.

    Regenerated scene definitions and a freshly settled physical reset must
    match the retained record. Valid self-consistent replacements are evaluated
    as the supplied task/controller; this record is not an authenticity proof.
    Replay receives a fresh full invocation budget and performs no search.
    """
    started = time.process_time()
    result = ReplayResult("REFUSED", 64, "INVALID_RETAINED_RESULT")
    budget: _Budget | None = None
    try:
        request, parameters = _admit_replay(source)
        result.request, result.selected_parameters = request, parameters
        budget = _Budget(request.limits, started, search=False)
        result.runtime = _runtime()
        budget.check()
        _replay_evaluate(result, source, request, budget)
    except RepairInputError as exc:
        result.reason, result.diagnostic = "INVALID_RETAINED_RESULT", _diagnostic(exc)
    except _RuntimeRefused as exc:
        result.reason, result.diagnostic = "UNSUPPORTED_RUNTIME", str(exc)
    except _Interrupted as exc:
        result.status, result.exit_code, result.reason = "INCOMPLETE", 20, str(exc)
    except Exception as exc:
        result.status, result.exit_code, result.reason = "ERROR", 70, "EVALUATOR_EXCEPTION"
        result.diagnostic = _diagnostic(exc)
    finally:
        _finish(result, budget, started, search_fraction=None)
    return result
