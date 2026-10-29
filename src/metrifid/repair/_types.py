"""In-memory measurements and outcomes for bounded controller repair.

These Python records are not an on-disk receipt format. Qualification describes
complete paired simulation under the retained supported task and runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ._physical_types import Measurement, Snapshot
from ._request import RepairLimits, RepairRequest


@dataclass
class ExecutionCosts:
    """Actual completed work, with unknown native calls represented separately."""

    attempts: int = 0
    integration_steps: int | None = 0
    known_completed_steps: int = 0
    unknown_step_calls: int = 0
    cpu_seconds: float = 0.0
    limits: RepairLimits | None = None
    search_cpu_fraction: float | None = None


@dataclass
class PairedMeasurement:
    """One controller measured at the nominal and finer timesteps."""

    parameters_normalized: tuple[float, ...]
    nominal: Measurement | None = None
    finer: Measurement | None = None
    objective: float | None = None
    classification: str = "INCOMPLETE"
    reason: str = "NOT_EVALUATED"
    complete: bool = False
    qualified: bool = False
    known_completed_steps: int = 0
    unknown_step_calls: int = 0


@dataclass
class EngineResult:
    """Retained repair inputs, evidence, selected candidate and truthful outcome.

    A selected controller can be useful without being qualified. Its screening
    evidence and confirmation are distinct; only complete paired confirmation
    can produce ``REPAIRED`` or ``ALREADY_QUALIFIED``.
    """

    status: str
    exit_code: int
    reason: str
    request: RepairRequest | None = None
    original_parameters: tuple[float, ...] | None = None
    selected_parameters: tuple[float, ...] | None = None
    profile: dict[str, object] | None = None
    nominal_xml: str | None = None
    finer_xml: str | None = None
    timestep_identities: dict[str, object] | None = None
    reset: Snapshot | None = None
    runtime: dict[str, str] = field(default_factory=dict)
    initialization: Measurement | None = None
    initial_screening: PairedMeasurement | None = None
    selected_screening: PairedMeasurement | None = None
    screening_measurements: list[PairedMeasurement] = field(default_factory=list)
    confirmation: PairedMeasurement | None = None
    confirmation_complete: bool = False
    qualified: bool = False
    costs: ExecutionCosts = field(default_factory=ExecutionCosts)
    diagnostic: str | None = None


@dataclass
class ReplayResult:
    """Fresh paired evaluation of a retained controller, without optimization."""

    status: str
    exit_code: int
    reason: str
    request: RepairRequest | None = None
    selected_parameters: tuple[float, ...] | None = None
    measurement: PairedMeasurement | None = None
    qualified: bool = False
    runtime: dict[str, str] = field(default_factory=dict)
    costs: ExecutionCosts = field(default_factory=ExecutionCosts)
    diagnostic: str | None = None
