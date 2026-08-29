"""Product-specific tests for the fixed Runtime Review execution journey."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from metrifid.runtime_review import _execution as execution
from metrifid.runtime_review._status import RuntimeReviewStatus

_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_SHA_E = "e" * 64
_SHA_F = "f" * 64


class _RetainedProcess:
    """Minimal successful retained-process projection used by orchestration tests."""

    def to_primitive(self) -> dict[str, object]:
        """Return the process fields required by a completed execution record."""
        return {
            "command_locator": "command.json",
            "command_sha256": _SHA_A,
            "stdout_locator": "stdout.txt",
            "stdout_sha256": _SHA_B,
            "stderr_locator": "stderr.txt",
            "stderr_sha256": _SHA_C,
            "exit_code_locator": "exit_code.txt",
            "exit_code_sha256": _SHA_D,
            "exit_code": 0,
            "no_exit_status": None,
        }


class _ProcessOutput:
    """Test-local output owner recording the semantic order of process attempts."""

    def __init__(self, root: Path, events: list[tuple[object, ...]]) -> None:
        """Retain one root and mutable event sink for assertions."""
        self.root = root
        self.events = events

    def new_profile_identity_path(self, role: str) -> Path:
        """Return the canonical absent profile identity locator."""
        self.events.append(("identity-path", role))
        return self.root / "profile_identities" / f"{role}.json"

    def write_profile_preflight(self, role: str, **observation: object) -> _RetainedProcess:
        """Record one retained preflight observation."""
        self.events.append(("retain-preflight", role, observation["exit_code"]))
        return _RetainedProcess()

    def new_evidence_cell_path(self, role: str, step_dt: str, repeat_id: int) -> Path:
        """Return one canonical evidence-cell path while retaining request order."""
        self.events.append(("cell-path", role, step_dt, repeat_id))
        token = {"0.004": "0p004", "0.002": "0p002", "0.001": "0p001"}[step_dt]
        return self.root / "captured_evidence" / role / token / f"repeat_{repeat_id}"

    def write_evidence_attempt(
        self, role: str, step_dt: str, repeat_id: int, **observation: object
    ) -> _RetainedProcess:
        """Record one retained evidence-process observation."""
        self.events.append(("retain-cell", role, step_dt, repeat_id, observation["exit_code"]))
        return _RetainedProcess()


class _PartialRunOutput(_ProcessOutput):
    """Capture retained failure truth and any forbidden post-cell publication stage."""

    def __init__(self, root: Path, events: list[tuple[object, ...]]) -> None:
        """Initialize retained observations and the generated-config output locator."""
        super().__init__(root, events)
        self.generated_runtime_review_config = root / "generated_runtime_review_config.json"
        self.retained_observations: list[dict[str, object]] = []

    def write_evidence_attempt(
        self, role: str, step_dt: str, repeat_id: int, **observation: object
    ) -> _RetainedProcess:
        """Retain the exact first process observation before orchestration refuses it."""
        self.retained_observations.append(dict(observation))
        return super().write_evidence_attempt(
            role,
            step_dt,
            repeat_id,
            **observation,
        )

    def write_generated_runtime_review_configuration(self, raw: bytes) -> SimpleNamespace:
        """Record an invalid attempt to publish a generated decision configuration."""
        self.events.append(("generated-config", raw))
        return SimpleNamespace(sha256=_SHA_A)

    def publish_completed_run_record(self, record: object) -> SimpleNamespace:
        """Record an invalid attempt to publish a completed run record."""
        self.events.append(("completed-record", record))
        return SimpleNamespace(path=self.root / "runtime_review_run.json", run_sha256=_SHA_B)


def _interpreter(path: Path) -> SimpleNamespace:
    """Create the interpreter projection consumed by orchestration helpers."""
    return SimpleNamespace(
        lexical_path=path,
        resolved_path=path,
        resolved_sha256=_SHA_A,
        to_primitive=lambda: {
            "lexical_path": path.as_posix(),
            "resolved_path": path.as_posix(),
            "resolved_sha256": _SHA_A,
        },
    )


def _admitted(tmp_path: Path) -> SimpleNamespace:
    """Create one admitted run projection with two explicit interpreter paths."""
    baseline = _interpreter(tmp_path / "baseline" / "bin" / "python")
    candidate = _interpreter(tmp_path / "candidate" / "bin" / "python")
    return SimpleNamespace(
        config=SimpleNamespace(fixture_id="smooth_pendulum"),
        baseline_interpreter=baseline,
        candidate_interpreter=candidate,
        manifest_path=tmp_path / "manifest.json",
        manifest_identity=SimpleNamespace(sha256=_SHA_B),
    )


def _resource(path: Path, digest: str = _SHA_C) -> execution._ResourceIdentity:
    """Construct one installed-resource projection without filesystem access."""
    return execution._ResourceIdentity(path.absolute(), digest)


def _identity(profile_role: str, interpreter: SimpleNamespace) -> dict[str, object]:
    """Construct the minimal admitted v2 identity used after patched validation."""
    return {
        "profile_role": profile_role,
        "package_version": "3.12.0",
        "native_version": "3.12.0",
        "native_version_integer": 3_012_000,
        "support_tier": "ADMITTED_CAPABILITY_COMPATIBLE_PROFILE",
        "profile_identity_sha256": _SHA_D,
        "sentinel": {
            "status": "PASS",
            "sentinel_identity_sha256": _SHA_C,
        },
        "python": {
            "executable": interpreter.lexical_path.as_posix(),
            "resolved_executable": interpreter.resolved_path.as_posix(),
            "resolved_executable_sha256": interpreter.resolved_sha256,
        },
        "native_smoke": {
            "fixture_id": "smooth_pendulum",
            "manifest_raw_sha256": _SHA_B,
        },
    }


def _result_identities() -> dict[str, object]:
    """Return the six decision-bearing identities from a first worker result."""
    return {
        "subject": {
            "fixture_id": "smooth_pendulum",
            "source_closure": {"closure_sha256": _SHA_A},
            "fixture_manifest_sha256": _SHA_B,
        },
        "workload": {
            "semantic_sha256": _SHA_C,
            "initial_state": {"semantic_sha256": _SHA_D},
            "action_program": {"semantic_sha256": _SHA_E},
        },
    }


def _successful_observation() -> execution._CompletedProcessObservation:
    """Return one successful one-shot process outcome."""
    return execution._CompletedProcessObservation(b"", b"", 0, None)


def _preflight_pair(admitted: SimpleNamespace, tmp_path: Path) -> tuple[SimpleNamespace, ...]:
    """Return admitted baseline and candidate preflights for cell-loop isolation."""
    return (
        SimpleNamespace(
            role="baseline",
            identity_path=tmp_path / "baseline.json",
            identity_file_sha256=_SHA_E,
            identity=_identity("baseline", admitted.baseline_interpreter),
        ),
        SimpleNamespace(
            role="candidate",
            identity_path=tmp_path / "candidate.json",
            identity_file_sha256=_SHA_F,
            identity=_identity("candidate", admitted.candidate_interpreter),
        ),
    )


def _forbidden_stage(events: list[tuple[object, ...]], stage: str) -> object:
    """Fail immediately while recording any post-cell stage reached by a partial run."""
    events.append((stage,))
    raise AssertionError(f"partial run reached forbidden stage: {stage}")


def test_profile_preflights_complete_before_evidence_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run baseline then candidate preflights and admit both before returning."""
    events: list[tuple[object, ...]] = []
    admitted = _admitted(tmp_path)
    output = _ProcessOutput(tmp_path / "run", events)
    worker = _resource(tmp_path / "worker", execution._FROZEN_WORKER_SHA256)
    collector = _resource(tmp_path / "collector")
    launched: list[tuple[str, ...]] = []

    monkeypatch.setattr(execution, "_recheck_before_process", lambda *args: None)
    monkeypatch.setattr(execution, "canonical_json_bytes", lambda value: b"identity")

    def execute_once(
        argv: tuple[str, ...], environment: dict[str, str], working_directory: Path
    ) -> execution._CompletedProcessObservation:
        """Capture each exact collector argv before returning success."""
        launched.append(argv)
        return _successful_observation()

    monkeypatch.setattr(execution, "_execute_process", execute_once)
    monkeypatch.setattr(execution, "read_bounded_regular_file", lambda *args: b"identity\n")

    def load_identity(path: Path, *, expected_profile_role: str, **kwargs: object) -> object:
        """Return the matching identity and retain strict load order."""
        events.append(("load-identity", expected_profile_role))
        interpreter = (
            admitted.baseline_interpreter
            if expected_profile_role == "baseline"
            else admitted.candidate_interpreter
        )
        return _identity(expected_profile_role, interpreter)

    monkeypatch.setattr(execution, "load_native_profile_identity_v2", load_identity)
    monkeypatch.setattr(execution, "_bind_profile_identity_to_run", lambda *args: None)
    monkeypatch.setattr(
        execution,
        "require_compatible_profile_identities_v2",
        lambda baseline, candidate: events.append(("pair-admitted",)),
    )

    completed = execution._run_profile_preflights(
        admitted,
        output,
        worker,
        collector,  # type: ignore[arg-type]
    )

    assert [item.role for item in completed] == ["baseline", "candidate"]
    assert [event for event in events if event[0] == "load-identity"] == [
        ("load-identity", "baseline"),
        ("load-identity", "candidate"),
    ]
    assert launched == [
        (
            admitted.baseline_interpreter.lexical_path.as_posix(),
            collector.path.as_posix(),
            "--worker",
            worker.path.as_posix(),
            "--manifest",
            admitted.manifest_path.as_posix(),
            "--fixture-id",
            "smooth_pendulum",
            "--profile-role",
            "baseline",
            "--output",
            (output.root / "profile_identities" / "baseline.json").as_posix(),
        ),
        (
            admitted.candidate_interpreter.lexical_path.as_posix(),
            collector.path.as_posix(),
            "--worker",
            worker.path.as_posix(),
            "--manifest",
            admitted.manifest_path.as_posix(),
            "--fixture-id",
            "smooth_pendulum",
            "--profile-role",
            "candidate",
            "--output",
            (output.root / "profile_identities" / "candidate.json").as_posix(),
        ),
    ]
    assert events[-1] == ("pair-admitted",)


def test_swapped_profile_interpreters_are_refused_before_evidence_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse a baseline launcher measured as the candidate before any evidence cell."""
    events: list[tuple[object, ...]] = []
    admitted = _admitted(tmp_path)
    output = _ProcessOutput(tmp_path / "run", events)
    worker = _resource(tmp_path / "worker", execution._FROZEN_WORKER_SHA256)
    collector = _resource(tmp_path / "collector")
    monkeypatch.setattr(execution, "_recheck_before_process", lambda *args: None)
    monkeypatch.setattr(execution, "_execute_process", lambda *args: _successful_observation())
    monkeypatch.setattr(execution, "read_bounded_regular_file", lambda *args: b"identity\n")

    def refuse_swapped_role(*args: object, **kwargs: object) -> object:
        """Model the strict collector loader rejecting the measured candidate role."""
        raise execution.ProfileIdentityRefusal("profile identity has the wrong runtime role")

    monkeypatch.setattr(execution, "load_native_profile_identity_v2", refuse_swapped_role)

    with pytest.raises(execution._ExecutionRefusal):
        execution._run_profile_preflights(
            admitted,
            output,
            worker,
            collector,  # type: ignore[arg-type]
        )

    assert [event for event in events if event[0] == "retain-preflight"] == [
        ("retain-preflight", "baseline", 0)
    ]
    assert not [event for event in events if event[0] == "cell-path"]


def test_failed_complete_state_sentinel_starts_zero_evidence_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain one failed profile sentinel and stop before every cross-profile cell."""
    events: list[tuple[object, ...]] = []
    admitted = _admitted(tmp_path)
    output = _PartialRunOutput(tmp_path / "run", events)
    worker = _resource(tmp_path / "worker", execution._FROZEN_WORKER_SHA256)
    collector = _resource(tmp_path / "collector")
    identity = _identity("baseline", admitted.baseline_interpreter)
    identity["sentinel"] = {
        "status": "FAIL",
        "sentinel_identity_sha256": _SHA_C,
    }

    monkeypatch.setattr(execution, "_recheck_before_process", lambda *args: None)
    monkeypatch.setattr(execution, "_execute_process", lambda *args: _successful_observation())
    monkeypatch.setattr(execution, "read_bounded_regular_file", lambda *args: b"identity\n")
    monkeypatch.setattr(execution, "canonical_json_bytes", lambda value: b"identity")
    monkeypatch.setattr(
        execution,
        "load_native_profile_identity_v2",
        lambda *args, **kwargs: identity,
    )
    monkeypatch.setattr(execution, "_bind_profile_identity_to_run", lambda *args: None)
    monkeypatch.setattr(
        execution,
        "_run_evidence_cells",
        lambda *args: _forbidden_stage(events, "evidence-cells"),
    )

    with pytest.raises(execution._ExecutionRefusal) as caught:
        execution._execute_owned_run(
            admitted,
            output,  # type: ignore[arg-type]
            worker,
            collector,  # type: ignore[arg-type]
        )

    assert "sentinel did not pass" in str(caught.value)
    assert [event for event in events if event[0] == "retain-preflight"] == [
        ("retain-preflight", "baseline", 0)
    ]
    assert not [event for event in events if event[0] in {"cell-path", "evidence-cells"}]


def test_evidence_cells_run_once_in_canonical_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Execute every role, step, and repeat exactly once in frozen order."""
    events: list[tuple[object, ...]] = []
    admitted = _admitted(tmp_path)
    output = _ProcessOutput(tmp_path / "run", events)
    worker = _resource(tmp_path / "worker", execution._FROZEN_WORKER_SHA256)
    baseline_identity = _identity("baseline", admitted.baseline_interpreter)
    candidate_identity = _identity("candidate", admitted.candidate_interpreter)
    preflights = (
        SimpleNamespace(
            role="baseline",
            identity_path=tmp_path / "baseline.json",
            identity_file_sha256=_SHA_E,
            identity=baseline_identity,
        ),
        SimpleNamespace(
            role="candidate",
            identity_path=tmp_path / "candidate.json",
            identity_file_sha256=_SHA_F,
            identity=candidate_identity,
        ),
    )
    config = execution.RuntimeReviewConfigV2.from_primitive(
        execution._generated_review_configuration_primitive(
            _result_identities(),
            preflights,  # type: ignore[arg-type]
        )
    )
    provisional = SimpleNamespace(config=config)
    admitted_cell = SimpleNamespace(
        fixture_id="smooth_pendulum",
        manifest_raw_sha256=_SHA_B,
        member_sha256={"result.json": _SHA_A, "CHECKSUMS.sha256": _SHA_B},
        runtime={"runtime_identity_sha256": _SHA_F},
    )
    launched: list[tuple[str, ...]] = []

    monkeypatch.setattr(execution, "_recheck_before_process", lambda *args: None)

    def execute_once(
        argv: tuple[str, ...], environment: dict[str, str], working_directory: Path
    ) -> execution._CompletedProcessObservation:
        """Capture each worker argv before returning one successful observation."""
        launched.append(argv)
        return _successful_observation()

    monkeypatch.setattr(execution, "_execute_process", execute_once)
    monkeypatch.setattr(
        execution,
        "_load_worker_result",
        lambda path: (_result_identities(), _SHA_A),
    )
    monkeypatch.setattr(execution, "_provisional_review_configuration", lambda *args: provisional)
    monkeypatch.setattr(execution, "_admit_one_cell", lambda *args: admitted_cell)
    monkeypatch.setattr(execution, "_bind_cell_to_profile", lambda *args: None)
    monkeypatch.setattr(
        execution,
        "canonical_json_bytes",
        lambda value: b"{}",
    )

    completed, generated = execution._run_evidence_cells(
        admitted,
        output,
        worker,
        preflights,  # type: ignore[arg-type]
    )

    expected = [
        (role, step_dt, repeat_id)
        for role in ("baseline", "candidate")
        for step_dt in ("0.004", "0.002", "0.001")
        for repeat_id in (0, 1)
    ]
    assert [(item.role, item.step_dt, item.repeat_id) for item in completed] == expected
    assert [event[1:4] for event in events if event[0] == "retain-cell"] == expected
    assert [
        (
            argv[0],
            argv[2:],
        )
        for argv in launched
    ] == [
        (
            (
                admitted.baseline_interpreter.lexical_path.as_posix()
                if role == "baseline"
                else admitted.candidate_interpreter.lexical_path.as_posix()
            ),
            (
                "--manifest",
                admitted.manifest_path.as_posix(),
                "--fixture-id",
                "smooth_pendulum",
                "--profile-role",
                role,
                "--step-dt",
                step_dt,
                "--repeat-id",
                str(repeat_id),
                "--output",
                (
                    output.root
                    / "captured_evidence"
                    / role
                    / {"0.004": "0p004", "0.002": "0p002", "0.001": "0p001"}[step_dt]
                    / f"repeat_{repeat_id}"
                ).as_posix(),
            ),
        )
        for role, step_dt, repeat_id in expected
    ]
    assert all(argv[1] == worker.path.as_posix() for argv in launched)
    assert generated == b"{}\n"


def test_failed_evidence_cell_stops_later_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain the first refused attempt and never launch a later evidence slot."""
    events: list[tuple[object, ...]] = []
    admitted = _admitted(tmp_path)
    output = _ProcessOutput(tmp_path / "run", events)
    worker = _resource(tmp_path / "worker", execution._FROZEN_WORKER_SHA256)
    preflights = (
        SimpleNamespace(
            role="baseline",
            identity_path=tmp_path / "baseline.json",
            identity_file_sha256=_SHA_E,
            identity={},
        ),
        SimpleNamespace(
            role="candidate",
            identity_path=tmp_path / "candidate.json",
            identity_file_sha256=_SHA_F,
            identity={},
        ),
    )
    launches = 0

    def refused_process(*args: object) -> execution._CompletedProcessObservation:
        """Return one bounded worker refusal and count the single launch."""
        nonlocal launches
        launches += 1
        return execution._CompletedProcessObservation(b"", b"REFUSED\n", 2, None)

    monkeypatch.setattr(execution, "_recheck_before_process", lambda *args: None)
    monkeypatch.setattr(execution, "_execute_process", refused_process)

    with pytest.raises(execution._ExecutionRefusal):
        execution._run_evidence_cells(
            admitted,
            output,
            worker,
            preflights,  # type: ignore[arg-type]
        )

    assert launches == 1
    assert [event for event in events if event[0] == "retain-cell"] == [
        ("retain-cell", "baseline", "0.004", 0, 2)
    ]


def test_timed_out_evidence_cell_stops_all_later_slots_and_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retain one timeout exactly and stop all later slots and decision publication."""
    events: list[tuple[object, ...]] = []
    admitted = _admitted(tmp_path)
    output = _PartialRunOutput(tmp_path / "run", events)
    worker = _resource(tmp_path / "worker", execution._FROZEN_WORKER_SHA256)
    collector = _resource(tmp_path / "collector")
    preflights = _preflight_pair(admitted, tmp_path)
    launched: list[tuple[str, ...]] = []

    def accept_recheck(*args: object) -> None:
        """Admit unchanged test-local resources before the first worker launch."""

    def accept_preflights(*args: object) -> tuple[SimpleNamespace, ...]:
        """Supply two already-admitted preflights to isolate evidence orchestration."""
        return preflights

    def time_out_once(
        argv: tuple[str, ...], environment: dict[str, str], working_directory: Path
    ) -> execution._CompletedProcessObservation:
        """Return one truthful timeout observation with retained partial streams."""
        launched.append(argv)
        return execution._CompletedProcessObservation(
            b"partial stdout",
            b"partial stderr",
            None,
            "TIMEOUT",
        )

    def forbid_reload(*args: object) -> object:
        """Reject any generated-configuration reload after the timed-out cell."""
        return _forbidden_stage(events, "strict-reload")

    def forbid_referee(*args: object) -> object:
        """Reject any scientific referee call after the timed-out cell."""
        return _forbidden_stage(events, "referee")

    def forbid_record(*args: object) -> object:
        """Reject any completed-record construction after the timed-out cell."""
        return _forbidden_stage(events, "record")

    monkeypatch.setattr(execution, "_recheck_before_process", accept_recheck)
    monkeypatch.setattr(execution, "_run_profile_preflights", accept_preflights)
    monkeypatch.setattr(execution, "_execute_process", time_out_once)
    monkeypatch.setattr(execution, "_strict_reload_generated_configuration", forbid_reload)
    monkeypatch.setattr(execution, "_call_runtime_reviewer", forbid_referee)
    monkeypatch.setattr(execution, "_build_completed_run_record", forbid_record)

    with pytest.raises(execution._ExecutionRefusal) as caught:
        execution._execute_owned_run(
            admitted,
            output,  # type: ignore[arg-type]
            worker,
            collector,  # type: ignore[arg-type]
        )

    refusal = caught.value
    assert refusal.code is execution.OperationalReasonCode.CHILD_PROCESS_SUPERVISION_FAILED
    assert refusal.field == "evidence_attempt.baseline.0p004.repeat_0"
    assert refusal.evidence == {
        "timeout_seconds": 300,
        "message": "child process exceeded the fixed timeout",
    }
    assert len(launched) == 1
    assert launched[0][9:12] == ("0.004", "--repeat-id", "0")
    assert [event for event in events if event[0] == "cell-path"] == [
        ("cell-path", "baseline", "0.004", 0)
    ]
    assert [event for event in events if event[0] == "retain-cell"] == [
        ("retain-cell", "baseline", "0.004", 0, None)
    ]
    assert len(output.retained_observations) == 1
    retained = output.retained_observations[0]
    assert retained["stdout"] == b"partial stdout"
    assert retained["stderr"] == b"partial stderr"
    assert retained["exit_code"] is None
    assert retained["no_exit_status"] == "TIMEOUT"
    assert not [
        event
        for event in events
        if event[0]
        in {"generated-config", "strict-reload", "referee", "record", "completed-record"}
    ]


def test_malformed_successful_evidence_cell_stops_all_later_slots_and_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stop after admitting malformed output from one successful child and publish nothing."""
    events: list[tuple[object, ...]] = []
    admitted = _admitted(tmp_path)
    output = _PartialRunOutput(tmp_path / "run", events)
    worker = _resource(tmp_path / "worker", execution._FROZEN_WORKER_SHA256)
    collector = _resource(tmp_path / "collector")
    preflights = _preflight_pair(admitted, tmp_path)
    launched: list[tuple[str, ...]] = []

    def accept_recheck(*args: object) -> None:
        """Admit unchanged test-local resources before the first worker launch."""

    def accept_preflights(*args: object) -> tuple[SimpleNamespace, ...]:
        """Supply two already-admitted preflights to isolate evidence orchestration."""
        return preflights

    def succeed_once(
        argv: tuple[str, ...], environment: dict[str, str], working_directory: Path
    ) -> execution._CompletedProcessObservation:
        """Return one successful child observation before result admission fails."""
        launched.append(argv)
        return execution._CompletedProcessObservation(b"worker stdout", b"", 0, None)

    def reject_malformed_result(path: Path) -> object:
        """Model malformed successful worker output at the strict result loader."""
        raise ValueError(f"malformed worker result at {path.name}")

    def forbid_reload(*args: object) -> object:
        """Reject any generated-configuration reload after malformed output."""
        return _forbidden_stage(events, "strict-reload")

    def forbid_referee(*args: object) -> object:
        """Reject any scientific referee call after malformed output."""
        return _forbidden_stage(events, "referee")

    def forbid_record(*args: object) -> object:
        """Reject any completed-record construction after malformed output."""
        return _forbidden_stage(events, "record")

    monkeypatch.setattr(execution, "_recheck_before_process", accept_recheck)
    monkeypatch.setattr(execution, "_run_profile_preflights", accept_preflights)
    monkeypatch.setattr(execution, "_execute_process", succeed_once)
    monkeypatch.setattr(execution, "_load_worker_result", reject_malformed_result)
    monkeypatch.setattr(execution, "_strict_reload_generated_configuration", forbid_reload)
    monkeypatch.setattr(execution, "_call_runtime_reviewer", forbid_referee)
    monkeypatch.setattr(execution, "_build_completed_run_record", forbid_record)

    with pytest.raises(execution._ExecutionRefusal) as caught:
        execution._execute_owned_run(
            admitted,
            output,  # type: ignore[arg-type]
            worker,
            collector,  # type: ignore[arg-type]
        )

    refusal = caught.value
    assert refusal.code is execution.OperationalReasonCode.CONFIGURATION_PARSE_FAILED
    assert refusal.field == "evidence_cells"
    assert refusal.evidence["exception_type"] == "ValueError"
    assert refusal.evidence["role"] == "baseline"
    assert refusal.evidence["step_dt"] == "0.004"
    assert refusal.evidence["repeat_id"] == 0
    assert len(launched) == 1
    assert launched[0][9:12] == ("0.004", "--repeat-id", "0")
    assert [event for event in events if event[0] == "cell-path"] == [
        ("cell-path", "baseline", "0.004", 0)
    ]
    assert [event for event in events if event[0] == "retain-cell"] == [
        ("retain-cell", "baseline", "0.004", 0, 0)
    ]
    assert len(output.retained_observations) == 1
    retained = output.retained_observations[0]
    assert retained["stdout"] == b"worker stdout"
    assert retained["stderr"] == b""
    assert retained["exit_code"] == 0
    assert retained["no_exit_status"] is None
    assert not [
        event
        for event in events
        if event[0]
        in {"generated-config", "strict-reload", "referee", "record", "completed-record"}
    ]


def test_generated_review_configuration_uses_admitted_evidence_identities() -> None:
    """Derive all six subject and workload hashes from the first result without inference."""
    admitted = _admitted(Path("/tmp/runtime-review-test"))
    preflights = _preflight_pair(admitted, Path("/tmp/runtime-review-test"))
    primitive = execution._generated_review_configuration_primitive(
        _result_identities(),
        preflights,  # type: ignore[arg-type]
    )

    assert primitive["expected_subject"] == {
        "fixture_id": "smooth_pendulum",
        "source_closure_sha256": _SHA_A,
        "fixture_manifest_sha256": _SHA_B,
    }
    assert primitive["expected_workload"] == {
        "semantic_sha256": _SHA_C,
        "initial_state_semantic_sha256": _SHA_D,
        "action_program_semantic_sha256": _SHA_E,
    }
    assert primitive["baseline_profile"] == {
        "profile_role": "baseline",
        "package_version": "3.12.0",
        "native_version": "3.12.0",
        "native_version_integer": 3_012_000,
        "profile_identity_sha256": _SHA_D,
        "identity_file": "profile_identities/baseline.json",
    }
    assert primitive["candidate_profile"] == {
        "profile_role": "candidate",
        "package_version": "3.12.0",
        "native_version": "3.12.0",
        "native_version_integer": 3_012_000,
        "profile_identity_sha256": _SHA_D,
        "identity_file": "profile_identities/candidate.json",
    }
    assert len(primitive["cells"]) == 12  # type: ignore[arg-type]
    assert primitive["output_dir"] == "decision"

    provisional = execution._provisional_review_configuration(
        admitted,  # type: ignore[arg-type]
        Path("/tmp/runtime-review-test"),
        _result_identities(),
        preflights,  # type: ignore[arg-type]
    )
    assert provisional.profile_identity_file_sha256 == (_SHA_E, _SHA_F)


def test_completed_decision_propagates_existing_runtime_review_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Call the referee once after all cells and propagate its receipt unchanged."""
    events: list[str] = []
    receipt = {"receipt_sha256": _SHA_A}
    referee_result = SimpleNamespace(
        status=RuntimeReviewStatus.UNRESOLVED_NEAR_BOUNDARY,
        reason_code=None,
        receipt=receipt,
        receipt_sha256=_SHA_A,
        runtime_review_json=tmp_path / "decision" / "runtime_review" / "runtime_review.json",
        runtime_review_markdown=tmp_path / "decision" / "runtime_review" / "runtime_review.md",
    )
    output = SimpleNamespace(
        root=tmp_path,
        generated_runtime_review_config=tmp_path / "generated_runtime_review_config.json",
        write_generated_runtime_review_configuration=lambda raw: SimpleNamespace(sha256=_SHA_B),
        publish_completed_run_record=lambda record: (
            events.append("publish")
            or SimpleNamespace(path=tmp_path / "runtime_review_run.json", run_sha256=_SHA_C)
        ),
    )
    admitted = _admitted(tmp_path)
    worker = _resource(tmp_path / "worker", execution._FROZEN_WORKER_SHA256)
    collector = _resource(tmp_path / "collector")
    monkeypatch.setattr(
        execution,
        "_run_profile_preflights",
        lambda *args: events.append("preflights") or (object(), object()),
    )
    monkeypatch.setattr(
        execution,
        "_run_evidence_cells",
        lambda *args: events.append("cells") or ((), b"{}\n"),
    )
    monkeypatch.setattr(
        execution,
        "_strict_reload_generated_configuration",
        lambda path: events.append("strict-reload") or SimpleNamespace(path=path),
    )
    monkeypatch.setattr(
        execution,
        "_call_runtime_reviewer",
        lambda path: events.append("referee") or referee_result,
    )
    monkeypatch.setattr(
        execution,
        "_recheck_final_inputs",
        lambda *args: events.append("final-recheck"),
    )
    monkeypatch.setattr(
        execution,
        "_build_completed_run_record",
        lambda *args: events.append("record") or {},
    )

    completed = execution._execute_owned_run(
        admitted,
        output,
        worker,
        collector,  # type: ignore[arg-type]
    )

    assert completed.receipt is receipt
    assert completed.status is RuntimeReviewStatus.UNRESOLVED_NEAR_BOUNDARY
    assert completed.reason_code is None
    assert completed.exit_code == 30
    assert events == [
        "preflights",
        "cells",
        "strict-reload",
        "referee",
        "final-recheck",
        "record",
        "publish",
    ]


def test_partial_run_never_publishes_completed_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stop a refused cell before any generated config, referee, or record publication hook."""
    events: list[str] = []
    admitted = _admitted(tmp_path)
    output = SimpleNamespace(
        root=tmp_path / "run",
        write_generated_runtime_review_configuration=lambda raw: events.append("generated"),
        publish_completed_run_record=lambda record: events.append("published"),
    )
    worker = _resource(tmp_path / "worker", execution._FROZEN_WORKER_SHA256)
    collector = _resource(tmp_path / "collector")
    monkeypatch.setattr(execution, "_run_profile_preflights", lambda *args: (object(), object()))

    def stop_cells(*args: object) -> object:
        """Represent the first retained cell refusal."""
        raise execution._ExecutionRefusal(
            execution.OperationalReasonCode.CHILD_PROCESS_SUPERVISION_FAILED,
            "evidence_cells",
            "refused",
        )

    monkeypatch.setattr(execution, "_run_evidence_cells", stop_cells)

    with pytest.raises(execution._ExecutionRefusal):
        execution._execute_owned_run(
            admitted,
            output,
            worker,
            collector,  # type: ignore[arg-type]
        )

    assert events == []


def test_subprocess_seam_uses_fixed_supervision_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pass argv directly with a closed environment, no shell, and the fixed timeout."""
    captured: dict[str, object] = {}

    def fake_run(argv: tuple[str, ...], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        """Capture the standard-library process call without starting a child."""
        captured["argv"] = argv
        captured.update(kwargs)
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)
    argv = ("/profiles/baseline/bin/python", "/installed/worker")
    environment = {"LANG": "C"}

    execution._run_subprocess(
        argv,
        env=environment,
        timeout=300,
        cwd=tmp_path,
    )

    assert captured["argv"] == argv
    assert captured["shell"] is False
    assert captured["check"] is False
    assert captured["timeout"] == 300
    assert captured["env"] == environment


def test_timed_out_process_preserves_partial_streams_without_an_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Represent a killed one-shot timeout truthfully and refuse without a retry."""
    launches = 0

    def time_out(*args: object, **kwargs: object) -> object:
        """Raise one timeout carrying the child's partial byte streams."""
        nonlocal launches
        launches += 1
        raise subprocess.TimeoutExpired(
            cmd=("python", "worker"),
            timeout=300,
            output=b"partial stdout",
            stderr=b"partial stderr",
        )

    monkeypatch.setattr(execution, "_run_subprocess", time_out)
    observed = execution._execute_process(("python", "worker"), {"LANG": "C"}, tmp_path)

    assert launches == 1
    assert observed.stdout == b"partial stdout"
    assert observed.stderr == b"partial stderr"
    assert observed.exit_code is None
    assert observed.no_exit_status == "TIMEOUT"
    with pytest.raises(execution._ExecutionRefusal):
        execution._require_success(observed, "evidence_attempt")


def test_child_refusal_maps_to_a_bounded_operational_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map a retained child refusal to the public operational exit without losing its cause."""
    observation = execution.OperationalToolObservation(
        "test-version", "VERIFIED_INSTALLED_DISTRIBUTION", _SHA_A
    )
    monkeypatch.setattr(execution, "_tool", lambda: observation)

    failure = execution._refuse(
        execution.OperationalReasonCode.CHILD_PROCESS_SUPERVISION_FAILED,
        field="profile_preflight.baseline",
        exit_code=2,
        message="child process returned a bounded refusal",
    ).failure

    assert failure.reason.code is execution.OperationalReasonCode.OUTPUT_WRITE_FAILED
    assert failure.reason.evidence["child_process_reason_code"] == (
        execution.OperationalReasonCode.CHILD_PROCESS_SUPERVISION_FAILED.value
    )
    assert failure.reason.evidence["exit_code"] == 2
    assert int(failure.exit_code) == 64


def test_profile_identity_binds_runtime_and_worker(tmp_path: Path) -> None:
    """Require the profile smoke to bind the selected fixture, manifest, and launcher."""
    admitted = _admitted(tmp_path)
    identity = _identity("baseline", admitted.baseline_interpreter)

    execution._bind_profile_identity_to_run(
        admitted,
        admitted.baseline_interpreter,
        identity,
    )

    identity["native_smoke"] = dict(identity["native_smoke"])  # type: ignore[arg-type]
    identity["native_smoke"]["manifest_raw_sha256"] = _SHA_F  # type: ignore[index]
    with pytest.raises(execution.ProfileIdentityRefusal):
        execution._bind_profile_identity_to_run(
            admitted,
            admitted.baseline_interpreter,
            identity,
        )


def test_profile_identity_wrong_smoke_fixture_is_refused_before_cells(tmp_path: Path) -> None:
    """Reject a validly shaped profile identity measured against another fixture."""
    admitted = _admitted(tmp_path)
    identity = _identity("baseline", admitted.baseline_interpreter)
    identity["native_smoke"] = dict(identity["native_smoke"])  # type: ignore[arg-type]
    identity["native_smoke"]["fixture_id"] = "different_fixture"  # type: ignore[index]

    with pytest.raises(execution.ProfileIdentityRefusal):
        execution._bind_profile_identity_to_run(
            admitted,
            admitted.baseline_interpreter,
            identity,
        )


def test_fixed_child_environment_excludes_caller_injection(tmp_path: Path) -> None:
    """Build only the frozen allowlist and add the fresh identity path for evidence cells."""
    interpreter = _interpreter(tmp_path / "profile" / "bin" / "python")
    identity = tmp_path / "run" / "profile_identities" / "baseline.json"

    preflight_environment = execution._child_environment(interpreter, profile_identity=None)
    environment = execution._child_environment(interpreter, profile_identity=identity)

    assert set(preflight_environment) == set(environment) - {
        "METRIFID_NATIVE_UPGRADE_PROFILE_IDENTITY"
    }
    assert set(environment) == {
        "PATH",
        "LANG",
        "LC_ALL",
        "PYTHONNOUSERSITE",
        "PYTHONHASHSEED",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "METRIFID_NATIVE_UPGRADE_PROFILE_IDENTITY",
    }
    assert "PYTHONPATH" not in environment
    assert "PYTHONHOME" not in environment
    assert preflight_environment["PATH"] == (
        f"{interpreter.lexical_path.parent.as_posix()}:/usr/bin:/bin:/usr/sbin:/sbin"
    )
    assert environment["METRIFID_NATIVE_UPGRADE_PROFILE_IDENTITY"] == identity.as_posix()


def test_resource_hash_is_checked_before_subprocess(tmp_path: Path) -> None:
    """Refuse a changed packaged resource without entering the subprocess seam."""
    resource_path = tmp_path / "worker"
    resource_path.write_bytes(b"changed")
    stale = execution._ResourceIdentity(resource_path, hashlib.sha256(b"original").hexdigest())

    with pytest.raises(ValueError):
        execution._recheck_resource(stale)


def test_result_derivation_is_bound_to_deeply_admitted_bytes() -> None:
    """Reject a result substitution between initial identity derivation and deep admission."""
    admitted_cell = SimpleNamespace(member_sha256={"result.json": _SHA_A})

    execution._require_observed_result_unchanged(admitted_cell, _SHA_A)
    with pytest.raises(ValueError):
        execution._require_observed_result_unchanged(admitted_cell, _SHA_B)
