"""Collect NPZ loader scenarios that pin decisions and evidence boundaries."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from metrifid import _npz
from metrifid._npz import ArtifactAdmissionRefusal, LoadedNpz
from metrifid.operational import OperationalReasonCode


def test_valid_npz_is_loaded_from_exact_bytes_with_immutable_mapping(tmp_path: Path) -> None:
    """Preserve the declared workload's numerical meaning.

    This scenario exercises valid NPZ is loaded from exact bytes with immutable mapping;
    malformed arrays, names, or dimensions must fail before comparison evidence is produced.
    """
    path = tmp_path / "valid.npz"
    np.savez(path, x=np.array([1.0, 2.0], dtype="<f8"))
    result = _npz.load_npz_arrays(
        path,
        expected_members=frozenset({"x.npy"}),
        invalid_reason=OperationalReasonCode.STATE_ARTIFACT_INVALID,
    )
    assert result.raw_file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert isinstance(result.arrays, MappingProxyType)
    assert np.array_equal(result.arrays["x"], np.array([1.0, 2.0]))
    with pytest.raises(TypeError):
        result.arrays["y"] = np.array([3.0])  # type: ignore[index]


def test_refusal_object_is_frozen_and_serializable() -> None:
    """Preserve the declared workload's numerical meaning.

    This scenario exercises refusal object is frozen and serializable; malformed arrays, names,
    or dimensions must fail before comparison evidence is produced.
    """
    refusal = _npz.refuse(
        OperationalReasonCode.NPZ_MEMBER_SET_INVALID,
        "comparison",
        issue="test",
        count=1,
    )
    assert isinstance(refusal, ArtifactAdmissionRefusal)
    assert refusal.to_primitive() == {
        "reason": "NPZ_MEMBER_SET_INVALID",
        "role": "comparison",
        "evidence": {"count": 1, "issue": "test"},
    }
    with pytest.raises(TypeError):
        refusal.evidence["new"] = "value"  # type: ignore[index]
    with pytest.raises(ValueError):
        ArtifactAdmissionRefusal(OperationalReasonCode.NPZ_MEMBER_SET_INVALID, "bad")  # type: ignore[arg-type]


def test_loaded_npz_constructor_normalizes_mapping_and_checks_hash() -> None:
    """Preserve the declared workload's numerical meaning.

    This scenario exercises loaded NPZ constructor normalizes mapping and checks hash; malformed
    arrays, names, or dimensions must fail before comparison evidence is produced.
    """
    value = LoadedNpz("a" * 64, {"x": np.array([1])})
    assert isinstance(value.arrays, MappingProxyType)
    with pytest.raises(ValueError):
        LoadedNpz("short", {})


def test_artifact_read_failure_and_growth_after_stat_are_deliberate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve the declared workload's numerical meaning.

    This scenario exercises artifact read failure and growth after stat are deliberate;
    malformed arrays, names, or dimensions must fail before comparison evidence is produced.
    """
    path = tmp_path / "artifact.npz"
    path.write_bytes(b"x")

    def read_error(_descriptor: int, _count: int) -> bytes:
        """Inject the deterministic read error branch required by this scenario.

        The NPZ loader test can assert failure delivery for artifact read failure and growth
        after stat are deliberate without depending on incidental runtime errors.
        """
        raise OSError("refused")

    monkeypatch.setattr(_npz.os, "read", read_error)
    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        _npz.load_npz_arrays(
            path,
            expected_members=frozenset({"x.npy"}),
            invalid_reason=OperationalReasonCode.STATE_ARTIFACT_INVALID,
        )
    assert exc.value.reason is OperationalReasonCode.STATE_ARTIFACT_INVALID
    monkeypatch.undo()

    path.write_bytes(b"x" * 11)
    monkeypatch.setattr(_npz, "MAX_NPZ_BYTES", 10)
    real_fstat = _npz.os.fstat

    def understated_size(descriptor: int) -> os.stat_result:
        """Report the pre-read file as one byte so the bounded growth branch is exercised."""
        values = list(real_fstat(descriptor))
        values[6] = 1
        return os.stat_result(values)

    monkeypatch.setattr(_npz.os, "fstat", understated_size)
    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        _npz.load_npz_arrays(
            path,
            expected_members=frozenset({"x.npy"}),
            invalid_reason=OperationalReasonCode.STATE_ARTIFACT_INVALID,
        )
    assert exc.value.reason is OperationalReasonCode.NPZ_SIZE_BUDGET_EXCEEDED


def test_npz_admission_never_reads_beyond_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Bound the open descriptor read itself, including growth after the fstat precheck."""
    path = tmp_path / "growing.npz"
    path.write_bytes(b"x" * 32)
    monkeypatch.setattr(_npz, "MAX_NPZ_BYTES", 8)
    real_fstat = _npz.os.fstat
    real_read = _npz.os.read
    requested: list[int] = []

    def understated_size(descriptor: int) -> os.stat_result:
        """Understate the open file so the bounded descriptor read detects growth."""
        values = list(real_fstat(descriptor))
        values[6] = 1
        return os.stat_result(values)

    def recording_read(descriptor: int, count: int) -> bytes:
        """Record the exact bounded read request before delegating to the real descriptor."""
        requested.append(count)
        return real_read(descriptor, count)

    monkeypatch.setattr(_npz.os, "fstat", understated_size)
    monkeypatch.setattr(_npz.os, "read", recording_read)
    monkeypatch.setattr(Path, "read_bytes", lambda _self: pytest.fail("unbounded path read"))
    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        _npz._read_bounded_bytes(path, OperationalReasonCode.STATE_ARTIFACT_INVALID)
    assert exc.value.reason is OperationalReasonCode.NPZ_SIZE_BUDGET_EXCEEDED
    assert requested == [9]


def test_generic_numpy_value_error_is_mapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve the declared workload's numerical meaning.

    This scenario exercises generic numpy value error is mapped; malformed arrays, names, or
    dimensions must fail before comparison evidence is produced.
    """
    path = tmp_path / "valid-container.npz"
    np.savez(path, x=np.array([1.0], dtype="<f8"))

    def fail_load(*_args: object, **_kwargs: object) -> object:
        """Inject the deterministic fail load branch required by this scenario.

        The NPZ loader test can assert failure delivery for generic numpy value error is mapped
        without depending on incidental runtime errors.
        """
        raise ValueError("unrelated parser failure")

    monkeypatch.setattr(_npz.np, "load", fail_load)
    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        _npz.load_npz_arrays(
            path,
            expected_members=frozenset({"x.npy"}),
            invalid_reason=OperationalReasonCode.ACTIONS_ARTIFACT_INVALID,
        )
    assert exc.value.reason is OperationalReasonCode.ACTIONS_ARTIFACT_INVALID
