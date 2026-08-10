"""Collect model closure scenarios that pin decisions and evidence boundaries."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from metrifid import _model_closure as closure
from metrifid import _model_refusal as refusal
from metrifid.operational import OperationalReasonCode
from metrifid.schemas import TargetReference


def _model_root(tmp_path: Path, *, extra: bytes = b"asset") -> Path:
    """Construct the model root fixture used by model closure scenarios.

    Deterministic setup isolates model closure without bypassing the contract boundary under
    assertion.
    """
    root = tmp_path / "model"
    (root / "nested").mkdir(parents=True)
    (root / "model.xml").write_text("<mujoco/>", encoding="utf-8")
    (root / "nested" / "asset.bin").write_bytes(extra)
    return root


def _assert_refusal(
    exc: pytest.ExceptionInfo[closure.ModelAdmissionRefusal], reason: OperationalReasonCode
) -> None:
    """Construct the assert refusal fixture used by model closure scenarios.

    Deterministic setup isolates model closure without bypassing the contract boundary under
    assertion.
    """
    assert exc.value.reason is reason
    assert exc.value.to_primitive()["reason"] == reason.value


def test_refusal_is_frozen_and_role_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises refusal is frozen and role checked; the assertions bind admission to
    exact model bytes, resource boundaries, or an explicit refusal reason.
    """
    item = closure.refuse(OperationalReasonCode.MODEL_ROOT_INVALID, "baseline", paths=["a"])
    assert item.role == "baseline"
    assert item.to_primitive() == {
        "reason": "MODEL_ROOT_INVALID",
        "role": "baseline",
        "evidence": {"paths": ["a"]},
    }
    with pytest.raises(ValueError):
        closure.ModelAdmissionRefusal(OperationalReasonCode.MODEL_ROOT_INVALID, "other")  # type: ignore[arg-type]
    monkeypatch.setattr(refusal, "freeze_canonical", lambda value: "not-an-object")
    with pytest.raises(TypeError):
        closure.ModelAdmissionRefusal(OperationalReasonCode.MODEL_ROOT_INVALID, "candidate")


@pytest.mark.parametrize(
    ("factory", "args"),
    [
        (closure._require_exact_object_fields, ([], {"a"}, "x")),
        (closure._strict_name, (1, "x")),
        (closure._strict_name, ("", "x")),
        (closure._strict_name, ("bad\x00name", "x")),
        (closure._strict_name, ("\ud800", "x")),
        (closure._nonnegative_int, (True, "x")),
        (closure._nonnegative_int, (-1, "x")),
        (closure._strict_array, ((), "x")),
        (closure._completed_hash, (None, "x")),
        (closure._slice, ([0], "x")),
        (closure._slice, ([0, 0], "x")),
    ],
)
def test_strict_helpers_reject_invalid_values(factory: Any, args: tuple[Any, ...]) -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises strict helpers reject invalid values; the assertions bind admission
    to exact model bytes, resource boundaries, or an explicit refusal reason.
    """
    with pytest.raises((TypeError, ValueError, UnicodeError)):
        factory(*args)


def test_strict_helpers_accept_optional_and_exact_values() -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises strict helpers accept optional and exact values; the assertions bind
    admission to exact model bytes, resource boundaries, or an explicit refusal reason.
    """
    assert closure._require_exact_object_fields({"a": 1}, {"a"}, "x") == {"a": 1}
    with pytest.raises(ValueError):
        closure._require_exact_object_fields({"a": 1, "b": 2}, {"a"}, "x")
    assert closure._strict_name(None, "x", optional=True) is None
    assert closure._strict_name("name", "x") == "name"
    assert closure._nonnegative_int(None, "x", optional=True) is None
    assert closure._nonnegative_int(0, "x") == 0
    assert closure._strict_array([], "x") == []
    assert closure._slice([2, 3], "x") == (2, 3)
    assert closure._completed_hash("0" * 64, "x") == "0" * 64


def test_aligned_joint_roundtrip_and_validation() -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises aligned joint roundtrip and validation; the assertions bind
    admission to exact model bytes, resource boundaries, or an explicit refusal reason.
    """
    item = closure.AlignedJoint("j", "HINGE", (0, 1), (3, 1), (0, 1), (2, 1))
    assert closure.AlignedJoint.from_primitive(item.to_primitive()) == item
    with pytest.raises(ValueError):
        closure.AlignedJoint("j", "BAD", (0, 1), (0, 1), (0, 1), (0, 1))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        closure._validate_slice([0, 1], "x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        closure._validate_slice((0, 0), "x")
    with pytest.raises(ValueError):
        closure._validate_slice((0, 2), "x", 1)


def test_aligned_actuator_roundtrip_and_target_shapes() -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises aligned actuator roundtrip and target shapes; the assertions bind
    admission to exact model bytes, resource boundaries, or an explicit refusal reason.
    """
    joint = (TargetReference("JOINT", "j"),)
    item = closure.AlignedActuator("a", "JOINT", joint, "NONE", 0, 0, 2, None, None)
    assert closure.AlignedActuator.from_primitive(item.to_primitive()) == item
    assert closure._valid_target_shape("JOINT", joint)
    assert closure._valid_target_shape(
        "SLIDERCRANK", (TargetReference("SITE", "a"), TargetReference("SITE", "b"))
    )
    assert not closure._valid_target_shape("UNKNOWN", joint)
    with pytest.raises(ValueError):
        closure.AlignedActuator("a", "JOINT", (), "NONE", 0, 0, 0, None, None)
    with pytest.raises(ValueError):
        closure.AlignedActuator("a", "JOINT", joint, "USER", 0, 0, 0, None, None)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        closure.AlignedActuator("a", "JOINT", joint, "INTEGRATOR", 1, 0, 0, None, 0)


def test_name_order_and_joint_target_canonicalization() -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises name order and joint target canonicalization; the assertions bind
    admission to exact model bytes, resource boundaries, or an explicit refusal reason.
    """
    first = closure.AlignedJoint("a", "HINGE", (0, 1), (0, 1), (0, 1), (0, 1))
    second = closure.AlignedJoint("b", "HINGE", (1, 1), (1, 1), (1, 1), (1, 1))
    assert closure._unique_sorted_names((first, second))
    assert not closure._unique_sorted_names((second, first))
    assert not closure._unique_sorted_names((first, first))
    targets = (TargetReference("JOINT", "old"), TargetReference("SITE", "site"))
    assert closure._canonical_targets(targets, {"old": "new"}) == (
        TargetReference("JOINT", "new"),
        TargetReference("SITE", "site"),
    )


def test_measure_snapshot_and_full_root_identity(tmp_path: Path) -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises measure snapshot and full root identity; the assertions bind
    admission to exact model bytes, resource boundaries, or an explicit refusal reason.
    """
    root = _model_root(tmp_path)
    first = closure.measure_model_closure(root.resolve(), "model.xml", "baseline")
    assert [item.path for item in first.members] == ["model.xml", "nested/asset.bin"]
    assert first.member_count == 2
    with closure.create_model_closure_snapshot(root.resolve(), "model.xml", "baseline") as snap:
        assert snap.snapshot_entrypoint.read_bytes() == (root / "model.xml").read_bytes()
        assert (snap.snapshot_root / "nested" / "asset.bin").read_bytes() == b"asset"
        assert snap.__enter__() is snap
        closure.verify_model_closure_unchanged(snap, "baseline")
    assert not snap.snapshot_root.exists()
    (root / "unused.txt").write_text("unused", encoding="utf-8")
    second = closure.measure_model_closure(root.resolve(), "model.xml", "baseline")
    assert second.sha256() != first.sha256()
    (root / "unused.txt").write_text("changed", encoding="utf-8")
    third = closure.measure_model_closure(root.resolve(), "model.xml", "baseline")
    assert third.sha256() != second.sha256()


def test_closure_mutation_is_detected_and_wrapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises closure mutation is detected and wrapped; the assertions bind
    admission to exact model bytes, resource boundaries, or an explicit refusal reason.
    """
    root = _model_root(tmp_path)
    with closure.create_model_closure_snapshot(root.resolve(), "model.xml", "candidate") as snap:
        (root / "nested" / "asset.bin").write_bytes(b"changed")
        with pytest.raises(closure.ModelAdmissionRefusal) as exc:
            closure.verify_model_closure_unchanged(snap, "candidate")
        _assert_refusal(exc, OperationalReasonCode.MODEL_CLOSURE_MUTATED)

    with closure.create_model_closure_snapshot(root.resolve(), "model.xml", "baseline") as snap:
        failure = closure.refuse(
            OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID, "baseline", issue="x"
        )
        monkeypatch.setattr(
            closure, "measure_model_closure", lambda *args, **kwargs: (_ for _ in ()).throw(failure)
        )
        with pytest.raises(closure.ModelAdmissionRefusal) as exc:
            closure.verify_model_closure_unchanged(snap, "baseline")
        _assert_refusal(exc, OperationalReasonCode.MODEL_CLOSURE_MUTATED)

    with closure.create_model_closure_snapshot(root.resolve(), "model.xml", "baseline") as snap:
        direct = closure.refuse(OperationalReasonCode.MODEL_CLOSURE_MUTATED, "baseline")
        monkeypatch.setattr(
            closure, "measure_model_closure", lambda *args, **kwargs: (_ for _ in ()).throw(direct)
        )
        with pytest.raises(closure.ModelAdmissionRefusal) as exc:
            closure.verify_model_closure_unchanged(snap, "baseline")
        assert exc.value is direct


def test_budget_refuses_before_reading(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises budget refuses before reading; the assertions bind admission to
    exact model bytes, resource boundaries, or an explicit refusal reason.
    """
    root = _model_root(tmp_path, extra=b"12345")
    monkeypatch.setattr(
        closure,
        "_read_member",
        lambda *args, **kwargs: pytest.fail("oversized closure must refuse before reading"),
    )
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        closure.measure_model_closure(root.resolve(), "model.xml", "baseline", max_bytes=1)
    _assert_refusal(exc, OperationalReasonCode.MODEL_CLOSURE_BUDGET_EXCEEDED)
    with pytest.raises(ValueError):
        closure.measure_model_closure(root.resolve(), "model.xml", "baseline", max_bytes=-1)
    with pytest.raises(ValueError):
        closure.measure_model_closure(root.resolve(), "model.xml", "baseline", max_bytes=True)


def test_read_member_detects_open_stat_and_size_races(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises read member detects open stat and size races; the assertions bind
    admission to exact model bytes, resource boundaries, or an explicit refusal reason.
    """
    path = tmp_path / "member"
    path.write_bytes(b"abc")
    meta = path.stat()
    member = closure._EnumeratedMember("member", path, 3, meta.st_dev, meta.st_ino, meta.st_mode)

    real_open = closure.os.open
    monkeypatch.setattr(
        closure.os, "open", lambda *args, **kwargs: (_ for _ in ()).throw(OSError())
    )
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        closure._read_member(member, "baseline")
    _assert_refusal(exc, OperationalReasonCode.MODEL_CLOSURE_MUTATED)
    monkeypatch.setattr(closure.os, "open", real_open)

    changed = closure._EnumeratedMember(
        "member", path, 3, meta.st_dev, meta.st_ino + 1, meta.st_mode
    )
    with pytest.raises(closure.ModelAdmissionRefusal):
        closure._read_member(changed, "baseline")

    calls = 0
    real_read = closure.os.read

    def short_read(fd: int, size: int) -> bytes:
        """Construct the short read fixture used by model closure scenarios.

        Deterministic setup isolates read member detects open stat and size races without
        bypassing the contract boundary under assertion.
        """
        nonlocal calls
        calls += 1
        if calls == 1:
            return b"a"
        return b""

    monkeypatch.setattr(closure.os, "read", short_read)
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        closure._read_member(member, "baseline")
    _assert_refusal(exc, OperationalReasonCode.MODEL_CLOSURE_MUTATED)
    monkeypatch.setattr(closure.os, "read", real_read)


def test_metadata_and_directory_errors_are_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises metadata and directory errors are typed; the assertions bind
    admission to exact model bytes, resource boundaries, or an explicit refusal reason.
    """
    root = _model_root(tmp_path)
    entry = SimpleNamespace(stat=lambda **kwargs: (_ for _ in ()).throw(OSError("bad")))
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        closure._metadata(entry, "baseline", "x")  # type: ignore[arg-type]
    _assert_refusal(exc, OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID)
    monkeypatch.setattr(closure.os, "scandir", lambda path: (_ for _ in ()).throw(OSError("bad")))
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        closure._enumerate_members(root.resolve(), "baseline", 100)
    _assert_refusal(exc, OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID)


def test_snapshot_cleanup_on_write_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises snapshot cleanup on write failure; the assertions bind admission to
    exact model bytes, resource boundaries, or an explicit refusal reason.
    """
    root = _model_root(tmp_path)
    original = Path.write_bytes

    def fail(self: Path, data: bytes) -> int:
        """Inject the deterministic fail branch required by this scenario.

        The model closure test can assert failure delivery for snapshot cleanup on write failure
        without depending on incidental runtime errors.
        """
        if "metrifid_baseline_model_" in str(self):
            raise OSError("write failed")
        return original(self, data)

    monkeypatch.setattr(Path, "write_bytes", fail)
    with pytest.raises(OSError):
        closure.create_model_closure_snapshot(root.resolve(), "model.xml", "baseline")
