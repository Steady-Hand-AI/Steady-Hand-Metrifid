"""Collect model closure read bounds scenarios that pin decisions and evidence boundaries."""

from __future__ import annotations

import hashlib
import stat
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from metrifid import _dependency_reader as dependencies
from metrifid import _model_closure as closure
from metrifid.operational import OperationalReasonCode
from metrifid.schemas import ModelClosureMember


class _BoundedSource:
    """Represent bounded source."""

    def __init__(self, data: bytes, *, fail: bool = False) -> None:
        """Construct the init fixture used by model closure read bounds scenarios.

        Deterministic setup isolates BoundedSource without bypassing the contract boundary under
        assertion.
        """
        self.data = data
        self.fail = fail
        self.offset = 0
        self.consumed = 0
        self.requests: list[int] = []

    def read(self, _descriptor: int, size: int) -> bytes:
        """Construct the read fixture used by model closure read bounds scenarios.

        Deterministic setup isolates BoundedSource without bypassing the contract boundary under
        assertion.
        """
        self.requests.append(size)
        if self.fail:
            raise OSError("controlled read failure")
        chunk = self.data[self.offset : self.offset + size]
        self.offset += len(chunk)
        self.consumed += len(chunk)
        return chunk


def _metadata(*, size: int, inode: int = 7) -> SimpleNamespace:
    """Construct the metadata fixture used by model closure read bounds scenarios.

    Deterministic setup isolates model closure read bounds without bypassing the contract
    boundary under assertion.
    """
    return SimpleNamespace(st_dev=3, st_ino=inode, st_mode=stat.S_IFREG | 0o600, st_size=size)


def _patch_descriptor_io(
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    source: _BoundedSource,
    before: SimpleNamespace,
    after: SimpleNamespace,
) -> None:
    """Construct the patch descriptor io fixture used by model closure read bounds scenarios.

    Deterministic setup isolates model closure read bounds without bypassing the contract
    boundary under assertion.
    """
    calls = iter((before, after))
    io = module.os
    monkeypatch.setattr(io, "open", lambda *_: 41)
    monkeypatch.setattr(io, "fstat", lambda _: next(calls))
    monkeypatch.setattr(io, "read", source.read)
    monkeypatch.setattr(io, "close", lambda _: None)


def _enumerated(tmp_path: Path, expected: int = 3) -> closure._EnumeratedMember:
    """Construct the enumerated fixture used by model closure read bounds scenarios.

    Deterministic setup isolates model closure read bounds without bypassing the contract
    boundary under assertion.
    """
    return closure._EnumeratedMember(
        "member.bin",
        tmp_path / "member.bin",
        expected,
        3,
        7,
        stat.S_IFREG | 0o600,
    )


@pytest.mark.parametrize(
    ("data", "before", "after", "accepted"),
    [
        (b"abc", _metadata(size=3), _metadata(size=3), True),
        (b"ab", _metadata(size=3), _metadata(size=3), False),
        (b"abc" + b"x" * 100, _metadata(size=3), _metadata(size=3), False),
        (b"abcd", _metadata(size=3), _metadata(size=4), False),
        (b"ab", _metadata(size=3), _metadata(size=2), False),
        (b"abc", _metadata(size=3), _metadata(size=3, inode=8), False),
    ],
)
def test_original_member_reader_never_consumes_more_than_expected_plus_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    data: bytes,
    before: SimpleNamespace,
    after: SimpleNamespace,
    accepted: bool,
) -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises original member reader never consumes more than expected plus one;
    the assertions bind admission to exact model bytes, resource boundaries, or an explicit
    refusal reason.
    """
    source = _BoundedSource(data)
    _patch_descriptor_io(monkeypatch, closure, source, before, after)
    member = _enumerated(tmp_path)
    if accepted:
        assert closure._read_member(member, "baseline") == b"abc"
    else:
        with pytest.raises(closure.ModelAdmissionRefusal) as exc:
            closure._read_member(member, "baseline")
        assert exc.value.reason is OperationalReasonCode.MODEL_CLOSURE_MUTATED
    assert source.consumed <= member.size_bytes + 1
    assert all(size <= member.size_bytes + 1 for size in source.requests)


def test_original_member_read_error_is_typed_and_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises original member read error is typed and bounded; the assertions bind
    admission to exact model bytes, resource boundaries, or an explicit refusal reason.
    """
    source = _BoundedSource(b"abc", fail=True)
    _patch_descriptor_io(monkeypatch, closure, source, _metadata(size=3), _metadata(size=3))
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        closure._read_member(_enumerated(tmp_path), "candidate")
    assert exc.value.reason is OperationalReasonCode.MODEL_CLOSURE_MUTATED
    assert exc.value.evidence["issue"] == "member_read_failed_after_enumeration"
    assert source.consumed == 0


def _snapshot_member(path: Path, data: bytes = b"abc") -> ModelClosureMember:
    """Construct the snapshot member fixture used by model closure read bounds scenarios.

    Deterministic setup isolates model closure read bounds without bypassing the contract
    boundary under assertion.
    """
    path.write_bytes(data)
    return ModelClosureMember(path.name, len(data), hashlib.sha256(data).hexdigest())


def _run_snapshot_read(
    path: Path,
    member: ModelClosureMember,
    monkeypatch: pytest.MonkeyPatch,
    source: _BoundedSource,
    mutate_after: Callable[[SimpleNamespace], SimpleNamespace] | None = None,
) -> bytes:
    """Construct the run snapshot read fixture used by model closure read bounds scenarios.

    Deterministic setup isolates model closure read bounds without bypassing the contract
    boundary under assertion.
    """
    actual = path.lstat()
    before = SimpleNamespace(
        st_dev=actual.st_dev,
        st_ino=actual.st_ino,
        st_mode=actual.st_mode,
        st_size=actual.st_size,
    )
    after = mutate_after(before) if mutate_after is not None else before
    _patch_descriptor_io(monkeypatch, dependencies, source, before, after)
    snapshot = SimpleNamespace(entrypoint="model.xml")
    return dependencies._read_dependency_member(
        path,
        member,
        "baseline",
        snapshot,  # type: ignore[arg-type]
        str(path),
        member.path,
    )


def test_private_snapshot_dependency_reader_is_bounded_and_hash_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises private snapshot dependency reader is bounded and hash bound; the
    assertions bind admission to exact model bytes, resource boundaries, or an explicit refusal
    reason.
    """
    path = tmp_path / "member.bin"
    member = _snapshot_member(path)
    exact = _BoundedSource(b"abc")
    assert _run_snapshot_read(path, member, monkeypatch, exact) == b"abc"
    assert exact.consumed <= member.size_bytes + 1


@pytest.mark.parametrize("data", [b"ab", b"abd", b"abc" + b"x" * 100])
def test_private_snapshot_dependency_short_hash_and_long_content_refuse_with_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    data: bytes,
) -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises private snapshot dependency short hash and long content refuse with
    bound; the assertions bind admission to exact model bytes, resource boundaries, or an
    explicit refusal reason.
    """
    path = tmp_path / "member.bin"
    member = _snapshot_member(path)
    source = _BoundedSource(data)
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        _run_snapshot_read(path, member, monkeypatch, source)
    assert exc.value.reason is OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID
    assert exc.value.evidence["issue"] == "dependency_does_not_match_measured_member"
    assert source.consumed <= member.size_bytes + 1


def test_private_snapshot_dependency_metadata_and_read_errors_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises private snapshot dependency metadata and read errors refuse; the
    assertions bind admission to exact model bytes, resource boundaries, or an explicit refusal
    reason.
    """
    path = tmp_path / "member.bin"
    member = _snapshot_member(path)
    changed = _BoundedSource(b"abc")

    def replace_inode(value: SimpleNamespace) -> SimpleNamespace:
        """Construct the replace inode fixture used by model closure read bounds scenarios.

        Deterministic setup isolates private snapshot dependency metadata and read errors refuse
        without bypassing the contract boundary under assertion.
        """
        return SimpleNamespace(**{**vars(value), "st_ino": value.st_ino + 1})

    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        _run_snapshot_read(path, member, monkeypatch, changed, replace_inode)
    assert exc.value.reason is OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID

    failing = _BoundedSource(b"abc", fail=True)
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        _run_snapshot_read(path, member, monkeypatch, failing)
    assert exc.value.reason is OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID
    assert failing.consumed == 0
