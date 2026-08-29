"""Collect NPZ safety scenarios that pin decisions and evidence boundaries."""

from __future__ import annotations

import io
import os
import struct
import subprocess
import sys
import warnings
import zlib
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import numpy as np
import pytest

from metrifid import _npz
from metrifid._npz import ArtifactAdmissionRefusal
from metrifid.operational import OperationalReasonCode

EXPECTED = frozenset({"x.npy"})


def _npy_bytes(array: np.ndarray[tuple[int, ...], np.dtype[np.generic]]) -> bytes:
    """Construct the npy bytes fixture used by NPZ safety scenarios.

    Deterministic setup isolates NPZ safety without bypassing the contract boundary under
    assertion.
    """
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=True)
    return stream.getvalue()


def _zip_bytes(members: list[tuple[str, bytes]], *, compression: int = ZIP_STORED) -> bytes:
    """Construct the zip bytes fixture used by NPZ safety scenarios.

    Deterministic setup isolates NPZ safety without bypassing the contract boundary under
    assertion.
    """
    stream = io.BytesIO()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with ZipFile(stream, "w", compression=compression) as archive:
            for name, data in members:
                archive.writestr(name, data)
    return stream.getvalue()


def _zip_bytes_with_empty_name(payload: bytes) -> bytes:
    """Build one stored ZIP member whose filename length is zero.

    ``zipfile.writestr`` rejects an empty name on some Python minors before Metrifid can inspect
    the archive. Constructing the minimal valid records directly keeps the hostile input identical
    across the supported interpreter matrix.
    """
    crc32 = zlib.crc32(payload) & 0xFFFFFFFF
    local = (
        struct.pack(
            "<IHHHHHIIIHH",
            0x04034B50,
            20,
            0,
            0,
            0,
            0,
            crc32,
            len(payload),
            len(payload),
            0,
            0,
        )
        + payload
    )
    central = struct.pack(
        "<IHHHHHHIIIHHHHHII",
        0x02014B50,
        20,
        20,
        0,
        0,
        0,
        0,
        crc32,
        len(payload),
        len(payload),
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    )
    end = struct.pack("<IHHHHIIH", 0x06054B50, 0, 0, 1, 1, len(central), len(local), 0)
    return local + central + end


def _write(path: Path, raw: bytes) -> Path:
    """Write write data into the isolated test workspace.

    The NPZ safety scenario observes real bytes and filesystem effects for NPZ safety.
    """
    path.write_bytes(raw)
    return path


def _refusal_reason(exc: pytest.ExceptionInfo[ArtifactAdmissionRefusal]) -> OperationalReasonCode:
    """Extract the admission reason emitted for an unsafe NPZ artifact."""
    return exc.value.reason


def _mark_encrypted(raw: bytes) -> bytes:
    """Construct the mark encrypted fixture used by NPZ safety scenarios.

    Deterministic setup isolates NPZ safety without bypassing the contract boundary under
    assertion.
    """
    data = bytearray(raw)
    local = data.find(b"PK\x03\x04")
    central = data.find(b"PK\x01\x02")
    assert local >= 0
    assert central >= 0
    local_flags = struct.unpack_from("<H", data, local + 6)[0] | 0x1
    central_flags = struct.unpack_from("<H", data, central + 8)[0] | 0x1
    struct.pack_into("<H", data, local + 6, local_flags)
    struct.pack_into("<H", data, central + 8, central_flags)
    return bytes(data)


def test_duplicate_member_refuses_before_numpy_loading(tmp_path: Path) -> None:
    """Keep hostile archive structure out of workload evidence.

    This scenario exercises duplicate member refuses before numpy loading; bounded parsing must
    refuse unsafe members before NumPy can interpret attacker-controlled content.
    """
    raw = _zip_bytes(
        [("x.npy", _npy_bytes(np.array([1.0]))), ("x.npy", _npy_bytes(np.array([2.0])))]
    )
    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        _npz.load_npz_arrays(
            _write(tmp_path / "duplicate.npz", raw),
            expected_members=EXPECTED,
            invalid_reason=OperationalReasonCode.STATE_ARTIFACT_INVALID,
        )
    assert _refusal_reason(exc) is OperationalReasonCode.NPZ_DUPLICATE_MEMBER
    assert exc.value.evidence["duplicate_members"] == ("x.npy",)


def test_encrypted_member_refuses(tmp_path: Path) -> None:
    """Keep hostile archive structure out of workload evidence.

    This scenario exercises encrypted member refuses; bounded parsing must refuse unsafe members
    before NumPy can interpret attacker-controlled content.
    """
    raw = _zip_bytes([("x.npy", _npy_bytes(np.array([1.0])))])
    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        _npz.load_npz_arrays(
            _write(tmp_path / "encrypted.npz", _mark_encrypted(raw)),
            expected_members=EXPECTED,
            invalid_reason=OperationalReasonCode.STATE_ARTIFACT_INVALID,
        )
    assert _refusal_reason(exc) is OperationalReasonCode.NPZ_ENCRYPTED_MEMBER


@pytest.mark.parametrize(
    "name",
    ["../x.npy", "dir/x.npy", "dir\\x.npy", "x..npy", "/x.npy", "x.npy/", ""],
)
def test_path_bearing_and_directory_members_refuse(tmp_path: Path, name: str) -> None:
    """Keep hostile archive structure out of workload evidence.

    This scenario exercises path bearing and directory members refuse; bounded parsing must
    refuse unsafe members before NumPy can interpret attacker-controlled content.
    """
    payload = _npy_bytes(np.array([1.0]))
    raw = _zip_bytes_with_empty_name(payload) if name == "" else _zip_bytes([(name, payload)])
    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        _npz.load_npz_arrays(
            _write(tmp_path / "path.npz", raw),
            expected_members=EXPECTED,
            invalid_reason=OperationalReasonCode.STATE_ARTIFACT_INVALID,
        )
    assert _refusal_reason(exc) is OperationalReasonCode.NPZ_PATH_MEMBER_INVALID


def test_missing_and_extra_member_sets_refuse(tmp_path: Path) -> None:
    """Keep hostile archive structure out of workload evidence.

    This scenario exercises missing and extra member sets refuse; bounded parsing must refuse
    unsafe members before NumPy can interpret attacker-controlled content.
    """
    for name, members in (
        ("missing", []),
        ("extra", [("x.npy", _npy_bytes(np.array([1.0]))), ("y.npy", b"bad")]),
    ):
        path = _write(tmp_path / f"{name}.npz", _zip_bytes(members))
        with pytest.raises(ArtifactAdmissionRefusal) as exc:
            _npz.load_npz_arrays(
                path,
                expected_members=EXPECTED,
                invalid_reason=OperationalReasonCode.STATE_ARTIFACT_INVALID,
            )
        assert _refusal_reason(exc) is OperationalReasonCode.NPZ_MEMBER_SET_INVALID


def test_raw_and_total_uncompressed_budgets_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep hostile archive structure out of workload evidence.

    This scenario exercises raw and total uncompressed budgets refuse; bounded parsing must
    refuse unsafe members before NumPy can interpret attacker-controlled content.
    """
    raw_path = tmp_path / "raw.npz"
    raw_path.write_bytes(b"x" * 11)
    monkeypatch.setattr(_npz, "MAX_NPZ_BYTES", 10)
    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        _npz.load_npz_arrays(
            raw_path,
            expected_members=EXPECTED,
            invalid_reason=OperationalReasonCode.STATE_ARTIFACT_INVALID,
        )
    assert _refusal_reason(exc) is OperationalReasonCode.NPZ_SIZE_BUDGET_EXCEEDED
    assert exc.value.evidence["issue"] == "raw_file"

    compressed = _zip_bytes(
        [("x.npy", _npy_bytes(np.zeros(10_000, dtype="<f8")))], compression=ZIP_DEFLATED
    )
    monkeypatch.setattr(_npz, "MAX_NPZ_BYTES", len(compressed) + 1)
    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        _npz.load_npz_arrays(
            _write(tmp_path / "expanded.npz", compressed),
            expected_members=EXPECTED,
            invalid_reason=OperationalReasonCode.STATE_ARTIFACT_INVALID,
        )
    assert _refusal_reason(exc) is OperationalReasonCode.NPZ_SIZE_BUDGET_EXCEEDED
    assert exc.value.evidence["issue"] == "uncompressed_content"


def test_object_and_pickle_backed_array_refuses(tmp_path: Path) -> None:
    """Keep hostile archive structure out of workload evidence.

    This scenario exercises object and pickle backed array refuses; bounded parsing must refuse
    unsafe members before NumPy can interpret attacker-controlled content.
    """
    raw = _zip_bytes([("x.npy", _npy_bytes(np.array([{"x": 1}], dtype=object)))])
    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        _npz.load_npz_arrays(
            _write(tmp_path / "object.npz", raw),
            expected_members=EXPECTED,
            invalid_reason=OperationalReasonCode.STATE_ARTIFACT_INVALID,
        )
    assert _refusal_reason(exc) is OperationalReasonCode.NPZ_OBJECT_ARRAY_REFUSED


def test_malformed_zip_and_npy_expose_deliberate_reasons(tmp_path: Path) -> None:
    """Keep hostile archive structure out of workload evidence.

    This scenario exercises malformed zip and npy expose deliberate reasons; bounded parsing
    must refuse unsafe members before NumPy can interpret attacker-controlled content.
    """
    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        _npz.load_npz_arrays(
            _write(tmp_path / "bad-zip.npz", b"not a zip"),
            expected_members=EXPECTED,
            invalid_reason=OperationalReasonCode.ACTIONS_ARTIFACT_INVALID,
        )
    assert _refusal_reason(exc) is OperationalReasonCode.NPZ_MEMBER_SET_INVALID

    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        _npz.load_npz_arrays(
            _write(tmp_path / "bad-npy.npz", _zip_bytes([("x.npy", b"not npy")])),
            expected_members=EXPECTED,
            invalid_reason=OperationalReasonCode.ACTIONS_ARTIFACT_INVALID,
        )
    assert _refusal_reason(exc) is OperationalReasonCode.ACTIONS_ARTIFACT_INVALID


def test_nonexistent_artifact_and_invalid_path_type_refuse(tmp_path: Path) -> None:
    """Keep hostile archive structure out of workload evidence.

    This scenario exercises nonexistent artifact and invalid path type refuse; bounded parsing
    must refuse unsafe members before NumPy can interpret attacker-controlled content.
    """
    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        _npz.load_npz_arrays(
            tmp_path / "missing.npz",
            expected_members=EXPECTED,
            invalid_reason=OperationalReasonCode.STATE_ARTIFACT_INVALID,
        )
    assert _refusal_reason(exc) is OperationalReasonCode.STATE_ARTIFACT_INVALID
    with pytest.raises(ArtifactAdmissionRefusal) as exc:
        _npz.load_npz_arrays(
            object(),  # type: ignore[arg-type]
            expected_members=EXPECTED,
            invalid_reason=OperationalReasonCode.STATE_ARTIFACT_INVALID,
        )
    assert _refusal_reason(exc) is OperationalReasonCode.STATE_ARTIFACT_INVALID


def test_fifo_refuses_without_blocking(tmp_path: Path) -> None:
    """Refuse a FIFO through one nonblocking open instead of waiting for a writer."""
    fifo = tmp_path / "state.npz"
    os.mkfifo(fifo)
    command = [
        sys.executable,
        "-B",
        "-c",
        (
            "from metrifid._npz import _read_bounded_bytes; "
            "from metrifid.operational import OperationalReasonCode; "
            f"_read_bounded_bytes({str(fifo)!r}, "
            "OperationalReasonCode.STATE_ARTIFACT_INVALID)"
        ),
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except subprocess.TimeoutExpired:
        pytest.fail("NPZ FIFO admission blocked in open")
    assert completed.returncode != 0
    assert "ArtifactAdmissionRefusal" in completed.stderr
