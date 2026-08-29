"""Complete-MJB artifact identity, header validation and chunked byte comparison."""

from __future__ import annotations

import hashlib
import os
import struct
from pathlib import Path

import mujoco  # type: ignore[import-untyped]
import numpy as np
import pytest

from metrifid._model_closure import ModelAdmissionRefusal
from metrifid.certify import _artifact as artifact_module
from metrifid.certify._artifact import (
    ARTIFACT_IDENTITY_SCHEMA,
    COMPLETE_MJB_METHOD,
    MAX_SERIALIZED_ARTIFACT_BYTES,
    read_header_words,
    serialize_complete_artifact,
)
from metrifid.certify._bytes import compare_artifact_bytes
from metrifid.operational import OperationalReasonCode

_SPHERE = """
<mujoco model="unit">
  <worldbody>
    <body name="b" pos="0 0 1">
      <geom name="g" type="sphere" size="0.1" rgba="1 0 0 1" mass="2"/>
      <joint name="j" type="hinge" axis="0 0 1" damping="0.5"/>
    </body>
  </worldbody>
</mujoco>
"""

_RUNTIME_DIGEST = "a" * 64


def _compile_test_model(xml: str = _SPHERE) -> mujoco.MjModel:
    """Compile a MuJoCo model used to exercise artifact serialization."""
    return mujoco.MjModel.from_xml_string(xml)


def test_the_digest_covers_every_serialized_byte(tmp_path: Path) -> None:
    """Protect the certification artifact assurance boundary from behavioral drift.

    This scenario exercises the digest covers every serialized byte; the assertions pin the
    user-visible result and the evidence needed to explain that result.
    """
    model = _compile_test_model()
    artifact = serialize_complete_artifact(model, "baseline", tmp_path)
    size = mujoco.mj_sizeModel(model)
    buffer = np.empty(size, dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    complete_mjb = buffer.tobytes()
    assert artifact.mjb_size_bytes == size == len(complete_mjb)
    assert artifact.mjb_sha256 == hashlib.sha256(complete_mjb).hexdigest()
    assert artifact.retained.read_exact(0, artifact.mjb_size_bytes) == complete_mjb
    assert artifact.retained.measured_digest() == artifact.mjb_sha256


def test_the_written_file_is_private_and_matches_the_recorded_identity(tmp_path: Path) -> None:
    """Protect the certification artifact assurance boundary from behavioral drift.

    This scenario exercises the written file is private and matches the recorded identity; the
    assertions pin the user-visible result and the evidence needed to explain that result.
    """
    artifact = serialize_complete_artifact(_compile_test_model(), "candidate", tmp_path)
    # The artifact is private for the moment it exists as a name, and nameless afterwards, so
    # no same-user process is left a directory entry it could redirect at other bytes.
    assert artifact.path.name == "candidate.mjb"
    assert not artifact.path.exists()
    assert list(tmp_path.iterdir()) == []
    descriptor = os.fstat(artifact.retained.fd)
    assert descriptor.st_nlink == 0
    assert descriptor.st_mode & 0o777 == 0o600
    assert descriptor.st_size == artifact.mjb_size_bytes
    identity = artifact.identity(_RUNTIME_DIGEST)
    assert identity.schema == ARTIFACT_IDENTITY_SCHEMA
    assert identity.schema_version == 1
    assert identity.method == COMPLETE_MJB_METHOD
    assert identity.runtime_identity_sha256 == _RUNTIME_DIGEST


def test_the_header_is_five_native_words_with_the_documented_meanings(tmp_path: Path) -> None:
    """Protect the certification artifact assurance boundary from behavioral drift.

    This scenario exercises the header is five native words with the documented meanings; the
    assertions pin the user-visible result and the evidence needed to explain that result.
    """
    artifact = serialize_complete_artifact(_compile_test_model(), "baseline", tmp_path)
    assert len(artifact.header_words) == 5
    assert artifact.header_words[0] == 54321
    assert artifact.header_words[1] == artifact.sizeof_mjtnum == 8
    native_version_integer = mujoco.mj_version()
    assert artifact.header_words[3] == native_version_integer
    header_source = Path(artifact.retained.descriptor_path())
    assert read_header_words(header_source, "baseline") == artifact.header_words
    identity = artifact.identity(_RUNTIME_DIGEST)
    assert identity.magic_decimal == 54321
    assert identity.magic_hex == "0x0000d431"
    assert identity.mujoco_version_integer == native_version_integer


def test_the_header_words_are_a_build_property_not_a_model_property(tmp_path: Path) -> None:
    """Protect the certification artifact assurance boundary from behavioral drift.

    This scenario exercises the header words are a build property not a model property; the
    assertions pin the user-visible result and the evidence needed to explain that result.
    """
    simple = serialize_complete_artifact(_compile_test_model(), "baseline", tmp_path)
    complex_xml = (
        "<mujoco><worldbody>"
        + "".join(
            f"<body name='b{index}'><geom size='0.1'/>"
            f"<joint name='j{index}' type='hinge' axis='0 0 1'/></body>"
            for index in range(20)
        )
        + "</worldbody></mujoco>"
    )
    other = serialize_complete_artifact(_compile_test_model(complex_xml), "candidate", tmp_path)
    assert simple.header_words == other.header_words
    assert simple.mjb_size_bytes != other.mjb_size_bytes


def test_an_oversized_artifact_refuses_with_the_size_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Protect the certification artifact assurance boundary from behavioral drift.

    This scenario exercises an oversized artifact refuses with the size reason; the assertions
    pin the user-visible result and the evidence needed to explain that result.
    """
    monkeypatch.setattr(
        mujoco, "mj_sizeModel", lambda _compile_test_model: MAX_SERIALIZED_ARTIFACT_BYTES + 1
    )
    with pytest.raises(ModelAdmissionRefusal) as caught:
        serialize_complete_artifact(_compile_test_model(), "baseline", tmp_path)
    assert caught.value.reason is OperationalReasonCode.COMPILED_ARTIFACT_SIZE_EXCEEDED
    assert caught.value.evidence["limit_bytes"] == 512 * 1024 * 1024
    assert not list(tmp_path.iterdir())


def test_the_size_bound_is_exclusive_at_exactly_the_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A model whose artifact is exactly the bound is admitted; one byte more is refused."""
    model = _compile_test_model()
    exact = mujoco.mj_sizeModel(model)
    monkeypatch.setattr(artifact_module, "MAX_SERIALIZED_ARTIFACT_BYTES", exact)
    assert serialize_complete_artifact(model, "baseline", tmp_path).mjb_size_bytes == exact
    monkeypatch.setattr(artifact_module, "MAX_SERIALIZED_ARTIFACT_BYTES", exact - 1)
    with pytest.raises(ModelAdmissionRefusal) as caught:
        serialize_complete_artifact(model, "candidate", tmp_path)
    assert caught.value.reason is OperationalReasonCode.COMPILED_ARTIFACT_SIZE_EXCEEDED
    assert caught.value.evidence["mjb_size_bytes"] == exact


@pytest.mark.parametrize(
    ("word_index", "replacement", "issue"),
    [
        (0, 12345, "magic_word_mismatch"),
        (1, 4, "mjtnum_width_mismatch"),
        pytest.param(3, None, "version_word_mismatch", id="runtime-version-word-mismatch"),
    ],
)
def test_a_corrupt_header_word_refuses_as_an_invalid_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    word_index: int,
    replacement: int | None,
    issue: str,
) -> None:
    """Protect the certification artifact assurance boundary from behavioral drift.

    This scenario exercises a corrupt header word refuses as an invalid artifact; the assertions
    pin the user-visible result and the evidence needed to explain that result.
    """
    real_save = mujoco.mj_saveModel

    def corrupt(model: object, filename: object, buffer: np.ndarray) -> None:
        """Construct the corrupt fixture used by certification artifact scenarios.

        Deterministic setup isolates a corrupt header word refuses as an invalid artifact
        without bypassing the contract boundary under assertion.
        """
        real_save(model, filename, buffer)
        changed_word = int(mujoco.mj_version()) + 1 if replacement is None else replacement
        struct.pack_into("=i", buffer.data, word_index * 4, changed_word)

    monkeypatch.setattr(mujoco, "mj_saveModel", corrupt)
    with pytest.raises(ModelAdmissionRefusal) as caught:
        serialize_complete_artifact(_compile_test_model(), "baseline", tmp_path)
    assert caught.value.reason is OperationalReasonCode.COMPILED_ARTIFACT_INVALID
    assert caught.value.evidence["issue"] == issue
    assert not list(tmp_path.iterdir())


def test_a_truncated_artifact_refuses_before_it_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Protect the certification artifact assurance boundary from behavioral drift.

    This scenario exercises a truncated artifact refuses before it is written; the assertions
    pin the user-visible result and the evidence needed to explain that result.
    """
    monkeypatch.setattr(mujoco, "mj_sizeModel", lambda _compile_test_model: 4)
    monkeypatch.setattr(mujoco, "mj_saveModel", lambda *_args: None)
    with pytest.raises(ModelAdmissionRefusal) as caught:
        serialize_complete_artifact(_compile_test_model(), "baseline", tmp_path)
    assert caught.value.reason is OperationalReasonCode.COMPILED_ARTIFACT_INVALID
    assert caught.value.evidence["issue"] == "artifact_shorter_than_header"


def test_reading_a_header_from_a_truncated_file_refuses(tmp_path: Path) -> None:
    """Protect the certification artifact assurance boundary from behavioral drift.

    This scenario exercises reading a header from a truncated file refuses; the assertions pin
    the user-visible result and the evidence needed to explain that result.
    """
    path = tmp_path / "short.mjb"
    path.write_bytes(b"\x00\x01\x02")
    with pytest.raises(ModelAdmissionRefusal) as caught:
        read_header_words(path, "candidate")
    assert caught.value.reason is OperationalReasonCode.COMPILED_ARTIFACT_INVALID


def test_identical_artifacts_compare_equal(tmp_path: Path) -> None:
    """Protect the certification artifact assurance boundary from behavioral drift.

    This scenario exercises identical artifacts compare equal; the assertions pin the user-
    visible result and the evidence needed to explain that result.
    """
    left = tmp_path / "left.mjb"
    right = tmp_path / "right.mjb"
    payload = bytes(range(256)) * 8192
    left.write_bytes(payload)
    right.write_bytes(payload)
    result = compare_artifact_bytes(left, right)
    assert result.equal is True
    assert result.first_differing_byte_offset is None
    assert result.differing_byte_count == 0
    assert result.compared_byte_count == len(payload)


def test_a_single_changed_byte_is_located_exactly(tmp_path: Path) -> None:
    """Protect the certification artifact assurance boundary from behavioral drift.

    This scenario exercises a single changed byte is located exactly; the assertions pin the
    user-visible result and the evidence needed to explain that result.
    """
    payload = bytearray(bytes(range(256)) * 8192)
    left = tmp_path / "left.mjb"
    right = tmp_path / "right.mjb"
    left.write_bytes(bytes(payload))
    payload[1_500_003] ^= 0xFF
    right.write_bytes(bytes(payload))
    result = compare_artifact_bytes(left, right)
    assert result.equal is False
    assert result.first_differing_byte_offset == 1_500_003
    assert result.differing_byte_count == 1


def test_a_length_difference_is_counted_and_never_reported_as_equal(tmp_path: Path) -> None:
    """Protect the certification artifact assurance boundary from behavioral drift.

    This scenario exercises a length difference is counted and never reported as equal; the
    assertions pin the user-visible result and the evidence needed to explain that result.
    """
    left = tmp_path / "left.mjb"
    right = tmp_path / "right.mjb"
    left.write_bytes(b"abcdef")
    right.write_bytes(b"abcd")
    result = compare_artifact_bytes(left, right)
    assert result.equal is False
    assert result.compared_byte_count == 4
    assert result.first_differing_byte_offset == 4
    assert result.differing_byte_count == 2


def test_differences_are_counted_across_chunk_boundaries(tmp_path: Path) -> None:
    """Protect the certification artifact assurance boundary from behavioral drift.

    This scenario exercises differences are counted across chunk boundaries; the assertions pin
    the user-visible result and the evidence needed to explain that result.
    """
    size = (1 << 20) * 2 + 17
    left = tmp_path / "left.mjb"
    right = tmp_path / "right.mjb"
    left.write_bytes(b"\x00" * size)
    other = bytearray(b"\x00" * size)
    for offset in (10, (1 << 20) - 1, 1 << 20, (1 << 20) * 2 + 16):
        other[offset] = 0x7F
    right.write_bytes(bytes(other))
    result = compare_artifact_bytes(left, right)
    assert result.first_differing_byte_offset == 10
    assert result.differing_byte_count == 4


def test_a_transient_retained_model_load_is_retried_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Retry the observed transient descriptor-path MJB open without rerunning the decision."""
    artifact = serialize_complete_artifact(_compile_test_model(), "baseline", tmp_path)
    real_loader = artifact_module._load_model_from_binary_path
    attempts = 0

    def flaky_loader(path: str) -> mujoco.MjModel:
        """Fail exactly once with the observed MuJoCo error, then load the same retained bytes."""
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("mj_loadModel: failed to load from mjb")
        return real_loader(path)

    monkeypatch.setattr(artifact_module, "_load_model_from_binary_path", flaky_loader)
    loaded = artifact_module.load_subject_model(artifact.retained)
    assert attempts == 2
    artifact_module._require_loaded_model_matches_subject(loaded, artifact.retained)


def test_a_retried_retained_model_load_reverifies_the_subject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-verify the retained subject between attempts instead of trusting the first check.

    The retry exists because one descriptor open lost a race, not because the artifact is assumed
    intact. Without this the retry could reuse a subject whose bytes changed between attempts, and
    nothing else in the bounded-retry contract would notice.
    """
    artifact = serialize_complete_artifact(_compile_test_model(), "baseline", tmp_path)
    real_loader = artifact_module._load_model_from_binary_path
    real_verify = artifact_module.RetainedCompiledArtifact.verify
    attempts = 0
    verifications: list[int] = []

    def counting_verify(self: artifact_module.RetainedCompiledArtifact) -> None:
        """Record how many load attempts had been made when each verification ran."""
        verifications.append(attempts)
        real_verify(self)

    def flaky_loader(path: str) -> mujoco.MjModel:
        """Fail exactly once with the observed MuJoCo error, then load the same retained bytes."""
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ValueError("mj_loadModel: failed to load from mjb")
        return real_loader(path)

    monkeypatch.setattr(artifact_module.RetainedCompiledArtifact, "verify", counting_verify)
    monkeypatch.setattr(artifact_module, "_load_model_from_binary_path", flaky_loader)
    artifact_module.load_subject_model(artifact.retained)
    assert attempts == 2
    assert verifications == [0, 1, 2]


def test_a_persistent_retained_model_load_failure_stops_after_one_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A persistent MJB-open failure remains a hard failure after exactly two attempts."""
    artifact = serialize_complete_artifact(_compile_test_model(), "baseline", tmp_path)
    attempts = 0

    def broken_loader(_path: str) -> mujoco.MjModel:
        """Return the same retryable failure on every bounded attempt."""
        nonlocal attempts
        attempts += 1
        raise ValueError("mj_loadModel: failed to load from mjb")

    monkeypatch.setattr(artifact_module, "_load_model_from_binary_path", broken_loader)
    with pytest.raises(ValueError, match="failed to load from mjb"):
        artifact_module.load_subject_model(artifact.retained)
    assert attempts == 2


def test_a_nontransient_retained_model_load_failure_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not retry a different loader error whose cause is not the observed descriptor race."""
    artifact = serialize_complete_artifact(_compile_test_model(), "baseline", tmp_path)
    attempts = 0

    def rejected_loader(_path: str) -> mujoco.MjModel:
        """Raise one nonretryable parser failure."""
        nonlocal attempts
        attempts += 1
        raise ValueError("different model-load failure")

    monkeypatch.setattr(artifact_module, "_load_model_from_binary_path", rejected_loader)
    with pytest.raises(ValueError, match="different model-load failure"):
        artifact_module.load_subject_model(artifact.retained)
    assert attempts == 1
