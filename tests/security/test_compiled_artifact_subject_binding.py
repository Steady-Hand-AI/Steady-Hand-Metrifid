"""Same-UID substitution regression for the private compiled artifacts of one review.

The defended case is a distinct process running under the same operating-system user that can
rename or replace entries in the private temporary directory holding the serialized complete-MJB
artifacts. Nothing here claims protection against a privileged kernel compromise or against
arbitrary mutation of this process's memory.

Every substitution below is performed by a real separate process at a controlled orchestration
boundary, so the reproduction is deterministic and never depends on timing luck. The required
product property is that a completed Model Change Gate result can only be derived from the exact
compiled subjects bound by the embedded Certify receipt: a review either publishes exactly the
result of the unsubstituted run, or it fails closed and publishes no completed pair at all.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mujoco  # type: ignore[import-untyped]
import numpy as np
import pytest

from metrifid.compare._failure import ComparisonOperationError
from metrifid.model_release import _run as run_module
from metrifid.model_release import review_model_release

_BASELINE_MASS = "1.0"
_CANDIDATE_MASS = "1.25"
# A third distinct compiled subject. It is a valid artifact for this exact MuJoCo runtime, so a
# substitution is only detectable by subject binding, never by a load failure.
_DECOY_MASS = "4.5"


def _slider_model(mass: str) -> str:
    """Return one fully named model whose single geom mass is the only authored variable."""
    return f"""
<mujoco model="release-slider">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="payload_body">
      <joint name="payload_slide" type="slide" axis="1 0 0" damping="0"/>
      <geom name="payload_geom" type="box" size="0.1 0.1 0.1" mass="{mass}"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="payload_motor" joint="payload_slide" gear="1"/>
  </actuator>
</mujoco>
"""


def _compiled_mjb_bytes(mass: str) -> bytes:
    """Serialize one compiled model exactly the way the product serializes an admitted role."""
    model = mujoco.MjModel.from_xml_string(_slider_model(mass))
    try:
        size = int(mujoco.mj_sizeModel(model))
        buffer = np.empty(size, dtype=np.uint8)
        mujoco.mj_saveModel(model, None, buffer)
        return bytes(buffer.tobytes())
    finally:
        del model


def _write_pair(root: Path) -> tuple[Path, Path]:
    """Write both admitted source roots so each closure identity stays role-local."""
    baseline_root = root / "baseline"
    candidate_root = root / "candidate"
    baseline_root.mkdir(parents=True)
    candidate_root.mkdir(parents=True)
    baseline = baseline_root / "model.xml"
    candidate = candidate_root / "model.xml"
    baseline.write_text(_slider_model(_BASELINE_MASS), encoding="utf-8")
    candidate.write_text(_slider_model(_CANDIDATE_MASS), encoding="utf-8")
    return baseline, candidate


def _write_policy(path: Path, baseline_mjb_sha256: str) -> Path:
    """Bind one rule-free discovery policy to the exact learned baseline subject."""
    path.write_text(
        json.dumps(
            {
                "schema": "metrifid.model_release_policy",
                "schema_version": 1,
                "baseline_compiled_sha256": baseline_mjb_sha256,
                "candidate_compiled_sha256": None,
                "rules": [],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


# A distinct process under the same operating-system user. It takes the exact private artifact
# pathname and installs unrelated bytes under that name, hard-linking the original object aside
# first when the name still resolves so the original bytes stay recoverable. When the product has
# already unlinked the name there is nothing to preserve, and the decoy is simply left where a
# pathname-reloading consumer would find it.
_SUBSTITUTE = """
import os, sys
target, decoy, keep = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    os.link(target, keep)
    print("REPLACED")
except FileNotFoundError:
    print("NAME_ALREADY_QUARANTINED")
os.replace(decoy, target)
"""

_RESTORE = """
import os, sys
target, keep = sys.argv[1], sys.argv[2]
if os.path.exists(keep):
    os.replace(keep, target)
else:
    os.unlink(target)
"""


def _run_other_process(program: str, *arguments: str) -> str:
    """Perform one private-directory mutation from a genuinely separate same-UID process."""
    completed = subprocess.run(
        [sys.executable, "-c", program, *arguments],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed.stdout.strip()


class _Substitution:
    """One controlled same-UID replace, and optional restore, of a private artifact name."""

    def __init__(self, workspace: Path, role: str, *, restore: bool) -> None:
        """Record which role is attacked and whether the original name is put back."""
        self.workspace = workspace
        self.role = role
        self.restore = restore
        self.performed = False
        self.restored = False
        self.quarantined = False
        self.substituted_a_live_subject = False
        self.target: Path | None = None
        self.decoy_sha256 = hashlib.sha256(_compiled_mjb_bytes(_DECOY_MASS)).hexdigest()

    def substitute(self, target: Path) -> None:
        """Install a different valid compiled artifact at one private artifact pathname."""
        decoy = self.workspace / f"decoy-{self.role}.mjb"
        decoy.write_bytes(_compiled_mjb_bytes(_DECOY_MASS))
        keep = self.workspace / f"original-{self.role}.mjb"
        outcome = _run_other_process(_SUBSTITUTE, str(target), str(decoy), str(keep))
        self.quarantined = outcome == "NAME_ALREADY_QUARANTINED"
        self.substituted_a_live_subject = outcome == "REPLACED"
        self.target = target
        self.performed = True
        assert outcome in {"REPLACED", "NAME_ALREADY_QUARANTINED"}, outcome
        assert target.exists(), "the decoy was not installed at the private artifact pathname"

    def put_back(self) -> None:
        """Restore the original artifact object before the run reaches its final boundary."""
        if not self.restore or not self.performed or self.restored or self.target is None:
            return
        keep = self.workspace / f"original-{self.role}.mjb"
        _run_other_process(_RESTORE, str(self.target), str(keep))
        self.restored = True


@contextmanager
def _substituted_run(
    monkeypatch: pytest.MonkeyPatch, workspace: Path, role: str, *, restore: bool
) -> Iterator[_Substitution]:
    """Attack one private artifact between the built receipt and the final publication.

    The substitution boundary is the return of the embedded Certify receipt, which is the exact
    point after every receipt-bound identity has been measured and before any later snapshot,
    field or policy consumer reads a subject. The optional restore happens immediately before
    publication, which is what defeats a design that only hashes the pathname twice.
    """
    substitution = _Substitution(workspace, role, restore=restore)
    real_receipt = run_module._certification_receipt
    real_publish = run_module.publish_paired_results

    def receipt_then_substitute(
        tool: Any, baseline: Any, candidate: Any, runtime: Any
    ) -> dict[str, Any]:
        """Build the real receipt, then let another process replace one private artifact."""
        built = real_receipt(tool, baseline, candidate, runtime)
        artifact = baseline if role == "baseline" else candidate
        substitution.substitute(Path(artifact.serialized.path))
        return built

    def restore_then_publish(*arguments: Any, **keywords: Any) -> Any:
        """Put the original artifact back just before the run publishes its result pair."""
        substitution.put_back()
        return real_publish(*arguments, **keywords)

    monkeypatch.setattr(run_module, "_certification_receipt", receipt_then_substitute)
    monkeypatch.setattr(run_module, "publish_paired_results", restore_then_publish)
    yield substitution


def _clean_result(root: Path, label: str) -> tuple[str, bytes, Path]:
    """Run one unattacked review and return its status and exact canonical receipt bytes."""
    baseline, candidate = _pair_for(root, label)
    policy = root / f"{label}-policy.json"
    _write_policy(policy, _baseline_subject(root, label))
    output = root / f"{label}-out"
    result = review_model_release(str(baseline), str(candidate), str(policy), str(output))
    return str(result.status), (output / "model_release.json").read_bytes(), policy


def _pair_for(root: Path, label: str) -> tuple[Path, Path]:
    """Return one isolated source pair for a single named scenario."""
    workspace = root / label
    workspace.mkdir(parents=True, exist_ok=True)
    baseline = workspace / "baseline" / "model.xml"
    if not baseline.exists():
        _write_pair(workspace)
    return baseline, workspace / "candidate" / "model.xml"


def _baseline_subject(root: Path, label: str) -> str:
    """Return the exact complete-MJB SHA-256 the product will measure for the baseline role."""
    return hashlib.sha256(_compiled_mjb_bytes(_BASELINE_MASS)).hexdigest()


def _decision_facts(receipt_bytes: bytes) -> Mapping[str, Any]:
    """Decode one published Model Change Gate receipt."""
    decoded = json.loads(receipt_bytes.decode("utf-8"))
    assert type(decoded) is dict
    return decoded


def _published_subject_digests(receipt: Mapping[str, Any]) -> tuple[str, str]:
    """Return the two receipt-bound compiled subject digests a reader would trust."""
    certification = receipt["certification_receipt"]
    assert type(certification) is dict
    baseline = certification["baseline"]["compiled_artifact"]["mjb_sha256"]
    candidate = certification["candidate"]["compiled_artifact"]["mjb_sha256"]
    return str(baseline), str(candidate)


@pytest.mark.parametrize(
    ("role", "restore"),
    [
        pytest.param("baseline", False, id="swap_baseline_artifact_only"),
        pytest.param("candidate", False, id="swap_candidate_artifact_only"),
        pytest.param("baseline", True, id="swap_and_restore_before_final_boundary"),
        pytest.param("candidate", True, id="swap_and_restore_candidate_before_boundary"),
    ],
)
def test_a_same_uid_artifact_substitution_cannot_complete_a_mismatched_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str, restore: bool
) -> None:
    """Refuse or reproduce exactly: a replaced private artifact never changes a completed result.

    A distinct same-UID process replaces one private complete-MJB pathname after the embedded
    Certify receipt has measured every subject identity. Whether or not the original object is
    put back before publication, the review must either fail closed with no completed pair, or
    publish byte-for-byte the same receipt the unattacked run publishes. Deriving a completed
    decision from the substituted bytes while reporting the original receipt digests is the
    defect this regression exists to forbid.
    """
    label = f"{role}-{'restored' if restore else 'replaced'}"
    clean_status, clean_receipt, _ = _clean_result(tmp_path, f"clean-{label}")
    clean = _decision_facts(clean_receipt)

    baseline, candidate = _pair_for(tmp_path, f"attacked-{label}")
    policy = _write_policy(
        tmp_path / f"attacked-{label}-policy.json", _baseline_subject(tmp_path, label)
    )
    output = tmp_path / f"attacked-{label}-out"
    workspace = tmp_path / f"attack-{label}"
    workspace.mkdir(parents=True)

    with _substituted_run(monkeypatch, workspace, role, restore=restore) as substitution:
        try:
            result = review_model_release(str(baseline), str(candidate), str(policy), str(output))
        except ComparisonOperationError as caught:
            # Failing closed is a correct outcome, but only for the right reason. A refusal
            # caused by anything other than the substituted subject would pass a weaker check.
            evidence = caught.failure.reason.evidence
            assert str(evidence.get("issue", "")).startswith("retained_artifact_"), evidence
            assert not (output / "model_release.json").exists()
            assert not (output / "model_release.md").exists()
            return
        published = (output / "model_release.json").read_bytes()

    attacked = _decision_facts(published)

    # Exactly one of two things happened, and the test distinguishes them rather than accepting
    # either silently. Either the pathname still named the compiled subject and a real
    # substitution landed, or the product had already quarantined that name so there was nothing
    # to redirect. The first is the pre-correction world, the second is the corrected one.
    assert substitution.quarantined == (not substitution.substituted_a_live_subject), (
        "the reproduction could not tell a landed substitution from a quarantined name"
    )

    # The published receipt advertises the subjects measured before the substitution. If any
    # decision fact below disagrees with the unattacked run, the completed result was derived
    # from bytes the receipt never identified, which is the defect this regression forbids.
    assert _published_subject_digests(attacked) == _published_subject_digests(clean)
    assert attacked["decision_sha256"] == clean["decision_sha256"], (
        f"the {role} decision was derived from a subject the receipt does not identify: "
        f"receipt subjects {_published_subject_digests(attacked)} are unchanged while "
        f"decision_sha256 moved from {clean['decision_sha256']} to {attacked['decision_sha256']}"
    )
    assert attacked["changes"] == clean["changes"]
    assert attacked["first_unexpected_witness"] == clean["first_unexpected_witness"]
    assert attacked["first_missing_required_witness"] == clean["first_missing_required_witness"]
    assert str(result.status) == clean_status
    assert attacked == clean
    assert published == clean_receipt


def test_a_serialized_subject_keeps_no_pathname_a_same_user_process_could_redirect(
    tmp_path: Path,
) -> None:
    """Pin that a serialized compiled subject is nameless and reachable only by descriptor."""
    from metrifid.certify._artifact import serialize_complete_artifact

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    model = mujoco.MjModel.from_xml_string(_slider_model(_BASELINE_MASS))
    try:
        artifact = serialize_complete_artifact(model, "baseline", scratch)
    finally:
        del model

    # No directory entry survives serialization, so there is nothing left to rename or replace.
    assert list(scratch.iterdir()) == []
    assert not artifact.path.exists()
    assert os.fstat(artifact.retained.fd).st_nlink == 0

    # The subject is still completely readable, and still exactly the bytes that were measured.
    assert artifact.retained.measured_digest() == artifact.mjb_sha256
    assert (
        hashlib.sha256(artifact.retained.read_exact(0, artifact.mjb_size_bytes)).hexdigest()
        == artifact.mjb_sha256
    )
    artifact.verify()

    # The only pathname it answers to is the operating system's view of our own descriptor.
    descriptor_path = artifact.retained.descriptor_path()
    assert descriptor_path in {
        f"/proc/self/fd/{artifact.retained.fd}",
        f"/dev/fd/{artifact.retained.fd}",
    }
    reloaded = mujoco.MjModel.from_binary_path(descriptor_path)
    try:
        size = int(mujoco.mj_sizeModel(reloaded))
        buffer = np.empty(size, dtype=np.uint8)
        mujoco.mj_saveModel(reloaded, None, buffer)
        assert hashlib.sha256(buffer.data).hexdigest() == artifact.mjb_sha256
    finally:
        del reloaded
    artifact.retained.close()


def test_in_place_mutation_of_a_retained_subject_fails_closed(tmp_path: Path) -> None:
    """Refuse a retained subject whose bytes were rewritten through an older descriptor.

    Quarantining the name removes every way to reach the subject by pathname, but a process that
    obtained a writable descriptor before the name was removed still holds the object itself.
    That residual case must be detected, not decided from.
    """
    from metrifid._model_closure import ModelAdmissionRefusal
    from metrifid.certify._artifact import serialize_complete_artifact

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    model = mujoco.MjModel.from_xml_string(_slider_model(_BASELINE_MASS))
    try:
        artifact = serialize_complete_artifact(model, "baseline", scratch)
    finally:
        del model
    artifact.verify()

    writable = os.open(artifact.retained.descriptor_path(), os.O_WRONLY)
    try:
        original = artifact.retained.read_exact(0, 1)
        os.pwrite(writable, bytes([original[0] ^ 0xFF]), 0)
    finally:
        os.close(writable)

    with pytest.raises(ModelAdmissionRefusal) as caught:
        artifact.verify()
    assert caught.value.evidence["issue"] == "retained_artifact_bytes_changed"
    assert caught.value.evidence["expected_mjb_sha256"] == artifact.mjb_sha256
    artifact.retained.close()


def test_the_reproduction_uses_three_distinct_valid_compiled_subjects() -> None:
    """Pin that the decoy is a real, loadable, and genuinely different compiled artifact."""
    digests = {
        mass: hashlib.sha256(_compiled_mjb_bytes(mass)).hexdigest()
        for mass in (_BASELINE_MASS, _CANDIDATE_MASS, _DECOY_MASS)
    }
    assert len(set(digests.values())) == 3
    assert len({len(_compiled_mjb_bytes(mass)) for mass in digests}) == 1


# ------------------------------------------------------------------------------------------------
# Exact-consumer binding.
#
# Removing the directory entry stops one process substituting a *different object* for the subject.
# It does not stop a same-user process mutating the retained object itself and putting the original
# bytes back: on Linux it can reopen the nameless inode through /proc/<pid>/fd. Verifying before and
# after a consumer therefore proves only that the bytes were right at two instants, never that the
# consumer read them. That is the same reasoning that rejected pre/post hashing of a pathname.
#
# The tests below bind the two consumers to what they actually consumed: the model that was loaded,
# and the byte streams that were compared. Every mutation here is a deterministic os.pwrite against
# the retained descriptor at an exact monkeypatched boundary. No sleeps, no polling, no race timing.
# ------------------------------------------------------------------------------------------------


def _retained_subject(scratch: Path, role: str, mass: str) -> Any:
    """Serialize one real compiled model into a retained, nameless private subject."""
    from metrifid.certify._artifact import serialize_complete_artifact

    model = mujoco.MjModel.from_xml_string(_slider_model(mass))
    try:
        return serialize_complete_artifact(model, role, scratch)
    finally:
        del model


def _longer_model() -> str:
    """Return a structurally larger model, so its complete MJB is genuinely longer.

    Mass alone never changes the serialized length; extra named bodies do, which is what the
    non-overlap tail regression needs.
    """
    bodies = "".join(
        f"""
    <body name="extra_body_{index}" pos="0 0 {index * 0.05:.2f}">
      <joint name="extra_slide_{index}" type="slide" axis="1 0 0" damping="0"/>
      <geom name="extra_geom_{index}" type="box" size="0.05 0.05 0.05" mass="1.0"/>
    </body>"""
        for index in range(12)
    )
    motors = "\n".join(
        f'    <motor name="extra_motor_{index}" joint="extra_slide_{index}" gear="1"/>'
        for index in range(12)
    )
    return f"""
<mujoco model="release-slider">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <worldbody>
    <body name="payload_body">
      <joint name="payload_slide" type="slide" axis="1 0 0" damping="0"/>
      <geom name="payload_geom" type="box" size="0.1 0.1 0.1" mass="1.0"/>
    </body>{bodies}
  </worldbody>
  <actuator>
    <motor name="payload_motor" joint="payload_slide" gear="1"/>
{motors}
  </actuator>
</mujoco>
"""


def _retained_longer_subject(scratch: Path, role: str) -> Any:
    """Serialize the structurally larger model into a retained, nameless private subject."""
    from metrifid.certify._artifact import serialize_complete_artifact

    model = mujoco.MjModel.from_xml_string(_longer_model())
    try:
        return serialize_complete_artifact(model, role, scratch)
    finally:
        del model


class _TransientMutation:
    """Overwrite a retained subject in place, then put the original bytes back.

    The decoy is a real MJB from the active runtime with the same length and a different digest, so
    nothing here is detectable as a malformed artifact or a size change. Only a digest over what the
    consumer actually read can catch it.
    """

    def __init__(self, subject: Any, decoy_mass: str) -> None:
        """Capture the subject's expected bytes and prepare a same-size decoy."""
        self.subject = subject
        self.expected = subject.retained.read_exact(0, subject.mjb_size_bytes)
        self.decoy = _compiled_mjb_bytes(decoy_mass)
        assert len(self.decoy) == len(self.expected), (
            "the decoy must not change the artifact length"
        )
        assert hashlib.sha256(self.decoy).hexdigest() != subject.mjb_sha256
        self.substituted = False
        self.restored = False

    def substitute(self) -> None:
        """Write the decoy over the retained descriptor."""
        os.pwrite(self.subject.retained.fd, self.decoy, 0)
        os.fsync(self.subject.retained.fd)
        self.substituted = True

    def restore(self) -> None:
        """Put the exact expected bytes back."""
        os.pwrite(self.subject.retained.fd, self.expected, 0)
        os.fsync(self.subject.retained.fd)
        self.restored = True


def test_loaded_model_digest_rejects_transient_mutate_restore(tmp_path: Path) -> None:
    """Refuse a model whose own serialization disagrees with the receipt-bound subject.

    The subject holds its expected bytes at both surrounding verifications and holds the decoy for
    exactly the moment MuJoCo reads it. Only reserializing the loaded model can tell the difference.
    """
    from metrifid._model_closure import ModelAdmissionRefusal
    from metrifid.certify._artifact import RetainedCompiledArtifact, load_subject_model
    from metrifid.operational import OperationalReasonCode

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    subject = _retained_subject(scratch, "baseline", _BASELINE_MASS)
    mutation = _TransientMutation(subject, _DECOY_MASS)

    original_verify = RetainedCompiledArtifact.verify
    calls: list[int] = []

    def verify_then_substitute(self: Any) -> None:
        """Pass the real verification, then move the bytes under the consumer."""
        if self is not subject.retained:
            original_verify(self)
            return
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            original_verify(self)
            mutation.substitute()
            return
        mutation.restore()
        original_verify(self)

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(RetainedCompiledArtifact, "verify", verify_then_substitute)
            with pytest.raises(ModelAdmissionRefusal) as caught:
                try:
                    load_subject_model(subject.retained)
                finally:
                    mutation.restore()

        assert caught.value.reason is OperationalReasonCode.COMPILED_ARTIFACT_INVALID
        assert caught.value.evidence["issue"] == "loaded_model_digest_mismatch"
        assert caught.value.evidence["expected_mjb_sha256"] == subject.mjb_sha256
        assert (
            caught.value.evidence["observed_loaded_model_mjb_sha256"]
            == hashlib.sha256(mutation.decoy).hexdigest()
        )
        assert mutation.substituted, "the reproduction never moved the bytes"

        # The subject is intact again, so the refusal came from the loaded model and not from a
        # residual difference the surrounding verifications could have caught.
        subject.verify()
    finally:
        subject.retained.close()


def test_byte_comparison_digest_rejects_transient_mutate_restore(tmp_path: Path) -> None:
    """Refuse a comparison whose consumed stream disagrees with the receipt-bound subject.

    The comparator is handed decoy bytes for its first chunk and the original bytes for every other
    read, so the subject is intact before and after. Only a digest over the consumed stream differs.
    """
    from metrifid._model_closure import ModelAdmissionRefusal
    from metrifid.certify._artifact import RetainedCompiledArtifact
    from metrifid.certify._bytes import compare_retained_artifacts
    from metrifid.operational import OperationalReasonCode

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    baseline = _retained_subject(scratch, "baseline", _BASELINE_MASS)
    candidate = _retained_subject(scratch, "candidate", _CANDIDATE_MASS)
    mutation = _TransientMutation(baseline, _DECOY_MASS)

    original_read = RetainedCompiledArtifact.read_exact
    served: list[int] = []

    def read_then_restore(self: Any, offset: int, span: int) -> bytes:
        """Serve the decoy for the attacked role's first chunk, then restore immediately."""
        if self is not baseline.retained or offset != 0 or served:
            return original_read(self, offset, span)
        served.append(offset)
        mutation.substitute()
        try:
            return original_read(self, offset, span)
        finally:
            mutation.restore()

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(RetainedCompiledArtifact, "read_exact", read_then_restore)
            with pytest.raises(ModelAdmissionRefusal) as caught:
                compare_retained_artifacts(baseline.retained, candidate.retained)

        assert caught.value.reason is OperationalReasonCode.COMPILED_ARTIFACT_INVALID
        assert caught.value.evidence["issue"] == "consumed_artifact_digest_mismatch"
        assert caught.value.role in {"baseline", "candidate"}
        assert caught.value.evidence["expected_mjb_sha256"] == baseline.mjb_sha256
        assert caught.value.evidence["observed_mjb_sha256"] != baseline.mjb_sha256
        assert served, "the reproduction never served a substituted chunk"

        # Both subjects still hold exactly their measured bytes, so the refusal is about what the
        # comparator consumed rather than about any surviving difference on the descriptor.
        baseline.verify()
        candidate.verify()
    finally:
        baseline.retained.close()
        candidate.retained.close()


def test_byte_comparison_digest_includes_nonoverlap_tail(tmp_path: Path) -> None:
    """Require the comparison to read the longer subject through its final byte.

    A comparison that stops at the overlap has not consumed the artifact it claims to identify, so
    its stream digest could never be checked against the receipt-bound digest.
    """
    from metrifid.certify._artifact import RetainedCompiledArtifact
    from metrifid.certify._bytes import compare_retained_artifacts

    scratch = tmp_path / "scratch"
    scratch.mkdir()
    baseline = _retained_subject(scratch, "baseline", _BASELINE_MASS)
    candidate = _retained_longer_subject(scratch, "candidate")

    assert baseline.mjb_size_bytes != candidate.mjb_size_bytes, (
        "this regression needs two compiled artifacts of different lengths"
    )
    longer = candidate if candidate.mjb_size_bytes > baseline.mjb_size_bytes else baseline

    original_read = RetainedCompiledArtifact.read_exact
    spans: list[tuple[int, int]] = []

    def record(self: Any, offset: int, span: int) -> bytes:
        """Record every positional read the comparator performs on the longer subject."""
        if self is longer.retained:
            spans.append((offset, span))
        return original_read(self, offset, span)

    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(RetainedCompiledArtifact, "read_exact", record)
            comparison = compare_retained_artifacts(baseline.retained, candidate.retained)

        assert sum(span for _offset, span in spans) == longer.mjb_size_bytes
        assert max(offset + span for offset, span in spans) == longer.mjb_size_bytes
        assert comparison.compared_byte_count == min(
            baseline.mjb_size_bytes, candidate.mjb_size_bytes
        )
        assert comparison.differing_byte_count >= abs(
            baseline.mjb_size_bytes - candidate.mjb_size_bytes
        )
        assert comparison.equal is False
        assert comparison.first_differing_byte_offset is not None
    finally:
        baseline.retained.close()
        candidate.retained.close()


__all__: Sequence[str] = ()
_UNUSED: tuple[Callable[..., object], ...] = ()
