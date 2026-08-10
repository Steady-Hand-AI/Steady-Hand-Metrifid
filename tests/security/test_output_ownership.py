"""Adversarial ownership, cleanup, descriptor-lifetime, and decision-order tests."""

from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import metrifid._atomic_output as atomic_output
import metrifid._owned_artifacts as owned
import metrifid.workload_writers as writers
from metrifid import _audit_execution as audit_execution
from metrifid import write_actions_artifact
from metrifid._atomic_output import (
    PairedOutputDirectory,
    PairedOutputNames,
    cleanup_paired_output_after_failure,
    prepare_paired_output_directory,
    publish_paired_results,
)
from metrifid._audit_artifacts import AuditArtifactRegistry
from metrifid._npz import ArtifactAdmissionRefusal
from metrifid.certify import _run as certify_run
from metrifid.certify._status import CertifyStatus
from metrifid.compare import _orchestrator as compare_orchestrator

_PAIR_NAMES = PairedOutputNames("result.json", "result.md")
_AUDIT_NAMES = PairedOutputNames("timestep_audit.json", "timestep_audit.md")


class _SpyCompareOutput:
    """Record comparison output verification and descriptor closure order."""

    def __init__(self, events: list[str], root: Path) -> None:
        """Bind event storage and stable result paths."""
        self.events = events
        self.json_path = root / "comparison.json"
        self.markdown_path = root / "comparison.md"

    def _verify_retained_pair(self) -> None:
        """Record the final retained-output verification."""
        self.events.append("outputs")

    def _close_after_success(self) -> None:
        """Record descriptor closure after the success result is constructed."""
        self.events.append("close")


class _SpyRetained:
    """Record certification pair verification, cleanup, and closure."""

    def __init__(self, events: list[str]) -> None:
        """Retain the shared event list."""
        self.events = events

    def verify(self) -> None:
        """Record final output verification."""
        self.events.append("outputs")

    def cleanup(self) -> None:
        """Record unexpected failure cleanup."""
        self.events.append("cleanup")

    def close(self) -> None:
        """Record retained-pair descriptor closure."""
        self.events.append("pair-close")


class _SpyPublished:
    """Provide certification paths and record output-directory closure."""

    def __init__(self, events: list[str], root: Path) -> None:
        """Bind event storage and stable certification paths."""
        self.events = events
        self.json_path = root / "certification.json"
        self.markdown_path = root / "certification.md"

    def close(self) -> None:
        """Record output-directory descriptor closure."""
        self.events.append("dir-close")


class _AuditOrderSpy:
    """Record the final candidate, tree, and aggregate verification order."""

    def __init__(self, events: list[str]) -> None:
        """Retain the shared event list."""
        self.events = events

    def verify_candidate_evidence(self) -> None:
        """Record candidate evidence verification."""
        self.events.append("candidate")

    def verify_public_tree(self) -> None:
        """Record exact public-tree verification."""
        self.events.append("tree")

    def verify_aggregate(self) -> None:
        """Record aggregate pair verification."""
        self.events.append("aggregate")


def _paired_output(tmp_path: Path, name: str = "out") -> PairedOutputDirectory:
    """Create one empty descriptor-bound output directory for ownership tests."""
    return prepare_paired_output_directory(tmp_path / name, _PAIR_NAMES)


def _audit_registry(
    tmp_path: Path, name: str = "audit"
) -> tuple[PairedOutputDirectory, AuditArtifactRegistry]:
    """Create one admitted audit root and its empty ownership registry."""
    output = prepare_paired_output_directory(tmp_path / name, _AUDIT_NAMES)
    return output, AuditArtifactRegistry(output)


def _write_actions(path: Path) -> None:
    """Write the smallest valid deterministic public actions artifact."""
    write_actions_artifact(path, actuator_names=("motor",), values=np.zeros((1, 1)))


def _replace_final(directory_fd: int, name: str, payload: bytes) -> None:
    """Replace one final entry with a foreign regular file under the retained parent."""
    os.unlink(name, dir_fd=directory_fd)
    descriptor = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=directory_fd)
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)


def _replace_output_root(
    public: Path, displaced: Path, attacker: Path, names: tuple[str, str]
) -> None:
    """Replace one public output directory and seed the replacement with attacker bytes."""
    public.rename(displaced)
    public.symlink_to(attacker, target_is_directory=True)
    for name in names:
        (attacker / name).write_bytes(b"ATTACKER")


def _certified_decision(*_args: object) -> tuple[CertifyStatus, dict[str, object]]:
    """Return the fixed green certification decision used by output-lifecycle tests."""
    return CertifyStatus.CERTIFIED_COMPILED_EQUIVALENCE, {}


def _descriptor_is_closed(descriptor: int) -> bool:
    """Return whether fstat proves one captured descriptor has been closed."""
    try:
        os.fstat(descriptor)
    except OSError:
        return True
    return False


def test_generic_artifact_cleanup_never_deletes_a_committed_final(tmp_path: Path) -> None:
    """Preserve a committed public final when generic failure cleanup runs later."""
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    artifact = owned.create_owned_artifact(parent_fd, ".result.")
    try:
        owned.write_owned_bytes(artifact, b"COMMITTED")
        owned.commit_owned_artifact(artifact, "result.bin")
        artifact.cleanup()
        assert (tmp_path / "result.bin").read_bytes() == b"COMMITTED"
    finally:
        artifact.close()
        os.close(parent_fd)


def test_paired_cleanup_preserves_committed_finals_and_unrelated_entries(tmp_path: Path) -> None:
    """Preserve an exact pair final, a replacement, and unrelated caller data."""
    output = _paired_output(tmp_path)
    pair = publish_paired_results(output, json_bytes=b"{}", markdown_text="report\n")
    _replace_final(output.directory_fd, _PAIR_NAMES.json_name, b"FOREIGN")
    unrelated = output.path / "caller.txt"
    unrelated.write_bytes(b"CALLER")
    pair.cleanup()
    pair.close()
    output.close()
    assert (output.path / _PAIR_NAMES.json_name).read_bytes() == b"FOREIGN"
    assert (output.path / _PAIR_NAMES.markdown_name).read_bytes() == b"report\n"
    assert unrelated.read_bytes() == b"CALLER"


def test_paired_descriptors_close_on_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Close retained pair and directory descriptors on explicit success and failure exits."""
    success = _paired_output(tmp_path, "success")
    pair = publish_paired_results(success, json_bytes=b"{}", markdown_text="ok\n")
    descriptors = [pair.first.fd, pair.first.parent_fd, pair.second.fd, pair.second.parent_fd]
    cleanup_paired_output_after_failure(success, pair)
    assert pair.closed
    assert success._handle.closed
    assert all(_descriptor_is_closed(descriptor) for descriptor in descriptors)

    created: list[owned.OwnedArtifact] = []
    real_create = atomic_output.create_owned_artifact
    real_link = owned.os.link

    def recording_create(parent_fd: int, prefix: str) -> owned.OwnedArtifact:
        """Record each retained temporary created for the failing pair."""
        artifact = real_create(parent_fd, prefix)
        created.append(artifact)
        return artifact

    calls = 0

    def failing_link(source: object, target: object, **kwargs: object) -> None:
        """Create the first link and interrupt the second link."""
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("forced")
        real_link(source, target, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(atomic_output, "create_owned_artifact", recording_create)
    monkeypatch.setattr(owned.os, "link", failing_link)
    failure = _paired_output(tmp_path, "failure")
    with pytest.raises(ArtifactAdmissionRefusal):
        publish_paired_results(failure, json_bytes=b"{}", markdown_text="bad\n")
    cleanup_paired_output_after_failure(failure)
    assert created
    assert all(artifact.closed for artifact in created)
    assert failure._handle.closed


def test_paired_in_place_mutation_is_detected_and_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse an in-place final mutation without deleting its changed inode."""
    output = _paired_output(tmp_path)
    real_link = owned.os.link
    attacked = False

    def mutate(source: object, target: object, **kwargs: object) -> None:
        """Mutate the first linked final through the shared inode before verification."""
        nonlocal attacked
        real_link(source, target, **kwargs)  # type: ignore[arg-type]
        if not attacked:
            attacked = True
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_TRUNC,
                dir_fd=kwargs["dst_dir_fd"],  # type: ignore[arg-type]
            )
            os.write(descriptor, b"MODIFIED")
            os.close(descriptor)

    monkeypatch.setattr(owned.os, "link", mutate)
    with pytest.raises(ArtifactAdmissionRefusal):
        publish_paired_results(output, json_bytes=b"{}", markdown_text="ok\n")
    assert (output.path / _PAIR_NAMES.json_name).read_bytes() == b"MODIFIED"
    output.close()


def test_paired_concurrent_final_injection_is_refused_and_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve a final injected after temporary creation and refuse no-clobber publication."""
    output = _paired_output(tmp_path)
    real_link = owned.os.link
    injected = False

    def inject(source: object, target: object, **kwargs: object) -> None:
        """Create a foreign destination immediately before the first real link."""
        nonlocal injected
        if not injected:
            injected = True
            descriptor = os.open(
                target,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=kwargs["dst_dir_fd"],  # type: ignore[arg-type]
            )
            os.write(descriptor, b"FOREIGN")
            os.close(descriptor)
        real_link(source, target, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(owned.os, "link", inject)
    with pytest.raises(ArtifactAdmissionRefusal):
        publish_paired_results(output, json_bytes=b"{}", markdown_text="ok\n")
    assert (output.path / _PAIR_NAMES.json_name).read_bytes() == b"FOREIGN"
    assert not (output.path / _PAIR_NAMES.markdown_name).exists()
    output.close()


def test_paired_existing_final_is_refused_and_preserved(tmp_path: Path) -> None:
    """Preserve an existing final observed after output-directory admission."""
    output = _paired_output(tmp_path)
    existing = output.path / _PAIR_NAMES.json_name
    existing.write_bytes(b"FOREIGN")
    with pytest.raises(ArtifactAdmissionRefusal):
        publish_paired_results(output, json_bytes=b"{}", markdown_text="ok\n")
    assert existing.read_bytes() == b"FOREIGN"
    output.close()


def test_paired_replacement_is_detected_and_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse a replaced linked final and preserve the foreign replacement."""
    output = _paired_output(tmp_path)
    real_link = owned.os.link
    attacked = False

    def replace(source: object, target: object, **kwargs: object) -> None:
        """Replace the first destination immediately after its real hard link."""
        nonlocal attacked
        real_link(source, target, **kwargs)  # type: ignore[arg-type]
        if not attacked:
            attacked = True
            _replace_final(kwargs["dst_dir_fd"], str(target), b"FOREIGN")  # type: ignore[arg-type]

    monkeypatch.setattr(owned.os, "link", replace)
    with pytest.raises(ArtifactAdmissionRefusal):
        publish_paired_results(output, json_bytes=b"{}", markdown_text="ok\n")
    assert (output.path / _PAIR_NAMES.json_name).read_bytes() == b"FOREIGN"
    output.close()


def test_post_commit_writer_failure_preserves_the_committed_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve exact writer bytes and unrelated caller data after a later parent-check failure."""
    expected = tmp_path / "expected.npz"
    _write_actions(expected)
    expected_bytes = expected.read_bytes()
    target = tmp_path / "actions.npz"
    real_verify = writers._verify_writer_parent
    calls = 0

    def fail_after_commit(parent: Path, identity: tuple[int, int]) -> None:
        """Fail the final parent check after creating unrelated caller evidence."""
        nonlocal calls
        calls += 1
        real_verify(parent, identity)
        if calls == 3:
            (parent / "caller.txt").write_bytes(b"CALLER")
            raise ValueError("forced after commit")

    monkeypatch.setattr(writers, "_verify_writer_parent", fail_after_commit)
    with pytest.raises(ValueError, match="forced"):
        _write_actions(target)
    assert target.read_bytes() == expected_bytes
    assert (tmp_path / "caller.txt").read_bytes() == b"CALLER"


def test_writer_descriptors_close_on_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Close writer artifact and parent descriptors after both publication outcomes."""
    artifacts: list[owned.OwnedArtifact] = []
    parents: list[int] = []
    real_open = writers._open_writer_temp
    real_bind = writers._bind_writer_parent

    def recording_open(parent_fd: int, target_name: str) -> owned.OwnedArtifact:
        """Record each writer artifact handle before returning it."""
        artifact = real_open(parent_fd, target_name)
        artifacts.append(artifact)
        return artifact

    def recording_bind(parent: Path) -> tuple[int, tuple[int, int]]:
        """Record each retained writer parent descriptor."""
        result = real_bind(parent)
        parents.append(result[0])
        return result

    monkeypatch.setattr(writers, "_open_writer_temp", recording_open)
    monkeypatch.setattr(writers, "_bind_writer_parent", recording_bind)
    _write_actions(tmp_path / "success.npz")
    monkeypatch.setattr(
        writers,
        "_commit_writer_artifact",
        lambda _artifact, _name: (_ for _ in ()).throw(ValueError("forced")),
    )
    with pytest.raises(ValueError, match="forced"):
        _write_actions(tmp_path / "failure.npz")
    assert len(artifacts) == 2
    assert all(artifact.closed for artifact in artifacts)
    assert len(parents) == 2
    assert all(_descriptor_is_closed(descriptor) for descriptor in parents)


def test_writer_destination_must_be_absent(tmp_path: Path) -> None:
    """Refuse and preserve regular, directory, and symlink destination objects."""
    regular = tmp_path / "regular.npz"
    regular.write_bytes(b"REGULAR")
    directory = tmp_path / "directory.npz"
    directory.mkdir()
    link_target = tmp_path / "target.npz"
    link_target.write_bytes(b"TARGET")
    link = tmp_path / "link.npz"
    link.symlink_to(link_target)
    for target in (regular, directory, link):
        with pytest.raises(ValueError, match="absent"):
            _write_actions(target)
    assert regular.read_bytes() == b"REGULAR"
    assert directory.is_dir()
    assert link.is_symlink()
    assert link_target.read_bytes() == b"TARGET"


def test_writer_in_place_mutation_is_detected_and_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse writer success and preserve a final changed through its linked inode."""
    target = tmp_path / "actions.npz"
    real_link = owned.os.link

    def mutate(source: object, name: object, **kwargs: object) -> None:
        """Mutate the linked writer final before finalization verifies it."""
        real_link(source, name, **kwargs)  # type: ignore[arg-type]
        descriptor = os.open(
            name,
            os.O_WRONLY | os.O_TRUNC,
            dir_fd=kwargs["dst_dir_fd"],  # type: ignore[arg-type]
        )
        os.write(descriptor, b"MODIFIED")
        os.close(descriptor)

    monkeypatch.setattr(owned.os, "link", mutate)
    with pytest.raises(ValueError, match="changed"):
        _write_actions(target)
    assert target.read_bytes() == b"MODIFIED"


def test_writer_replacement_is_detected_and_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse writer success and preserve a foreign final replacement."""
    target = tmp_path / "actions.npz"
    real_link = owned.os.link

    def replace(source: object, name: object, **kwargs: object) -> None:
        """Replace the linked writer destination before final verification."""
        real_link(source, name, **kwargs)  # type: ignore[arg-type]
        _replace_final(kwargs["dst_dir_fd"], str(name), b"FOREIGN")  # type: ignore[arg-type]

    monkeypatch.setattr(owned.os, "link", replace)
    with pytest.raises(ValueError, match="changed"):
        _write_actions(target)
    assert target.read_bytes() == b"FOREIGN"


def test_audit_failure_cleanup_preserves_public_evidence_and_caller_entries(
    tmp_path: Path,
) -> None:
    """Close retained Audit handles without deleting candidate evidence or caller entries."""
    output, registry = _audit_registry(tmp_path)
    candidates = registry.create_directory((), "candidates", group="candidate")
    case = registry.create_directory(candidates.key, "case", group="candidate")
    owned_file = registry.write_file(case.key, "comparison.json", b"{}", group="candidate")
    caller_root = output.path / "caller.txt"
    caller_case = case.path / "caller.bin"
    caller_root.write_bytes(b"ROOT")
    caller_case.write_bytes(b"CASE")
    registry.cleanup()
    output.close()
    assert owned_file.closed
    assert (case.path / "comparison.json").read_bytes() == b"{}"
    assert caller_root.read_bytes() == b"ROOT"
    assert caller_case.read_bytes() == b"CASE"


def test_audit_descriptors_close_on_success_and_failure(tmp_path: Path) -> None:
    """Close all registry and root descriptors on both success-style and failure cleanup exits."""
    success, success_registry = _audit_registry(tmp_path, "success-audit")
    success_dir = success_registry.create_directory((), "candidates", group="candidate")
    success_file = success_registry.write_file(
        success_dir.key, "failure.json", b"{}", group="candidate"
    )
    success_registry.close()
    success.close()
    assert success_dir.closed
    assert success_file.closed
    assert success._handle.closed

    failure, failure_registry = _audit_registry(tmp_path, "failure-audit")
    failure_dir = failure_registry.create_directory((), "candidates", group="candidate")
    failure_file = failure_registry.write_file(
        failure_dir.key, "failure.json", b"{}", group="candidate"
    )
    failure_registry.cleanup()
    failure.close()
    assert failure_dir.closed
    assert failure_file.closed
    assert failure._handle.closed


def test_audit_preserves_unrelated_caller_files(tmp_path: Path) -> None:
    """Reject an injected root entry and preserve it during audit failure cleanup."""
    output, registry = _audit_registry(tmp_path)
    registry.create_directory((), "candidates", group="candidate")
    caller = output.path / "caller.txt"
    caller.write_bytes(b"CALLER")
    with pytest.raises(ArtifactAdmissionRefusal):
        registry.verify_public_tree()
    registry.cleanup()
    output.close()
    assert caller.read_bytes() == b"CALLER"


def test_audit_remove_private_deletes_only_exact_private_workspace(tmp_path: Path) -> None:
    """Remove exact private evidence while retaining registered public evidence."""
    output, registry = _audit_registry(tmp_path)
    candidates = registry.create_directory((), "candidates", group="candidate")
    public = registry.write_file(candidates.key, "failure.json", b"PUBLIC", group="candidate")
    workspace = registry.create_directory((), ".audit_workspace", group="private")
    private = registry.write_file(workspace.key, "state.npz", b"PRIVATE", group="private")

    registry.remove_private()

    assert private.closed
    assert not workspace.path.exists()
    assert (candidates.path / "failure.json").read_bytes() == b"PUBLIC"
    registry.verify_candidate_evidence()
    registry.verify_public_tree()
    registry.cleanup()
    output.close()
    assert public.closed


def test_audit_private_directories_require_exclusive_creation(tmp_path: Path) -> None:
    """Refuse and preserve a pre-existing private product directory instead of adopting it."""
    output, registry = _audit_registry(tmp_path)
    private = output.path / ".audit_workspace"
    private.mkdir()
    marker = private / "caller.txt"
    marker.write_bytes(b"CALLER")
    with pytest.raises(ArtifactAdmissionRefusal):
        registry.create_directory((), private.name, group="private")
    registry.cleanup()
    output.close()
    assert marker.read_bytes() == b"CALLER"


def test_audit_reverifies_candidate_evidence_before_success(tmp_path: Path) -> None:
    """Detect and preserve candidate evidence mutated before the final audit decision."""
    output, registry = _audit_registry(tmp_path)
    candidates = registry.create_directory((), "candidates", group="candidate")
    case = registry.create_directory(candidates.key, "case", group="candidate")
    artifact = registry.write_file(case.key, "comparison.json", b"{}", group="candidate")
    (case.path / "comparison.json").write_bytes(b"MODIFIED")
    with pytest.raises(ArtifactAdmissionRefusal):
        registry.verify_candidate_evidence()
    registry.cleanup()
    output.close()
    assert artifact.closed
    assert (case.path / "comparison.json").read_bytes() == b"MODIFIED"


def test_audit_verifies_public_tree_after_aggregate_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Make the public path/tree check the final audit success boundary."""
    events: list[str] = []

    def verify_inputs(_campaign: object) -> None:
        """Record the final live input verification boundary."""
        events.append("inputs")

    monkeypatch.setattr(audit_execution, "_verify_campaign_live", verify_inputs)
    audit_execution._verify_before_success(object(), _AuditOrderSpy(events))  # type: ignore[arg-type]
    assert events == ["inputs", "candidate", "aggregate", "tree"]


def test_compare_verifies_outputs_after_source_reverification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Place final comparison output verification after the final live source check."""
    events: list[str] = []
    live = SimpleNamespace(verify_sources_unchanged=lambda: events.append("sources"))
    receipt = SimpleNamespace(to_primitive=lambda: {})
    output = _SpyCompareOutput(events, tmp_path)
    monkeypatch.setattr(
        compare_orchestrator, "open_live_model_pair", lambda **_kwargs: nullcontext(live)
    )
    monkeypatch.setattr(
        compare_orchestrator, "_compare_within_live_pair", lambda **_kwargs: receipt
    )
    monkeypatch.setattr(compare_orchestrator, "render_markdown", lambda _receipt, _path: "report\n")
    monkeypatch.setattr(
        compare_orchestrator,
        "publish_results",
        lambda _output, **_kwargs: events.append("publish"),
    )
    role = SimpleNamespace(model_root="model", entrypoint="model.xml")
    config = SimpleNamespace(aliases=None, baseline=role, candidate=role)
    compare_orchestrator._execute_comparison(
        tmp_path / "comparison-config.json",
        b"{}",
        config,  # type: ignore[arg-type]
        tmp_path,
        output,  # type: ignore[arg-type]
        None,
        "d" * 64,
        SimpleNamespace(),  # type: ignore[arg-type]
        [],
    )
    assert events == ["sources", "publish", "sources", "outputs", "close"]


def test_certify_verifies_outputs_after_source_reverification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Place final certification output verification after both final source checks."""
    events: list[str] = []
    snapshots = {role: SimpleNamespace(role=role) for role in ("baseline", "candidate")}
    role_artifact = SimpleNamespace(serialized=SimpleNamespace(header_words=(1, 2, 3, 4)))
    published = _SpyPublished(events, tmp_path)

    def snapshot_context(_root: Path, _entrypoint: str, role: str) -> Any:
        """Return a no-op retained snapshot context for the requested role."""
        return nullcontext(snapshots[role])

    def publish(_output: object, **_kwargs: object) -> _SpyRetained:
        """Record publication and return a retained pair spy."""
        events.append("publish")
        return _SpyRetained(events)

    def verify_source(_snapshot: object, role: str) -> None:
        """Record each role's source revalidation."""
        events.append(f"source:{role}")

    monkeypatch.setattr(certify_run, "create_model_closure_snapshot", snapshot_context)
    monkeypatch.setattr(certify_run, "_certify_role", lambda *_args: role_artifact)
    monkeypatch.setattr(certify_run, "build_certify_runtime_identity", lambda _words: object())
    monkeypatch.setattr(certify_run, "_certify_decision", _certified_decision)
    monkeypatch.setattr(certify_run, "verify_model_closure_unchanged", verify_source)
    monkeypatch.setattr(certify_run, "publish_paired_results", publish)
    monkeypatch.setattr(certify_run, "render_markdown", lambda _receipt: "report\n")
    monkeypatch.setattr(
        certify_run,
        "verify_paired_results",
        lambda _output, _pair: events.append("outputs"),
    )
    certify_run._certify(
        SimpleNamespace(),  # type: ignore[arg-type]
        SimpleNamespace(model_root=tmp_path, entrypoint="baseline.xml"),  # type: ignore[arg-type]
        SimpleNamespace(model_root=tmp_path, entrypoint="candidate.xml"),  # type: ignore[arg-type]
        published,  # type: ignore[arg-type]
    )
    assert events == [
        "source:baseline",
        "source:candidate",
        "publish",
        "source:baseline",
        "source:candidate",
        "outputs",
        "pair-close",
        "dir-close",
    ]


def test_cleanup_preserves_reoccupied_temporary_name_after_finalization(tmp_path: Path) -> None:
    """Never delete a caller entry that reuses a relinquished private temporary name."""
    parent_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    artifact = owned.create_owned_artifact(parent_fd, ".owned.")
    temporary_name = artifact.temporary_name
    try:
        owned.write_owned_bytes(artifact, b"SEALED")
        owned.commit_owned_artifact(artifact, "result.bin")
        os.link(
            "result.bin",
            temporary_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        artifact.cleanup()
        assert (tmp_path / temporary_name).read_bytes() == b"SEALED"
        assert (tmp_path / "result.bin").read_bytes() == b"SEALED"
    finally:
        artifact.close()
        os.close(parent_fd)


def test_compare_rejects_replaced_public_output_path_after_source_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse success when Compare's public path is replaced after source revalidation."""
    output = compare_orchestrator.prepare_output_directory(tmp_path / "comparison")
    displaced = tmp_path / "comparison-displaced"
    attacker = tmp_path / "comparison-attacker"
    attacker.mkdir()

    class ReplacingContext:
        """Replace the public directory as the live model context closes."""

        def __enter__(self) -> SimpleNamespace:
            """Return a lightweight live model pair with a no-op source verifier."""
            return SimpleNamespace(verify_sources_unchanged=lambda: None)

        def __exit__(self, *_exc: object) -> None:
            """Swap the public path after product outputs exist but before final verification."""
            _replace_output_root(
                output.path, displaced, attacker, ("comparison.json", "comparison.md")
            )

    receipt = SimpleNamespace(to_primitive=lambda: {})
    monkeypatch.setattr(
        compare_orchestrator, "open_live_model_pair", lambda **_kwargs: ReplacingContext()
    )
    monkeypatch.setattr(
        compare_orchestrator, "_compare_within_live_pair", lambda **_kwargs: receipt
    )
    monkeypatch.setattr(compare_orchestrator, "render_markdown", lambda _receipt, _path: "GOOD\n")
    role = SimpleNamespace(model_root="model", entrypoint="model.xml")
    config = SimpleNamespace(aliases=None, baseline=role, candidate=role)

    with pytest.raises(ArtifactAdmissionRefusal):
        compare_orchestrator._execute_comparison(
            tmp_path / "comparison.json",
            b"{}",
            config,  # type: ignore[arg-type]
            tmp_path,
            output,
            None,
            "d" * 64,
            SimpleNamespace(),  # type: ignore[arg-type]
            [],
        )

    assert (attacker / "comparison.json").read_bytes() == b"ATTACKER"
    assert (attacker / "comparison.md").read_bytes() == b"ATTACKER"
    assert (displaced / "comparison.json").read_bytes() == b"{}"
    assert (displaced / "comparison.md").read_bytes() == b"GOOD\n"
    output._cleanup_retained_pair()
    output._paired().close()


def test_certify_rejects_replaced_public_output_path_after_source_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse success when Certify's public path is replaced after closure revalidation."""
    published = prepare_paired_output_directory(
        tmp_path / "certification", certify_run.CERTIFY_OUTPUT_NAMES
    )
    displaced = tmp_path / "certification-displaced"
    attacker = tmp_path / "certification-attacker"
    attacker.mkdir()
    snapshots = {role: SimpleNamespace(role=role) for role in ("baseline", "candidate")}
    role_artifact = SimpleNamespace(serialized=SimpleNamespace(header_words=(1, 2, 3, 4)))

    class SnapshotContext:
        """Return one lightweight snapshot and replace the public path on final context exit."""

        def __init__(self, role: str) -> None:
            """Retain the requested role identity."""
            self.role = role

        def __enter__(self) -> object:
            """Return the fixed snapshot for this role."""
            return snapshots[self.role]

        def __exit__(self, *_exc: object) -> None:
            """Replace the path when the outer baseline context exits."""
            if self.role == "baseline":
                _replace_output_root(
                    published.path,
                    displaced,
                    attacker,
                    ("certification.json", "certification.md"),
                )

    monkeypatch.setattr(
        certify_run,
        "create_model_closure_snapshot",
        lambda _root, _entrypoint, role: SnapshotContext(role),
    )
    monkeypatch.setattr(certify_run, "_certify_role", lambda *_args: role_artifact)
    monkeypatch.setattr(certify_run, "build_certify_runtime_identity", lambda _words: object())
    monkeypatch.setattr(certify_run, "_certify_decision", _certified_decision)
    monkeypatch.setattr(certify_run, "verify_model_closure_unchanged", lambda *_args: None)
    monkeypatch.setattr(certify_run, "render_markdown", lambda _receipt: "GOOD\n")

    with pytest.raises(ArtifactAdmissionRefusal):
        certify_run._certify(
            SimpleNamespace(),  # type: ignore[arg-type]
            SimpleNamespace(model_root=tmp_path, entrypoint="baseline.xml"),  # type: ignore[arg-type]
            SimpleNamespace(model_root=tmp_path, entrypoint="candidate.xml"),  # type: ignore[arg-type]
            published,
        )

    assert {(path.name, path.read_bytes()) for path in attacker.iterdir()} == {
        ("certification.json", b"ATTACKER"),
        ("certification.md", b"ATTACKER"),
    }
    assert (displaced / "certification.json").read_bytes() == b"{}\n"
    assert (displaced / "certification.md").read_bytes() == b"GOOD\n"
    published.close()


def test_audit_rejects_public_path_replaced_during_aggregate_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Make Audit recheck the public root after aggregate-byte verification completes."""
    output, registry = _audit_registry(tmp_path)
    aggregate = publish_paired_results(output, json_bytes=b"{}", markdown_text="GOOD\n")
    registry.register_pair((), aggregate, group="aggregate")
    displaced = tmp_path / "audit-displaced"
    attacker = tmp_path / "audit-attacker"
    attacker.mkdir()
    real_verify_aggregate = registry.verify_aggregate

    def replace_after_aggregate() -> None:
        """Verify the aggregate, then replace the public root before the final tree check."""
        real_verify_aggregate()
        _replace_output_root(
            output.path,
            displaced,
            attacker,
            ("timestep_audit.json", "timestep_audit.md"),
        )

    monkeypatch.setattr(registry, "verify_aggregate", replace_after_aggregate)
    monkeypatch.setattr(audit_execution, "_verify_campaign_live", lambda _campaign: None)

    with pytest.raises(ArtifactAdmissionRefusal):
        audit_execution._verify_before_success(object(), registry)  # type: ignore[arg-type]

    assert (attacker / "timestep_audit.json").read_bytes() == b"ATTACKER"
    assert (attacker / "timestep_audit.md").read_bytes() == b"ATTACKER"
    assert (displaced / "timestep_audit.json").read_bytes() == b"{}"
    assert (displaced / "timestep_audit.md").read_bytes() == b"GOOD\n"
    registry.close()
    output.close()
