"""Integration coverage for portable, independently owned evidence snapshots."""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from metrifid.runtime_review import _owned_output
from metrifid.runtime_review._owned_output import (
    OwnedEvidenceCell,
    OwnedRuntimeReviewOutputError,
    PublishedRuntimeReviewOutput,
    prepare_owned_runtime_review_output,
    verify_published_runtime_review_output,
)
from metrifid.runtime_review._receipt_validation import load_and_validate_runtime_review_receipt

_MEMBER_NAMES = (
    "fixture.xml",
    "input_manifest.json",
    "model.mjb",
    "result.json",
    "trace.npz",
)


@dataclass(frozen=True, slots=True)
class _AdmittedCell:
    """Represent one already measured six-member source cell for snapshot tests."""

    profile_role: str
    step_dt: str
    repeat_id: int
    source_directory: Path
    member_sha256: dict[str, str]


def _source_grid(root: Path) -> tuple[_AdmittedCell, ...]:
    """Create one complete deterministic grid of byte-distinct synthetic source cells."""
    cells: list[_AdmittedCell] = []
    for ordinal, (role, step, repeat) in enumerate(
        (role, step, repeat)
        for role in ("baseline", "candidate")
        for step in ("0.004", "0.002", "0.001")
        for repeat in (0, 1)
    ):
        directory = root / f"cell_{ordinal:02d}"
        directory.mkdir(parents=True)
        hashes: dict[str, str] = {}
        for name in _MEMBER_NAMES:
            payload = f"synthetic {ordinal} {name}\n".encode()
            (directory / name).write_bytes(payload)
            hashes[name] = hashlib.sha256(payload).hexdigest()
        manifest = "".join(f"{hashes[name]}  {name}\n" for name in _MEMBER_NAMES).encode()
        (directory / "CHECKSUMS.sha256").write_bytes(manifest)
        hashes["CHECKSUMS.sha256"] = hashlib.sha256(manifest).hexdigest()
        cells.append(_AdmittedCell(role, step, repeat, directory, hashes))
    return tuple(cells)


def _inject_foreign_member_and_reject(staging: Path) -> None:
    """Add caller-owned bytes to staging before simulating an independent refusal."""
    (staging / "foreign").write_bytes(b"preserve")
    raise ValueError("independent decision reconstruction failed")


def _publish_synthetic_snapshot(tmp_path: Path) -> PublishedRuntimeReviewOutput:
    """Publish one exact synthetic tree for post-publication integrity mutations."""
    sources = _source_grid(tmp_path / "source")
    output_dir = tmp_path / "published"
    with prepare_owned_runtime_review_output(output_dir, b"{}\n") as staging:
        staging.copy_evidence_cells(sources)
        return staging.publish({"receipt_sha256": "0" * 64}, "# Synthetic snapshot\n")


def test_published_snapshot_is_a_portable_byte_copy_after_sources_disappear(
    tmp_path: Path,
) -> None:
    """Preserve all cells without links and reverify them after removing every source cell."""
    sources = _source_grid(tmp_path / "source")
    source_inodes = {
        (cell.profile_role, cell.step_dt, cell.repeat_id, name): (cell.source_directory / name)
        .stat()
        .st_ino
        for cell in sources
        for name in ("CHECKSUMS.sha256", *_MEMBER_NAMES)
    }
    output_dir = tmp_path / "published"
    with prepare_owned_runtime_review_output(output_dir, b"{}\n") as staging:
        records = staging.copy_evidence_cells(sources)
        published = staging.publish({"receipt_sha256": "0" * 64}, "# Synthetic snapshot\n")

    shutil.rmtree(tmp_path / "source")
    verify_published_runtime_review_output(published)
    assert published.admitted_configuration.read_bytes() == b"{}\n"
    assert len(records) == 12
    for cell in records:
        for member in cell.members:
            copied = published.root / cell.directory / member.name
            key = (cell.profile_role, cell.step_dt, cell.repeat_id, member.name)
            assert copied.is_file()
            assert not copied.is_symlink()
            assert copied.stat().st_ino != source_inodes[key]
            assert copied.stat().st_nlink == 1
            assert hashlib.sha256(copied.read_bytes()).hexdigest() == member.sha256


def test_role_profile_identities_are_owned_with_the_portable_snapshot(tmp_path: Path) -> None:
    """Copy both v2 preflight identities so replay does not depend on original paths."""
    sources = _source_grid(tmp_path / "source")
    identity_sources: dict[str, tuple[Path, str]] = {}
    for role in ("baseline", "candidate"):
        path = tmp_path / f"{role}.json"
        payload = f'{{"profile_role":"{role}"}}\n'.encode()
        path.write_bytes(payload)
        identity_sources[role] = (path, hashlib.sha256(payload).hexdigest())
    output_dir = tmp_path / "published"
    with prepare_owned_runtime_review_output(output_dir, b"{}\n") as staging:
        staging.copy_evidence_cells(sources)
        identities = staging.copy_profile_identities(identity_sources)
        published = staging.publish({"receipt_sha256": "0" * 64}, "# Synthetic snapshot\n")

    for path, _digest in identity_sources.values():
        path.unlink()
    verify_published_runtime_review_output(published)
    assert [identity.profile_role for identity in identities] == ["baseline", "candidate"]
    assert [identity.locator.as_posix() for identity in identities] == [
        "profile_identities/baseline.json",
        "profile_identities/candidate.json",
    ]


def test_role_profile_identity_substitution_is_detected_after_publication(
    tmp_path: Path,
) -> None:
    """Reject changed owned v2 preflight bytes at the final verification boundary."""
    sources = _source_grid(tmp_path / "source")
    identity_sources: dict[str, tuple[Path, str]] = {}
    for role in ("baseline", "candidate"):
        path = tmp_path / f"{role}.json"
        payload = f'{{"profile_role":"{role}"}}\n'.encode()
        path.write_bytes(payload)
        identity_sources[role] = (path, hashlib.sha256(payload).hexdigest())
    output_dir = tmp_path / "published"
    with prepare_owned_runtime_review_output(output_dir, b"{}\n") as staging:
        staging.copy_evidence_cells(sources)
        staging.copy_profile_identities(identity_sources)
        published = staging.publish({"receipt_sha256": "0" * 64}, "# Synthetic snapshot\n")
    target = published.root / "profile_identities" / "candidate.json"
    target.write_bytes(b"substituted\n")

    with pytest.raises(OwnedRuntimeReviewOutputError, match="published profile identity changed"):
        verify_published_runtime_review_output(published)


def test_existing_output_is_preserved_without_clobbering(tmp_path: Path) -> None:
    """Refuse an existing configured root while preserving its exact bytes."""
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel"
    sentinel.write_bytes(b"keep")

    with pytest.raises(OwnedRuntimeReviewOutputError, match="already exists"):
        prepare_owned_runtime_review_output(output_dir, b"{}\n")

    assert sentinel.read_bytes() == b"keep"


def test_changed_owned_member_fails_without_deleting_the_publication(tmp_path: Path) -> None:
    """Detect post-publication evidence mutation while preserving the failed evidence tree."""
    sources = _source_grid(tmp_path / "source")
    output_dir = tmp_path / "published"
    with prepare_owned_runtime_review_output(output_dir, b"{}\n") as staging:
        records = staging.copy_evidence_cells(sources)
        published = staging.publish({"receipt_sha256": "0" * 64}, "# Synthetic snapshot\n")
    changed = published.root / records[0].directory / "trace.npz"
    changed.write_bytes(b"substituted")

    with pytest.raises(OwnedRuntimeReviewOutputError, match="do not match"):
        verify_published_runtime_review_output(published)

    assert changed.read_bytes() == b"substituted"
    assert output_dir.exists()


def test_public_replay_refuses_and_preserves_a_postpublication_root_extra(
    tmp_path: Path,
) -> None:
    """Reject an extra root member before receipt parsing without deleting foreign bytes."""
    published = _publish_synthetic_snapshot(tmp_path)
    foreign = published.root / "foreign"
    foreign.write_bytes(b"preserve")

    with pytest.raises(OwnedRuntimeReviewOutputError, match="invalid closed shape"):
        load_and_validate_runtime_review_receipt(published.runtime_review_json)

    assert foreign.read_bytes() == b"preserve"


def test_final_verification_refuses_and_preserves_an_intermediate_extra(
    tmp_path: Path,
) -> None:
    """Apply the same closed-tree rule at the final SDK success boundary."""
    published = _publish_synthetic_snapshot(tmp_path)
    foreign = published.root / "evidence" / "baseline" / "foreign"
    foreign.write_bytes(b"preserve")

    with pytest.raises(OwnedRuntimeReviewOutputError, match="invalid closed shape"):
        verify_published_runtime_review_output(published)

    assert foreign.read_bytes() == b"preserve"


def test_final_verification_rebinds_the_tree_after_all_cell_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catch an intermediate member inserted through the last per-cell validation seam."""
    published = _publish_synthetic_snapshot(tmp_path)
    original = _owned_output.validate_owned_evidence_cell
    foreign = published.root / "evidence" / "baseline" / "foreign"
    calls = 0

    def validate_then_inject(root: Path, cell: OwnedEvidenceCell) -> Path:
        """Insert one foreign intermediate member after the final cell recheck."""
        nonlocal calls
        result = original(root, cell)
        calls += 1
        if calls == len(published.evidence_cells):
            foreign.write_bytes(b"preserve")
        return result

    monkeypatch.setattr(_owned_output, "validate_owned_evidence_cell", validate_then_inject)

    with pytest.raises(OwnedRuntimeReviewOutputError, match="invalid closed shape"):
        verify_published_runtime_review_output(published)

    assert calls == 12
    assert foreign.read_bytes() == b"preserve"


def test_public_replay_refuses_a_missing_nested_member_without_cleanup(tmp_path: Path) -> None:
    """Reject an incomplete leaf while retaining the rest of the incomplete publication."""
    published = _publish_synthetic_snapshot(tmp_path)
    member = published.root / "evidence" / "baseline" / "0p004" / "repeat_0" / "trace.npz"
    preserved = tmp_path / "preserved-trace.npz"
    member.rename(preserved)

    with pytest.raises(OwnedRuntimeReviewOutputError, match="invalid closed shape"):
        load_and_validate_runtime_review_receipt(published.runtime_review_json)

    assert published.root.is_dir()
    assert preserved.read_bytes() == b"synthetic 0 trace.npz\n"


def test_public_replay_refuses_a_symlinked_nested_member_without_cleanup(tmp_path: Path) -> None:
    """Reject a leaf symlink through no-follow descriptors and preserve its target."""
    published = _publish_synthetic_snapshot(tmp_path)
    member = published.root / "evidence" / "baseline" / "0p004" / "repeat_0" / "trace.npz"
    target = tmp_path / "foreign-trace.npz"
    target.write_bytes(b"preserve")
    member.unlink()
    member.symlink_to(target)

    with pytest.raises(OwnedRuntimeReviewOutputError, match="unavailable"):
        load_and_validate_runtime_review_receipt(published.runtime_review_json)

    assert member.is_symlink()
    assert target.read_bytes() == b"preserve"


def test_public_replay_refuses_a_hardlinked_owned_member_without_cleanup(tmp_path: Path) -> None:
    """Require each supposedly copied member to retain exactly one filesystem link."""
    published = _publish_synthetic_snapshot(tmp_path)
    member = published.root / "evidence" / "baseline" / "0p004" / "repeat_0" / "trace.npz"
    foreign_link = tmp_path / "foreign-hardlink.npz"
    os.link(member, foreign_link)

    with pytest.raises(OwnedRuntimeReviewOutputError, match="exactly one link"):
        load_and_validate_runtime_review_receipt(published.runtime_review_json)

    assert member.read_bytes() == b"synthetic 0 trace.npz\n"
    assert foreign_link.read_bytes() == b"synthetic 0 trace.npz\n"


def test_public_replay_refuses_wholesale_markdown_tampering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind the human report bytes exactly to their canonical receipt rendering."""
    from metrifid.runtime_review import _receipt_validation

    published = _publish_synthetic_snapshot(tmp_path)
    published.runtime_review_markdown.write_bytes(b"# Substituted report\n")
    monkeypatch.setattr(_receipt_validation, "_validate_document_schema", lambda _receipt: None)
    monkeypatch.setattr(_receipt_validation, "validate_self_hash", lambda *_args: None)
    monkeypatch.setattr(
        _receipt_validation,
        "render_runtime_review_markdown",
        lambda _receipt: "# Synthetic snapshot\n",
    )

    with pytest.raises(ValueError, match="Markdown"):
        load_and_validate_runtime_review_receipt(published.runtime_review_json)

    assert published.runtime_review_markdown.read_bytes() == b"# Substituted report\n"


def test_symlinked_source_member_refuses_and_preserves_private_staging(tmp_path: Path) -> None:
    """Reject a source symlink without recursively deleting private failure evidence."""
    sources = list(_source_grid(tmp_path / "source"))
    source = sources[0]
    member = source.source_directory / "trace.npz"
    member.unlink()
    member.symlink_to(source.source_directory / "fixture.xml")
    output_dir = tmp_path / "published"

    with pytest.raises(OwnedRuntimeReviewOutputError, match="unavailable"):
        with prepare_owned_runtime_review_output(output_dir, b"{}\n") as staging:
            staging.copy_evidence_cells(sources)

    assert (output_dir / ".runtime_review.staging").is_dir()
    assert not (output_dir / "runtime_review").exists()
    assert member.is_symlink()


def test_independent_refusal_preserves_foreign_staging_bytes_without_a_completed_output(
    tmp_path: Path,
) -> None:
    """Preserve caller-inserted staging bytes and leave the public final absent on refusal."""
    sources = _source_grid(tmp_path / "source")
    output_dir = tmp_path / "published"

    with pytest.raises(ValueError, match="decision reconstruction"):
        with prepare_owned_runtime_review_output(output_dir, b"{}\n") as staging:
            staging.copy_evidence_cells(sources)
            staging.publish(
                {"receipt_sha256": "0" * 64},
                "# Synthetic snapshot\n",
                prepublication_validator=_inject_foreign_member_and_reject,
            )

    staging = output_dir / ".runtime_review.staging"
    assert (staging / "foreign").read_bytes() == b"preserve"
    assert not (output_dir / "runtime_review").exists()


def test_raced_in_final_directory_is_never_replaced_or_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refuse an atomic publication race and preserve the competing directory inode and bytes."""
    sources = _source_grid(tmp_path / "source")
    output_dir = tmp_path / "published"
    original = _owned_output._rename_directory_noreplace
    raced_inode: int | None = None

    def inject_competing_directory(parent_fd: int, source: str, destination: str) -> None:
        """Create a competing nonempty final immediately before atomic publication."""
        nonlocal raced_inode
        os.mkdir(destination, mode=0o700, dir_fd=parent_fd)
        final_fd = os.open(destination, os.O_RDONLY | os.O_DIRECTORY, dir_fd=parent_fd)
        try:
            sentinel_fd = os.open(
                "foreign",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=final_fd,
            )
            try:
                os.write(sentinel_fd, b"preserve")
            finally:
                os.close(sentinel_fd)
        finally:
            os.close(final_fd)
        raced_inode = os.stat(destination, dir_fd=parent_fd, follow_symlinks=False).st_ino
        original(parent_fd, source, destination)

    monkeypatch.setattr(
        _owned_output,
        "_rename_directory_noreplace",
        inject_competing_directory,
    )
    with pytest.raises(OwnedRuntimeReviewOutputError, match="already exists"):
        with prepare_owned_runtime_review_output(output_dir, b"{}\n") as staging:
            staging.copy_evidence_cells(sources)
            staging.publish({"receipt_sha256": "0" * 64}, "# Synthetic snapshot\n")

    final = output_dir / "runtime_review"
    assert raced_inode is not None
    assert final.stat().st_ino == raced_inode
    assert (final / "foreign").read_bytes() == b"preserve"
    assert (output_dir / ".runtime_review.staging").is_dir()


def test_raced_in_empty_final_directory_is_not_replaced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the atomic exclusion flag against an empty competing directory."""
    sources = _source_grid(tmp_path / "source")
    output_dir = tmp_path / "published"
    original = _owned_output._rename_directory_noreplace
    raced_inode: int | None = None

    def inject_empty_directory(parent_fd: int, source: str, destination: str) -> None:
        """Create a competing empty final immediately before atomic publication."""
        nonlocal raced_inode
        os.mkdir(destination, mode=0o700, dir_fd=parent_fd)
        raced_inode = os.stat(destination, dir_fd=parent_fd, follow_symlinks=False).st_ino
        original(parent_fd, source, destination)

    monkeypatch.setattr(
        _owned_output,
        "_rename_directory_noreplace",
        inject_empty_directory,
    )
    with pytest.raises(OwnedRuntimeReviewOutputError, match="already exists"):
        with prepare_owned_runtime_review_output(output_dir, b"{}\n") as staging:
            staging.copy_evidence_cells(sources)
            staging.publish({"receipt_sha256": "0" * 64}, "# Synthetic snapshot\n")

    final = output_dir / "runtime_review"
    assert raced_inode is not None
    assert final.stat().st_ino == raced_inode
    assert tuple(final.iterdir()) == ()
    assert (output_dir / ".runtime_review.staging").is_dir()


def test_final_verification_refuses_a_replaced_published_path(tmp_path: Path) -> None:
    """Preserve a foreign final-directory replacement at the last success boundary."""
    sources = _source_grid(tmp_path / "source")
    output_dir = tmp_path / "published"
    with prepare_owned_runtime_review_output(output_dir, b"{}\n") as staging:
        staging.copy_evidence_cells(sources)
        published = staging.publish({"receipt_sha256": "0" * 64}, "# Synthetic snapshot\n")

    original = output_dir / "original_runtime_review"
    published.root.rename(original)
    published.root.mkdir()
    sentinel = published.root / "foreign"
    sentinel.write_bytes(b"preserve")

    with pytest.raises(OwnedRuntimeReviewOutputError, match="pathname was replaced"):
        verify_published_runtime_review_output(published)

    assert sentinel.read_bytes() == b"preserve"
    assert original.is_dir()


def test_verification_preserves_a_foreign_output_root_sibling(tmp_path: Path) -> None:
    """Ignore an unrelated output-root sibling while verifying the exact published final."""
    sources = _source_grid(tmp_path / "source")
    output_dir = tmp_path / "published"
    with prepare_owned_runtime_review_output(output_dir, b"{}\n") as staging:
        staging.copy_evidence_cells(sources)
        published = staging.publish({"receipt_sha256": "0" * 64}, "# Synthetic snapshot\n")
    sibling = output_dir / "foreign"
    sibling.write_bytes(b"preserve")

    verify_published_runtime_review_output(published)

    assert sibling.read_bytes() == b"preserve"
    assert published.root.exists()


def test_nested_missing_output_parents_publish_one_verified_tree(tmp_path: Path) -> None:
    """Create and preserve a portable nested output path through newly owned ancestors."""
    sources = _source_grid(tmp_path / "source")
    output_dir = tmp_path / "new" / "nested" / "published"
    with prepare_owned_runtime_review_output(output_dir, b"{}\n") as staging:
        staging.copy_evidence_cells(sources)
        published = staging.publish({"receipt_sha256": "0" * 64}, "# Synthetic snapshot\n")

    verify_published_runtime_review_output(published)
    assert published.root.is_dir()
