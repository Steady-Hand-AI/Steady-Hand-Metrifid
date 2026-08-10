"""Tests for the shared strict two-file atomic publication primitive."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import metrifid._owned_artifacts as owned
from metrifid._atomic_output import (
    PairedOutputDirectory,
    PairedOutputNames,
    cleanup_paired_output_after_failure,
    prepare_paired_output_directory,
    publish_paired_results,
)
from metrifid._npz import ArtifactAdmissionRefusal
from metrifid.compare._output import (
    COMPARISON_OUTPUT_NAMES,
    OutputDirectory,
    cleanup_output_after_failure,
    prepare_output_directory,
    publish_results,
)
from metrifid.operational import OperationalReasonCode

_CERTIFY_NAMES = PairedOutputNames("certification.json", "certification.md")


def test_the_comparison_wrapper_publishes_the_accepted_names_and_bytes(tmp_path: Path) -> None:
    """Prevent partial or ambiguous assurance artifacts from reaching users.

    This scenario exercises the comparison wrapper publishes the accepted names and bytes;
    publication must remain atomic, deterministic, and recoverable after a failed commit.
    """
    output = prepare_output_directory(tmp_path / "out")
    assert isinstance(output, OutputDirectory)
    publish_results(output, json_bytes=b'{"a":1}', markdown_text="# report\n")
    assert output.json_path.name == "comparison.json"
    assert output.markdown_path.name == "comparison.md"
    assert output.json_path.read_bytes() == b'{"a":1}'
    assert output.markdown_path.read_bytes() == b"# report\n"


def test_the_two_publishers_write_identical_bytes_for_the_same_payload(tmp_path: Path) -> None:
    """Prevent partial or ambiguous assurance artifacts from reaching users.

    This scenario exercises the two publishers write identical bytes for the same payload;
    publication must remain atomic, deterministic, and recoverable after a failed commit.
    """
    payload = b'{"status":"X","n":[1,2,3]}'
    markdown = "# heading\n\n- item\n"
    comparison = prepare_output_directory(tmp_path / "comparison")
    publish_results(comparison, json_bytes=payload, markdown_text=markdown)
    certification = prepare_paired_output_directory(tmp_path / "certification", _CERTIFY_NAMES)
    publish_paired_results(certification, json_bytes=payload, markdown_text=markdown)
    assert comparison.json_path.read_bytes() == certification.json_path.read_bytes() == payload
    assert comparison.markdown_path.read_bytes() == certification.markdown_path.read_bytes()


def test_the_primitive_publishes_whichever_pair_of_names_it_is_given(tmp_path: Path) -> None:
    """Prevent partial or ambiguous assurance artifacts from reaching users.

    This scenario exercises the primitive publishes whichever pair of names it is given;
    publication must remain atomic, deterministic, and recoverable after a failed commit.
    """
    output = prepare_paired_output_directory(tmp_path / "out", _CERTIFY_NAMES)
    publish_paired_results(output, json_bytes=b"{}", markdown_text="x\n")
    assert sorted(item.name for item in output.path.iterdir()) == [
        "certification.json",
        "certification.md",
    ]


def test_an_absent_directory_is_created_with_the_accepted_mode(tmp_path: Path) -> None:
    """Prevent partial or ambiguous assurance artifacts from reaching users.

    This scenario exercises an absent directory is created with the accepted mode; publication
    must remain atomic, deterministic, and recoverable after a failed commit.
    """
    output = prepare_paired_output_directory(tmp_path / "fresh", _CERTIFY_NAMES)
    assert output.path.is_dir()
    assert output.path.stat().st_mode & 0o777 == 0o755


def test_an_existing_empty_directory_is_admitted(tmp_path: Path) -> None:
    """Prevent partial or ambiguous assurance artifacts from reaching users.

    This scenario exercises an existing empty directory is admitted; publication must remain
    atomic, deterministic, and recoverable after a failed commit.
    """
    target = tmp_path / "empty"
    target.mkdir()
    assert prepare_paired_output_directory(target, _CERTIFY_NAMES).path == target.absolute()


def test_a_nonempty_directory_refuses(tmp_path: Path) -> None:
    """Prevent partial or ambiguous assurance artifacts from reaching users.

    This scenario exercises a nonempty directory refuses; publication must remain atomic,
    deterministic, and recoverable after a failed commit.
    """
    target = tmp_path / "used"
    target.mkdir()
    (target / "leftover").write_text("x", encoding="utf-8")
    with pytest.raises(ArtifactAdmissionRefusal) as caught:
        prepare_paired_output_directory(target, _CERTIFY_NAMES)
    assert caught.value.reason is OperationalReasonCode.OUTPUT_DIRECTORY_NOT_EMPTY


def test_a_symlinked_output_path_refuses(tmp_path: Path) -> None:
    """Prevent partial or ambiguous assurance artifacts from reaching users.

    This scenario exercises a symlinked output path refuses; publication must remain atomic,
    deterministic, and recoverable after a failed commit.
    """
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(ArtifactAdmissionRefusal) as caught:
        prepare_paired_output_directory(link, _CERTIFY_NAMES)
    assert caught.value.reason is OperationalReasonCode.OUTPUT_PATH_INVALID


def test_a_symlinked_parent_refuses(tmp_path: Path) -> None:
    """Prevent partial or ambiguous assurance artifacts from reaching users.

    This scenario exercises a symlinked parent refuses; publication must remain atomic,
    deterministic, and recoverable after a failed commit.
    """
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    with pytest.raises(ArtifactAdmissionRefusal) as caught:
        prepare_paired_output_directory(link / "out", _CERTIFY_NAMES)
    assert caught.value.reason is OperationalReasonCode.OUTPUT_PATH_INVALID


def test_a_directory_that_filled_after_admission_refuses_at_publish(tmp_path: Path) -> None:
    """Prevent partial or ambiguous assurance artifacts from reaching users.

    This scenario exercises a directory that filled after admission refuses at publish;
    publication must remain atomic, deterministic, and recoverable after a failed commit.
    """
    output = prepare_paired_output_directory(tmp_path / "out", _CERTIFY_NAMES)
    (output.path / "raced").write_text("x", encoding="utf-8")
    with pytest.raises(ArtifactAdmissionRefusal) as caught:
        publish_paired_results(output, json_bytes=b"{}", markdown_text="x\n")
    assert caught.value.reason is OperationalReasonCode.OUTPUT_DIRECTORY_NOT_EMPTY
    assert not output.json_path.exists()
    assert not output.markdown_path.exists()


def test_paired_output_does_not_follow_replaced_directory(tmp_path: Path) -> None:
    """Keep publication and cleanup confined to the directory admitted by descriptor."""
    public = tmp_path / "public"
    admitted = tmp_path / "admitted"
    outside = tmp_path / "outside"
    outside.mkdir()
    output = prepare_paired_output_directory(public, _CERTIFY_NAMES)
    public.rename(admitted)
    public.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ArtifactAdmissionRefusal) as caught:
        publish_paired_results(output, json_bytes=b"{}", markdown_text="x\n")

    assert caught.value.reason is OperationalReasonCode.OUTPUT_PATH_INVALID
    assert list(admitted.iterdir()) == []
    assert list(outside.iterdir()) == []


def test_failed_second_link_preserves_first_and_injected_second_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Preserve both public links when an injected second hardlink causes EEXIST."""
    output = prepare_paired_output_directory(tmp_path / "out", _CERTIFY_NAMES)
    real_link = owned.os.link
    calls = 0

    def inject_second(source: object, target: object, **kwargs: object) -> None:
        """Inject the second final before its real no-clobber link."""
        nonlocal calls
        calls += 1
        if calls == 2:
            real_link(source, target, **kwargs)  # type: ignore[arg-type]
        real_link(source, target, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(owned.os, "link", inject_second)
    with pytest.raises(ArtifactAdmissionRefusal) as caught:
        publish_paired_results(output, json_bytes=b"{}", markdown_text="x\n")
    assert caught.value.reason is OperationalReasonCode.OUTPUT_WRITE_FAILED
    assert calls == 2
    assert output.json_path.read_bytes() == b"{}"
    assert output.markdown_path.read_bytes() == b"x\n"


def test_cleanup_preserves_public_finals_for_either_name_pair(tmp_path: Path) -> None:
    """Preserve each publisher's committed pair and every unrelated caller entry."""
    certification = prepare_paired_output_directory(tmp_path / "certification", _CERTIFY_NAMES)
    certify_pair = publish_paired_results(
        certification, json_bytes=b"{}", markdown_text="certified\n"
    )
    certify_foreign = certification.path / ".certification.json.foreign.tmp"
    certify_foreign.write_bytes(b"foreign")
    cleanup_paired_output_after_failure(certification, certify_pair)
    assert certification.json_path.read_bytes() == b"{}"
    assert certification.markdown_path.read_bytes() == b"certified\n"
    assert certify_foreign.read_bytes() == b"foreign"

    comparison = prepare_output_directory(tmp_path / "comparison")
    publish_results(comparison, json_bytes=b"{}", markdown_text="compared\n")
    comparison_foreign = comparison.path / "unrelated"
    comparison_foreign.write_bytes(b"foreign")
    cleanup_output_after_failure(comparison)
    assert comparison.json_path.read_bytes() == b"{}"
    assert comparison.markdown_path.read_bytes() == b"compared\n"
    assert comparison_foreign.read_bytes() == b"foreign"


def test_cleanup_accepts_an_absent_output(tmp_path: Path) -> None:
    """Allow failure cleanup before any output directory was admitted."""
    cleanup_paired_output_after_failure(None)
    cleanup_output_after_failure(None)


def test_the_temporary_prefix_is_derived_from_the_final_name(tmp_path: Path) -> None:
    """Preserve an unregistered lookalike temporary derived from a final name."""
    output = prepare_paired_output_directory(tmp_path / "out", _CERTIFY_NAMES)
    stray = output.path / ".certification.json.zzz.tmp"
    stray.write_text("t", encoding="utf-8")
    cleanup_paired_output_after_failure(output)
    assert stray.read_text(encoding="utf-8") == "t"


def test_paired_names_must_be_plain_and_distinct() -> None:
    """Reject path-bearing, duplicate, and parent-like paired final names."""
    with pytest.raises(ValueError):
        PairedOutputNames("a/b.json", "c.md")
    with pytest.raises(ValueError):
        PairedOutputNames("same.json", "same.json")
    with pytest.raises(ValueError):
        PairedOutputNames("..", "c.md")


def test_json_bytes_must_be_bytes(tmp_path: Path) -> None:
    """Refuse text at the exact-byte JSON publication boundary."""
    output = prepare_paired_output_directory(tmp_path / "out", _CERTIFY_NAMES)
    with pytest.raises(TypeError):
        publish_paired_results(
            output,
            json_bytes="not bytes",
            markdown_text="x\n",  # type: ignore[arg-type]
        )


def test_the_comparison_directory_type_still_exposes_the_accepted_attributes(
    tmp_path: Path,
) -> None:
    """Keep the accepted comparison path attributes on both directory wrappers."""
    output = OutputDirectory(tmp_path)
    assert output.path == tmp_path
    assert output.json_path == tmp_path / "comparison.json"
    assert output.markdown_path == tmp_path / "comparison.md"
    paired = PairedOutputDirectory(tmp_path, COMPARISON_OUTPUT_NAMES)
    assert paired.json_path == output.json_path
    assert paired.markdown_path == output.markdown_path
    output._paired().close()
    paired.close()


def test_existing_final_is_refused_and_preserved(tmp_path: Path) -> None:
    """Refuse and preserve every existing final instead of overwriting it."""
    output = prepare_paired_output_directory(tmp_path / "out", _CERTIFY_NAMES)
    output.json_path.write_bytes(b"stale receipt")
    output.markdown_path.write_text("stale markdown\n", encoding="utf-8")
    with pytest.raises(ArtifactAdmissionRefusal):
        publish_paired_results(output, json_bytes=b"{}", markdown_text="fresh\n")
    assert output.json_path.read_bytes() == b"stale receipt"
    assert output.markdown_path.read_text(encoding="utf-8") == "stale markdown\n"


def test_replacement_is_atomic_from_a_reader_point_of_view(tmp_path: Path) -> None:
    """Expose only complete sealed temporary bytes at each no-clobber link boundary."""
    output = prepare_paired_output_directory(tmp_path / "out", _CERTIFY_NAMES)
    payload = b'{"receipt":"complete"}'
    observed: list[bytes] = []
    real_link = owned.os.link

    def watching(source: object, target: object, **kwargs: object) -> None:
        """Read the complete private source immediately before creating its hard link."""
        descriptor = os.open(
            source,
            os.O_RDONLY,
            dir_fd=kwargs["src_dir_fd"],  # type: ignore[arg-type]
        )
        with os.fdopen(descriptor, "rb") as stream:
            observed.append(stream.read())
        real_link(source, target, **kwargs)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(owned.os, "link", watching)
        retained = publish_paired_results(output, json_bytes=payload, markdown_text="done\n")
    assert observed == [payload, b"done\n"]
    assert output.json_path.read_bytes() == payload
    assert output.markdown_path.read_text(encoding="utf-8") == "done\n"
    retained.close()
    output.close()


def test_failed_pair_link_preserves_injected_hardlink_final(tmp_path: Path) -> None:
    """Preserve an injected first hardlink when the real no-clobber link gets EEXIST."""
    output = prepare_paired_output_directory(tmp_path / "out", _CERTIFY_NAMES)
    real_link = owned.os.link
    injected = False

    def inject_first(source: object, target: object, **kwargs: object) -> None:
        """Inject the first final immediately before its real no-clobber link."""
        nonlocal injected
        if not injected:
            injected = True
            real_link(source, target, **kwargs)  # type: ignore[arg-type]
        real_link(source, target, **kwargs)  # type: ignore[arg-type]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(owned.os, "link", inject_first)
        with pytest.raises(ArtifactAdmissionRefusal) as caught:
            publish_paired_results(output, json_bytes=b"{}", markdown_text="x\n")
    assert caught.value.reason is OperationalReasonCode.OUTPUT_WRITE_FAILED
    assert output.json_path.read_bytes() == b"{}"
    assert not output.markdown_path.exists()


def test_the_comparison_publisher_writes_the_accepted_bytes_unchanged(tmp_path: Path) -> None:
    """Correction A may not move a single byte of an accepted comparison output."""
    payload = b'{"status":"NO_MATERIAL_DIFFERENCE_ON_DECLARED_WORKLOAD"}'
    markdown = "# comparison\n\n- unchanged\n"
    output = prepare_output_directory(tmp_path / "comparison")
    publish_results(output, json_bytes=payload, markdown_text=markdown)
    assert output.json_path.name == "comparison.json"
    assert output.markdown_path.name == "comparison.md"
    assert output.json_path.read_bytes() == payload
    assert output.markdown_path.read_bytes() == markdown.encode("utf-8")
