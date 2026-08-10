"""Output-directory and paired-publication failure-path tests for comparison."""

from __future__ import annotations

from pathlib import Path

import pytest

from metrifid import _owned_artifacts
from metrifid._npz import ArtifactAdmissionRefusal
from metrifid.compare._output import prepare_output_directory, publish_results
from metrifid.operational import OperationalReasonCode


def test_absent_and_empty_directories_publish_complete_pair(tmp_path: Path) -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises absent and empty directories publish complete pair; status,
    numerical evidence, and artifact publication must remain stable for the declared workload.
    """
    output = prepare_output_directory(tmp_path / "new")
    publish_results(output, json_bytes=b"{}", markdown_text="# report\n")
    assert output.json_path.read_bytes() == b"{}"
    assert output.markdown_path.read_text(encoding="utf-8") == "# report\n"

    empty = tmp_path / "empty"
    empty.mkdir()
    admitted = prepare_output_directory(empty)
    assert admitted.path == empty.absolute()


def test_file_symlink_and_nonempty_directory_are_refused(tmp_path: Path) -> None:
    """Keep workload comparison decisions reproducible.

    This scenario exercises file symlink and nonempty directory are refused; status, numerical
    evidence, and artifact publication must remain stable for the declared workload.
    """
    file_path = tmp_path / "file"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(ArtifactAdmissionRefusal) as file_error:
        prepare_output_directory(file_path)
    assert file_error.value.reason is OperationalReasonCode.OUTPUT_PATH_INVALID

    link = tmp_path / "link"
    link.symlink_to(file_path)
    with pytest.raises(ArtifactAdmissionRefusal) as link_error:
        prepare_output_directory(link)
    assert link_error.value.reason is OperationalReasonCode.OUTPUT_PATH_INVALID

    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "entry").write_text("x", encoding="utf-8")
    with pytest.raises(ArtifactAdmissionRefusal) as nonempty_error:
        prepare_output_directory(nonempty)
    assert nonempty_error.value.reason is OperationalReasonCode.OUTPUT_DIRECTORY_NOT_EMPTY


def test_second_no_clobber_link_conflict_preserves_public_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse a second-link conflict without deleting either public entry.

    The injected Markdown entry is created from Metrifid's sealed temporary immediately before
    the real no-clobber link attempt. Publication must fail, while both the already-linked JSON
    and the concurrently linked Markdown retain the exact intended bytes.
    """
    output = prepare_output_directory(tmp_path / "out")
    original_link = _owned_artifacts.os.link
    calls = 0

    def inject_second_final(*args: object, **kwargs: object) -> None:
        """Create the second final first so Metrifid's no-clobber link deterministically fails."""
        nonlocal calls
        calls += 1
        if calls == 2:
            original_link(*args, **kwargs)
        original_link(*args, **kwargs)

    monkeypatch.setattr(_owned_artifacts.os, "link", inject_second_final)
    with pytest.raises(ArtifactAdmissionRefusal) as error:
        publish_results(output, json_bytes=b"{}", markdown_text="# report")

    assert calls == 2
    assert error.value.reason is OperationalReasonCode.OUTPUT_WRITE_FAILED
    assert output.json_path.read_bytes() == b"{}"
    assert output.markdown_path.read_bytes() == b"# report"
    assert sorted(path.name for path in output.path.iterdir()) == [
        "comparison.json",
        "comparison.md",
    ]
