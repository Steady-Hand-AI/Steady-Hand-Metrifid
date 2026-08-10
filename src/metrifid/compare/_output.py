"""Comparison output admission plus retained no-clobber result publication."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .._atomic_output import (
    PairedOutputDirectory,
    PairedOutputNames,
    cleanup_paired_output_after_failure,
    prepare_paired_output_directory,
    publish_paired_results,
    verify_paired_results,
)
from .._owned_artifacts import RetainedArtifactPair

COMPARISON_OUTPUT_NAMES = PairedOutputNames("comparison.json", "comparison.md")


@dataclass(slots=True, init=False)
class OutputDirectory:
    """Own one admitted comparison directory and any retained published result pair."""

    _paired_output: PairedOutputDirectory
    _retained_pair: RetainedArtifactPair | None = field(default=None, repr=False)

    def __init__(self, path: Path | PairedOutputDirectory) -> None:
        """Retain an existing shared binding or bind a legacy direct path once."""
        self._paired_output = (
            path
            if isinstance(path, PairedOutputDirectory)
            else PairedOutputDirectory(path, COMPARISON_OUTPUT_NAMES)
        )
        self._retained_pair = None

    @property
    def path(self) -> Path:
        """Return the public display path of the descriptor-bound directory."""
        return self._paired_output.path

    @property
    def json_path(self) -> Path:
        """Return the final canonical receipt path."""
        return self.path / COMPARISON_OUTPUT_NAMES.json_name

    @property
    def markdown_path(self) -> Path:
        """Return the final human-readable report path."""
        return self.path / COMPARISON_OUTPUT_NAMES.markdown_name

    def _paired(self) -> PairedOutputDirectory:
        """Adapt this comparison directory to the shared paired-output publisher."""
        return self._paired_output

    def _store_retained_pair(self, pair: RetainedArtifactPair) -> None:
        """Take private ownership of the one pair published into this directory."""
        if self._retained_pair is not None:
            pair.cleanup()
            pair.close()
            raise RuntimeError("comparison output pair is already retained")
        self._retained_pair = pair

    def _verify_retained_pair(self) -> None:
        """Reverify the public output path and exact committed pair before success."""
        if self._retained_pair is None:
            raise RuntimeError("comparison output pair is not retained")
        verify_paired_results(self._paired_output, self._retained_pair)

    def _take_retained_pair(self) -> RetainedArtifactPair:
        """Transfer the retained pair to an enclosing audit registry."""
        if self._retained_pair is None:
            raise RuntimeError("comparison output pair is not retained")
        pair = self._retained_pair
        self._retained_pair = None
        return pair

    def _cleanup_retained_pair(self) -> None:
        """Clean private temporaries and close the pair without deleting public finals."""
        if self._retained_pair is None:
            return
        self._retained_pair.cleanup()
        self._retained_pair.close()
        self._retained_pair = None

    def _close_after_success(self) -> None:
        """Close retained pair and directory descriptors without deleting committed files."""
        if self._retained_pair is not None:
            self._retained_pair.close()
            self._retained_pair = None
        self._paired_output.close()


def prepare_output_directory(path: Path) -> OutputDirectory:
    """Create an absent final directory or admit one empty real directory."""
    return OutputDirectory(prepare_paired_output_directory(path, COMPARISON_OUTPUT_NAMES))


def publish_results(
    output: OutputDirectory,
    *,
    json_bytes: bytes,
    markdown_text: str,
) -> RetainedArtifactPair:
    """Publish both complete outputs and retain their private ownership handle."""
    pair = publish_paired_results(
        output._paired(), json_bytes=json_bytes, markdown_text=markdown_text
    )
    output._store_retained_pair(pair)
    return pair


def cleanup_output_after_failure(output: OutputDirectory | None) -> None:
    """Clean private temporaries, preserve public finals, and close retained descriptors."""
    if output is None:
        return
    output._cleanup_retained_pair()
    cleanup_paired_output_after_failure(output._paired())


__all__ = [
    "OutputDirectory",
    "cleanup_output_after_failure",
    "prepare_output_directory",
    "publish_results",
]
