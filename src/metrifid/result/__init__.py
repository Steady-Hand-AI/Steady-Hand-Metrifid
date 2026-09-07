"""One small presented result shared by the ``show`` and ``diff`` commands.

The canonical receipts remain the authority.  This package adds a documented, tested presentation
over them: source-shaped findings, categorized limitations, bounded JSON and a readable text view.
Importing it pulls no native library, so a saved receipt can be read where MuJoCo is not installed.
"""

from __future__ import annotations

from ._contract import (
    RESULT_SCHEMA,
    RESULT_SCHEMA_VERSION,
    build_document,
    build_failure_document,
)
from ._diff import DiffOutcome, run_diff
from ._emit import DEFAULT_JSON_MAX_BYTES, encode_result, render_text
from ._show import show_receipt

__all__ = [
    "DEFAULT_JSON_MAX_BYTES",
    "RESULT_SCHEMA",
    "RESULT_SCHEMA_VERSION",
    "DiffOutcome",
    "build_document",
    "build_failure_document",
    "encode_result",
    "render_text",
    "run_diff",
    "show_receipt",
]
