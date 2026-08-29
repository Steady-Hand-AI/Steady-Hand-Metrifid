"""Render user-controlled labels into Markdown without letting them become Markdown.

Every semantic label in a qualification is user text: ``workload_id``, ``probe_id``, ``parameter``,
``magnitude_semantics``, and the explanations built from them. Rendered naively, one pipe turns a
label into two table cells, one backtick closes a code span, a leading ``#`` becomes a heading, and
angle brackets become raw HTML.

Escaping happens in exactly one place so no report path can forget it. The canonical JSON output is
untouched: it carries the admitted string exactly, through ordinary JSON escaping, and this module
only governs how that same string is displayed.
"""

from __future__ import annotations

import unicodedata
from typing import Final

_CONTROL_REPLACEMENTS: Final[dict[str, str]] = {
    "\r": "\\r",
    "\n": "\\n",
    "\t": "\\t",
}
_MARKDOWN_ESCAPES: Final[tuple[tuple[str, str], ...]] = (
    ("\\", "\\\\"),
    ("`", "\\`"),
    ("|", "\\|"),
    ("<", "\\<"),
    (">", "\\>"),
    ("*", "\\*"),
    ("_", "\\_"),
    ("[", "\\["),
    ("]", "\\]"),
    ("#", "\\#"),
)


def markdown_label(value: object) -> str:
    """Return one user label rendered as inert Markdown text.

    Line breaks become visible escapes rather than real breaks, so a label can never open a new row,
    close a fence, or start a block. Structural characters are backslash-escaped. Other Unicode is
    preserved, except format and control code points, which are replaced by their code point so a
    bidirectional override cannot reorder the surrounding report.
    """
    text = "" if value is None else str(value)
    for needle, replacement in _CONTROL_REPLACEMENTS.items():
        text = text.replace(needle, replacement)
    text = "".join(
        character
        if unicodedata.category(character) not in {"Cc", "Cf", "Co", "Cs", "Cn"}
        else f"\\u{ord(character):04x}"
        for character in text
    )
    for needle, replacement in _MARKDOWN_ESCAPES:
        text = text.replace(needle, replacement)
    return text


def markdown_code(value: object) -> str:
    """Return one label inside an inline code span, with the span made unbreakable."""
    return f"`{markdown_label(value)}`"


def markdown_block(value: object) -> str:
    """Return one label for a fenced block, with fence terminators neutralized."""
    text = "" if value is None else str(value)
    for needle, replacement in _CONTROL_REPLACEMENTS.items():
        text = text.replace(needle, replacement)
    return text.replace("```", "\\`\\`\\`")
