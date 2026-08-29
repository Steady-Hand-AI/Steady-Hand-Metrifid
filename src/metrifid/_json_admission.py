"""Bounded strict JSON admission for every file-based trust boundary.

Metrifid reads JSON from callers at three boundaries: comparison configurations, alias files, and
timestep-audit configurations, plus certification receipts supplied for revalidation. Those inputs
are untrusted text. This module is the single admission point for all of them.

It layers two guarantees over :func:`metrifid.json_values.strict_json_loads`. First, the bytes are
read through one descriptor opened with ``O_NOFOLLOW`` and confirmed regular by ``fstat``, with a
hard ceiling applied before any parsing happens. Second, the parsed document is bounded in depth,
node count, and string length, so a syntactically strict document still cannot exhaust memory or
the interpreter stack.

That second guarantee is published separately as :func:`enforce_json_structure` because other
private trust boundaries parse with their own strict hooks and cannot route through
:func:`bounded_strict_json_loads`. Runtime-review evidence is one such boundary: it decodes worker
results with binary64 floats and input manifests with lexical :class:`~decimal.Decimal` values, and
must keep those representations. Sharing this one iterative walk is what makes the depth bound a
declared product policy instead of whatever nesting the running interpreter's parser happens to
tolerate.

Nothing here imports MuJoCo or NumPy: admission must work in a pure environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .json_values import CanonicalValue, strict_json_loads

__all__ = [
    "CONFIG_JSON_LIMITS",
    "RECEIPT_JSON_LIMITS",
    "JsonAdmissionError",
    "JsonAdmissionLimits",
    "bounded_strict_json_loads",
    "enforce_json_structure",
    "read_bounded_regular_file",
    "read_bounded_strict_json",
]

_MIB = 1024 * 1024


class JsonAdmissionError(ValueError):
    """Raised when untrusted JSON exceeds an admission bound or violates strict syntax.

    This derives from :class:`ValueError` so existing call sites that already convert strict-parse
    ``ValueError`` into a typed operational refusal keep working unchanged.
    """


@dataclass(frozen=True, slots=True)
class JsonAdmissionLimits:
    """Exact bounds applied to one untrusted JSON document.

    Attributes:
        max_bytes: Maximum strict UTF-8 byte length of the whole document.
        max_depth: Maximum nesting depth, where the root value has depth zero.
        max_nodes: Maximum node count, counting every value and every object member name.
        max_string_bytes: Maximum strict UTF-8 byte length of any single key or string value.
    """

    max_bytes: int
    max_depth: int
    max_nodes: int
    max_string_bytes: int

    def __post_init__(self) -> None:
        """Reject a limit set that could never admit a document."""
        for name in ("max_bytes", "max_depth", "max_nodes", "max_string_bytes"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


CONFIG_JSON_LIMITS = JsonAdmissionLimits(
    max_bytes=4 * _MIB,
    max_depth=64,
    max_nodes=100_000,
    max_string_bytes=1 * _MIB,
)

RECEIPT_JSON_LIMITS = JsonAdmissionLimits(
    max_bytes=64 * _MIB,
    max_depth=128,
    max_nodes=1_000_000,
    max_string_bytes=8 * _MIB,
)


def read_bounded_regular_file(path: str | os.PathLike[str], max_bytes: int) -> bytes:
    """Read at most ``max_bytes`` from one regular file through a single no-follow descriptor.

    The descriptor is opened once with ``O_RDONLY`` plus ``O_NOFOLLOW`` and ``O_NONBLOCK`` where the
    platform provides them, so a symbolic link is refused and a FIFO cannot block. ``fstat`` runs on
    that same descriptor, which removes the window between checking the path and reading it.

    Args:
        path: Filesystem path to admit.
        max_bytes: Inclusive maximum number of content bytes.

    Returns:
        The exact file content as bytes.

    Raises:
        JsonAdmissionError: The path is not a regular file, cannot be opened, or exceeds the bound.
    """
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    target = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(target, flags)
    except OSError as exc:
        raise JsonAdmissionError(
            f"input file could not be opened as a regular no-follow file: {target}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not _is_regular(metadata.st_mode):
            raise JsonAdmissionError(f"input file is not a regular file: {target}")
        payload = _read_exactly(descriptor, max_bytes + 1)
    finally:
        os.close(descriptor)
    if len(payload) > max_bytes:
        raise JsonAdmissionError(
            f"input file exceeds the {max_bytes} byte admission limit: {target}"
        )
    return payload


def bounded_strict_json_loads(data: bytes | str, limits: JsonAdmissionLimits) -> CanonicalValue:
    """Parse strict JSON and enforce byte, depth, node, and string bounds.

    Strict semantics come from :func:`metrifid.json_values.strict_json_loads`: duplicate object
    names, raw float tokens, and nonstandard constants such as ``NaN`` are rejected there. This
    function adds the size bounds and converts every failure into :class:`JsonAdmissionError`.

    Args:
        data: UTF-8 bytes or text to admit.
        limits: The exact bounds to enforce.

    Returns:
        The validated canonical value.

    Raises:
        JsonAdmissionError: Any bound was exceeded or the document was not strict canonical JSON.
    """
    if not isinstance(limits, JsonAdmissionLimits):
        raise TypeError("limits must be a JsonAdmissionLimits")
    payload = _as_bounded_bytes(data, limits.max_bytes)
    try:
        parsed = strict_json_loads(payload)
    except RecursionError as exc:
        raise JsonAdmissionError("JSON document nesting exhausted the interpreter stack") from exc
    except (ValueError, TypeError) as exc:
        raise JsonAdmissionError(f"strict JSON admission failed: {exc}") from exc
    enforce_json_structure(parsed, limits)
    return parsed


def read_bounded_strict_json(
    path: str | os.PathLike[str], limits: JsonAdmissionLimits
) -> CanonicalValue:
    """Read one regular file and admit it as bounded strict JSON.

    Args:
        path: Filesystem path to admit.
        limits: The exact bounds to enforce.

    Returns:
        The validated canonical value.

    Raises:
        JsonAdmissionError: The file or its content failed admission.
    """
    if not isinstance(limits, JsonAdmissionLimits):
        raise TypeError("limits must be a JsonAdmissionLimits")
    return bounded_strict_json_loads(read_bounded_regular_file(path, limits.max_bytes), limits)


def _is_regular(mode: int) -> bool:
    """Report whether one stat mode denotes a regular file."""
    return (mode & 0o170000) == 0o100000


def _read_exactly(descriptor: int, limit: int) -> bytes:
    """Read up to ``limit`` bytes from one open descriptor without following further links."""
    chunks: list[bytes] = []
    remaining = limit
    while remaining > 0:
        try:
            block = os.read(descriptor, min(remaining, _MIB))
        except BlockingIOError as exc:
            raise JsonAdmissionError("input file would block on read") from exc
        except OSError as exc:
            raise JsonAdmissionError(f"input file could not be read: {exc}") from exc
        if not block:
            break
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def _as_bounded_bytes(data: bytes | str, max_bytes: int) -> bytes:
    """Return the strict UTF-8 encoding of one input after enforcing the byte ceiling."""
    if isinstance(data, str):
        try:
            payload = data.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise JsonAdmissionError("JSON text is not encodable as strict UTF-8") from exc
    elif isinstance(data, bytes):
        payload = data
    else:
        raise TypeError("bounded_strict_json_loads accepts UTF-8 text or bytes only")
    if len(payload) > max_bytes:
        raise JsonAdmissionError(
            f"JSON document exceeds the {max_bytes} byte admission limit: {len(payload)} bytes"
        )
    return payload


def enforce_json_structure(value: object, limits: JsonAdmissionLimits) -> None:
    """Enforce depth, node, and string bounds over one already-parsed JSON document.

    The walk is iterative, so a deeply nested document is refused by the declared depth bound rather
    than by whatever nesting the running interpreter's parser happens to tolerate. That is the whole
    point of applying it after parsing: two supported CPython versions must admit and refuse the same
    documents.

    The value is typed as :class:`object` rather than :data:`~metrifid.json_values.CanonicalValue`
    because callers with their own strict parse hooks reuse this bound. Only containers are
    traversed and only strings are measured, so a :class:`float` or :class:`~decimal.Decimal` leaf is
    counted once and returned to its caller unchanged.

    Args:
        value: One parsed JSON document.
        limits: The exact bounds to enforce.

    Raises:
        JsonAdmissionError: The document exceeded the depth, node, or string bound.
    """
    nodes = 0
    pending: list[tuple[object, int]] = [(value, 0)]
    while pending:
        current, depth = pending.pop()
        if depth > limits.max_depth:
            raise JsonAdmissionError(
                f"JSON document exceeds the maximum nesting depth of {limits.max_depth}"
            )
        nodes += 1
        if nodes > limits.max_nodes:
            raise JsonAdmissionError(
                f"JSON document exceeds the maximum of {limits.max_nodes} nodes"
            )
        if isinstance(current, dict):
            for name, member in current.items():
                nodes += 1
                if nodes > limits.max_nodes:
                    raise JsonAdmissionError(
                        f"JSON document exceeds the maximum of {limits.max_nodes} nodes"
                    )
                _enforce_string(name, limits.max_string_bytes)
                pending.append((member, depth + 1))
        elif isinstance(current, list):
            for member in current:
                pending.append((member, depth + 1))
        elif isinstance(current, str):
            _enforce_string(current, limits.max_string_bytes)


def _enforce_string(value: str, max_string_bytes: int) -> None:
    """Enforce strict UTF-8 encodability and the byte ceiling for one key or string value.

    ``json`` accepts an escaped lone surrogate and yields a ``str`` that has no
    strict UTF-8 encoding. Measuring that string raises :class:`UnicodeEncodeError`, which is a
    parser-shaped exception escaping an admission boundary; callers that translate
    :class:`JsonAdmissionError` would let it through untyped. Translating it here keeps one refusal
    vocabulary at the boundary and preserves the original error as the cause.
    """
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as error:
        raise JsonAdmissionError("JSON string is not encodable as strict UTF-8") from error
    if len(encoded) > max_string_bytes:
        raise JsonAdmissionError(f"JSON string exceeds the {max_string_bytes} byte admission limit")
