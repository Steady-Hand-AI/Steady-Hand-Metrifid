#!/usr/bin/env python3
"""Recreate and verify the exact watertight concave L-prism used by this case."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

EXPECTED_SHA256 = "cc0dc3171271a9721cac0c7d6d0d77285d31ee62679fcf0386c796add77463a0"
VERTICES = (
    (0, 0, -0.5),
    (2, 0, -0.5),
    (2, 1, -0.5),
    (1, 1, -0.5),
    (1, 2, -0.5),
    (0, 2, -0.5),
    (0, 0, 0.5),
    (2, 0, 0.5),
    (2, 1, 0.5),
    (1, 1, 0.5),
    (1, 2, 0.5),
    (0, 2, 0.5),
)
FACES = (
    (3, 1, 0),
    (3, 2, 1),
    (5, 3, 0),
    (5, 4, 3),
    (6, 7, 9),
    (7, 8, 9),
    (6, 9, 11),
    (9, 10, 11),
    (0, 1, 7),
    (0, 7, 6),
    (1, 2, 8),
    (1, 8, 7),
    (2, 3, 9),
    (2, 9, 8),
    (3, 4, 10),
    (3, 10, 9),
    (4, 5, 11),
    (4, 11, 10),
    (5, 0, 6),
    (5, 6, 11),
)


def encoded() -> bytes:
    """Return canonical OBJ bytes after proving every manifold edge has incidence two."""
    edges: Counter[tuple[int, int]] = Counter()
    for a, b, c in FACES:
        for left, right in ((a, b), (b, c), (c, a)):
            edge = (left, right) if left < right else (right, left)
            edges[edge] += 1
    if set(edges.values()) != {2}:
        raise RuntimeError("mesh edge incidence is not exactly two")
    lines = ["# Metrifid complaint-backed watertight concave L-prism"]
    lines.extend(f"v {x:.1f} {y:.1f} {z:.1f}" for x, y, z in VERTICES)
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in FACES)
    payload = ("\n".join(lines) + "\n").encode("ascii")
    if hashlib.sha256(payload).hexdigest() != EXPECTED_SHA256:
        raise RuntimeError("generated mesh bytes do not match the frozen identity")
    return payload


def main() -> int:
    """Verify both tracked copies and refuse any byte drift."""
    payload = encoded()
    root = Path(__file__).resolve().parent
    for role in ("baseline", "candidate"):
        path = root / role / "concave_shape.obj"
        if path.read_bytes() != payload:
            raise RuntimeError(f"{role} mesh bytes differ from the frozen generator")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
