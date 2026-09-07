"""A self-contained first-use demonstration that runs from a wheel installation alone.

``python -m metrifid.demo`` needs no repository checkout, no arguments, no network access, and no
packaged model assets. It writes three tiny MJCF models into a temporary directory and runs the two
comparisons a new user most needs to see, through the same ``diff`` operation the command line
uses:

* two source-different files that compile to the same model, exit 0;
* one changed mass compiles differently, exit 40.

The models are temporary; the results are not. Each comparison retains its receipt, Markdown and
offline report through the ordinary retained-run allocator, and the demonstration prints those real
paths so they can be opened and read afterwards.
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .result import run_diff

__all__ = ["main"]

# Two spellings of one model. The attribute order and number formatting differ; the compiled
# artifact does not, which is exactly the distinction Certify exists to make.
_BASELINE_MJCF = """<mujoco model="demo">
  <worldbody>
    <body name="arm" pos="0 0 1">
      <geom name="link" type="capsule" size="0.04 0.2" mass="1.5"/>
      <joint name="shoulder" type="hinge" axis="0 1 0"/>
    </body>
  </worldbody>
</mujoco>
"""

_EQUIVALENT_MJCF = """<mujoco model="demo">
  <worldbody>
    <body pos="0 0 1" name="arm">
      <geom mass="1.50" name="link" size="0.04 0.2" type="capsule"/>
      <joint axis="0 1 0" name="shoulder" type="hinge"/>
    </body>
  </worldbody>
</mujoco>
"""

# One physical change: the link is heavier. The compiled bytes must differ.
_CHANGED_MJCF = """<mujoco model="demo">
  <worldbody>
    <body name="arm" pos="0 0 1">
      <geom name="link" type="capsule" size="0.04 0.2" mass="1.6"/>
      <joint name="shoulder" type="hinge" axis="0 1 0"/>
    </body>
  </worldbody>
</mujoco>
"""

_IDENTICAL_EXIT = 0
_DIFFERENT_EXIT = 40


def _write_model(directory: Path, text: str) -> Path:
    """Write one model file into its own root directory and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model.xml"
    path.write_text(text, encoding="utf-8")
    return path


@dataclass(frozen=True, slots=True)
class _Comparison:
    """One completed demonstration comparison and where its evidence was retained."""

    label: str
    exit_code: int
    state: str
    receipt: str | None
    report: str | None
    presentation: tuple[str, ...]


def _compare(label: str, baseline: Path, candidate: Path) -> _Comparison:
    """Run one comparison through the ordinary diff operation and read what it retained."""
    outcome = run_diff(str(baseline), str(candidate))
    document = outcome.document
    comparison = document["compiled_comparison"]
    artifacts = document["artifacts"]
    limitations = document["limitations"]
    state = str(comparison["state"]) if isinstance(comparison, dict) else "not_established"
    receipt = artifacts.get("receipt") if isinstance(artifacts, dict) else None
    report = artifacts.get("html") if isinstance(artifacts, dict) else None
    warnings = (
        tuple(
            str(item["message"])
            for item in limitations
            if isinstance(item, dict) and item.get("category") == "presentation"
        )
        if isinstance(limitations, list)
        else ()
    )
    return _Comparison(
        label,
        outcome.exit_code,
        state,
        receipt if isinstance(receipt, str) else None,
        report if isinstance(report, str) else None,
        warnings,
    )


def _run(workspace: Path) -> tuple[_Comparison, _Comparison]:
    """Write the three models into one temporary workspace and compare both pairs."""
    baseline = _write_model(workspace / "baseline", _BASELINE_MJCF)
    equivalent = _write_model(workspace / "equivalent", _EQUIVALENT_MJCF)
    changed = _write_model(workspace / "changed", _CHANGED_MJCF)
    return (
        _compare("different source, same compiled model", baseline, equivalent),
        _compare("one changed mass", baseline, changed),
    )


def _problems(equivalent: _Comparison, changed: _Comparison) -> list[str]:
    """Return every expectation the demonstration did not meet."""
    problems: list[str] = []
    for comparison, expected, state in (
        (equivalent, _IDENTICAL_EXIT, "identical"),
        (changed, _DIFFERENT_EXIT, "different"),
    ):
        if comparison.exit_code != expected:
            problems.append(
                f"{comparison.label} exited {comparison.exit_code}, expected {expected}"
            )
        elif comparison.state != state:
            problems.append(f"{comparison.label} reported {comparison.state}, expected {state}")
        elif comparison.receipt is None:
            problems.append(f"{comparison.label} retained no receipt")
    return problems


def _describe(comparison: _Comparison) -> None:
    """Print one comparison's outcome and the evidence it actually retained."""
    print(f"{comparison.label:38s}: exit {comparison.exit_code} ({comparison.state})")
    print(f"{'  receipt':38s}: {comparison.receipt}")
    if comparison.report is not None:
        print(f"{'  report':38s}: {comparison.report}")
    else:
        print(f"{'  report':38s}: not written")
    for warning in comparison.presentation:
        print(f"{'  note':38s}: {warning}")


def main() -> int:
    """Run the bundled demonstration and report whether every expectation held.

    The temporary models are removed when the workspace is discarded. The retained results are
    not: their printed paths stay readable afterwards with ``metrifid show``.

    Returns:
        ``0`` when the equivalent pair exits 0 and the changed pair exits 40, and each retained a
        receipt; a nonzero code otherwise.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="metrifid-demo-") as raw:
            # resolve(): the platform temporary directory is often reached through a symbolic
            # link, and Metrifid refuses to admit a model root through one.
            equivalent, changed = _run(Path(raw).resolve())
    except Exception as exc:
        print(f"metrifid demo failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    problems = _problems(equivalent, changed)
    if problems:
        print(f"metrifid demo failed: {'; '.join(problems)}", file=sys.stderr)
        return 1

    _describe(equivalent)
    _describe(changed)
    print("Metrifid demo passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
