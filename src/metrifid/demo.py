"""A self-contained first-use demonstration that runs from a wheel installation alone.

``python -m metrifid.demo`` needs no repository checkout, no arguments, no network access, and no
packaged model assets. It writes two tiny MJCF models into a temporary directory and runs the two
certification outcomes a new user most needs to see:

* two source-different files that compile to the same artifact certify, exit 0;
* one changed mass compiles differently, exit 40.

Both published receipts are then reloaded through the public raw loader, which is the same strict
admission path an independent reader would use.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from .certify import certify_models, load_and_validate_certification_receipt

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

_EQUIVALENT_STATUS = "CERTIFIED_COMPILED_EQUIVALENCE"
_CHANGED_STATUS = "NOT_CERTIFIED_COMPILED_DIFFERS"


def _write_model(directory: Path, text: str) -> Path:
    """Write one model file into its own root directory and return its path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "model.xml"
    path.write_text(text, encoding="utf-8")
    return path


def _certify_pair(baseline: Path, candidate: Path, output: Path) -> tuple[str, Path, Path]:
    """Certify one model pair and return its status with both published paths."""
    result = certify_models(str(baseline), str(candidate), str(output))
    return str(result.status), result.certification_json, result.certification_markdown


def _run(workspace: Path) -> tuple[str, str, int, int]:
    """Run both certifications inside one workspace and revalidate every published receipt.

    Returns:
        The equivalent status, the changed status, the number of receipts revalidated through the
        public raw loader, and the number of Markdown renderings found on disk.
    """
    baseline = _write_model(workspace / "baseline", _BASELINE_MJCF)
    equivalent = _write_model(workspace / "equivalent", _EQUIVALENT_MJCF)
    changed = _write_model(workspace / "changed", _CHANGED_MJCF)

    equivalent_status, equivalent_json, equivalent_markdown = _certify_pair(
        baseline, equivalent, workspace / "out_equivalent"
    )
    changed_status = _CHANGED_STATUS
    try:
        changed_status, changed_json, changed_markdown = _certify_pair(
            baseline, changed, workspace / "out_changed"
        )
    except Exception:
        # A differing pair is a completed decision, not an error, so any exception here is real.
        raise

    validated = 0
    for payload in (equivalent_json, changed_json):
        load_and_validate_certification_receipt(payload.read_bytes())
        validated += 1
    markdown = sum(1 for path in (equivalent_markdown, changed_markdown) if path.is_file())
    return equivalent_status, changed_status, validated, markdown


def main() -> int:
    """Run the bundled demonstration and report whether every expectation held.

    Returns:
        ``0`` when both certifications produced their expected status, both receipts revalidated,
        and both Markdown renderings exist; a nonzero code otherwise.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="metrifid-demo-") as raw:
            # resolve(): the platform temporary directory is often reached through a symbolic
            # link, and Metrifid refuses to publish through one.
            workspace = Path(raw).resolve()
            equivalent_status, changed_status, validated, markdown = _run(workspace)
    except Exception as exc:
        print(f"metrifid demo failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    problems: list[str] = []
    if equivalent_status != _EQUIVALENT_STATUS:
        problems.append(f"equivalent pair reported {equivalent_status}")
    if changed_status != _CHANGED_STATUS:
        problems.append(f"changed pair reported {changed_status}")
    if validated != 2:
        problems.append(f"revalidated {validated} receipts, expected 2")
    if markdown != 2:
        problems.append(f"found {markdown} Markdown renderings, expected 2")
    if problems:
        print(f"metrifid demo failed: {'; '.join(problems)}", file=sys.stderr)
        return 1

    print(f"different source, same compiled model : {equivalent_status}")
    print(f"one changed mass                      : {changed_status}")
    print("Metrifid demo passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
