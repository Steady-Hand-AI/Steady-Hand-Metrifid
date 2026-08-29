"""Run the six complaint-backed public cases through a clean installed Metrifid distribution.

Each case answers the same release question — *the model changed; what can I actually approve?* —
by running Certify, a Model Review discovery pass, a candidate-bound declared review built from that
discovery, and one direct-MuJoCo control that observes the compiled mechanism the complaint
described.

Usage:
    python examples/public_cases/run_all.py --output /absolute/absent/output

The six cases are independently authored mechanism analogues. None of them executes Newton, USD,
Isaac, or Robosuite code, and none proves cross-backend equivalence or physical correctness.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast

_CASE_DIRECTORY = Path(__file__).resolve().parent
if str(_CASE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_CASE_DIRECTORY))

from _shared import (  # noqa: E402  - the gallery directory must be importable first
    CaseExpectationError,
    execute_case,
    load_case_manifest,
    require_absent_directory,
    write_checksum_manifest,
    write_new_bytes,
    write_new_json,
)

GALLERY_RESULT_SCHEMA = "metrifid.public_case_gallery_result"
GALLERY_RESULT_SCHEMA_VERSION = 1
SUCCESS_TOKEN = "METRIFID_PUBLIC_CASE_GALLERY_PASSED"
FAILURE_TOKEN = "METRIFID_PUBLIC_CASE_GALLERY_FAILED"


def read_case_order(gallery_directory: Path) -> list[str]:
    """Read the frozen case order from the tracked case index.

    Args:
        gallery_directory: Directory holding ``case_index.json``.

    Returns:
        The frozen case identifiers, in their frozen order.

    Raises:
        RuntimeError: If the index is missing or not the expected schema.
    """
    index_path = gallery_directory / "case_index.json"
    if not index_path.is_file():
        raise RuntimeError(f"case index is missing: {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(index, dict) or index.get("schema") != "metrifid.public_case_index":
        raise RuntimeError(f"case index has an unexpected schema: {index_path}")
    order = index.get("case_order")
    if not isinstance(order, list) or not all(isinstance(item, str) for item in order):
        raise RuntimeError("case index case_order is not a list of strings")
    return list(order)


def _containing_checkout() -> Path | None:
    """Return the Metrifid source checkout holding this gallery, when there is one.

    The gallery is also runnable after being copied somewhere else entirely, so the checkout is
    found by its own marker files rather than by counting parent directories. Counting silently
    resolves to the filesystem root once the gallery moves, which makes every install look like a
    source-tree import.
    """
    for candidate in _CASE_DIRECTORY.parents:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "metrifid").is_dir():
            return candidate
    return None


def require_installed_distribution() -> str:
    """Prove Metrifid is imported from a real installation rather than a source tree.

    Returns:
        The resolved ``metrifid`` package location.

    Raises:
        RuntimeError: If Metrifid resolves inside the checkout that contains this gallery, or does
            not resolve inside an installed site directory at all.
    """
    import metrifid

    location = Path(metrifid.__file__ or "").resolve()
    checkout = _containing_checkout()
    if checkout is not None and checkout in location.parents:
        raise RuntimeError(
            "the gallery must run against an installed metrifid distribution, but metrifid "
            f"resolved inside the source checkout at {location}"
        )
    if not any(part in {"site-packages", "dist-packages"} for part in location.parts):
        raise RuntimeError(
            "the gallery must run against a noneditable installed metrifid distribution, but "
            f"metrifid resolved outside any installed site directory at {location}"
        )
    return str(location)


def index_case_directories(gallery_directory: Path) -> dict[str, Path]:
    """Map every tracked case identifier to the directory that declares it.

    A case identifier is a family and a variant joined by ``.``, but a family with a single variant
    is stored at the family directory itself, so the identifier is not a path. The manifests are the
    authority: each one names its own case, and no two may name the same one.

    Args:
        gallery_directory: Root of the tracked gallery.

    Returns:
        Case identifier to case directory.

    Raises:
        RuntimeError: If two manifests declare the same case identifier.
    """
    index: dict[str, Path] = {}
    for manifest_path in sorted(gallery_directory.rglob("case_manifest.json")):
        directory = manifest_path.parent
        case_id = str(load_case_manifest(directory)["case_id"])
        if case_id in index:
            raise RuntimeError(
                f"case {case_id!r} is declared twice: {index[case_id]} and {directory}"
            )
        index[case_id] = directory
    return index


def build_stable_projection(
    case_results: list[dict[str, object]], manifests: list[dict[str, object]]
) -> list[dict[str, object]]:
    """Project the case results onto the frozen cross-runtime comparison rows.

    Only these values are compared across separate workspaces and compatible MuJoCo versions. MJB
    hashes, receipt hashes, runtime identities, and explanatory floating-point diagnostics stay bound
    to their exact run.

    The two ``required_*_observed`` members carry the manifest's required subsets, which
    ``execute_case`` has already proved were present in the discovery projection for this run.

    Args:
        case_results: The six case result mappings, in frozen order.
        manifests: The six case manifests, in the same order.

    Returns:
        One projection row per case, in the same order.
    """
    rows: list[dict[str, object]] = []
    for result, manifest in zip(case_results, manifests, strict=True):
        product = cast("Mapping[str, Mapping[str, object]]", result["product_results"])
        control = cast("Mapping[str, object]", result["independent_control"])
        required_fields = cast("list[str]", manifest.get("required_compiled_fields") or [])
        required_selectors = cast(
            "list[Mapping[str, str]]", manifest.get("required_semantic_selectors") or []
        )
        rows.append(
            {
                "case_id": result["case_id"],
                "certify_status": product["certify"]["status"],
                "certify_exit_code": product["certify"]["completed_exit_code"],
                "discovery_status": product["discovery"]["status"],
                "discovery_exit_code": product["discovery"]["completed_exit_code"],
                "declared_status": product["declared"]["status"],
                "declared_exit_code": product["declared"]["completed_exit_code"],
                "required_public_fields_observed": sorted(required_fields),
                "required_semantic_selectors_observed": [dict(item) for item in required_selectors],
                "independent_control_classification": control["classification"],
                "claim_boundary": result["claim_boundary"],
            }
        )
    return rows


def render_summary(rows: list[dict[str, object]]) -> bytes:
    """Render the human-readable gallery summary from the stable projection.

    Args:
        rows: The stable projection rows.

    Returns:
        UTF-8 Markdown bytes.
    """
    lines = [
        "# Complaint-backed public case gallery",
        "",
        "Each row is one independently authored mechanism analogue run through the accepted",
        "Metrifid product journey. A completed exit `40` here is a referee decision, not a crash:",
        "Certify reports that the complete compiled artifacts differ, and the discovery review",
        "reports that the observed change is not yet declared by any policy.",
        "",
        "The declared policy in each case is generated mechanically from the discovery receipt to",
        "demonstrate the two-pass workflow. It is not release authority. A human maintainer must",
        "inspect and justify every declared rule before approving a real model change.",
        "",
        "| case | certify | discovery | declared | independent control |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| `{row['case_id']}` "
            f"| {row['certify_status']} ({row['certify_exit_code']}) "
            f"| {row['discovery_status']} ({row['discovery_exit_code']}) "
            f"| {row['declared_status']} ({row['declared_exit_code']}) "
            f"| {row['independent_control_classification']} |"
        )
    lines.extend(
        [
            "",
            "## Claim boundaries",
            "",
        ]
    )
    for row in rows:
        lines.append(f"- `{row['case_id']}` — {row['claim_boundary']}")
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def main() -> int:
    """Run all six cases into one fresh output root and publish the gallery evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        help="absolute path to an absent directory to publish the gallery into",
    )
    arguments = parser.parse_args()

    installed_location = require_installed_distribution()
    gallery_directory = _CASE_DIRECTORY
    case_order = read_case_order(gallery_directory)
    case_directories = index_case_directories(gallery_directory)
    missing = [case_id for case_id in case_order if case_id not in case_directories]
    if missing:
        raise RuntimeError(f"the case index names cases with no manifest: {missing}")
    unlisted = sorted(set(case_directories) - set(case_order))
    if unlisted:
        raise RuntimeError(f"these cases have a manifest but are not in the case index: {unlisted}")
    output_root = require_absent_directory(Path(arguments.output))

    case_results: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    relative_result_paths: list[str] = []
    divergences: list[CaseExpectationError] = []
    for case_id in case_order:
        case_directory = case_directories[case_id]
        manifests.append(load_case_manifest(case_directory))
        case_output = output_root.joinpath("cases", *case_id.split("."))
        result_path = case_output / "case_result.json"
        try:
            case_results.append(execute_case(case_directory, case_output))
        except CaseExpectationError as error:
            # The case published every receipt before judging itself. Keep its evidence, record the
            # divergence, and run the remaining cases so one report covers all six.
            divergences.append(error)
            case_results.append(json.loads(result_path.read_text(encoding="utf-8")))
        relative_result_paths.append(result_path.relative_to(output_root).as_posix())

    rows = build_stable_projection(case_results, manifests)
    write_new_json(
        output_root / "gallery_result.json",
        {
            "schema": GALLERY_RESULT_SCHEMA,
            "schema_version": GALLERY_RESULT_SCHEMA_VERSION,
            "case_order": case_order,
            "case_results": relative_result_paths,
            "stable_projection": rows,
        },
    )
    write_new_bytes(output_root / "gallery_summary.md", render_summary(rows))
    write_checksum_manifest(output_root)

    print(f"installed metrifid: {installed_location}")
    print(f"gallery output:     {output_root}")
    for row in rows:
        print(
            f"  {row['case_id']:42s} "
            f"certify {row['certify_status']} ({row['certify_exit_code']}) | "
            f"discovery {row['discovery_status']} ({row['discovery_exit_code']}) | "
            f"declared {row['declared_status']} ({row['declared_exit_code']})"
        )
    if divergences:
        print("")
        print(f"{len(divergences)} of {len(case_order)} cases diverged from the frozen journey:")
        for divergence in divergences:
            for mismatch in divergence.mismatches:
                print(f"  {divergence.case_id}: {mismatch}")
        print(FAILURE_TOKEN)
        return 1
    print(SUCCESS_TOKEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
