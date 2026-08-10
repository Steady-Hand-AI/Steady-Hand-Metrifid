"""The installed `metrifid certify` command: contract, exits, refusals and publication."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

_BASELINE_XML = """
<mujoco model="cli-fixture">
  <option timestep="0.002"/>
  <worldbody>
    <body name="b" pos="0 0 1">
      <geom name="g" type="sphere" size="0.1" rgba="1 0 0 1" mass="2"/>
      <joint name="j" type="hinge" axis="0 0 1" damping="0.5"/>
    </body>
  </worldbody>
</mujoco>
"""

_CANDIDATE_XML = _BASELINE_XML.replace('mass="2"', 'mass="3"')


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Construct the run fixture used by certification cli scenarios.

    Deterministic setup isolates certification cli without bypassing the contract boundary under
    assertion.
    """
    return subprocess.run(
        [sys.executable, "-m", "metrifid.cli", "certify", *arguments],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )


def _tree(
    root: Path, baseline: str = _BASELINE_XML, candidate: str = _CANDIDATE_XML
) -> tuple[Path, Path]:
    """Construct the tree fixture used by certification cli scenarios.

    Deterministic setup isolates certification cli without bypassing the contract boundary under
    assertion.
    """
    baseline_dir = root / "baseline"
    candidate_dir = root / "candidate"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = baseline_dir / "model.xml"
    candidate_path = candidate_dir / "model.xml"
    baseline_path.write_text(baseline, encoding="utf-8")
    candidate_path.write_text(candidate, encoding="utf-8")
    return baseline_path, candidate_path


def _failure(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    """Inject the deterministic failure branch required by this scenario.

    The certification cli test can assert failure delivery for certification cli without
    depending on incidental runtime errors.
    """
    return json.loads(result.stderr.strip().splitlines()[-1])


def test_identical_sources_certify_and_exit_zero(tmp_path: Path) -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises identical sources certify and exit zero; the observable command or
    import contract is pinned without relying on repository layout.
    """
    baseline, _ = _tree(tmp_path)
    result = _run(str(baseline), str(baseline), "--output", str(tmp_path / "out"))
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["status"] == "CERTIFIED_COMPILED_EQUIVALENCE"
    receipt = json.loads((tmp_path / "out" / "certification.json").read_text(encoding="utf-8"))
    assert receipt["receipt_sha256"] == summary["receipt_sha256"]
    assert (tmp_path / "out" / "certification.md").is_file()


def test_differing_sources_are_not_certified_and_exit_forty(tmp_path: Path) -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises differing sources are not certified and exit forty; the observable
    command or import contract is pinned without relying on repository layout.
    """
    baseline, candidate = _tree(tmp_path)
    result = _run(str(baseline), str(candidate), "--output", str(tmp_path / "out"))
    assert result.returncode == 40, result.stderr
    assert json.loads(result.stdout)["status"] == "NOT_CERTIFIED_COMPILED_DIFFERS"


def test_two_distinct_trees_with_identical_bytes_still_certify(tmp_path: Path) -> None:
    """The claim is about compiled artifacts, so two separate source trees may certify."""
    baseline, candidate = _tree(tmp_path, _BASELINE_XML, _BASELINE_XML)
    assert baseline != candidate
    result = _run(str(baseline), str(candidate), "--output", str(tmp_path / "out"))
    assert result.returncode == 0, result.stderr
    receipt = json.loads((tmp_path / "out" / "certification.json").read_text(encoding="utf-8"))
    assert (
        receipt["baseline"]["source_closure_sha256"]
        == receipt["candidate"]["source_closure_sha256"]
    )
    assert (
        receipt["baseline"]["compiled_artifact"]["mjb_sha256"]
        == receipt["candidate"]["compiled_artifact"]["mjb_sha256"]
    )


def test_different_source_text_that_compiles_identically_still_certifies(tmp_path: Path) -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises different source text that compiles identically still certifies; the
    observable command or import contract is pinned without relying on repository layout.
    """
    commented = _BASELINE_XML.replace(
        "<worldbody>", "<!-- a comment that changes the source bytes -->\n  <worldbody>"
    )
    baseline, candidate = _tree(tmp_path, _BASELINE_XML, commented)
    result = _run(str(baseline), str(candidate), "--output", str(tmp_path / "out"))
    assert result.returncode == 0, result.stderr
    receipt = json.loads((tmp_path / "out" / "certification.json").read_text(encoding="utf-8"))
    assert (
        receipt["baseline"]["source_closure_sha256"]
        != receipt["candidate"]["source_closure_sha256"]
    )
    assert receipt["status"] == "CERTIFIED_COMPILED_EQUIVALENCE"


def test_repeated_invocations_publish_byte_identical_artifacts(tmp_path: Path) -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises repeated invocations publish byte identical artifacts; the
    observable command or import contract is pinned without relying on repository layout.
    """
    baseline, candidate = _tree(tmp_path)
    for name in ("first", "second"):
        assert (
            _run(str(baseline), str(candidate), "--output", str(tmp_path / name)).returncode == 40
        )
    for filename in ("certification.json", "certification.md"):
        first = (tmp_path / "first" / filename).read_bytes()
        second = (tmp_path / "second" / filename).read_bytes()
        assert first == second


def test_no_absolute_or_temporary_path_appears_in_the_published_artifacts(tmp_path: Path) -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises no absolute or temporary path appears in the published artifacts;
    the observable command or import contract is pinned without relying on repository layout.
    """
    baseline, candidate = _tree(tmp_path)
    assert _run(str(baseline), str(candidate), "--output", str(tmp_path / "out")).returncode == 40
    for filename in ("certification.json", "certification.md"):
        text = (tmp_path / "out" / filename).read_text(encoding="utf-8")
        assert str(tmp_path) not in text
        assert "/private/" not in text
        assert "/tmp/" not in text
        assert "metrifid-certify-" not in text


def test_an_explicit_root_yields_a_relative_posix_entrypoint(tmp_path: Path) -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises an explicit root yields a relative posix entrypoint; the observable
    command or import contract is pinned without relying on repository layout.
    """
    root = tmp_path / "tree"
    nested = root / "models" / "arm"
    nested.mkdir(parents=True)
    (nested / "model.xml").write_text(_BASELINE_XML, encoding="utf-8")
    result = _run(
        str(nested / "model.xml"),
        str(nested / "model.xml"),
        "--output",
        str(tmp_path / "out"),
        "--baseline-root",
        str(root),
        "--candidate-root",
        str(root),
    )
    assert result.returncode == 0, result.stderr
    receipt = json.loads((tmp_path / "out" / "certification.json").read_text(encoding="utf-8"))
    assert receipt["baseline"]["source_closure"]["entrypoint"] == "models/arm/model.xml"


def test_an_omitted_root_uses_the_parent_directory_and_basename(tmp_path: Path) -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises an omitted root uses the parent directory and basename; the
    observable command or import contract is pinned without relying on repository layout.
    """
    baseline, _ = _tree(tmp_path)
    result = _run(str(baseline), str(baseline), "--output", str(tmp_path / "out"))
    assert result.returncode == 0, result.stderr
    receipt = json.loads((tmp_path / "out" / "certification.json").read_text(encoding="utf-8"))
    assert receipt["baseline"]["source_closure"]["entrypoint"] == "model.xml"


def test_an_entrypoint_outside_the_supplied_root_refuses(tmp_path: Path) -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises an entrypoint outside the supplied root refuses; the observable
    command or import contract is pinned without relying on repository layout.
    """
    baseline, candidate = _tree(tmp_path)
    result = _run(
        str(baseline),
        str(candidate),
        "--output",
        str(tmp_path / "out"),
        "--baseline-root",
        str(tmp_path / "candidate"),
    )
    assert result.returncode == 64
    assert _failure(result)["reason"]["code"] == "MODEL_CLOSURE_PATH_ESCAPE"
    assert not (tmp_path / "out").exists()


def test_a_missing_entrypoint_refuses_before_any_output_exists(tmp_path: Path) -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises a missing entrypoint refuses before any output exists; the
    observable command or import contract is pinned without relying on repository layout.
    """
    baseline, _ = _tree(tmp_path)
    result = _run(str(baseline), str(tmp_path / "absent.xml"), "--output", str(tmp_path / "out"))
    assert result.returncode == 64
    failure = _failure(result)
    assert failure["reason"]["code"] == "MODEL_ENTRYPOINT_INVALID"
    assert failure["operation"] == "certify"
    assert not (tmp_path / "out").exists()


def test_a_symlinked_entrypoint_refuses(tmp_path: Path) -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises a symlinked entrypoint refuses; the observable command or import
    contract is pinned without relying on repository layout.
    """
    baseline, _ = _tree(tmp_path)
    link = tmp_path / "baseline" / "link.xml"
    link.symlink_to(baseline)
    result = _run(str(link), str(baseline), "--output", str(tmp_path / "out"))
    assert result.returncode == 64
    assert _failure(result)["reason"]["code"] == "MODEL_CLOSURE_SYMLINK_REFUSED"


def test_a_nonempty_output_directory_refuses_and_is_left_untouched(tmp_path: Path) -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises a nonempty output directory refuses and is left untouched; the
    observable command or import contract is pinned without relying on repository layout.
    """
    baseline, candidate = _tree(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    (output / "existing.txt").write_text("keep me", encoding="utf-8")
    result = _run(str(baseline), str(candidate), "--output", str(output))
    assert result.returncode == 64
    assert _failure(result)["reason"]["code"] == "OUTPUT_DIRECTORY_NOT_EMPTY"
    assert sorted(item.name for item in output.iterdir()) == ["existing.txt"]


def test_a_model_that_fails_to_compile_refuses_with_the_accepted_reason(tmp_path: Path) -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises a model that fails to compile refuses with the accepted reason; the
    observable command or import contract is pinned without relying on repository layout.
    """
    baseline, candidate = _tree(tmp_path, _BASELINE_XML, "<mujoco><worldbody><body/>")
    result = _run(str(baseline), str(candidate), "--output", str(tmp_path / "out"))
    assert result.returncode == 64
    assert _failure(result)["reason"]["code"] == "CANDIDATE_MODEL_COMPILE_ERROR"
    # A compile refusal happens after the output directory is admitted, so the directory is
    # left in place exactly as `compare` leaves it - but with nothing that reads as a receipt.
    assert list((tmp_path / "out").iterdir()) == []


def test_a_missing_required_option_is_an_invocation_failure(tmp_path: Path) -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises a missing required option is an invocation failure; the observable
    command or import contract is pinned without relying on repository layout.
    """
    baseline, candidate = _tree(tmp_path)
    result = _run(str(baseline), str(candidate))
    assert result.returncode == 64
    failure = _failure(result)
    assert failure["reason"]["code"] == "INVALID_CLI_INVOCATION"
    assert failure["operation"] == "certify"


def test_a_missing_positional_argument_is_an_invocation_failure(tmp_path: Path) -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises a missing positional argument is an invocation failure; the
    observable command or import contract is pinned without relying on repository layout.
    """
    baseline, _ = _tree(tmp_path)
    result = _run(str(baseline), "--output", str(tmp_path / "out"))
    assert result.returncode == 64
    assert _failure(result)["reason"]["code"] == "INVALID_CLI_INVOCATION"


def test_the_command_accepts_no_json_configuration(tmp_path: Path) -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises the command accepts no json configuration; the observable command or
    import contract is pinned without relying on repository layout.
    """
    configuration = tmp_path / "certify.json"
    configuration.write_text("{}", encoding="utf-8")
    result = _run(str(configuration))
    assert result.returncode == 64
    assert _failure(result)["reason"]["code"] == "INVALID_CLI_INVOCATION"


@pytest.mark.parametrize("other", ["compare", "audit-timestep"])
def test_the_existing_commands_keep_their_own_operation_label(tmp_path: Path, other: str) -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises the existing commands keep their own operation label; the observable
    command or import contract is pinned without relying on repository layout.
    """
    result = subprocess.run(
        [sys.executable, "-m", "metrifid.cli", other, str(tmp_path / "absent.json")],
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )
    assert result.returncode in {64, 70}
    assert _failure(result)["operation"] == other


def test_certify_never_publishes_a_comparison_named_output(tmp_path: Path) -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises certify never publishes a comparison named output; the observable
    command or import contract is pinned without relying on repository layout.
    """
    baseline, candidate = _tree(tmp_path)
    assert _run(str(baseline), str(candidate), "--output", str(tmp_path / "out")).returncode == 40
    assert sorted(item.name for item in (tmp_path / "out").iterdir()) == [
        "certification.json",
        "certification.md",
    ]
