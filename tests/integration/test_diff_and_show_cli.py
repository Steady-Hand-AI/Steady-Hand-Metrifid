"""The installed `metrifid diff` and `metrifid show` journey: exits, retention, and refusals."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "certify"
_BASELINE = _EXAMPLES / "equivalent" / "baseline.xml"
_EQUIVALENT_CANDIDATE = _EXAMPLES / "equivalent" / "candidate.xml"
_CHANGED_CANDIDATE = _EXAMPLES / "changed.xml"
_FIXTURES = Path(__file__).parents[1] / "fixtures" / "result"

_CANONICAL_NAMES = ["model_release.json", "model_release.md"]
_RETAINED_NAMES = ["model_release.json", "model_release.md", "report.html"]
_RESULT_MEMBERS = frozenset(
    {
        "artifacts",
        "command",
        "compiled_comparison",
        "evaluation",
        "finding_count",
        "findings",
        "inputs",
        "limitations",
        "observation",
        "problems",
        "runtime",
        "schema",
        "schema_version",
        "truncated",
    }
)
_PRIVATE_DIRECTORY_MODE = 0o700

_SHARED_ASSET_XML = """<mujocoinclude>
  <worldbody>
    <body name="link" pos="0 0 1">
      <geom name="shaft" type="capsule" fromto="0 0 0 0 0 -0.4" size="0.04" mass="{mass}"/>
      <joint name="elbow" type="hinge" axis="0 1 0" damping="0.1"/>
    </body>
  </worldbody>
</mujocoinclude>
"""
_INCLUDING_MODEL_XML = """<mujoco model="arm">
  <option timestep="0.002"/>
  <include file="../assets/common.xml"/>
  <actuator>
    <motor joint="elbow" gear="20"/>
  </actuator>
</mujoco>
"""


def _run(*arguments: str, home: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Invoke the installed CLI once and capture both complete streams.

    Args:
        arguments: The command and its options, exactly as a user would type them.
        home: An isolated home directory, for the runs that retain evidence beneath it.

    Returns:
        The completed process, never raising on a nonzero exit.
    """
    environment = os.environ.copy()
    if home is not None:
        environment["HOME"] = str(home)
    return subprocess.run(
        [sys.executable, "-m", "metrifid.cli", *arguments],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )


def _failure(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    """Parse the one canonical operational failure a refusal writes to stderr."""
    parsed: dict[str, Any] = json.loads(result.stderr.strip().splitlines()[-1])
    return parsed


def _document(result: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    """Parse the one metrifid.result document a --json invocation writes to stdout."""
    parsed: dict[str, Any] = json.loads(result.stdout)
    return parsed


def _receipt(directory: Path) -> dict[str, Any]:
    """Read the canonical model-release receipt published into one output directory."""
    parsed: dict[str, Any] = json.loads((directory / "model_release.json").read_text("utf-8"))
    return parsed


def _copied_pair(root: Path) -> tuple[Path, Path]:
    """Copy the shipped differing example pair into one writable model root."""
    root.mkdir(parents=True, exist_ok=True)
    baseline = root / "baseline.xml"
    candidate = root / "candidate.xml"
    baseline.write_bytes(_BASELINE.read_bytes())
    candidate.write_bytes(_CHANGED_CANDIDATE.read_bytes())
    return baseline, candidate


def _shared_asset_tree(root: Path, mass: str) -> Path:
    """Build one model root whose entrypoint includes a file from a shared asset directory."""
    (root / "assets").mkdir(parents=True)
    (root / "models").mkdir(parents=True)
    (root / "assets" / "common.xml").write_text(
        _SHARED_ASSET_XML.format(mass=mass), encoding="utf-8"
    )
    (root / "assets" / "NOTES.txt").write_text("shared assets\n", encoding="utf-8")
    entrypoint = root / "models" / "arm.xml"
    entrypoint.write_text(_INCLUDING_MODEL_XML, encoding="utf-8")
    return entrypoint


def _compiled_field_names(document: dict[str, Any]) -> list[str]:
    """Return every compiled field name the result document's findings name."""
    return [
        finding["data"]["selector"]["object_name"]
        for finding in document["findings"]
        if finding["kind"] == "compiled_field"
    ]


def _file_count(root: Path) -> int:
    """Count every regular file beneath one admitted model root."""
    return len([entry for entry in root.rglob("*") if entry.is_file()])


def test_the_changed_pair_reports_a_compiled_change_and_retains_its_evidence(
    tmp_path: Path,
) -> None:
    """The differing shipped pair exits 40, says so, and retains only its two artifacts."""
    output = (tmp_path / "out").resolve()
    result = _run("diff", str(_BASELINE), str(_CHANGED_CANDIDATE), "--output", str(output))
    assert result.returncode == 40, result.stderr
    assert "Compiled model changed." in result.stdout
    assert "71 serialized byte(s) differ" in result.stdout
    assert sorted(entry.name for entry in output.iterdir()) == _RETAINED_NAMES


def test_the_equivalent_pair_reports_an_identical_compiled_model_and_exits_zero(
    tmp_path: Path,
) -> None:
    """The equivalent shipped pair exits 0 and states that every serialized byte matched."""
    output = (tmp_path / "out").resolve()
    result = _run("diff", str(_BASELINE), str(_EQUIVALENT_CANDIDATE), "--output", str(output))
    assert result.returncode == 0, result.stderr
    assert "Compiled model identical." in result.stdout
    assert "Every serialized byte of the two compiled artifacts is identical." in result.stdout
    assert sorted(entry.name for entry in output.iterdir()) == _RETAINED_NAMES


@pytest.mark.parametrize(
    ("candidate", "expected_exit"),
    [(_EQUIVALENT_CANDIDATE, 0), (_CHANGED_CANDIDATE, 40)],
    ids=["equivalent", "changed"],
)
def test_json_diff_emits_one_result_document_that_agrees_with_the_published_receipt(
    tmp_path: Path, candidate: Path, expected_exit: int
) -> None:
    """--json emits one complete metrifid.result whose counts come from the receipt itself."""
    output = (tmp_path / "out").resolve()
    result = _run("diff", str(_BASELINE), str(candidate), "--output", str(output), "--json")
    assert result.returncode == expected_exit, result.stderr
    document = _document(result)
    assert set(document) == _RESULT_MEMBERS
    assert document["schema"] == "metrifid.result"
    assert document["schema_version"] == 1
    assert document["command"] == {"name": "diff", "exit_code": result.returncode}
    assert document["observation"] == "produced_now"
    assert document["evaluation"]["policy_origin"] == "generated_empty"
    receipt = _receipt(output)
    assert document["finding_count"] == len(receipt["changes"])
    assert len(document["findings"]) == len(receipt["changes"])
    assert document["evaluation"]["receipt_sha256"] == receipt["receipt_sha256"]
    assert document["artifacts"]["output_dir"] == str(output)


def test_two_consecutive_runs_without_an_output_retain_private_directories_under_home(
    tmp_path: Path,
) -> None:
    """With no --output, each run lands in its own 0o700 directory beneath $HOME/.metrifid."""
    home = (tmp_path / "home").resolve()
    home.mkdir()
    models = (tmp_path / "models").resolve()
    baseline, candidate = _copied_pair(models)
    retained: list[Path] = []
    for _ in range(2):
        result = _run("diff", str(baseline), str(candidate), "--json", home=home)
        assert result.returncode == 40, result.stderr
        retained.append(Path(_document(result)["artifacts"]["output_dir"]))
    runs = home / ".metrifid" / "runs"
    assert retained[0] != retained[1]
    assert sorted(runs.iterdir()) == sorted(retained)
    for directory in (home / ".metrifid", runs, *retained):
        assert directory.stat().st_mode & 0o777 == _PRIVATE_DIRECTORY_MODE, directory
    for directory in retained:
        assert directory.parent == runs
        assert sorted(entry.name for entry in directory.iterdir()) == _RETAINED_NAMES


def test_show_reads_the_published_receipt_after_the_compared_models_are_deleted(
    tmp_path: Path,
) -> None:
    """A saved read is a read of the receipt alone, so it survives losing both inputs."""
    models = (tmp_path / "models").resolve()
    baseline, candidate = _copied_pair(models)
    output = (tmp_path / "out").resolve()
    produced = _run("diff", str(baseline), str(candidate), "--output", str(output), "--json")
    assert produced.returncode == 40, produced.stderr
    expected = _compiled_field_names(_document(produced))
    assert expected

    receipt_path = output / "model_release.json"
    before = _run("show", str(receipt_path), "--json")
    assert before.returncode == 0, before.stderr
    assert _compiled_field_names(_document(before)) == expected

    baseline.unlink()
    candidate.unlink()
    assert not baseline.exists()
    assert not candidate.exists()
    after = _run("show", str(receipt_path), "--json")
    assert after.returncode == 0, after.stderr
    assert _compiled_field_names(_document(after)) == expected
    assert after.stdout == before.stdout


def test_show_of_a_receipt_that_recorded_exit_forty_returns_zero_and_keeps_that_outcome(
    tmp_path: Path,
) -> None:
    """Reading a receipt succeeds; the recorded evaluation is reported, never replaced."""
    output = (tmp_path / "out").resolve()
    produced = _run("diff", str(_BASELINE), str(_CHANGED_CANDIDATE), "--output", str(output))
    assert produced.returncode == 40, produced.stderr
    result = _run("show", str(output / "model_release.json"), "--json")
    assert result.returncode == 0, result.stderr
    document = _document(result)
    assert set(document) == _RESULT_MEMBERS
    assert document["command"] == {"name": "show", "exit_code": 0}
    assert document["observation"] == "recorded"
    assert document["evaluation"]["exit_code"] == 40
    assert document["evaluation"]["status"] == _receipt(output)["status"]
    assert document["compiled_comparison"]["state"] == "different"


def test_an_explicit_empty_policy_reproduces_the_generated_discovery_receipt_byte_for_byte(
    tmp_path: Path,
) -> None:
    """Diff is review-model under a generated empty policy, so both publish the same bytes."""
    discovered = (tmp_path / "discovered").resolve()
    first = _run("diff", str(_BASELINE), str(_CHANGED_CANDIDATE), "--output", str(discovered))
    assert first.returncode == 40, first.stderr
    policy = _receipt(discovered)["policy"]
    assert policy["candidate_compiled_sha256"] is None
    assert policy["rules"] == []
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(
        json.dumps(
            {
                "schema": policy["schema"],
                "schema_version": policy["schema_version"],
                "baseline_compiled_sha256": policy["baseline_compiled_sha256"],
                "candidate_compiled_sha256": policy["candidate_compiled_sha256"],
                "rules": policy["rules"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    declared = (tmp_path / "declared").resolve()
    second = _run(
        "review-model",
        str(_BASELINE),
        str(_CHANGED_CANDIDATE),
        "--policy",
        str(policy_path),
        "--output",
        str(declared),
    )
    assert second.returncode == 40, second.stderr
    for name in _CANONICAL_NAMES:
        assert (discovered / name).read_bytes() == (declared / name).read_bytes(), name
    assert json.loads(second.stdout)["receipt_sha256"] == _receipt(discovered)["receipt_sha256"]


def test_a_missing_baseline_entrypoint_refuses_before_anything_is_published(
    tmp_path: Path,
) -> None:
    """An absent model is refused at the closure boundary, with exit 64 and no output."""
    output = (tmp_path / "out").resolve()
    result = _run(
        "diff", str(tmp_path / "absent.xml"), str(_CHANGED_CANDIDATE), "--output", str(output)
    )
    assert result.returncode == 64
    failure = _failure(result)
    assert failure["reason"]["code"] == "MODEL_ENTRYPOINT_INVALID"
    assert failure["reason"]["role"] == "baseline"
    assert failure["operation"] == "review-model"
    assert not output.exists()
    assert "Compiled comparison not established." in result.stdout


def test_an_output_directory_that_already_holds_a_file_refuses_and_is_left_untouched(
    tmp_path: Path,
) -> None:
    """A nonempty output is refused with exit 64, and its existing bytes are not disturbed."""
    output = (tmp_path / "out").resolve()
    output.mkdir()
    committed = output / "existing.txt"
    committed.write_bytes(b"keep me\n")
    result = _run("diff", str(_BASELINE), str(_CHANGED_CANDIDATE), "--output", str(output))
    assert result.returncode == 64
    assert _failure(result)["reason"]["code"] == "OUTPUT_DIRECTORY_NOT_EMPTY"
    assert sorted(entry.name for entry in output.iterdir()) == ["existing.txt"]
    assert committed.read_bytes() == b"keep me\n"


def test_an_output_directory_inside_a_model_root_refuses_with_an_invalid_output_path(
    tmp_path: Path,
) -> None:
    """Evidence may never be written into the tree being measured."""
    models = (tmp_path / "models").resolve()
    baseline, candidate = _copied_pair(models)
    output = models / "out"
    result = _run("diff", str(baseline), str(candidate), "--output", str(output))
    assert result.returncode == 64
    failure = _failure(result)
    assert failure["reason"]["code"] == "OUTPUT_PATH_INVALID"
    assert failure["reason"]["evidence"]["issue"] == "output_inside_model_root"
    assert not output.exists()
    assert sorted(entry.name for entry in models.iterdir()) == ["baseline.xml", "candidate.xml"]


def test_a_model_root_that_is_the_home_directory_refuses_and_names_a_remedy(
    tmp_path: Path,
) -> None:
    """A root that broad cannot be what the caller meant, so it is refused with a way out."""
    home = (tmp_path / "home").resolve()
    baseline, candidate = _copied_pair(home)
    output = (tmp_path / "out").resolve()
    result = _run("diff", str(baseline), str(candidate), "--output", str(output), home=home)
    assert result.returncode == 64
    failure = _failure(result)
    assert failure["reason"]["code"] == "MODEL_ROOT_INVALID"
    assert failure["reason"]["role"] == "baseline"
    assert failure["reason"]["evidence"]["issue"] == "model_root_is_home_directory"
    remedy = failure["reason"]["evidence"]["remedy"]
    assert isinstance(remedy, str)
    assert "--baseline-root" in remedy
    assert not output.exists()


def test_a_refused_run_publishes_no_receipt_and_deletes_nothing_already_committed(
    tmp_path: Path,
) -> None:
    """A refusal is inert: nothing is published anywhere and every prior byte survives."""
    output = (tmp_path / "out").resolve()
    output.mkdir()
    committed = output / "notes.md"
    committed.write_bytes(b"# already here\n")
    neighbour = (tmp_path / "neighbour.txt").resolve()
    neighbour.write_bytes(b"untouched\n")
    result = _run("diff", str(_BASELINE), str(_CHANGED_CANDIDATE), "--output", str(output))
    assert result.returncode == 64
    assert _failure(result)["reason"]["code"] == "OUTPUT_DIRECTORY_NOT_EMPTY"
    assert committed.read_bytes() == b"# already here\n"
    assert neighbour.read_bytes() == b"untouched\n"
    assert list(tmp_path.rglob("model_release.json")) == []
    assert list(tmp_path.rglob("model_release.md")) == []


def test_explicit_roots_admit_a_model_whose_shared_assets_sit_beside_it(tmp_path: Path) -> None:
    """An explicit root measures the whole tree, including files the model never references."""
    baseline_root = (tmp_path / "base").resolve()
    candidate_root = (tmp_path / "cand").resolve()
    baseline = _shared_asset_tree(baseline_root, "1.5")
    candidate = _shared_asset_tree(candidate_root, "2.0")
    output = (tmp_path / "out").resolve()
    result = _run(
        "diff",
        str(baseline),
        str(candidate),
        "--baseline-root",
        str(baseline_root),
        "--candidate-root",
        str(candidate_root),
        "--output",
        str(output),
        "--json",
    )
    assert result.returncode == 40, result.stderr
    document = _document(result)
    for role, root in (("baseline", baseline_root), ("candidate", candidate_root)):
        entry = document["inputs"][role]
        assert entry["entrypoint"] == "models/arm.xml"
        assert entry["root"] == str(root)
        assert entry["member_count"] == _file_count(root) == 3
    assert document["compiled_comparison"]["state"] == "different"
    assert sorted(item.name for item in output.iterdir()) == _RETAINED_NAMES


def test_the_same_shared_asset_model_cannot_be_compiled_without_its_explicit_root(
    tmp_path: Path,
) -> None:
    """Without the explicit root the included asset is outside the measured closure."""
    baseline_root = (tmp_path / "base").resolve()
    candidate_root = (tmp_path / "cand").resolve()
    baseline = _shared_asset_tree(baseline_root, "1.5")
    candidate = _shared_asset_tree(candidate_root, "2.0")
    output = (tmp_path / "out").resolve()
    result = _run("diff", str(baseline), str(candidate), "--output", str(output))
    assert result.returncode == 64
    failure = _failure(result)
    assert failure["reason"]["code"] == "BASELINE_MODEL_COMPILE_ERROR"
    assert failure["reason"]["role"] == "baseline"
    assert list(output.iterdir()) == []


def test_a_refusal_before_allocation_never_creates_a_run_directory(tmp_path: Path) -> None:
    """A request refused before any output is allocated leaves the retention tree untouched."""
    home = tmp_path.resolve()
    refused = _run("diff", str(tmp_path / "absent.xml"), str(_CHANGED_CANDIDATE), home=home)
    assert refused.returncode == 64, refused.stderr
    runs = home / ".metrifid" / "runs"
    assert not runs.exists() or list(runs.iterdir()) == []
    completed = _run("diff", str(_BASELINE), str(_CHANGED_CANDIDATE), home=home)
    assert completed.returncode == 40, completed.stderr
    assert len(list(runs.iterdir())) == 1


def test_an_empty_home_variable_is_refused_instead_of_retaining_under_the_root() -> None:
    """An empty HOME expands to the filesystem root here, and that is never a retention root."""
    environment = os.environ.copy()
    environment["HOME"] = ""
    refused = subprocess.run(
        [sys.executable, "-m", "metrifid.cli", "diff", str(_BASELINE), str(_CHANGED_CANDIDATE)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode == 64, refused.stdout
    reason = json.loads(refused.stderr)["reason"]
    assert reason["code"] == "OUTPUT_PATH_INVALID"
    assert reason["evidence"]["issue"] == "home_directory_unavailable"
    assert "--output" in reason["evidence"]["remedy"]
    assert not Path("/.metrifid").exists()


def test_a_symlinked_tool_directory_is_refused_and_nothing_is_written_through_it(
    tmp_path: Path,
) -> None:
    """Refuse a retained-run root that is not a real directory the tool owns."""
    home = (tmp_path / "home").resolve()
    home.mkdir()
    elsewhere = (tmp_path / "elsewhere").resolve()
    elsewhere.mkdir()
    (home / ".metrifid").symlink_to(elsewhere)
    refused = _run("diff", str(_BASELINE), str(_CHANGED_CANDIDATE), home=home)
    assert refused.returncode == 64, refused.stdout
    reason = json.loads(refused.stderr)["reason"]
    assert reason["code"] == "OUTPUT_PATH_INVALID"
    assert reason["evidence"]["issue"] == "output_path_not_real_directory"
    assert list(elsewhere.iterdir()) == []


def test_the_declared_policy_route_still_refuses_a_missing_policy_path(tmp_path: Path) -> None:
    """Running without a declared policy is a separate call, never a null argument falling through.

    The discovery route exists for callers who mean it. A caller of the declared-policy API who
    supplies nothing is refused exactly as it was before that route existed, because a review
    evaluated against no policy at all must never be reported as a review that passed.
    """
    source = textwrap.dedent(
        f"""
        import pathlib
        import tempfile

        from metrifid.model_release import ModelReleaseOperationError, review_model_release

        baseline = {str(_BASELINE)!r}
        candidate = {str(_EQUIVALENT_CANDIDATE)!r}
        for supplied in (None, "", "/nonexistent/policy.json"):
            output = pathlib.Path(tempfile.mkdtemp()).resolve() / "out"
            try:
                review_model_release(baseline, candidate, supplied, str(output))
            except ModelReleaseOperationError as exc:
                assert int(exc.failure.exit_code) == 64, exc.failure.exit_code
                assert exc.failure.reason.code.value == "CONFIGURATION_PARSE_FAILED"
                continue
            raise AssertionError(f"a policy path of {{supplied!r}} was accepted")
        print("DECLARED_ROUTE_STILL_REFUSES")
        """
    )
    script = tmp_path / "declared.py"
    script.write_text(source, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "DECLARED_ROUTE_STILL_REFUSES" in completed.stdout


def _uncompilable_pair(root: Path) -> tuple[Path, Path]:
    """Write one compilable baseline beside a candidate MuJoCo refuses, in separate roots."""
    root.mkdir(parents=True, exist_ok=True)
    baseline_root = root / "before"
    candidate_root = root / "after"
    baseline_root.mkdir()
    candidate_root.mkdir()
    baseline = baseline_root / "model.xml"
    baseline.write_text(_BASELINE.read_text(encoding="utf-8"), encoding="utf-8")
    candidate = candidate_root / "model.xml"
    candidate.write_text(
        '<mujoco model="broken"><worldbody><body name="x">'
        '<geom type="nosuchtype" size="1"/></body></worldbody></mujoco>',
        encoding="utf-8",
    )
    return baseline, candidate


def test_a_failure_after_allocation_reports_the_known_run_and_probes_nothing(
    tmp_path: Path,
) -> None:
    """A core failure reports the run it already knows, leaves it alone, and claims no receipt.

    Nothing is inspected or removed on the way out. Emptiness is not ownership, so a directory
    substituted after allocation must never be adopted or deleted by a cleanup pass.
    """
    home = (tmp_path / "home").resolve()
    home.mkdir()
    baseline, candidate = _uncompilable_pair((tmp_path / "models").resolve())
    refused = _run("diff", str(baseline), str(candidate), "--json", home=home)
    assert refused.returncode == 64, refused.stdout
    document = json.loads(refused.stdout)
    assert [problem["code"] for problem in document["problems"]] == [
        "CANDIDATE_MODEL_COMPILE_ERROR"
    ]
    reported = document["artifacts"]["output_dir"]
    runs = home / ".metrifid" / "runs"
    retained = sorted(runs.iterdir())
    assert len(retained) == 1
    assert reported == str(retained[0])
    assert list(retained[0].iterdir()) == []
    for name in ("receipt", "markdown", "html", "receipt_sha256"):
        assert document["artifacts"][name] is None


def test_an_explicit_output_directory_is_still_named_when_the_run_refuses(
    tmp_path: Path,
) -> None:
    """A directory the caller chose is theirs, so it is reported even on a refusal."""
    baseline, candidate = _uncompilable_pair((tmp_path / "models").resolve())
    output = (tmp_path / "chosen").resolve()
    refused = _run("diff", str(baseline), str(candidate), "--output", str(output), "--json")
    assert refused.returncode == 64, refused.stdout
    document = json.loads(refused.stdout)
    assert document["artifacts"]["output_dir"] == str(output)
    assert output.is_dir()


def test_a_refusal_keeps_its_own_reason_and_exit_code_on_stderr(tmp_path: Path) -> None:
    """The core refusal is the message; nothing about rendering a report may replace it."""
    home = (tmp_path / "home").resolve()
    home.mkdir()
    baseline, candidate = _uncompilable_pair((tmp_path / "models").resolve())
    refused = _run("diff", str(baseline), str(candidate), home=home)
    assert refused.returncode == 64, refused.stdout
    reason = json.loads(refused.stderr)["reason"]
    assert reason["code"] == "CANDIDATE_MODEL_COMPILE_ERROR"
    assert json.loads(refused.stderr)["exit_code"] == 64


@pytest.mark.parametrize(
    "arguments",
    [
        ("diff", "--json"),
        ("diff", "only-one.xml", "--json"),
        ("show", "--json"),
        ("show", str(_BASELINE), "--json", "--nonsense"),
    ],
)
def test_the_installed_command_answers_a_usage_error_in_json_with_one_document(
    arguments: tuple[str, ...],
) -> None:
    """An agent that asked for JSON can parse an ordinary usage error from the real executable."""
    refused = _run(*arguments)
    assert refused.returncode == 64, refused.stderr
    document = json.loads(refused.stdout)
    assert document["schema"] == "metrifid.result"
    assert document["command"]["name"] == arguments[0]
    assert document["command"]["exit_code"] == refused.returncode
    assert document["compiled_comparison"]["state"] == "not_established"
    assert document["problems"][0]["code"] == "INVALID_CLI_INVOCATION"
    assert json.loads(refused.stderr)["reason"]["code"] == "INVALID_CLI_INVOCATION"


@pytest.mark.parametrize(
    ("arguments", "command"),
    [(("--", "diff", "--json"), "diff"), (("--", "show", "--json"), "show")],
)
def test_a_leading_end_of_options_marker_is_answered_as_this_interpreter_parses_it(
    tmp_path: Path, arguments: tuple[str, ...], command: str
) -> None:
    """A marker in command position is answered the way the running argparse reads it.

    Python 3.12 and later consume one leading marker, so these are ordinary missing-operand usage
    errors for a recognized command and the caller gets its result document. Python 3.11 refuses
    the marker as an invalid command choice, so nothing was recognized and stdout stays empty.
    Either way the operational failure is on stderr, the exit is 64, and nothing is compiled or
    published. The installed interpreter decides which shape applies, so both are admitted here
    and the wrong one is never accepted for the interpreter in use.
    """
    before = sorted(p.name for p in tmp_path.iterdir())
    refused = _run(*arguments, home=tmp_path)
    assert refused.returncode == 64, refused.stdout + refused.stderr
    failure = json.loads(refused.stderr)
    assert failure["reason"]["code"] == "INVALID_CLI_INVOCATION"
    assert "Traceback" not in refused.stderr

    if sys.version_info >= (3, 12):
        document = json.loads(refused.stdout)
        assert document["schema"] == "metrifid.result"
        assert document["command"]["name"] == command
        assert document["command"]["exit_code"] == 64
        assert document["observation"] == "none"
        assert document["compiled_comparison"]["state"] == "not_established"
        assert document["problems"][0]["code"] == "INVALID_CLI_INVOCATION"
    else:
        # The parser never recognized a command, so nothing may be promised on stdout.
        assert refused.stdout == "", refused.stdout

    # No comparison was dispatched and no run directory was created under the isolated home.
    assert sorted(p.name for p in tmp_path.iterdir()) == before


def test_a_handled_read_refusal_in_json_mode_also_returns_one_document(tmp_path: Path) -> None:
    """A refusal the reader handles is reported in the requested format, not only on stderr."""
    absent = (tmp_path / "absent.json").resolve()
    refused = _run("show", str(absent), "--json")
    assert refused.returncode == 64, refused.stderr
    document = json.loads(refused.stdout)
    assert document["problems"][0]["code"] == "CONFIGURATION_IO_FAILED"
    assert document["observation"] == "none"


def test_the_installed_command_leaves_stdout_empty_when_no_document_was_requested() -> None:
    """Without an actual JSON option the usage error stays on stderr alone."""
    refused = _run("diff")
    assert refused.returncode == 64
    assert refused.stdout == ""
    assert json.loads(refused.stderr)["reason"]["code"] == "INVALID_CLI_INVOCATION"


def test_a_receipt_named_like_a_flag_after_the_separator_is_read_not_reformatted(
    tmp_path: Path,
) -> None:
    """End-of-options semantics survive: a file named --json is read and rendered readably."""
    receipt = (tmp_path / "--json").resolve()
    receipt.write_bytes(_FIXTURES.joinpath("equal_model_release.json").read_bytes())
    completed = _run("show", "--", str(receipt))
    assert completed.returncode == 0, completed.stderr
    assert not completed.stdout.lstrip().startswith("{")
    assert "Compiled model identical." in completed.stdout


def test_full_output_returns_the_inventories_and_omission_rows_the_receipt_retained() -> None:
    """A full export must not quietly drop detail the producer actually recorded."""
    source = _FIXTURES / "changed_model_release.json"
    original = json.loads(source.read_text(encoding="utf-8"))
    certification = original["certification_receipt"]
    exported = _run("show", str(source), "--json", "--full")
    assert exported.returncode == 0, exported.stderr
    document = json.loads(exported.stdout)
    assert document["truncated"] is False
    for role in ("baseline", "candidate"):
        assert (
            document["inputs"][role]["members"] == certification[role]["source_closure"]["members"]
        )
    producer = [item for item in document["limitations"] if item["category"] == "producer"]
    assert len(producer) == 1
    assert (
        producer[0]["detail"]["omitted_fields"] == certification["field_report"]["omitted_fields"]
    )


def _published(directory: Path) -> list[str]:
    """Return the file names a completed comparison left in one output directory."""
    return sorted(entry.name for entry in directory.iterdir())


@pytest.mark.parametrize(
    ("candidate", "expected"), [(_EQUIVALENT_CANDIDATE, 0), (_CHANGED_CANDIDATE, 40)]
)
def test_a_completed_comparison_publishes_the_offline_report_beside_its_receipt(
    tmp_path: Path, candidate: Path, expected: int
) -> None:
    """The report is the readable half of the same evidence, written into the same directory."""
    output = (tmp_path / "run").resolve()
    completed = _run("diff", str(_BASELINE), str(candidate), "--output", str(output), "--json")
    assert completed.returncode == expected, completed.stderr
    assert _published(output) == ["model_release.json", "model_release.md", "report.html"]
    document = json.loads(completed.stdout)
    assert document["artifacts"]["html"] == str(output / "report.html")
    assert not [item for item in document["limitations"] if item["category"] == "presentation"]
    report = (output / "report.html").read_text(encoding="utf-8")
    assert "<script" not in report
    assert "Not a statement about behaviour, safety, or approval." in report


def test_a_report_that_cannot_be_written_keeps_the_comparison_and_its_evidence(
    tmp_path: Path,
) -> None:
    """A presentation failure is not a comparison failure, and it never claims a report exists."""
    source = textwrap.dedent(
        f"""
        import json
        import pathlib

        from metrifid.result import _diff, run_diff

        def refuse(document):
            raise ValueError("renderer unavailable")

        _diff.__dict__.setdefault("_TEST_HOOK", None)
        import metrifid.result._report as report_module
        report_module.render_report = refuse

        output = pathlib.Path({str(tmp_path / "refused")!r})
        outcome = run_diff({str(_BASELINE)!r}, {str(_CHANGED_CANDIDATE)!r},
                           output_directory=str(output))
        document = outcome.document
        assert outcome.exit_code == 40, outcome.exit_code
        assert document["artifacts"]["html"] is None
        presentation = [i for i in document["limitations"] if i["category"] == "presentation"]
        assert len(presentation) == 1, presentation
        names = sorted(p.name for p in output.iterdir())
        assert names == ["model_release.json", "model_release.md"], names
        print("PRESENTATION_ONLY")
        """
    )
    script = tmp_path / "refused.py"
    script.write_text(source, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(script)], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PRESENTATION_ONLY" in completed.stdout


def test_a_conflicting_report_name_is_never_overwritten(tmp_path: Path) -> None:
    """Publishing links the final name, so a report already present survives untouched."""
    source = textwrap.dedent(
        f"""
        import pathlib

        from metrifid.result import run_diff
        import metrifid.result._report as report_module

        original = report_module.render_report

        def plant_then_render(document):
            directory = pathlib.Path(str(document["artifacts"]["output_dir"]))
            (directory / "report.html").write_text("PRE-EXISTING", encoding="utf-8")
            return original(document)

        report_module.render_report = plant_then_render

        output = pathlib.Path({str(tmp_path / "conflict")!r})
        outcome = run_diff({str(_BASELINE)!r}, {str(_CHANGED_CANDIDATE)!r},
                           output_directory=str(output))
        assert outcome.exit_code == 40, outcome.exit_code
        assert outcome.document["artifacts"]["html"] is None
        kept = (output / "report.html").read_text(encoding="utf-8")
        assert kept == "PRE-EXISTING", kept
        names = sorted(p.name for p in output.iterdir())
        assert names == ["model_release.json", "model_release.md", "report.html"], names
        print("REPORT_NOT_OVERWRITTEN")
        """
    )
    script = tmp_path / "conflict.py"
    script.write_text(source, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(script)], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "REPORT_NOT_OVERWRITTEN" in completed.stdout


def test_repeated_comparisons_leak_no_descriptors(tmp_path: Path) -> None:
    """The bound output and the report handle both close, on the success path and the failure one."""
    source = textwrap.dedent(
        f"""
        import os
        import pathlib

        from metrifid.result import run_diff

        root = pathlib.Path({str(tmp_path / "leak")!r})
        root.mkdir(parents=True, exist_ok=True)

        def open_descriptors():
            return len(os.listdir("/dev/fd"))

        def run(index):
            run_diff({str(_BASELINE)!r}, {str(_CHANGED_CANDIDATE)!r},
                     output_directory=str(root / f"ok{{index}}"))

        def refused(index):
            try:
                run_diff({str(_BASELINE)!r}, str(root / "absent.xml"),
                         output_directory=str(root / f"no{{index}}"))
            except Exception:
                pass

        run(0); refused(0)
        before = open_descriptors()
        for index in range(1, 5):
            run(index); refused(index)
        after = open_descriptors()
        assert after == before, (before, after)
        print("NO_DESCRIPTOR_LEAK", before)
        """
    )
    script = tmp_path / "leak.py"
    script.write_text(source, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(script)], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "NO_DESCRIPTOR_LEAK" in completed.stdout


def test_a_directory_replaced_during_publication_is_never_announced_as_the_report_location(
    tmp_path: Path,
) -> None:
    """The published path is a claim about where the report is; a moved directory invalidates it.

    Rendering is the one unbounded step between publishing the canonical pair and writing the
    report. If the public pathname stops naming the bound directory while that runs, the run has
    nothing truthful to say about an HTML location: the descriptor still reaches the original
    directory, but the pathname now reaches something else. The comparison itself already
    completed, so it keeps its exit code and its canonical bytes, and the report is simply absent
    with a presentation limitation recording that.
    """
    source = textwrap.dedent(
        f"""
        import hashlib
        import pathlib

        from metrifid.result import run_diff
        import metrifid.result._report as report_module

        original = report_module.render_report
        output = pathlib.Path({str(tmp_path / "run")!r})
        moved = pathlib.Path({str(tmp_path / "run-moved")!r})
        digests = {{}}

        def move_then_render(document):
            for name in ("model_release.json", "model_release.md"):
                digests[name] = hashlib.sha256((output / name).read_bytes()).hexdigest()
            output.rename(moved)
            output.mkdir()
            return original(document)

        report_module.render_report = move_then_render

        outcome = run_diff({str(_BASELINE)!r}, {str(_CHANGED_CANDIDATE)!r},
                           output_directory=str(output))
        document = outcome.document

        assert outcome.exit_code == 40, outcome.exit_code
        assert document["artifacts"]["html"] is None, document["artifacts"]["html"]
        presentation = [i for i in document["limitations"] if i["category"] == "presentation"]
        assert len(presentation) == 1, presentation

        # The canonical pair travelled with the directory it was published into and is unchanged.
        kept = sorted(p.name for p in moved.iterdir())
        assert kept == ["model_release.json", "model_release.md"], kept
        for name, digest in digests.items():
            actual = hashlib.sha256((moved / name).read_bytes()).hexdigest()
            assert actual == digest, name

        # The directory now standing at the public pathname received nothing from this run.
        assert sorted(p.name for p in output.iterdir()) == [], sorted(output.iterdir())
        print("NO_STALE_REPORT_CLAIM")
        """
    )
    script = tmp_path / "moved.py"
    script.write_text(source, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(script)], check=False, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "NO_STALE_REPORT_CLAIM" in completed.stdout
