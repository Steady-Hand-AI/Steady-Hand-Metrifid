"""The frozen contract of the presented ``metrifid.result`` document.

The result document is a presentation over an already-authoritative receipt, so its member set,
its schema identity and the provenance of every member are contract rather than convenience.
These tests hold four lines: the fourteen members are always all present, compiled identity is
read from the certification byte comparison and never from a policy outcome, a recorded negative
evaluation survives a successful read, and the whole reader stays free of any native dependency.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from metrifid import cli
from metrifid.compare._failure import ComparisonOperationError
from metrifid.json_values import CanonicalValue
from metrifid.operational import OperationalReasonCode
from metrifid.result import (
    RESULT_SCHEMA,
    RESULT_SCHEMA_VERSION,
    DiffOutcome,
    build_document,
    build_failure_document,
    encode_result,
    render_text,
    run_diff,
    show_receipt,
)

_BLOCK_NATIVE = textwrap.dedent(
    """
    import sys

    class _BlockNative:
        def find_spec(self, name, path=None, target=None):
            root = name.split(".", 1)[0]
            if root in {"mujoco", "numpy"}:
                raise ModuleNotFoundError(f"{name} is unavailable in this environment")
            return None

    sys.meta_path.insert(0, _BlockNative())
    """
)

_RESULT_MEMBERS = [
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
]

_CERTIFICATION_SCHEMA = "metrifid.compiled_equivalence_receipt"
_MODEL_RELEASE_SCHEMA = "metrifid.model_release_receipt"
_ADMITTED_SCHEMAS = [_CERTIFICATION_SCHEMA, _MODEL_RELEASE_SCHEMA]
_STUB_DISTRIBUTION_SHA256 = "1" * 64

_BASELINE_XML = """
<mujoco model="result-contract">
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


def _bind_stub_distribution(patch: pytest.MonkeyPatch) -> None:
    """Bind one fixed distribution digest into every module that measures the installed wheel."""
    from metrifid import _runtime_identity
    from metrifid.certify import _run as certify_run
    from metrifid.model_release import _run as model_release_run
    from metrifid.result import _diff as diff_module
    from metrifid.result import _show as show_module

    modules = (_runtime_identity, certify_run, model_release_run, diff_module, show_module)
    for module in modules:
        patch.setattr(module, "installed_distribution_sha256", lambda: _STUB_DISTRIBUTION_SHA256)


def _written_pair(root: Path) -> tuple[Path, Path]:
    """Write one baseline and one candidate model into separate roots beneath one directory."""
    baseline_root = root / "baseline"
    candidate_root = root / "candidate"
    baseline_root.mkdir(parents=True)
    candidate_root.mkdir(parents=True)
    baseline = baseline_root / "model.xml"
    candidate = candidate_root / "model.xml"
    baseline.write_text(_BASELINE_XML, encoding="utf-8")
    candidate.write_text(_CANDIDATE_XML, encoding="utf-8")
    return baseline, candidate


def _run_pure(body: str) -> subprocess.CompletedProcess[str]:
    """Run one script in a fresh interpreter where MuJoCo and NumPy cannot be imported."""
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_NATIVE + textwrap.dedent(body)],
        check=False,
        capture_output=True,
        text=True,
    )


def _synthetic_receipt(
    *, status: str, exit_code: int, comparison: dict[str, CanonicalValue] | None
) -> dict[str, CanonicalValue]:
    """Build one model-release-shaped receipt carrying an exact status and byte comparison."""
    receipt: dict[str, CanonicalValue] = {
        "schema": _MODEL_RELEASE_SCHEMA,
        "schema_version": 1,
        "status": status,
        "completed_exit_code": exit_code,
        "receipt_sha256": "a" * 64,
        "changes": [],
        "changes_complete": True,
        "limitations": [],
        "missing_required_rules": [],
        "policy": {"raw_sha256": "b" * 64, "rule_count": 0},
    }
    if comparison is not None:
        receipt["certification_receipt"] = {"byte_comparison": comparison}
    return receipt


def _presented(receipt: dict[str, CanonicalValue]) -> dict[str, CanonicalValue]:
    """Present one receipt through the shared document builder with fixed surrounding facts."""
    return build_document(
        command="show",
        exit_code=0,
        observation="recorded",
        receipt=receipt,
        policy_origin="not_recorded",
        reader_version="9.9.9",
        inputs={"baseline": None, "candidate": None, "roots_correspond": None},
        artifacts={"output_dir": None, "receipt": None, "receipt_sha256": None},
    )


def _member(document: dict[str, CanonicalValue], name: str) -> dict[str, CanonicalValue]:
    """Return one object-valued member of a result document."""
    value = document[name]
    assert isinstance(value, dict), name
    return value


def _tampered_text(raw: str, member: str) -> str:
    """Increment one integer member of a serialized receipt without resealing its hashes."""
    marker = f'"{member}":'
    start = raw.index(marker) + len(marker)
    end = start
    while raw[end].isdigit():
        end += 1
    assert end > start, member
    return raw[:start] + str(int(raw[start:end]) + 1) + raw[end:]


@pytest.fixture(scope="module")
def differing_diff(tmp_path_factory: pytest.TempPathFactory) -> DiffOutcome:
    """Produce one completed diff of two models whose compiled artifacts differ."""
    root: Path = tmp_path_factory.mktemp("result-diff").resolve()
    baseline, candidate = _written_pair(root)
    with pytest.MonkeyPatch.context() as patch:
        _bind_stub_distribution(patch)
        return run_diff(str(baseline), str(candidate), output_directory=str(root / "out"))


@pytest.fixture(scope="module")
def model_release_receipt(differing_diff: DiffOutcome) -> Path:
    """Return the saved model-release receipt that the completed diff published."""
    artifacts = differing_diff.document["artifacts"]
    assert isinstance(artifacts, dict)
    receipt = artifacts["receipt"]
    assert isinstance(receipt, str)
    return Path(receipt)


@pytest.fixture(scope="module")
def certification_receipt(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Produce one standalone certification receipt for the same differing pair."""
    from metrifid.certify import certify_models

    root: Path = tmp_path_factory.mktemp("result-certify").resolve()
    baseline, candidate = _written_pair(root)
    with pytest.MonkeyPatch.context() as patch:
        _bind_stub_distribution(patch)
        result = certify_models(str(baseline), str(candidate), str(root / "out"))
    return Path(result.certification_json)


def test_the_result_schema_string_and_version_are_frozen(model_release_receipt: Path) -> None:
    """Pin the result schema identity as the exact string and integer the contract names."""
    assert RESULT_SCHEMA == "metrifid.result"
    assert RESULT_SCHEMA_VERSION == 1
    assert isinstance(RESULT_SCHEMA_VERSION, int)
    assert not isinstance(RESULT_SCHEMA_VERSION, bool)

    document = show_receipt(str(model_release_receipt))

    assert document["schema"] == "metrifid.result"
    assert document["schema_version"] == 1


def test_a_success_document_carries_exactly_the_fourteen_frozen_members(
    model_release_receipt: Path,
) -> None:
    """Hold the member set of a successful read to the exact fourteen-member contract."""
    document = show_receipt(str(model_release_receipt))

    assert sorted(document.keys()) == _RESULT_MEMBERS
    assert len(_RESULT_MEMBERS) == 14


def test_a_failure_document_carries_the_same_fourteen_members_with_truthful_empty_values() -> None:
    """Keep every member present on a failure, stating nothing rather than omitting members."""
    document = build_failure_document(
        command="diff",
        exit_code=64,
        reason_code="OUTPUT_WRITE_FAILED",
        message="the retained run directory could not be written",
        reader_version="9.9.9",
        inputs={"baseline": None, "candidate": None, "roots_correspond": None},
        artifacts={"output_dir": None, "receipt": None, "receipt_sha256": None},
        policy_origin="not_created",
    )

    assert sorted(document.keys()) == _RESULT_MEMBERS
    assert document["command"] == {"name": "diff", "exit_code": 64}
    assert document["observation"] == "none"
    assert document["compiled_comparison"] == {
        "state": "not_established",
        "equal": None,
        "first_differing_byte_offset": None,
        "differing_byte_count": None,
    }
    assert document["evaluation"] == {
        "status": None,
        "exit_code": None,
        "policy_origin": "not_created",
        "policy_raw_sha256": None,
        "policy_rule_count": None,
        "receipt_sha256": None,
    }
    assert document["runtime"] == {"reader": "9.9.9", "producer": None}
    assert document["findings"] == []
    assert document["finding_count"] == 0
    assert document["limitations"] == []
    assert document["truncated"] is False
    assert document["problems"] == [
        {
            "code": "OUTPUT_WRITE_FAILED",
            "message": "the retained run directory could not be written",
        }
    ]


def test_compiled_identity_is_identical_from_equal_bytes_despite_a_failing_status() -> None:
    """Read compiled identity from the byte comparison even when the receipt status failed."""
    receipt = _synthetic_receipt(
        status="REVIEW_REQUIRED",
        exit_code=40,
        comparison={
            "equal": True,
            "first_differing_byte_offset": None,
            "differing_byte_count": 0,
        },
    )

    document = _presented(receipt)

    assert _member(document, "compiled_comparison")["state"] == "identical"
    assert _member(document, "compiled_comparison")["equal"] is True
    assert _member(document, "evaluation")["status"] == "REVIEW_REQUIRED"
    assert _member(document, "evaluation")["exit_code"] == 40


def test_compiled_identity_is_different_from_unequal_bytes_despite_a_passing_status() -> None:
    """Refuse to call two differing artifacts identical because the policy outcome passed."""
    receipt = _synthetic_receipt(
        status="NO_COMPILED_CHANGE",
        exit_code=0,
        comparison={
            "equal": False,
            "first_differing_byte_offset": 1692,
            "differing_byte_count": 71,
        },
    )

    document = _presented(receipt)

    assert _member(document, "compiled_comparison") == {
        "state": "different",
        "equal": False,
        "first_differing_byte_offset": 1692,
        "differing_byte_count": 71,
    }
    assert _member(document, "evaluation")["status"] == "NO_COMPILED_CHANGE"
    assert _member(document, "evaluation")["exit_code"] == 0


def test_compiled_identity_is_not_established_with_null_members_without_a_certification() -> None:
    """State that nothing was compared, rather than guessing, when no certification exists."""
    receipt = _synthetic_receipt(status="REVIEW_REQUIRED", exit_code=40, comparison=None)

    document = _presented(receipt)

    assert _member(document, "compiled_comparison") == {
        "state": "not_established",
        "equal": None,
        "first_differing_byte_offset": None,
        "differing_byte_count": None,
    }
    assert _member(document, "evaluation")["status"] == "REVIEW_REQUIRED"


def test_a_first_differing_byte_offset_of_zero_survives_as_zero_and_not_as_null() -> None:
    """Carry a first differing offset of zero through the document, the JSON and the text."""
    receipt = _synthetic_receipt(
        status="REVIEW_REQUIRED",
        exit_code=40,
        comparison={
            "equal": False,
            "first_differing_byte_offset": 0,
            "differing_byte_count": 1,
        },
    )

    document = _presented(receipt)
    offset = _member(document, "compiled_comparison")["first_differing_byte_offset"]

    assert offset == 0
    assert offset is not None
    assert isinstance(offset, int)
    assert not isinstance(offset, bool)
    encoded = json.loads(encode_result(document, full=True).decode("utf-8"))
    assert encoded["compiled_comparison"]["first_differing_byte_offset"] == 0
    assert "first_differing_byte_offset" in encoded["compiled_comparison"]
    assert "1 serialized byte(s) differ, first at offset 0." in render_text(document)


def test_show_preserves_a_recorded_review_required_outcome_while_its_own_read_succeeds(
    model_release_receipt: Path,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """Report the recorded negative evaluation even though reading the receipt exits zero."""
    document = show_receipt(str(model_release_receipt))

    assert _member(document, "evaluation")["status"] == "REVIEW_REQUIRED"
    assert _member(document, "evaluation")["exit_code"] == 40
    assert document["observation"] == "recorded"
    assert document["command"] == {"name": "show", "exit_code": 0}

    assert cli.main(["show", str(model_release_receipt)]) == 0
    assert capsysbinary.readouterr().err == b""


def test_show_of_a_model_release_receipt_reports_the_policy_origin_as_not_recorded(
    model_release_receipt: Path,
) -> None:
    """Say that a saved model-release receipt retained no policy origin of its own."""
    document = show_receipt(str(model_release_receipt))

    assert _member(document, "evaluation")["policy_origin"] == "not_recorded"


def test_show_of_a_standalone_certification_reports_the_policy_origin_as_not_applicable(
    certification_receipt: Path,
) -> None:
    """Say that a certification never had a policy rather than that its policy was empty."""
    document = show_receipt(str(certification_receipt))

    assert _member(document, "evaluation")["policy_origin"] == "not_applicable"
    assert _member(document, "evaluation")["policy_raw_sha256"] is None
    assert _member(document, "evaluation")["policy_rule_count"] is None


def test_a_completed_diff_reports_the_policy_origin_as_generated_empty(
    differing_diff: DiffOutcome,
) -> None:
    """Say that a diff satisfied a policy it generated itself, holding no declared rule."""
    document = differing_diff.document

    assert differing_diff.failure is None
    assert differing_diff.exit_code == 40
    assert sorted(document.keys()) == _RESULT_MEMBERS
    assert _member(document, "evaluation")["policy_origin"] == "generated_empty"
    assert _member(document, "evaluation")["policy_rule_count"] == 0
    assert document["observation"] == "produced_now"
    assert _member(document, "command") == {"name": "diff", "exit_code": 40}
    assert _member(document, "compiled_comparison")["state"] == "different"


def test_a_diff_that_failed_before_any_policy_existed_reports_the_policy_origin_as_not_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Say that policy creation never happened when the comparison failed at admission."""
    _bind_stub_distribution(monkeypatch)
    root = tmp_path.resolve()
    _, candidate = _written_pair(root)
    missing = root / "absent" / "model.xml"

    outcome = run_diff(str(missing), str(candidate), output_directory=str(root / "out"))

    assert outcome.failure is not None
    assert outcome.exit_code == 64
    document = outcome.document
    assert sorted(document.keys()) == _RESULT_MEMBERS
    assert _member(document, "evaluation")["policy_origin"] == "not_created"
    assert document["observation"] == "none"
    assert _member(document, "compiled_comparison")["state"] == "not_established"
    assert document["problems"] == [
        {"code": "MODEL_ENTRYPOINT_INVALID", "message": "entrypoint_unavailable"}
    ]
    assert not (root / "out").exists()


def test_show_renders_a_saved_receipt_with_mujoco_and_numpy_blocked(
    model_release_receipt: Path,
) -> None:
    """Read and render one saved receipt in an interpreter where no native library imports."""
    completed = _run_pure(
        f"""
        import sys
        from metrifid.cli import main

        assert main(["show", {str(model_release_receipt)!r}]) == 0
        native = [m for m in sys.modules if m.split(".", 1)[0] in {{"mujoco", "numpy"}}]
        assert not native, native
        print("SHOW_TEXT_PURE")
        """
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "SHOW_TEXT_PURE" in completed.stdout
    assert "Compiled model changed." in completed.stdout


def test_show_json_encodes_a_saved_receipt_with_mujoco_and_numpy_blocked(
    model_release_receipt: Path,
) -> None:
    """Encode the bounded JSON result in an interpreter where no native library imports."""
    completed = _run_pure(
        f"""
        import sys
        from metrifid.cli import main

        assert main(["show", "--json", {str(model_release_receipt)!r}]) == 0
        native = [m for m in sys.modules if m.split(".", 1)[0] in {{"mujoco", "numpy"}}]
        assert not native, native
        print("SHOW_JSON_PURE")
        """
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "SHOW_JSON_PURE" in completed.stdout
    assert '"schema":"metrifid.result"' in completed.stdout


def test_importing_the_result_package_loads_no_native_module() -> None:
    """Keep the presented-result package itself free of MuJoCo and NumPy at import time."""
    completed = _run_pure(
        """
        import sys
        import importlib

        importlib.import_module("metrifid.result")
        native = [m for m in sys.modules if m.split(".", 1)[0] in {"mujoco", "numpy"}]
        assert not native, native
        print("RESULT_IMPORT_PURE")
        """
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "RESULT_IMPORT_PURE" in completed.stdout


def test_show_refuses_a_schema_that_is_neither_admitted_receipt_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsysbinary: pytest.CaptureFixture[bytes],
) -> None:
    """Refuse a policy document rather than guessing which receipt reader it belongs to."""
    _bind_stub_distribution(monkeypatch)
    document_path = tmp_path / "policy.json"
    document_path.write_text(
        json.dumps(
            {"schema": "metrifid.model_release_policy", "schema_version": 1},
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    with pytest.raises(ComparisonOperationError) as caught:
        show_receipt(str(document_path))

    failure = caught.value.failure
    assert int(failure.exit_code) == 64
    assert failure.reason.code is OperationalReasonCode.CONFIGURATION_PARSE_FAILED
    assert failure.reason.field == "schema"
    evidence = failure.reason.evidence
    assert evidence["issue"] == "receipt_schema_not_admitted"
    assert evidence["observed_schema"] == "metrifid.model_release_policy"
    assert list(evidence["admitted_schemas"]) == _ADMITTED_SCHEMAS

    assert cli.main(["show", str(document_path)]) == 64
    emitted = json.loads(capsysbinary.readouterr().err.decode("utf-8"))
    assert emitted["reason"]["evidence"]["admitted_schemas"] == _ADMITTED_SCHEMAS


def test_show_reads_a_certification_saved_under_a_model_release_filename(
    certification_receipt: Path, tmp_path: Path
) -> None:
    """Decide the reader from the receipt's own schema, never from the file's name."""
    misnamed = tmp_path / "model_release.json"
    misnamed.write_bytes(certification_receipt.read_bytes())

    document = show_receipt(str(misnamed))
    original = show_receipt(str(certification_receipt))

    assert _member(document, "evaluation")["policy_origin"] == "not_applicable"
    assert document["evaluation"] == original["evaluation"]
    assert document["findings"] == original["findings"]
    findings = document["findings"]
    assert isinstance(findings, list)
    assert findings
    for finding in findings:
        assert isinstance(finding, dict)
        assert finding["kind"] == "compiled_field"
        source = finding["source"]
        assert isinstance(source, str)
        assert source.startswith("/field_report/changed_fields/")


def test_show_refuses_a_receipt_whose_recorded_self_hash_no_longer_matches_its_content(
    certification_receipt: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse a receipt whose content was edited without resealing its recorded hash."""
    _bind_stub_distribution(monkeypatch)
    raw = certification_receipt.read_text(encoding="utf-8")
    tampered = tmp_path / "certification.json"
    tampered.write_text(_tampered_text(raw, "fields_compared_count"), encoding="utf-8")
    assert tampered.read_text(encoding="utf-8") != raw

    with pytest.raises(ComparisonOperationError) as caught:
        show_receipt(str(tampered))

    failure = caught.value.failure
    assert int(failure.exit_code) == 64
    assert failure.reason.code is OperationalReasonCode.CONFIGURATION_PARSE_FAILED
    assert failure.reason.evidence["issue"] == "certification_receipt_invalid"
    assert "receipt_sha256" in str(failure.reason.evidence["message"])
    assert show_receipt(str(certification_receipt))["schema"] == "metrifid.result"
