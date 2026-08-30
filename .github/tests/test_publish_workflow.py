"""Structural and behavioural contract for the publication workflow.

The publication workflow is the only path by which project bytes reach a public registry, so its
shape is checked here rather than trusted. Structural assertions cover the job graph, exact job
conditions, permissions, environments, action pins, step order and asset paths.

Structure alone is not enough. A step can contain the right words and still decide the wrong way, so
every decision-bearing validator embedded in the workflow is extracted and executed here against
small temporary fixtures, positive and negative. Where a validator is repeated in more than one
protected job, every copy is executed, because proving only the first copy proves nothing about the
others. Network calls are replaced by an injected stub, so the real decision logic runs offline
without the workflow carrying any test-only branch.

The workflow runs this file itself, so it is exercised on every publication attempt.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

import pytest
import yaml

REPOSITORY = Path(__file__).resolve().parents[2]
WORKFLOW = REPOSITORY / ".github/workflows/publish.yml"

VALIDATE_ONLY = "validate-only"
STAGED_RELEASE = "staged-release"
EXACT_EXTERNAL_CONDITION = "${{ inputs.mode == 'staged-release' }}"
EXACT_UPLOAD_CONDITION = "${{ steps.remote_state.outputs.upload == 'true' }}"

REGISTRY_PUBLISH_JOBS = ("testpypi_publish", "pypi_publish")
REGISTRY_VERIFY_JOBS = ("testpypi_verify", "pypi_verify")
PROTECTED_JOBS = ("testpypi_publish", "pypi_publish", "github_release")
PROTECTED_ENVIRONMENTS = ("testpypi", "pypi", "github-release")

IDENTITY_STEP = "Require public identity after approval"
FINAL_IDENTITY_STEP = "Require public identity again immediately before publication"
PREFLIGHT_STEP = "Decide the remote registry state"
POLL_STEP = "Poll the index and download both files"
PRECHECK_STEP = "Verify the complete release contract before publishing"
POSTCHECK_STEP = "Verify the complete release contract after publishing"
PUBLISH_STEP = "Publish the verified draft as the final mutation"
RELEASE_STATE_STEP = "Determine the release state"
RELEASE_MUTATING_SUBCOMMANDS = ("create", "edit", "delete", "upload")
CREATE_STEP = "Create the release as a draft when absent"
MANIFEST_STEP = "Record the public-safe promotion manifest and checksums"
SUITE_STEP = "Install the original wheel on the primary runtime and exercise the public surface"

EXACT_JOB_PERMISSIONS: dict[str, dict[str, str] | None] = {
    "workflow_contract": None,
    "build_and_validate": None,
    "testpypi_publish": {"contents": "read", "id-token": "write"},
    "testpypi_verify": None,
    "pypi_publish": {"contents": "read", "id-token": "write"},
    "pypi_verify": None,
    "github_release": {"contents": "write"},
}

HELP_SURFACES = (
    "certify",
    "compare",
    "audit-timestep",
    "review-model",
    "qualify-workload",
    "review-runtime",
    "run-runtime-review",
)

FORBIDDEN_TOKENS = ("0.2.1", "1372", "test_sdist_contains_every_documented_repository_path")

PRIMARY_MUJOCO = "3.12.0"

# The public promotion manifest is constrained by exact positive schemas: only these keys, at every
# level. Anything else would have to be added here deliberately and reviewed.
MANIFEST_TOP_LEVEL = {
    "schema_version",
    "project",
    "version",
    "tag",
    "mode",
    "commit",
    "tree",
    "repository",
    "workflow_sha256",
    "build",
    "suite_runtime",
    "artifacts",
    "outcomes",
}
MANIFEST_BUILD = {
    "backend",
    "backend_version",
    "frontend_version",
    "twine_version",
    "isolated",
    "runner",
}
MANIFEST_SUITE_RUNTIME = {
    "python",
    "pytest",
    "numpy",
    "mujoco",
    "mujoco_native_string",
    "mujoco_native_integer",
}
MANIFEST_ARTIFACT = {"filename", "size", "sha256"}
MANIFEST_OUTCOMES = {
    "observed_test_cases",
    "pip_check",
    "help_surfaces",
    "demo",
    "mjb_characterization",
}

CROSS_JOB_ARTIFACT_PATHS = [
    "dist/",
    "SHA256SUMS.txt",
    "RELEASE_NOTES.md",
    "dist-promotion-manifest.json",
]

ACCEPTED_SKIPS = (
    "tests.integration.test_mujoco_model_identity_native::test_dependencies_contain_both_hfield_pngs",
    "tests.integration.test_mujoco_model_identity_native::test_mujoco_compiles_official_scene",
    "tests.integration.test_mujoco_model_identity_native::test_source_model_bytes_unchanged",
    "tests.integration.test_mujoco_model_identity_native::test_official_scene_compiles_natively_and_is_admitted",
    "tests.integration.test_mujoco_model_identity_native::test_scene_29dof_remains_admitted_and_unaffected",
    "tests.integration.test_mujoco_model_identity_native::test_unrelated_checkout_xml_does_not_affect_the_dependency_result",
    "tests.integration.test_mujoco_model_identity_native::test_existing_scene_29dof_path_still_works",
    "tests.integration.test_mujoco_model_identity_native::test_every_dependency_is_a_contained_measured_member",
    "tests.integration.test_mujoco_model_identity_native::test_same_role_model_pair_identity_for_scene",
    "tests.integration.test_timestep_audit_native::test_bundled_real_policy_audit_reproduces_after_finalization",
)

FIXTURE_VERSION = "9.9.9"
FIXTURE_COMMIT = "a" * 40
FIXTURE_TREE = "b" * 40
FIXTURE_REPOSITORY = "example/metrifid"
FIXTURE_WORKFLOW = b"name: publish\n"
FIXTURE_WORKFLOW_SHA = hashlib.sha256(FIXTURE_WORKFLOW).hexdigest()

RELEASE_ASSET_NAMES = (
    f"metrifid-{FIXTURE_VERSION}-py3-none-any.whl",
    f"metrifid-{FIXTURE_VERSION}.tar.gz",
    "SHA256SUMS.txt",
)

# MuJoCo encodes its native version as major * 1_000_000 + minor * 1_000 + patch.
PRIMARY_MUJOCO_NATIVE = 3_012_000

# Fixture distributions installed on a private path, so an executed validator reports these exact
# versions whatever the host happens to have. Without them the manifest builder cannot run at all
# under the contract job's own environment, which installs only pytest and PyYAML.
STUB_BACKEND_VERSION = "1.99.0"
STUB_FRONTEND_VERSION = "2.99.0"
STUB_TWINE_VERSION = "3.99.0"
STUB_NUMPY_VERSION = "4.99.0"
STUB_PYTEST_VERSION = "5.99.0"

# Deliberately not the interpreter running this file: the manifest must copy the recorded suite
# runtime rather than describe the runner that builds the manifest.
FIXTURE_SUITE_RUNTIME = {
    "python": "3.12.99",
    "pytest": STUB_PYTEST_VERSION,
    "numpy": STUB_NUMPY_VERSION,
    "mujoco": PRIMARY_MUJOCO,
    "mujoco_native_string": PRIMARY_MUJOCO,
    "mujoco_native_integer": PRIMARY_MUJOCO_NATIVE,
}

# Replaces every network call inside an extracted validator with a scripted sequence, so the real
# decision logic runs offline. The workflow itself carries no test-only branch.
NETWORK_STUB = """
import base64 as _base64
import json as _json
import os as _os
import time as _time
import urllib.error as _urlerror
import urllib.request as _urlrequest

_PLAN = _json.loads(_os.environ["STUB_PLAN"])
_STATE = {"call": 0, "clock": 0.0}


class _Response:
    def __init__(self, body):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _urlopen(url, timeout=None):
    index = min(_STATE["call"], len(_PLAN) - 1)
    entry = _PLAN[index]
    _STATE["call"] += 1
    kind = entry["kind"]
    if kind == "http_error":
        raise _urlerror.HTTPError(str(url), entry["code"], "stub", None, None)
    if kind == "url_error":
        raise _urlerror.URLError("stub network failure")
    if kind == "json":
        return _Response(_json.dumps(entry["payload"]).encode("utf-8"))
    if kind == "raw":
        return _Response(entry["text"].encode("utf-8"))
    if kind == "b64":
        return _Response(_base64.b64decode(entry["b64"]))
    raise AssertionError(kind)


def _monotonic():
    _STATE["clock"] += float(_os.environ.get("STUB_CLOCK_STEP", "0"))
    return _STATE["clock"]


_urlrequest.urlopen = _urlopen
_time.sleep = lambda seconds: None
_time.monotonic = _monotonic
"""


# ---- Reading the workflow ------------------------------------------------------------------------


def _text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _document() -> dict[Any, Any]:
    loaded = yaml.safe_load(_text())
    assert isinstance(loaded, dict)
    return loaded


def _triggers(document: dict[Any, Any]) -> dict[str, Any]:
    """Return the `on:` mapping. PyYAML resolves the bare key `on` to the boolean ``True``."""
    triggers = document.get("on", document.get(True))
    assert isinstance(triggers, dict), "the workflow must declare a mapping of triggers"
    return triggers


def _jobs(document: dict[Any, Any]) -> dict[str, dict[str, Any]]:
    jobs = document["jobs"]
    assert isinstance(jobs, dict)
    return jobs


def _steps(job: dict[str, Any]) -> list[dict[str, Any]]:
    steps = job.get("steps", [])
    assert isinstance(steps, list)
    return [step for step in steps if isinstance(step, dict)]


def _run_scripts(job: dict[str, Any]) -> list[str]:
    return [step["run"] for step in _steps(job) if isinstance(step.get("run"), str)]


def _job_text(job: dict[str, Any]) -> str:
    return "\n".join(_run_scripts(job))


def _logical_lines(script: str) -> list[str]:
    """Join backslash-continued shell lines so a whole command is inspected as one string."""
    joined: list[str] = []
    buffer = ""
    for raw in script.splitlines():
        stripped = raw.rstrip()
        if stripped.endswith("\\"):
            buffer += stripped[:-1] + " "
            continue
        joined.append(buffer + stripped)
        buffer = ""
    if buffer:
        joined.append(buffer)
    return joined


def _step_named(job: dict[str, Any], name: str) -> dict[str, Any]:
    for step in _steps(job):
        if step.get("name") == name:
            return step
    raise AssertionError(f"no step named {name!r}")


def _step_index(job: dict[str, Any], name: str) -> int:
    names = [step.get("name") for step in _steps(job)]
    assert name in names, f"no step named {name!r}"
    return names.index(name)


def _inline_python(job: dict[str, Any], step_name: str) -> str:
    """Extract the embedded Python validator from one named step so it can be executed."""
    script = _step_named(job, step_name)["run"]
    assert isinstance(script, str)
    lines = script.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip().endswith("<<'PY'")]
    assert len(starts) == 1, f"{step_name} must embed exactly one validator, found {len(starts)}"
    start = starts[0]
    ends = [i for i in range(start + 1, len(lines)) if lines[i].strip() == "PY"]
    assert ends, f"{step_name} validator is not terminated"
    body = lines[start + 1 : ends[0]]
    indent = min(len(line) - len(line.lstrip()) for line in body if line.strip())
    return "\n".join(line[indent:] if line.strip() else "" for line in body) + "\n"


def _run_validator(
    code: str, workdir: Path, environment: dict[str, str], *, prelude: str = ""
) -> subprocess.CompletedProcess[str]:
    merged = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"}
    merged.update(environment)
    source = f"import os\n{prelude}\n{code}" if prelude else code
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=workdir,
        env=merged,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _needs(job: dict[str, Any]) -> set[str]:
    needs = job.get("needs", [])
    if isinstance(needs, str):
        return {needs}
    assert isinstance(needs, list)
    return set(needs)


def _external_job_ids(jobs: dict[str, dict[str, Any]]) -> set[str]:
    """Every job that can contact TestPyPI, PyPI or the GitHub Release surface."""
    external: set[str] = set()
    for job_id, job in jobs.items():
        if job.get("environment"):
            external.add(job_id)
            continue
        body = _job_text(job)
        uses = " ".join(str(step.get("uses", "")) for step in _steps(job))
        if "gh-action-pypi-publish" in uses:
            external.add(job_id)
        elif any(token in body for token in ("pypi.org", "gh release")):
            external.add(job_id)
    return external


@pytest.fixture(scope="module")
def document() -> dict[Any, Any]:
    """The parsed publication workflow."""
    return _document()


@pytest.fixture(scope="module")
def jobs(document: dict[Any, Any]) -> dict[str, dict[str, Any]]:
    """The workflow's job mapping."""
    return _jobs(document)


# ---- Nothing obsolete survives ---------------------------------------------------------------------


@pytest.mark.parametrize("token", FORBIDDEN_TOKENS)
def test_no_obsolete_pin_total_or_removed_test_survives(token: str) -> None:
    """A pinned old version, a stale total, or a deleted test name would silently misgate a run."""
    assert token not in _text(), token


# ---- Manual modes and exact gating ------------------------------------------------------------------


def test_the_only_trigger_is_manual_dispatch(document: dict[Any, Any]) -> None:
    """A tag push must not be able to start a publication."""
    assert set(_triggers(document)) == {"workflow_dispatch"}


def test_exactly_two_modes_with_the_safe_one_as_default(document: dict[Any, Any]) -> None:
    """The default must be the mode that cannot reach a registry."""
    mode = _triggers(document)["workflow_dispatch"]["inputs"]["mode"]
    assert mode["type"] == "choice"
    assert list(mode["options"]) == [VALIDATE_ONLY, STAGED_RELEASE]
    assert mode["default"] == VALIDATE_ONLY


def test_external_job_conditions_are_exact_not_merely_containing(jobs: dict[str, Any]) -> None:
    """`always() || …` still contains the safe text but runs in every mode, so equality is required."""
    external = _external_job_ids(jobs)
    assert external, "the workflow must declare at least one external job"
    for job_id in external:
        assert jobs[job_id].get("if") == EXACT_EXTERNAL_CONDITION, job_id


def test_no_external_job_uses_a_status_function_bypass(jobs: dict[str, Any]) -> None:
    """`always()`, `success()` and boolean short circuits must not appear in an external gate."""
    for job_id in _external_job_ids(jobs):
        condition = str(jobs[job_id].get("if", ""))
        for bypass in ("always(", "success(", "failure(", "cancelled(", "||"):
            assert bypass not in condition, (job_id, bypass, condition)


def test_the_header_does_not_claim_the_workflow_is_offline() -> None:
    """Validate-only mutates nothing external, but installing dependencies contacts indexes."""
    header = _text()[: _text().index("on:")]
    assert "mutates no external state" in header
    assert "contacts package indexes" in header


# ---- Least privilege ---------------------------------------------------------------------------------


def test_root_permissions_are_read_only(document: dict[Any, Any]) -> None:
    """The workflow starts with no write scope at all."""
    assert document["permissions"] == {"contents": "read"}


@pytest.mark.parametrize("job_id", sorted(EXACT_JOB_PERMISSIONS))
def test_every_job_declares_its_exact_least_privilege_map(
    jobs: dict[str, Any], job_id: str
) -> None:
    """A job-level map replaces the workflow default, so an omitted scope becomes none.

    The registry jobs call GitHub commit, ref and tag APIs after approval, so they need
    `contents: read` alongside `id-token: write`; neither may hold `contents: write`.
    """
    assert jobs[job_id].get("permissions") == EXACT_JOB_PERMISSIONS[job_id], job_id


def test_registry_jobs_never_hold_release_write(jobs: dict[str, Any]) -> None:
    """Registry upload privileges never carry the scope that can mutate a release."""
    for job_id in REGISTRY_PUBLISH_JOBS:
        assert (jobs[job_id]["permissions"]).get("contents") != "write", job_id


def test_every_third_party_action_is_pinned_to_a_full_commit_sha(jobs: dict[str, Any]) -> None:
    """A moving tag is a supply-chain hole in a job that touches release artifacts."""
    for job_id, job in jobs.items():
        for step in _steps(job):
            uses = step.get("uses")
            if not isinstance(uses, str):
                continue
            assert re.fullmatch(r"[^@]+@[0-9a-f]{40}", uses.split(" #")[0].strip()), (job_id, uses)


def test_three_distinct_protected_environments_are_used(jobs: dict[str, Any]) -> None:
    """Each external transition passes through its own protected environment."""
    observed = []
    for job in jobs.values():
        environment = job.get("environment")
        if isinstance(environment, dict):
            observed.append(environment["name"])
        elif isinstance(environment, str):
            observed.append(environment)
    assert sorted(observed) == sorted(PROTECTED_ENVIRONMENTS), observed


# ---- Step order around every mutation ------------------------------------------------------------------


@pytest.mark.parametrize("job_id", REGISTRY_PUBLISH_JOBS)
def test_identity_is_the_last_run_step_before_a_registry_upload(
    jobs: dict[str, Any], job_id: str
) -> None:
    """Preflight, then complete identity, then the upload action with nothing in between."""
    job = jobs[job_id]
    steps = _steps(job)
    preflight = _step_index(job, PREFLIGHT_STEP)
    identity = _step_index(job, IDENTITY_STEP)
    upload = next(
        index
        for index, step in enumerate(steps)
        if "gh-action-pypi-publish" in str(step.get("uses", ""))
    )
    assert preflight < identity < upload, (preflight, identity, upload)
    assert upload - identity == 1, "no run step may sit between identity and the upload"
    assert steps[upload].get("if") == EXACT_UPLOAD_CONDITION, job_id
    assert steps[identity].get("if") is None, "identity must run even on an idempotent remote state"


def test_the_release_job_checks_identity_before_and_immediately_before_publication(
    jobs: dict[str, Any],
) -> None:
    """The public draft is created after one check and published immediately after another."""
    job = jobs["github_release"]
    order = [
        _step_index(job, IDENTITY_STEP),
        _step_index(job, RELEASE_STATE_STEP),
        _step_index(job, CREATE_STEP),
        _step_index(job, PRECHECK_STEP),
        _step_index(job, FINAL_IDENTITY_STEP),
        _step_index(job, PUBLISH_STEP),
        _step_index(job, POSTCHECK_STEP),
    ]
    assert order == sorted(order), order
    assert order[5] - order[4] == 1, "identity must be the run step immediately before publication"


def test_no_unprotected_job_claims_release_identity_authority(jobs: dict[str, Any]) -> None:
    """Only jobs behind a protected environment may assert public identity."""
    for job_id, job in jobs.items():
        if job_id in PROTECTED_JOBS:
            continue
        assert "git/ref/tags/" not in _job_text(job), job_id


# ---- Executable: every protected identity copy -------------------------------------------------------


def _identity_sources(jobs: dict[str, Any]) -> list[tuple[str, str]]:
    sources = [(job_id, _inline_python(jobs[job_id], IDENTITY_STEP)) for job_id in PROTECTED_JOBS]
    sources.append(
        ("github_release_final", _inline_python(jobs["github_release"], FINAL_IDENTITY_STEP))
    )
    return sources


def _promotion_fixture(
    root: Path,
    *,
    tree: str = FIXTURE_TREE,
    workflow_sha: str = FIXTURE_WORKFLOW_SHA,
    digest_override: str | None = None,
    sums_lines: int = 2,
) -> None:
    artifacts = root / "release-artifacts"
    dist = artifacts / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    names = (f"metrifid-{FIXTURE_VERSION}-py3-none-any.whl", f"metrifid-{FIXTURE_VERSION}.tar.gz")
    rows = []
    sums = []
    for index, name in enumerate(names):
        payload = f"payload-{index}".encode()
        (dist / name).write_bytes(payload)
        digest = hashlib.sha256(payload).hexdigest()
        rows.append({"filename": name, "size": len(payload), "sha256": digest_override or digest})
        sums.append(f"{digest}  {name}")
    (artifacts / "SHA256SUMS.txt").write_text("\n".join(sums[:sums_lines]) + "\n", encoding="utf-8")
    (artifacts / "RELEASE_NOTES.md").write_text("## 9.9.9\n\nnotes\n", encoding="utf-8")
    (artifacts / "dist-promotion-manifest.json").write_text(
        json.dumps(
            {
                "commit": FIXTURE_COMMIT,
                "version": FIXTURE_VERSION,
                "tag": FIXTURE_VERSION,
                "repository": FIXTURE_REPOSITORY,
                "tree": tree,
                "workflow_sha256": workflow_sha,
                "artifacts": rows,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _identity_environment(**overrides: str) -> dict[str, str]:
    import base64

    environment = {
        "RELEASE_VERSION": FIXTURE_VERSION,
        "PLANNED_TAG": FIXTURE_VERSION,
        "EXPECTED_COMMIT": FIXTURE_COMMIT,
        "GITHUB_SHA": FIXTURE_COMMIT,
        "GITHUB_REPOSITORY": FIXTURE_REPOSITORY,
        "PUBLIC_MAIN_HEAD": FIXTURE_COMMIT,
        "REMOTE_TREE": FIXTURE_TREE,
        "REMOTE_WORKFLOW_B64": base64.b64encode(FIXTURE_WORKFLOW).decode("ascii"),
        "REMOTE_WORKFLOW_ENCODING": "base64",
        "TAG_OBJECT_TYPE": "tag",
        "TAG_PEELED": FIXTURE_COMMIT,
        "TAG_PEELED_TYPE": "commit",
    }
    environment.update(overrides)
    return environment


def test_every_identity_copy_accepts_a_completely_consistent_state(
    jobs: dict[str, Any], tmp_path: Path
) -> None:
    """Every protected copy must agree that a fully consistent state is acceptable."""
    for label, code in _identity_sources(jobs):
        workdir = tmp_path / f"green-{label}"
        workdir.mkdir()
        _promotion_fixture(workdir)
        done = _run_validator(code, workdir, _identity_environment())
        assert done.returncode == 0, (label, done.stdout + done.stderr)


IDENTITY_RED_CASES = [
    pytest.param({"EXPECTED_COMMIT": "abc123"}, {}, id="expected_not_full_sha"),
    pytest.param({"GITHUB_SHA": "d" * 40}, {}, id="expected_not_run_commit"),
    pytest.param({"PUBLIC_MAIN_HEAD": "e" * 40}, {}, id="public_main_is_other"),
    pytest.param({"TAG_OBJECT_TYPE": "commit"}, {}, id="lightweight_tag"),
    pytest.param({"TAG_PEELED": "f" * 40}, {}, id="tag_peels_to_other"),
    pytest.param({"TAG_PEELED_TYPE": "tree"}, {}, id="tag_peels_to_non_commit"),
    pytest.param({"RELEASE_VERSION": "8.8.8"}, {}, id="manifest_version_mismatch"),
    pytest.param({"PLANNED_TAG": "8.8.8"}, {}, id="manifest_tag_mismatch"),
    pytest.param({"GITHUB_REPOSITORY": "other/repo"}, {}, id="repository_mismatch"),
    pytest.param({"REMOTE_TREE": "c" * 40}, {}, id="public_tree_disagrees"),
    pytest.param({"REMOTE_TREE": "not-an-oid"}, {}, id="public_tree_malformed"),
    pytest.param({"REMOTE_WORKFLOW_B64": ""}, {}, id="public_workflow_empty"),
    pytest.param({}, {"tree": "9" * 40}, id="manifest_tree_wrong_but_nonempty"),
    pytest.param({}, {"workflow_sha": "9" * 64}, id="manifest_workflow_wrong_but_nonempty"),
    pytest.param({}, {"digest_override": "0" * 64}, id="stored_hash_tampered"),
    pytest.param({}, {"sums_lines": 1}, id="checksum_file_truncated"),
]


@pytest.mark.parametrize(("overrides", "fixture_kwargs"), IDENTITY_RED_CASES)
def test_every_identity_copy_fails_closed_on_any_broken_binding(
    jobs: dict[str, Any], tmp_path: Path, overrides: dict[str, str], fixture_kwargs: dict[str, Any]
) -> None:
    """Each protected copy is executed; one broken binding is enough to refuse the mutation."""
    for label, code in _identity_sources(jobs):
        workdir = tmp_path / f"red-{label}"
        workdir.mkdir()
        _promotion_fixture(workdir, **fixture_kwargs)
        done = _run_validator(code, workdir, _identity_environment(**overrides))
        assert done.returncode != 0, (label, done.stdout)


# ---- Executable: both registry preflight copies -------------------------------------------------------


def _index_payload(root: Path, *, mutate: str = "exact") -> dict[str, Any]:
    manifest = json.loads(
        (root / "release-artifacts/dist-promotion-manifest.json").read_text(encoding="utf-8")
    )
    rows = [
        {
            "filename": row["filename"],
            "digests": {"sha256": row["sha256"]},
            "url": f"https://x/{row['filename']}",
        }
        for row in manifest["artifacts"]
    ]
    if mutate == "partial":
        rows = rows[:1]
    elif mutate == "extra":
        rows = [
            *rows,
            {"filename": "other.whl", "digests": {"sha256": "0" * 64}, "url": "https://x/o"},
        ]
    elif mutate == "wrong_hash":
        rows[0]["digests"]["sha256"] = "0" * 64
    return {"urls": rows}


@pytest.mark.parametrize("job_id", REGISTRY_PUBLISH_JOBS)
def test_preflight_allows_upload_only_on_explicit_not_found(
    jobs: dict[str, Any], tmp_path: Path, job_id: str
) -> None:
    """A 404 is the only state that permits an upload."""
    code = _inline_python(jobs[job_id], PREFLIGHT_STEP)
    workdir = tmp_path / f"nf-{job_id}"
    workdir.mkdir()
    _promotion_fixture(workdir)
    output = workdir / "gh_output"
    done = _run_validator(
        code,
        workdir,
        {
            "INDEX_JSON_URL": "https://stub/index",
            "GITHUB_OUTPUT": str(output),
            "STUB_PLAN": json.dumps([{"kind": "http_error", "code": 404}]),
        },
        prelude=NETWORK_STUB,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "upload=true" in output.read_text(encoding="utf-8")


@pytest.mark.parametrize("job_id", REGISTRY_PUBLISH_JOBS)
def test_preflight_is_idempotent_on_exact_existing_state(
    jobs: dict[str, Any], tmp_path: Path, job_id: str
) -> None:
    """An exact match uploads nothing and still succeeds, so a rerun is safe."""
    code = _inline_python(jobs[job_id], PREFLIGHT_STEP)
    workdir = tmp_path / f"exact-{job_id}"
    workdir.mkdir()
    _promotion_fixture(workdir)
    output = workdir / "gh_output"
    done = _run_validator(
        code,
        workdir,
        {
            "INDEX_JSON_URL": "https://stub/index",
            "GITHUB_OUTPUT": str(output),
            "STUB_PLAN": json.dumps([{"kind": "json", "payload": _index_payload(workdir)}]),
        },
        prelude=NETWORK_STUB,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert "upload=false" in output.read_text(encoding="utf-8")


@pytest.mark.parametrize("job_id", REGISTRY_PUBLISH_JOBS)
@pytest.mark.parametrize(
    "plan_kind",
    ["partial", "extra", "wrong_hash", "malformed", "http_500", "network_error"],
)
def test_preflight_fails_closed_on_every_other_state(
    jobs: dict[str, Any], tmp_path: Path, job_id: str, plan_kind: str
) -> None:
    """Incomplete, different, malformed or failed lookups never become an upload."""
    code = _inline_python(jobs[job_id], PREFLIGHT_STEP)
    workdir = tmp_path / f"bad-{job_id}-{plan_kind}"
    workdir.mkdir()
    _promotion_fixture(workdir)
    plan: list[dict[str, Any]]
    if plan_kind == "malformed":
        plan = [{"kind": "json", "payload": {"urls": "not-a-list"}}]
    elif plan_kind == "http_500":
        plan = [{"kind": "http_error", "code": 500}]
    elif plan_kind == "network_error":
        plan = [{"kind": "url_error"}]
    else:
        plan = [{"kind": "json", "payload": _index_payload(workdir, mutate=plan_kind)}]
    done = _run_validator(
        code,
        workdir,
        {
            "INDEX_JSON_URL": "https://stub/index",
            "GITHUB_OUTPUT": str(workdir / "gh_output"),
            "STUB_PLAN": json.dumps(plan),
        },
        prelude=NETWORK_STUB,
    )
    assert done.returncode != 0, (job_id, plan_kind, done.stdout)


# ---- Executable: both poll copies ------------------------------------------------------------------------


@pytest.mark.parametrize("job_id", REGISTRY_VERIFY_JOBS)
def test_poll_accepts_matching_remote_bytes(
    jobs: dict[str, Any], tmp_path: Path, job_id: str
) -> None:
    """The index matches and both downloads are byte-identical to the originals."""
    code = _inline_python(jobs[job_id], POLL_STEP)
    workdir = tmp_path / f"poll-{job_id}"
    workdir.mkdir()
    _promotion_fixture(workdir)
    plan = [{"kind": "json", "payload": _index_payload(workdir)}]
    for index in range(2):
        plan.append({"kind": "raw", "text": f"payload-{index}"})
    done = _run_validator(
        code,
        workdir,
        {"INDEX_JSON_URL": "https://stub/index", "STUB_PLAN": json.dumps(plan)},
        prelude=NETWORK_STUB,
    )
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.parametrize("job_id", REGISTRY_VERIFY_JOBS)
def test_poll_reports_the_last_state_on_timeout(
    jobs: dict[str, Any], tmp_path: Path, job_id: str
) -> None:
    """A bounded wait ends with the last observed state, not a silent pass."""
    code = _inline_python(jobs[job_id], POLL_STEP)
    workdir = tmp_path / f"timeout-{job_id}"
    workdir.mkdir()
    _promotion_fixture(workdir)
    done = _run_validator(
        code,
        workdir,
        {
            "INDEX_JSON_URL": "https://stub/index",
            "STUB_PLAN": json.dumps([{"kind": "http_error", "code": 404}]),
            "STUB_CLOCK_STEP": "1000",
        },
        prelude=NETWORK_STUB,
    )
    assert done.returncode != 0
    assert "timed out waiting for the index" in done.stdout + done.stderr
    assert "404" in done.stdout + done.stderr


@pytest.mark.parametrize("job_id", REGISTRY_VERIFY_JOBS)
def test_poll_fails_closed_on_an_unexpected_error(
    jobs: dict[str, Any], tmp_path: Path, job_id: str
) -> None:
    """A 500 is not absence; it must stop immediately rather than keep polling."""
    code = _inline_python(jobs[job_id], POLL_STEP)
    workdir = tmp_path / f"poll500-{job_id}"
    workdir.mkdir()
    _promotion_fixture(workdir)
    done = _run_validator(
        code,
        workdir,
        {
            "INDEX_JSON_URL": "https://stub/index",
            "STUB_PLAN": json.dumps([{"kind": "http_error", "code": 500}]),
            "STUB_CLOCK_STEP": "0",
        },
        prelude=NETWORK_STUB,
    )
    assert done.returncode != 0
    assert "timed out" not in done.stdout + done.stderr


@pytest.mark.parametrize("job_id", REGISTRY_VERIFY_JOBS)
def test_poll_refuses_a_download_that_is_not_byte_identical(
    jobs: dict[str, Any], tmp_path: Path, job_id: str
) -> None:
    """A matching index digest is not enough; the bytes themselves are compared."""
    code = _inline_python(jobs[job_id], POLL_STEP)
    workdir = tmp_path / f"polldiff-{job_id}"
    workdir.mkdir()
    _promotion_fixture(workdir)
    plan = [
        {"kind": "json", "payload": _index_payload(workdir)},
        {"kind": "raw", "text": "tampered"},
    ]
    done = _run_validator(
        code,
        workdir,
        {"INDEX_JSON_URL": "https://stub/index", "STUB_PLAN": json.dumps(plan)},
        prelude=NETWORK_STUB,
    )
    assert done.returncode != 0, done.stdout


# ---- Executable: the complete release contract, before and after publication ---------------------------------


def _release_fixture(
    root: Path,
    *,
    is_draft: bool,
    byte_only_drift: str | None = None,
    hash_only_drift: str | None = None,
    assets: tuple[str, ...] | None = None,
    downloaded: tuple[str, ...] | None = None,
    title: str | None = None,
    body: str | None = None,
    prerelease: bool = False,
    tamper_sums: bool = False,
) -> None:
    _promotion_fixture(root)
    artifacts = root / "release-artifacts"
    state = root / "state"
    state.mkdir(parents=True, exist_ok=True)
    allowlist = (
        f"metrifid-{FIXTURE_VERSION}-py3-none-any.whl",
        f"metrifid-{FIXTURE_VERSION}.tar.gz",
        "SHA256SUMS.txt",
    )
    for name in downloaded if downloaded is not None else allowlist:
        source = artifacts / "dist" / name
        if source.is_file():
            (state / name).write_bytes(source.read_bytes())
        elif name == "SHA256SUMS.txt":
            payload = (artifacts / "SHA256SUMS.txt").read_bytes()
            (state / name).write_bytes(b"tampered\n" if tamper_sums else payload)
        else:
            (state / name).write_bytes(b"extra\n")
    if byte_only_drift is not None or hash_only_drift is not None:
        name = (
            f"metrifid-{FIXTURE_VERSION}-py3-none-any.whl"
            if (byte_only_drift or hash_only_drift) == "wheel"
            else f"metrifid-{FIXTURE_VERSION}.tar.gz"
        )
        replacement = b"drifted payload"
        manifest_path = artifacts / "dist-promotion-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if byte_only_drift is not None:
            # The published bytes differ from the stored original, yet every recorded digest agrees
            # with them, so only the byte comparison can notice.
            (state / name).write_bytes(replacement)
            digest = hashlib.sha256(replacement).hexdigest()
            for row in manifest["artifacts"]:
                if row["filename"] == name:
                    row["sha256"] = digest
                    row["size"] = len(replacement)
            sums = []
            for row in manifest["artifacts"]:
                sums.append(f"{row['sha256']}  {row['filename']}")
            (artifacts / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
            (state / "SHA256SUMS.txt").write_bytes((artifacts / "SHA256SUMS.txt").read_bytes())
        else:
            # The published bytes equal the stored original, but neither matches the accepted
            # digest, so only the hash comparison can notice.
            (state / name).write_bytes(replacement)
            (artifacts / "dist" / name).write_bytes(replacement)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    (state / "release.json").write_text(
        json.dumps(
            {
                "tagName": FIXTURE_VERSION,
                "isDraft": is_draft,
                "isPrerelease": prerelease,
                "name": title if title is not None else f"metrifid {FIXTURE_VERSION}",
                "body": body if body is not None else "## 9.9.9\n\nnotes",
                "assets": [{"name": name} for name in (assets or allowlist)],
            }
        ),
        encoding="utf-8",
    )


def _release_sources(jobs: dict[str, Any]) -> list[tuple[str, str, str]]:
    job = jobs["github_release"]
    return [
        ("precheck", _inline_python(job, PRECHECK_STEP), "true"),
        ("postcheck", _inline_python(job, POSTCHECK_STEP), "false"),
    ]


def test_the_same_release_contract_runs_before_and_after_publication(
    jobs: dict[str, Any],
) -> None:
    """The two checks differ only in the draft state they expect."""
    job = jobs["github_release"]
    precheck = _inline_python(job, PRECHECK_STEP)
    postcheck = _inline_python(job, POSTCHECK_STEP)
    assert precheck == postcheck, "the release contract must be identical before and after"


def test_each_release_contract_copy_accepts_its_correct_state(
    jobs: dict[str, Any], tmp_path: Path
) -> None:
    """A correct draft passes the precheck; a correct published release passes the postcheck."""
    for label, code, expect_draft in _release_sources(jobs):
        workdir = tmp_path / f"rel-green-{label}"
        workdir.mkdir()
        _release_fixture(workdir, is_draft=expect_draft == "true")
        done = _run_validator(
            code,
            workdir,
            _identity_environment(RELEASE_STATE=str(workdir / "state"), EXPECT_DRAFT=expect_draft),
        )
        assert done.returncode == 0, (label, done.stdout + done.stderr)


RELEASE_RED_CASES = [
    pytest.param({"body": "different notes"}, id="wrong_body"),
    pytest.param({"prerelease": True}, id="prerelease_true"),
    pytest.param({"title": "metrifid"}, id="wrong_title"),
    pytest.param({"tamper_sums": True}, id="tampered_checksums"),
    pytest.param(
        {"assets": (f"metrifid-{FIXTURE_VERSION}-py3-none-any.whl", "SHA256SUMS.txt")},
        id="missing_remote_asset",
    ),
    pytest.param(
        {
            "assets": (
                f"metrifid-{FIXTURE_VERSION}-py3-none-any.whl",
                f"metrifid-{FIXTURE_VERSION}.tar.gz",
                "SHA256SUMS.txt",
                "release-junit.xml",
            )
        },
        id="fourth_remote_asset",
    ),
    pytest.param(
        {
            "downloaded": (
                f"metrifid-{FIXTURE_VERSION}-py3-none-any.whl",
                f"metrifid-{FIXTURE_VERSION}.tar.gz",
                "SHA256SUMS.txt",
                "extra.txt",
            )
        },
        id="extra_downloaded_file",
    ),
]


@pytest.mark.parametrize("kwargs", RELEASE_RED_CASES)
def test_each_release_contract_copy_refuses_a_broken_state(
    jobs: dict[str, Any], tmp_path: Path, kwargs: dict[str, Any]
) -> None:
    """Both copies must refuse; proving only the prepublication copy proves nothing about the other."""
    for label, code, expect_draft in _release_sources(jobs):
        workdir = tmp_path / f"rel-red-{label}-{abs(hash(tuple(sorted(kwargs))))}"
        workdir.mkdir(parents=True, exist_ok=True)
        _release_fixture(workdir, is_draft=expect_draft == "true", **kwargs)
        done = _run_validator(
            code,
            workdir,
            _identity_environment(RELEASE_STATE=str(workdir / "state"), EXPECT_DRAFT=expect_draft),
        )
        assert done.returncode != 0, (label, done.stdout)


@pytest.mark.parametrize("label_index", [0, 1])
def test_each_release_contract_copy_refuses_the_wrong_draft_state(
    jobs: dict[str, Any], tmp_path: Path, label_index: int
) -> None:
    """A published release must not satisfy the draft check, and the reverse."""
    label, code, expect_draft = _release_sources(jobs)[label_index]
    workdir = tmp_path / f"rel-draft-{label}"
    workdir.mkdir()
    _release_fixture(workdir, is_draft=expect_draft != "true")
    done = _run_validator(
        code,
        workdir,
        _identity_environment(RELEASE_STATE=str(workdir / "state"), EXPECT_DRAFT=expect_draft),
    )
    assert done.returncode != 0, (label, done.stdout)


def test_no_release_step_clobbers_or_uploads_onto_existing_state(jobs: dict[str, Any]) -> None:
    """`--clobber` deletes existing assets and leaves unrelated ones behind."""
    body = _job_text(jobs["github_release"])
    assert "--clobber" not in body
    assert "gh release upload" not in body
    assert "gh release delete" not in body


# ---- Executable: the wheel archive and source correspondence ---------------------------------------------------


def _wheel_fixture(root: Path, members: dict[str, bytes]) -> None:
    dist = root / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    distinfo = f"metrifid-{FIXTURE_VERSION}.dist-info"
    with zipfile.ZipFile(dist / f"metrifid-{FIXTURE_VERSION}-py3-none-any.whl", "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
        archive.writestr(f"{distinfo}/METADATA", "Metadata-Version: 2.4\n")


def _source_fixture(root: Path, *, forced: str | None = "tools/worker.py") -> None:
    package = root / "src/metrifid"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_bytes(b"x = 1\n")
    (root / "tools").mkdir(parents=True, exist_ok=True)
    (root / "tools/worker.py").write_bytes(b"worker = True\n")
    include = ""
    if forced is not None:
        include = f'\n[tool.hatch.build.targets.wheel.force-include]\n"{forced}" = "metrifid/worker.py.txt"\n'
    (root / "pyproject.toml").write_text(
        '[tool.hatch.build.targets.wheel]\npackages = ["src/metrifid"]\n' + include,
        encoding="utf-8",
    )


@pytest.fixture(scope="module")
def force_include_validator(jobs: dict[str, Any]) -> str:
    """The embedded wheel/source correspondence validator, as executable source."""
    return _inline_python(
        jobs["build_and_validate"], "Require wheel package bytes to equal declared source"
    )


def test_force_include_validator_accepts_a_declared_forced_member(
    force_include_validator: str, tmp_path: Path
) -> None:
    """A declared force-include is part of the payload, not drift."""
    _source_fixture(tmp_path)
    _wheel_fixture(
        tmp_path, {"metrifid/__init__.py": b"x = 1\n", "metrifid/worker.py.txt": b"worker = True\n"}
    )
    done = _run_validator(force_include_validator, tmp_path, {"RELEASE_VERSION": FIXTURE_VERSION})
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.parametrize(
    ("members", "forced"),
    [
        pytest.param({"metrifid/__init__.py": b"x = 1\n"}, "tools/worker.py", id="missing"),
        pytest.param(
            {"metrifid/__init__.py": b"x = 1\n", "metrifid/worker.py.txt": b"tampered\n"},
            "tools/worker.py",
            id="wrong_bytes",
        ),
        pytest.param(
            {"metrifid/__init__.py": b"x = 1\n", "metrifid/worker.py.txt": b"worker = True\n"},
            None,
            id="undeclared",
        ),
        pytest.param(
            {"metrifid/__init__.py": b"x = 1\n", "metrifid/worker.py.txt": b"worker = True\n"},
            "tools/absent.py",
            id="missing_source",
        ),
    ],
)
def test_force_include_validator_refuses_every_mismatch(
    force_include_validator: str, tmp_path: Path, members: dict[str, bytes], forced: str | None
) -> None:
    """Missing, tampered, undeclared or unbacked forced members are all refused."""
    _source_fixture(tmp_path, forced=forced)
    _wheel_fixture(tmp_path, members)
    done = _run_validator(force_include_validator, tmp_path, {"RELEASE_VERSION": FIXTURE_VERSION})
    assert done.returncode != 0, done.stdout


@pytest.fixture(scope="module")
def archive_validator(jobs: dict[str, Any]) -> str:
    """The embedded wheel archive validator, as executable source."""
    return _inline_python(
        jobs["build_and_validate"], "Require complete, canonically named, safe wheel contents"
    )


def _record_wheel(root: Path, members: dict[str, bytes], licenses: bool = True) -> None:
    import base64
    import csv
    import io

    dist = root / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    distinfo = f"metrifid-{FIXTURE_VERSION}.dist-info"
    payloads = dict(members)
    payloads[f"{distinfo}/METADATA"] = (
        f"Metadata-Version: 2.4\nVersion: {FIXTURE_VERSION}\nLicense-Expression: Apache-2.0\n".encode()
    )
    payloads[f"{distinfo}/entry_points.txt"] = b"[console_scripts]\nmetrifid = metrifid.cli:main\n"
    if licenses:
        for stem in ("LICENSE", "NOTICE"):
            (root / stem).write_bytes(f"{stem} text\n".encode())
            payloads[f"{distinfo}/licenses/{stem}"] = f"{stem} text\n".encode()
    payloads["metrifid/py.typed"] = b""
    rows = io.StringIO()
    writer = csv.writer(rows, lineterminator="\n")
    for name, blob in payloads.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(blob).digest()).rstrip(b"=").decode()
        writer.writerow([name, f"sha256={digest}", len(blob)])
    writer.writerow([f"{distinfo}/RECORD", "", ""])
    payloads[f"{distinfo}/RECORD"] = rows.getvalue().encode()
    with zipfile.ZipFile(dist / f"metrifid-{FIXTURE_VERSION}-py3-none-any.whl", "w") as archive:
        for name, blob in payloads.items():
            archive.writestr(name, blob)


def test_archive_validator_accepts_a_canonical_wheel(
    archive_validator: str, tmp_path: Path
) -> None:
    """A wheel whose every member is already canonically named passes."""
    _record_wheel(tmp_path, {"metrifid/__init__.py": b"x = 1\n"})
    done = _run_validator(archive_validator, tmp_path, {"RELEASE_VERSION": FIXTURE_VERSION})
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.parametrize(
    "member",
    [
        pytest.param("metrifid/./__init__.py", id="dot_alias"),
        pytest.param("metrifid//__init__.py", id="repeated_separator"),
        pytest.param("./metrifid/__init__.py", id="leading_alias"),
        pytest.param("metrifid\\__init__.py", id="backslash"),
        pytest.param("/metrifid/__init__.py", id="absolute"),
        pytest.param("metrifid/../metrifid/__init__.py", id="traversal"),
    ],
)
def test_archive_validator_refuses_a_single_noncanonical_member(
    archive_validator: str, tmp_path: Path, member: str
) -> None:
    """A lone alias is refused even when no second member collides with it."""
    _record_wheel(tmp_path, {member: b"x = 1\n"})
    done = _run_validator(archive_validator, tmp_path, {"RELEASE_VERSION": FIXTURE_VERSION})
    assert done.returncode != 0, done.stdout


# ---- Executable: the exact skip set ------------------------------------------------------------------------


def _junit_fixture(path: Path, skipped: tuple[str, ...], *, failure: bool = False) -> None:
    parts = ['<?xml version="1.0" encoding="utf-8"?>', '<testsuite name="release">']
    parts.append('<testcase classname="tests.unit.test_a" name="test_passes"/>')
    for identity in skipped:
        classname, _, name = identity.partition("::")
        parts.append(f'<testcase classname="{classname}" name="{name}"><skipped/></testcase>')
    if failure:
        parts.append(
            '<testcase classname="tests.unit.test_a" name="test_breaks"><failure/></testcase>'
        )
    parts.append("</testsuite>")
    path.write_text("\n".join(parts), encoding="utf-8")


@pytest.fixture(scope="module")
def skip_validator(jobs: dict[str, Any]) -> str:
    """The embedded JUnit skip-set validator, as executable source."""
    return _inline_python(
        jobs["build_and_validate"],
        "Require the exact accepted skip set and nonempty required outputs",
    )


def _skip_environment(workdir: Path, *, mjb: str = "ok\n", demo: str = "ok\n") -> dict[str, str]:
    (workdir / "mjb.log").write_text(mjb, encoding="utf-8")
    (workdir / "demo.log").write_text(demo, encoding="utf-8")
    return {
        "JUNIT_REPORT": str(workdir / "junit.xml"),
        "MJB_LOG": str(workdir / "mjb.log"),
        "DEMO_LOG": str(workdir / "demo.log"),
    }


def test_skip_validator_accepts_exactly_the_accepted_set(
    skip_validator: str, tmp_path: Path
) -> None:
    """The ten accepted native-fixture skips are the whole permitted set."""
    _junit_fixture(tmp_path / "junit.xml", ACCEPTED_SKIPS)
    done = _run_validator(skip_validator, tmp_path, _skip_environment(tmp_path))
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.mark.parametrize(
    ("skipped", "failure"),
    [
        pytest.param((), False, id="zero_skips"),
        pytest.param(ACCEPTED_SKIPS[:-1], False, id="one_removed"),
        pytest.param(
            (*ACCEPTED_SKIPS[:-1], "tests.integration.test_timestep_audit_native::test_renamed"),
            False,
            id="one_renamed",
        ),
        pytest.param(
            (*ACCEPTED_SKIPS, "tests.integration.test_timestep_audit_native::test_extra"),
            False,
            id="one_added_in_allowed_module",
        ),
        pytest.param(ACCEPTED_SKIPS, True, id="failure_present"),
    ],
)
def test_skip_validator_refuses_any_other_set(
    skip_validator: str, tmp_path: Path, skipped: tuple[str, ...], failure: bool
) -> None:
    """Zero, removal, renaming, addition inside an allowed module, and any failure all fail."""
    _junit_fixture(tmp_path / "junit.xml", skipped, failure=failure)
    done = _run_validator(skip_validator, tmp_path, _skip_environment(tmp_path))
    assert done.returncode != 0, done.stdout


@pytest.mark.parametrize("empty", ["mjb", "demo"])
def test_skip_validator_requires_nonempty_characterization_and_demo_output(
    skip_validator: str, tmp_path: Path, empty: str
) -> None:
    """Both required outputs are mandatory; neither check may be skipped or unreachable."""
    _junit_fixture(tmp_path / "junit.xml", ACCEPTED_SKIPS)
    environment = _skip_environment(
        tmp_path, mjb="" if empty == "mjb" else "ok\n", demo="" if empty == "demo" else "ok\n"
    )
    done = _run_validator(skip_validator, tmp_path, environment)
    assert done.returncode != 0, done.stdout


def test_the_required_output_paths_are_actually_passed_to_the_validator(
    jobs: dict[str, Any],
) -> None:
    """An optional branch reading variables the step never receives would be unreachable."""
    step = _step_named(
        jobs["build_and_validate"],
        "Require the exact accepted skip set and nonempty required outputs",
    )
    for name in ("JUNIT_REPORT", "MJB_LOG", "DEMO_LOG"):
        assert name in step["env"], name


# ---- Build provenance and the primary runtime ------------------------------------------------------------------


def test_both_builds_use_the_constrained_backend_without_isolation(jobs: dict[str, Any]) -> None:
    """An isolated backend could be a different Hatchling than the one the manifest reports."""
    body = _job_text(jobs["build_and_validate"])
    lines = _logical_lines(body)
    original = [line for line in lines if "python -m build --sdist --wheel" in line]
    assert len(original) == 1, original
    assert "--no-isolation" in original[0]
    parity = [line for line in lines if "--outdir parity" in line]
    assert len(parity) == 1, parity
    assert "--no-isolation" in parity[0]
    assert "sdist-extract" in parity[0]
    for line in lines:
        assert line.strip() != "python -m build", (
            "a bare build would derive the wheel from the sdist"
        )


def test_the_suite_environment_pins_the_primary_runtime_under_the_constraint(
    jobs: dict[str, Any],
) -> None:
    """The suite must run on the declared primary profile with NumPy resolved under the constraint."""
    step = _step_named(jobs["build_and_validate"], SUITE_STEP)
    lines = _logical_lines(str(step["run"]))
    wheel_install = [
        line for line in lines if "pip install" in line and "metrifid-${RELEASE_VERSION}" in line
    ]
    assert len(wheel_install) == 1, wheel_install
    assert "quality-constraints.txt" in wheel_install[0], "the constraint must bind this install"
    assert "mujoco==${PRIMARY_MUJOCO}" in wheel_install[0], wheel_install[0]
    tools_install = [line for line in lines if "pip install" in line and "pytest-check" in line]
    assert tools_install, "the installed-wheel test environment must supply pytest-check"


def test_the_primary_runtime_is_the_declared_authority(document: dict[Any, Any]) -> None:
    """The workflow pins the primary release profile the project declares."""
    assert document["env"]["PRIMARY_MUJOCO"] == PRIMARY_MUJOCO


def test_runtime_identity_is_read_from_the_suite_interpreter(jobs: dict[str, Any]) -> None:
    """Recording the outer runner's versions would describe the wrong environment."""
    step = _step_named(jobs["build_and_validate"], SUITE_STEP)
    body = str(step["run"])
    assert '"$RELEASE_VENV/bin/python" - <<' in body, "runtime identity must be read in the venv"
    assert "mj_versionString" in body
    assert "mj_version()" in body
    assert "SUITE_RUNTIME" in step["env"]
    manifest_body = _job_text(jobs["build_and_validate"])
    assert 'suite["mujoco"]' in manifest_body, "the manifest must read the recorded suite runtime"


# ---- The public promotion manifest schema ------------------------------------------------------------------------


def _manifest_literal(jobs: dict[str, Any]) -> dict[str, Any]:
    code = _inline_python(jobs["build_and_validate"], MANIFEST_STEP)
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "manifest":
            assert isinstance(node.value, ast.Dict)
            return {"node": node.value}
    raise AssertionError("the manifest literal was not found")


def _dict_keys(node: ast.Dict) -> set[str]:
    keys: set[str] = set()
    for key in node.keys:
        assert isinstance(key, ast.Constant)
        assert isinstance(key.value, str)
        keys.add(key.value)
    return keys


def _nested(node: ast.Dict, name: str) -> ast.Dict:
    for key, value in zip(node.keys, node.values, strict=True):
        if isinstance(key, ast.Constant) and key.value == name:
            assert isinstance(value, ast.Dict), name
            return value
    raise AssertionError(f"{name} is not a nested table")


def test_the_public_manifest_matches_its_exact_schema(jobs: dict[str, Any]) -> None:
    """The manifest travels publicly, so its keys are an exact positive allowlist at every level."""
    node = _manifest_literal(jobs)["node"]
    assert _dict_keys(node) == MANIFEST_TOP_LEVEL
    assert _dict_keys(_nested(node, "build")) == MANIFEST_BUILD
    assert _dict_keys(_nested(node, "suite_runtime")) == MANIFEST_SUITE_RUNTIME
    assert _dict_keys(_nested(node, "outcomes")) == MANIFEST_OUTCOMES


def test_the_manifest_records_no_outcome_that_has_not_happened(jobs: dict[str, Any]) -> None:
    """The registry certify journeys run later, so the build manifest cannot report them."""
    outcomes = _dict_keys(_nested(_manifest_literal(jobs)["node"], "outcomes"))
    assert "certify_journey" not in outcomes


def test_the_manifest_artifact_rows_match_their_exact_schema(jobs: dict[str, Any]) -> None:
    """Each artifact row carries exactly a filename, a size and a digest."""
    code = _inline_python(jobs["build_and_validate"], MANIFEST_STEP)
    tree = ast.parse(code)
    rows = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Dict) and _dict_keys(node) == MANIFEST_ARTIFACT
    ]
    assert len(rows) == 1, "exactly one artifact row shape must be built"


# ---- The public artifact boundary ------------------------------------------------------------------------------------


def test_the_cross_job_artifact_carries_only_release_bound_files(jobs: dict[str, Any]) -> None:
    """Actions artifacts are readable in a public repository, so reports and logs stay on the runner."""
    uploads = [
        step
        for step in _steps(jobs["build_and_validate"])
        if "actions/upload-artifact" in str(step.get("uses", ""))
    ]
    assert len(uploads) == 1
    paths = [line.strip() for line in str(uploads[0]["with"]["path"]).splitlines() if line.strip()]
    assert paths == CROSS_JOB_ARTIFACT_PATHS, paths
    assert uploads[0]["with"]["retention-days"] == 1


@pytest.mark.parametrize("name", ["release-junit.xml", "mjb-characterization.log", "demo.log"])
def test_reports_and_logs_never_leave_the_runner(jobs: dict[str, Any], name: str) -> None:
    """No report or log may enter the cross-job artifact or a public release asset."""
    uploads = [
        step
        for step in _steps(jobs["build_and_validate"])
        if "actions/upload-artifact" in str(step.get("uses", ""))
    ]
    assert name not in str(uploads[0]["with"]["path"]), name
    for job_id in _external_job_ids(jobs):
        for command in _logical_lines(_job_text(jobs[job_id])):
            if "gh release create" in command:
                assert name not in command, (job_id, name)


def test_the_workflow_runs_its_own_contract_test(jobs: dict[str, Any]) -> None:
    """The workflow is the production consumer of this file, so it runs it on every attempt."""
    body = "\n".join(_job_text(job) for job in jobs.values())
    assert "python -m pytest -q .github/tests/test_publish_workflow.py" in body


# ---- Fail-closed shells, staging order and installed behaviour -----------------------------------------------------------


def test_every_shell_step_propagates_the_first_failure(jobs: dict[str, Any]) -> None:
    """A pipeline that swallows a failure turns a red release into a green one."""
    for job_id, job in jobs.items():
        for script in _run_scripts(job):
            assert script.lstrip().startswith("set -euo pipefail"), (job_id, script[:60])
            for swallow in ("|| true", "2>/dev/null ||", "|| :"):
                assert swallow not in script, (job_id, swallow)


def test_each_external_transition_depends_on_the_previous_verification(
    jobs: dict[str, Any],
) -> None:
    """Nothing is promoted before the previous stage was verified."""
    assert _needs(jobs["testpypi_publish"]) == {"build_and_validate"}
    assert _needs(jobs["testpypi_verify"]) == {"testpypi_publish"}
    assert _needs(jobs["pypi_publish"]) == {"testpypi_verify"}
    assert _needs(jobs["pypi_verify"]) == {"pypi_publish"}
    assert _needs(jobs["github_release"]) == {"pypi_verify"}


def test_every_installed_surface_and_journey_is_exercised(jobs: dict[str, Any]) -> None:
    """The root command plus seven subcommands, the demo, and one real product journey."""
    for job_id in ("build_and_validate", "testpypi_verify", "pypi_verify"):
        body = _job_text(jobs[job_id])
        assert "--help" in body, job_id
        for surface in HELP_SURFACES:
            assert surface in body, (job_id, surface)
        assert "metrifid.demo" in body, job_id
    for job_id in REGISTRY_VERIFY_JOBS:
        body = _job_text(jobs[job_id])
        assert "certify journey" in body, job_id
        assert "pip check" in body, job_id


def test_sdist_members_are_validated_before_extraction(jobs: dict[str, Any]) -> None:
    """A hostile archive must be refused before anything is written to disk."""
    body = _job_text(jobs["build_and_validate"])
    assert body.index("unsafe sdist path") < body.index("extractall")
    for rule in ("issym", "islnk", "ischr", "isblk", "isfifo", "duplicate normalized sdist path"):
        assert rule in body, rule


def test_the_wheel_record_and_licences_are_proved(jobs: dict[str, Any]) -> None:
    """RECORD must describe the whole wheel, and the licences must be the repository's own bytes."""
    body = _job_text(jobs["build_and_validate"])
    for rule in (
        "RECORD does not describe the wheel",
        "RECORD hash mismatch",
        "RECORD size mismatch",
        "RECORD contains a duplicate path",
        "RECORD self-row must be empty",
        "does not equal the repository",
    ):
        assert rule in body, rule


def test_no_publish_job_declares_an_unconsumed_output(jobs: dict[str, Any]) -> None:
    """A job output nothing reads is dead surface."""
    for job_id, job in jobs.items():
        for name in job.get("outputs") or {}:
            reference = f"needs.{job_id}.outputs.{name}"
            assert any(reference in yaml.dump(other) for other in jobs.values()), (job_id, name)


def test_the_contract_test_directory_is_outside_the_packaged_distributions() -> None:
    """`.github/` is CI configuration, not package payload, and must stay out of both artifacts."""
    project = (REPOSITORY / "pyproject.toml").read_text(encoding="utf-8")
    assert ".github/tests" not in project


# ---- Executing the acquisition shell itself, with a fake GitHub CLI -----------------------------------

FAKE_GH = """#!{python}
import json, os, pathlib, sys

fixture = json.loads(pathlib.Path(os.environ["GH_FIXTURE"]).read_text(encoding="utf-8"))
key = " ".join(sys.argv[1:])
with open(os.environ["GH_CALLS"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(sys.argv[1:]) + "\\n")
entry = fixture.get(key)
if entry is None:
    sys.stderr.write(f"fake gh: no fixture for {{key!r}}\\n")
    raise SystemExit(3)
sys.stdout.write(entry.get("stdout", ""))
sys.stderr.write(entry.get("stderr", ""))
raise SystemExit(int(entry.get("exit", 0)))
"""


def _gh_calls(path: Path) -> list[list[str]]:
    """Every recorded fake-CLI invocation as its complete argument vector."""
    calls: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        parsed = json.loads(line)
        assert isinstance(parsed, list), line
        calls.append([str(item) for item in parsed])
    return calls


def _release_trace(calls: list[list[str]]) -> list[str]:
    """Every `gh release` subcommand, in the order the route issued it."""
    return [call[1] for call in calls if len(call) >= 2 and call[0] == "release"]


def _release_mutations(calls: list[list[str]]) -> list[list[str]]:
    """Every recorded call that would change public release state."""
    return [
        call
        for call in calls
        if len(call) >= 2 and call[0] == "release" and call[1] in RELEASE_MUTATING_SUBCOMMANDS
    ]


def _fake_bin(root: Path, fixture: dict[str, Any]) -> tuple[Path, Path, Path]:
    """A disposable bin directory holding a scripted `gh` and a `python` shim."""
    binary = root / "fakebin"
    binary.mkdir(parents=True, exist_ok=True)
    fixture_path = root / "gh_fixture.json"
    fixture_path.write_text(json.dumps(fixture), encoding="utf-8")
    calls = root / "gh_calls.txt"
    calls.write_text("", encoding="utf-8")
    gh = binary / "gh"
    gh.write_text(FAKE_GH.format(python=sys.executable), encoding="utf-8")
    gh.chmod(0o755)
    shim = binary / "python"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    shim.chmod(0o755)
    return binary, fixture_path, calls


def _run_step_shell(
    script: str, workdir: Path, environment: dict[str, str], fixture: dict[str, Any]
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    binary, fixture_path, calls = _fake_bin(workdir, fixture)
    merged = {
        "PATH": f"{binary}:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "GH_FIXTURE": str(fixture_path),
        "GH_CALLS": str(calls),
        "RUNNER_TEMP": str(workdir / "runner-temp"),
    }
    (workdir / "runner-temp").mkdir(exist_ok=True)
    merged.update(environment)
    done = subprocess.run(
        ["bash", "-c", script], cwd=workdir, env=merged, capture_output=True, text=True, timeout=180
    )
    return done, _gh_calls(calls)


def _identity_gh_fixture(**overrides: str) -> dict[str, Any]:
    import base64

    repository = FIXTURE_REPOSITORY
    commit = overrides.get("commit", FIXTURE_COMMIT)
    tree = overrides.get("tree", FIXTURE_TREE)
    content = overrides.get("content", base64.b64encode(FIXTURE_WORKFLOW).decode("ascii"))
    encoding = overrides.get("encoding", "base64")
    tag_object = "1" * 40
    return {
        f"api repos/{repository} --jq .default_branch": {"stdout": "main\n"},
        f"api repos/{repository}/commits/main --jq .sha": {"stdout": commit + "\n"},
        f"api repos/{repository}/commits/{FIXTURE_COMMIT} --jq .commit.tree.sha": {
            "stdout": tree + "\n"
        },
        f"api repos/{repository}/contents/.github/workflows/publish.yml?ref={FIXTURE_COMMIT} --jq .content": {
            "stdout": content + "\n"
        },
        f"api repos/{repository}/contents/.github/workflows/publish.yml?ref={FIXTURE_COMMIT} --jq .encoding": {
            "stdout": encoding + "\n"
        },
        f"api repos/{repository}/git/ref/tags/{FIXTURE_VERSION} --jq .object.type": {
            "stdout": "tag\n"
        },
        f"api repos/{repository}/git/ref/tags/{FIXTURE_VERSION} --jq .object.sha": {
            "stdout": tag_object + "\n"
        },
        f"api repos/{repository}/git/tags/{tag_object} --jq .object.sha": {
            "stdout": FIXTURE_COMMIT + "\n"
        },
        f"api repos/{repository}/git/tags/{tag_object} --jq .object.type": {"stdout": "commit\n"},
    }


def _identity_shell_environment() -> dict[str, str]:
    return {
        "GH_TOKEN": "stub",
        "GH_REPO": FIXTURE_REPOSITORY,
        "RELEASE_VERSION": FIXTURE_VERSION,
        "PLANNED_TAG": FIXTURE_VERSION,
        "EXPECTED_COMMIT": FIXTURE_COMMIT,
        "GITHUB_SHA": FIXTURE_COMMIT,
        "GITHUB_REPOSITORY": FIXTURE_REPOSITORY,
    }


def _identity_shell_scripts(jobs: dict[str, Any]) -> list[tuple[str, str]]:
    scripts = [
        (job_id, str(_step_named(jobs[job_id], IDENTITY_STEP)["run"])) for job_id in PROTECTED_JOBS
    ]
    scripts.append(
        (
            "github_release_final",
            str(_step_named(jobs["github_release"], FINAL_IDENTITY_STEP)["run"]),
        )
    )
    return scripts


def _expected_identity_calls() -> list[list[str]]:
    """The complete ordered call list every identity copy must make, selectors included."""
    repository = FIXTURE_REPOSITORY
    commit = FIXTURE_COMMIT
    tag_object = "1" * 40
    contents = f"repos/{repository}/contents/.github/workflows/publish.yml?ref={commit}"
    return [
        ["api", f"repos/{repository}", "--jq", ".default_branch"],
        ["api", f"repos/{repository}/commits/main", "--jq", ".sha"],
        ["api", f"repos/{repository}/commits/{commit}", "--jq", ".commit.tree.sha"],
        ["api", contents, "--jq", ".content"],
        ["api", contents, "--jq", ".encoding"],
        ["api", f"repos/{repository}/git/ref/tags/{FIXTURE_VERSION}", "--jq", ".object.type"],
        ["api", f"repos/{repository}/git/ref/tags/{FIXTURE_VERSION}", "--jq", ".object.sha"],
        ["api", f"repos/{repository}/git/tags/{tag_object}", "--jq", ".object.sha"],
        ["api", f"repos/{repository}/git/tags/{tag_object}", "--jq", ".object.type"],
    ]


def test_every_identity_copy_makes_the_exact_public_calls_in_order(
    jobs: dict[str, Any], tmp_path: Path
) -> None:
    """The acquisition shell itself is executed, and its complete calls are compared in order.

    Endpoint, selector and multiplicity all count, so replacing an acquired value with `GITHUB_SHA`,
    the manifest or another locally available value fails here even when a different selector still
    queries the same endpoint.
    """
    expected = _expected_identity_calls()
    for label, script in _identity_shell_scripts(jobs):
        workdir = tmp_path / f"acq-{label}"
        workdir.mkdir()
        _promotion_fixture(workdir)
        done, observed = _run_step_shell(
            script, workdir, _identity_shell_environment(), _identity_gh_fixture()
        )
        assert done.returncode == 0, (label, done.stdout + done.stderr)
        assert observed == expected, (label, observed)


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"commit": "e" * 40}, id="public_main_is_other"),
        pytest.param({"tree": "9" * 40}, id="public_tree_disagrees"),
        pytest.param({"encoding": "utf-8"}, id="workflow_encoding_not_base64"),
        pytest.param({"content": ""}, id="workflow_content_empty"),
    ],
)
def test_every_identity_copy_fails_closed_on_a_bad_acquisition(
    jobs: dict[str, Any], tmp_path: Path, overrides: dict[str, str]
) -> None:
    """A wrong or unusable public answer stops the mutation in every copy."""
    for label, script in _identity_shell_scripts(jobs):
        workdir = tmp_path / f"acqbad-{label}-{'-'.join(sorted(overrides))}"
        workdir.mkdir(parents=True, exist_ok=True)
        _promotion_fixture(workdir)
        done, _ = _run_step_shell(
            script, workdir, _identity_shell_environment(), _identity_gh_fixture(**overrides)
        )
        assert done.returncode != 0, (label, done.stdout)


@pytest.mark.parametrize(
    "content",
    [
        pytest.param("bmFtZTogcHVibGlzaAo=@@@", id="nonalphabet_suffix"),
        pytest.param("bmFtZTogcHVibGlzaAo", id="bad_padding"),
        pytest.param("!!!!", id="not_base64_at_all"),
    ],
)
def test_every_identity_copy_decodes_the_workflow_strictly(
    jobs: dict[str, Any], tmp_path: Path, content: str
) -> None:
    """Appending garbage to otherwise valid Base64 must not decode into a matching digest."""
    for label, script in _identity_shell_scripts(jobs):
        workdir = tmp_path / f"decode-{label}-{abs(hash(content))}"
        workdir.mkdir(parents=True, exist_ok=True)
        _promotion_fixture(workdir)
        done, _ = _run_step_shell(
            script, workdir, _identity_shell_environment(), _identity_gh_fixture(content=content)
        )
        assert done.returncode != 0, (label, done.stdout)


def test_a_multi_line_base64_workflow_content_still_decodes(
    jobs: dict[str, Any], tmp_path: Path
) -> None:
    """GitHub wraps its Base64 content; only those documented separators are removed."""
    import base64

    wrapped = "\n".join(
        base64.b64encode(FIXTURE_WORKFLOW).decode("ascii")[i : i + 4]
        for i in range(0, len(base64.b64encode(FIXTURE_WORKFLOW).decode("ascii")), 4)
    )
    _, script = _identity_shell_scripts(jobs)[0]
    workdir = tmp_path / "wrapped"
    workdir.mkdir()
    _promotion_fixture(workdir)
    done, _ = _run_step_shell(
        script, workdir, _identity_shell_environment(), _identity_gh_fixture(content=wrapped)
    )
    assert done.returncode == 0, done.stdout + done.stderr


# ---- Executing the release-state route offline -----------------------------------------------------------


def _lookup_response(
    *, status: int, draft: bool | None = None, body: str | None = None
) -> dict[str, Any]:
    """One scripted `gh api --include` answer, exactly as the real CLI frames it."""
    if body is None:
        payload = "" if draft is None else json.dumps({"draft": draft, "tag_name": FIXTURE_VERSION})
    else:
        payload = body
    return {
        "response": f"HTTP/2.0 {status}\r\ncontent-type: application/json\r\n\r\n{payload}",
        "exit": 0 if status == 200 else 1,
    }


def _release_lookup_fixture(
    *, status: int, draft: bool | None = None, body: str | None = None
) -> dict[str, Any]:
    """That answer, keyed for the single-step fake CLI."""
    entry = _lookup_response(status=status, draft=draft, body=body)
    return {
        f"api --include repos/{FIXTURE_REPOSITORY}/releases/tags/{FIXTURE_VERSION}": {
            "stdout": entry["response"],
            "exit": entry["exit"],
        }
    }


@pytest.mark.parametrize(
    ("case", "fixture", "expected_state"),
    [
        pytest.param("absent", _release_lookup_fixture(status=404), "absent", id="absent"),
        pytest.param(
            "draft", _release_lookup_fixture(status=200, draft=True), "draft", id="existing_draft"
        ),
        pytest.param(
            "published",
            _release_lookup_fixture(status=200, draft=False),
            "published",
            id="existing_published",
        ),
    ],
)
def test_the_release_state_route_names_exactly_one_state(
    jobs: dict[str, Any], tmp_path: Path, case: str, fixture: dict[str, Any], expected_state: str
) -> None:
    """Absent, draft and published are the only outcomes, and each is reached deterministically."""
    script = str(_step_named(jobs["github_release"], RELEASE_STATE_STEP)["run"])
    workdir = tmp_path / f"state-{case}"
    workdir.mkdir()
    output = workdir / "gh_output"
    output.write_text("", encoding="utf-8")
    environment = _identity_shell_environment()
    environment["LOOKUP_RESPONSE"] = str(workdir / "lookup.http")
    environment["GITHUB_OUTPUT"] = str(output)
    done, observed = _run_step_shell(script, workdir, environment, fixture)
    assert done.returncode == 0, done.stdout + done.stderr
    assert f"state={expected_state}" in output.read_text(encoding="utf-8")
    assert _release_mutations(observed) == [], observed


@pytest.mark.parametrize(
    ("case", "fixture"),
    [
        pytest.param("auth", _release_lookup_fixture(status=401), id="authentication_error"),
        pytest.param("rate", _release_lookup_fixture(status=429), id="rate_limited"),
        pytest.param("server", _release_lookup_fixture(status=500), id="unexpected_status"),
        pytest.param(
            "malformed", _release_lookup_fixture(status=200, body="{not json"), id="malformed_json"
        ),
        pytest.param(
            "no_draft_field",
            _release_lookup_fixture(status=200, body='{"tag_name": "9.9.9"}'),
            id="missing_field",
        ),
        pytest.param(
            "no_status_line",
            {
                f"api --include repos/{FIXTURE_REPOSITORY}/releases/tags/{FIXTURE_VERSION}": {
                    "stdout": "garbage without a status line\r\n\r\n{}",
                    "exit": 1,
                }
            },
            id="no_status_line",
        ),
    ],
)
def test_the_release_state_route_is_fatal_on_anything_else(
    jobs: dict[str, Any], tmp_path: Path, case: str, fixture: dict[str, Any]
) -> None:
    """Only an exact not-found may mean absence; every other answer stops without mutation."""
    script = str(_step_named(jobs["github_release"], RELEASE_STATE_STEP)["run"])
    workdir = tmp_path / f"statebad-{case}"
    workdir.mkdir()
    output = workdir / "gh_output"
    output.write_text("", encoding="utf-8")
    environment = _identity_shell_environment()
    environment["LOOKUP_RESPONSE"] = str(workdir / "lookup.http")
    environment["GITHUB_OUTPUT"] = str(output)
    done, observed = _run_step_shell(script, workdir, environment, fixture)
    assert done.returncode != 0, (case, done.stdout)
    assert "state=" not in output.read_text(encoding="utf-8")
    assert _release_mutations(observed) == [], observed


def test_creation_and_publication_are_routed_by_the_release_state(jobs: dict[str, Any]) -> None:
    """Only an absent release is created, and only a non-published release is published."""
    job = jobs["github_release"]
    create = _step_named(job, CREATE_STEP)
    publish = _step_named(job, PUBLISH_STEP)
    assert create["if"] == "${{ steps.release_state.outputs.state == 'absent' }}"
    assert publish["if"] == "${{ steps.release_state.outputs.state != 'published' }}"
    precheck = _step_named(job, PRECHECK_STEP)
    assert (
        precheck["env"]["EXPECT_DRAFT"]
        == "${{ steps.release_state.outputs.state == 'published' && 'false' || 'true' }}"
    )
    assert _step_named(job, POSTCHECK_STEP)["env"]["EXPECT_DRAFT"] == "false"


def test_an_exact_published_release_is_a_no_op_that_still_verifies(jobs: dict[str, Any]) -> None:
    """A rerun after publication succeeded but a later query failed must recover without mutating."""
    job = jobs["github_release"]
    order = [
        _step_index(job, RELEASE_STATE_STEP),
        _step_index(job, CREATE_STEP),
        _step_index(job, PRECHECK_STEP),
        _step_index(job, FINAL_IDENTITY_STEP),
        _step_index(job, PUBLISH_STEP),
        _step_index(job, POSTCHECK_STEP),
    ]
    assert order == sorted(order), order
    assert _step_named(job, POSTCHECK_STEP).get("if") is None, "final verification always runs"


# ---- Executing the whole release route offline ------------------------------------------------------------

# A purpose-built release CLI, not a GitHub Actions emulator: it holds one release, answers the
# lookup from that state, and lets the real step scripts create, view, download and publish it.
STATEFUL_GH = """#!{python}
import json
import os
import pathlib
import sys

argv = sys.argv[1:]
with open(os.environ["GH_CALLS"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(argv) + "\\n")

state_path = pathlib.Path(os.environ["GH_STATE"])
state = json.loads(state_path.read_text(encoding="utf-8"))


def flag(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


def save():
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def refuse(message):
    sys.stderr.write("fake gh: " + message + "\\n")
    raise SystemExit(3)


if argv[:1] == ["api"]:
    if "--include" in argv:
        override = state.get("lookup_override")
        if override is not None:
            sys.stdout.write(override["response"])
            raise SystemExit(int(override.get("exit", 1)))
        if state["release"] is None:
            body = json.dumps({{"message": "Not Found"}})
            sys.stdout.write("HTTP/2.0 404\\r\\ncontent-type: application/json\\r\\n\\r\\n" + body)
            raise SystemExit(1)
        body = json.dumps(
            {{"draft": state["release"]["isDraft"], "tag_name": state["release"]["tagName"]}}
        )
        sys.stdout.write("HTTP/2.0 200\\r\\ncontent-type: application/json\\r\\n\\r\\n" + body)
        raise SystemExit(0)
    key = " ".join(argv)
    entry = state["identity"].get(key)
    if entry is None:
        refuse("no identity fixture for " + repr(key))
    sys.stdout.write(entry.get("stdout", ""))
    raise SystemExit(int(entry.get("exit", 0)))

if argv[:2] == ["release", "create"]:
    if state["release"] is not None:
        refuse("a release already exists for that tag")
    assets = []
    for token in reversed(argv):
        if token.startswith("--") or not pathlib.Path(token).is_file():
            break
        assets.append(token)
    assets.reverse()
    notes = pathlib.Path(flag("--notes-file")).read_text(encoding="utf-8")
    state["release"] = {{
        "tagName": argv[2],
        "isDraft": "--draft" in argv,
        "isPrerelease": "--prerelease" in argv,
        "name": flag("--title"),
        # GitHub stores the curated notes without their terminal newline.
        "body": notes.removesuffix("\\n"),
        "assets": {{pathlib.Path(p).name: pathlib.Path(p).read_bytes().hex() for p in assets}},
    }}
    save()
    raise SystemExit(0)

release = state["release"]
if release is None:
    sys.stderr.write("fake gh: release not found\\n")
    raise SystemExit(1)

if argv[:2] == ["release", "view"]:
    available = {{
        "tagName": release["tagName"],
        "isDraft": release["isDraft"],
        "isPrerelease": release["isPrerelease"],
        "name": release["name"],
        "body": release["body"],
        "assets": [{{"name": name}} for name in sorted(release["assets"])],
    }}
    fields = [name for name in flag("--json", "").split(",") if name]
    unmodelled = [name for name in fields if name not in available]
    if unmodelled:
        refuse("unmodelled --json fields " + repr(unmodelled))
    sys.stdout.write(json.dumps({{name: available[name] for name in fields}}))
    raise SystemExit(0)

if argv[:2] == ["release", "download"]:
    target = pathlib.Path(flag("--dir"))
    target.mkdir(parents=True, exist_ok=True)
    for name, payload in release["assets"].items():
        (target / name).write_bytes(bytes.fromhex(payload))
    raise SystemExit(0)

if argv[:2] == ["release", "edit"]:
    if "--draft=false" not in argv:
        refuse("unsupported release edit " + repr(argv))
    release["isDraft"] = False
    save()
    raise SystemExit(0)

refuse("unsupported call " + repr(argv))
"""

RELEASE_ROUTE_STEPS = (
    IDENTITY_STEP,
    RELEASE_STATE_STEP,
    CREATE_STEP,
    PRECHECK_STEP,
    FINAL_IDENTITY_STEP,
    PUBLISH_STEP,
    POSTCHECK_STEP,
)

# The route fixture resolves only these two expression shapes, and
# `test_creation_and_publication_are_routed_by_the_release_state` pins the exact strings they
# resolve, so a changed condition is a failure here rather than a silently skipped step.
STATE_CONDITION = re.compile(r"^\$\{\{ steps\.release_state\.outputs\.state (==|!=) '(\w+)' \}\}$")
STATE_TERNARY = re.compile(
    r"^\$\{\{ steps\.release_state\.outputs\.state == '(\w+)' && '(\w+)' \|\| '(\w+)' \}\}$"
)


def _release_step_runs(step: dict[str, Any], state: str) -> bool:
    """Apply the step's own condition to the state the real lookup script produced."""
    condition = step.get("if")
    if condition is None:
        return True
    match = STATE_CONDITION.match(str(condition))
    assert match is not None, f"unsupported step condition: {condition!r}"
    operator, expected = match.groups()
    return state == expected if operator == "==" else state != expected


def _release_step_env(step: dict[str, Any], state: str, workdir: Path) -> dict[str, str]:
    """Resolve the step's declared environment for the state the lookup script produced."""
    resolved: dict[str, str] = {}
    for name, raw in (step.get("env") or {}).items():
        value = str(raw)
        ternary = STATE_TERNARY.match(value)
        if ternary is not None:
            compared, when_true, when_false = ternary.groups()
            resolved[name] = when_true if state == compared else when_false
        elif value == "${{ github.token }}":
            resolved[name] = "stub"
        elif value == "${{ github.repository }}":
            resolved[name] = FIXTURE_REPOSITORY
        elif value.startswith("${{ runner.temp }}/"):
            resolved[name] = str(workdir / "runner-temp" / value.split("/", 1)[1])
        else:
            assert "${{" not in value, f"unsupported step env {name}={value!r}"
            resolved[name] = value
    return resolved


def _existing_release(workdir: Path, kind: str, mismatch: str | None) -> dict[str, Any]:
    """The release the fake CLI already holds before the route starts."""
    artifacts = workdir / "release-artifacts"
    assets = {
        name: (artifacts / "dist" / name).read_bytes().hex() for name in RELEASE_ASSET_NAMES[:2]
    }
    assets["SHA256SUMS.txt"] = (artifacts / "SHA256SUMS.txt").read_bytes().hex()
    if mismatch == "asset_bytes":
        assets[RELEASE_ASSET_NAMES[0]] = b"tampered wheel".hex()
    return {
        "tagName": FIXTURE_VERSION,
        "isDraft": kind == "draft",
        "isPrerelease": False,
        "name": "metrifid" if mismatch == "title" else f"metrifid {FIXTURE_VERSION}",
        "body": (artifacts / "RELEASE_NOTES.md").read_text(encoding="utf-8").removesuffix("\n"),
        "assets": assets,
    }


def _release_route_bin(workdir: Path, state: dict[str, Any]) -> dict[str, str]:
    """A disposable bin holding the stateful `gh` and the environment every step shares."""
    binary = workdir / "fakebin"
    binary.mkdir()
    gh = binary / "gh"
    gh.write_text(STATEFUL_GH.format(python=sys.executable), encoding="utf-8")
    gh.chmod(0o755)
    shim = binary / "python"
    shim.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n', encoding="utf-8")
    shim.chmod(0o755)
    (workdir / "gh_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    (workdir / "gh_calls.txt").write_text("", encoding="utf-8")
    (workdir / "github_output").write_text("", encoding="utf-8")
    (workdir / "runner-temp").mkdir()
    environment = {
        "PATH": f"{binary}:/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "GH_STATE": str(workdir / "gh_state.json"),
        "GH_CALLS": str(workdir / "gh_calls.txt"),
        "GITHUB_OUTPUT": str(workdir / "github_output"),
        "RUNNER_TEMP": str(workdir / "runner-temp"),
    }
    environment.update(_identity_shell_environment())
    return environment


def _stored_release(workdir: Path) -> dict[str, Any] | None:
    """Whatever release the fake CLI holds once the route has finished."""
    state = json.loads((workdir / "gh_state.json").read_text(encoding="utf-8"))
    assert isinstance(state, dict)
    release = state["release"]
    assert release is None or isinstance(release, dict)
    return release


RouteRecord = tuple[str, str, "subprocess.CompletedProcess[str] | None"]


def _run_release_route(
    jobs: dict[str, Any],
    workdir: Path,
    *,
    existing: str | None = None,
    mismatch: str | None = None,
    lookup_override: dict[str, Any] | None = None,
) -> tuple[list[RouteRecord], list[list[str]]]:
    """Run the real release step scripts in their real order against one shared release state."""
    job = jobs["github_release"]
    _promotion_fixture(workdir)
    base = _release_route_bin(
        workdir,
        {
            "identity": _identity_gh_fixture(),
            "release": _existing_release(workdir, existing, mismatch) if existing else None,
            "lookup_override": lookup_override,
        },
    )
    records: list[RouteRecord] = []
    state = ""
    broken = False
    for name in RELEASE_ROUTE_STEPS:
        step = _step_named(job, name)
        if broken:
            records.append((name, "unreached", None))
            continue
        if not _release_step_runs(step, state):
            records.append((name, "skipped", None))
            continue
        environment = dict(base)
        environment.update(_release_step_env(step, state, workdir))
        done = subprocess.run(
            ["bash", "-c", str(step["run"])],
            cwd=workdir,
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
        )
        records.append((name, "ran", done))
        if done.returncode != 0:
            broken = True
        elif name == RELEASE_STATE_STEP:
            outputs = (workdir / "github_output").read_text(encoding="utf-8")
            state = dict(line.split("=", 1) for line in outputs.splitlines() if "=" in line)[
                "state"
            ]
    return records, _gh_calls(workdir / "gh_calls.txt")


def _route_status(records: list[RouteRecord], name: str) -> str:
    for step_name, status, _ in records:
        if step_name == name:
            return status
    raise AssertionError(f"{name} is not part of the route")


def _route_process(records: list[RouteRecord], name: str) -> subprocess.CompletedProcess[str]:
    for step_name, _, done in records:
        if step_name == name and done is not None:
            return done
    raise AssertionError(f"{name} did not run")


def _assert_route_steps_passed(records: list[RouteRecord], expected: dict[str, str]) -> None:
    for name, status, done in records:
        assert status == expected[name], (name, status)
        if status == "ran":
            assert done is not None, name
            assert done.returncode == 0, (name, done.stdout + done.stderr)


ABSENT_TRACE = ["create", "view", "download", "edit", "view", "download"]
DRAFT_TRACE = ["view", "download", "edit", "view", "download"]
PUBLISHED_TRACE = ["view", "download", "view", "download"]
REFUSED_TRACE = ["view", "download"]


def test_the_release_route_creates_verifies_and_publishes_an_absent_release(
    jobs: dict[str, Any], tmp_path: Path
) -> None:
    """The real step scripts create one draft, verify it, publish it once and verify it again."""
    workdir = tmp_path / "route-absent"
    workdir.mkdir()
    records, calls = _run_release_route(jobs, workdir)
    _assert_route_steps_passed(records, dict.fromkeys(RELEASE_ROUTE_STEPS, "ran"))
    assert _release_trace(calls) == ABSENT_TRACE, calls
    create = next(call for call in calls if call[:2] == ["release", "create"])
    assert create[2] == FIXTURE_VERSION
    assert "--verify-tag" in create
    assert "--notes-file" in create
    assets = create[create.index("--draft") + 1 :]
    assert sorted(Path(item).name for item in assets) == sorted(RELEASE_ASSET_NAMES)
    published = _stored_release(workdir)
    assert published is not None
    assert published["isDraft"] is False


def test_the_release_route_publishes_an_existing_draft_without_creating_it(
    jobs: dict[str, Any], tmp_path: Path
) -> None:
    """A rerun that already left a matching draft behind publishes it and creates nothing."""
    workdir = tmp_path / "route-draft"
    workdir.mkdir()
    records, calls = _run_release_route(jobs, workdir, existing="draft")
    expected = dict.fromkeys(RELEASE_ROUTE_STEPS, "ran")
    expected[CREATE_STEP] = "skipped"
    _assert_route_steps_passed(records, expected)
    assert _release_trace(calls) == DRAFT_TRACE, calls
    published = _stored_release(workdir)
    assert published is not None
    assert published["isDraft"] is False


def test_the_release_route_treats_an_exact_published_release_as_a_verified_no_op(
    jobs: dict[str, Any], tmp_path: Path
) -> None:
    """An already-published release is neither created nor edited, yet is fully verified twice."""
    workdir = tmp_path / "route-published"
    workdir.mkdir()
    records, calls = _run_release_route(jobs, workdir, existing="published")
    expected = dict.fromkeys(RELEASE_ROUTE_STEPS, "ran")
    expected[CREATE_STEP] = "skipped"
    expected[PUBLISH_STEP] = "skipped"
    _assert_route_steps_passed(records, expected)
    assert _release_trace(calls) == PUBLISHED_TRACE, calls
    assert _release_mutations(calls) == [], calls


@pytest.mark.parametrize(
    ("existing", "mismatch"),
    [
        pytest.param("draft", "asset_bytes", id="existing_draft_holds_other_bytes"),
        pytest.param("published", "title", id="existing_published_has_another_title"),
    ],
)
def test_the_release_route_refuses_a_mismatched_release_before_any_mutation(
    jobs: dict[str, Any], tmp_path: Path, existing: str, mismatch: str
) -> None:
    """A release that does not match the accepted contract stops the route with nothing changed."""
    workdir = tmp_path / f"route-bad-{existing}"
    workdir.mkdir()
    records, calls = _run_release_route(jobs, workdir, existing=existing, mismatch=mismatch)
    assert _route_process(records, RELEASE_STATE_STEP).returncode == 0
    assert _route_status(records, CREATE_STEP) == "skipped"
    assert _route_process(records, PRECHECK_STEP).returncode != 0
    for name in (FINAL_IDENTITY_STEP, PUBLISH_STEP, POSTCHECK_STEP):
        assert _route_status(records, name) == "unreached", name
    assert _release_trace(calls) == REFUSED_TRACE, calls
    assert _release_mutations(calls) == [], calls
    held = _stored_release(workdir)
    assert held is not None
    assert held["isDraft"] is (existing == "draft")


@pytest.mark.parametrize(
    ("case", "override"),
    [
        pytest.param("auth", _lookup_response(status=401), id="authentication_error"),
        pytest.param("rate", _lookup_response(status=429), id="rate_limited"),
        pytest.param("server", _lookup_response(status=500), id="unexpected_status"),
        pytest.param(
            "malformed", _lookup_response(status=200, body="{not json"), id="malformed_json"
        ),
        pytest.param(
            "missing_field",
            _lookup_response(status=200, body='{"tag_name": "9.9.9"}'),
            id="missing_draft_field",
        ),
        pytest.param(
            "no_status",
            {"response": "garbage without a status line\r\n\r\n{}", "exit": 1},
            id="network_without_a_status_line",
        ),
    ],
)
def test_the_release_route_stops_before_any_mutation_on_a_lookup_failure(
    jobs: dict[str, Any], tmp_path: Path, case: str, override: dict[str, Any]
) -> None:
    """Only an exact not-found may authorize creation; nothing else reaches a release command."""
    workdir = tmp_path / f"route-lookup-{case}"
    workdir.mkdir()
    records, calls = _run_release_route(jobs, workdir, lookup_override=override)
    assert _route_process(records, IDENTITY_STEP).returncode == 0
    assert _route_process(records, RELEASE_STATE_STEP).returncode != 0
    for name in (CREATE_STEP, PRECHECK_STEP, FINAL_IDENTITY_STEP, PUBLISH_STEP, POSTCHECK_STEP):
        assert _route_status(records, name) == "unreached", name
    assert _release_trace(calls) == [], calls
    assert _stored_release(workdir) is None


# ---- Registry rows: duplicates, malformed rows and immediate failure ------------------------------------


def _duplicate_payload(root: Path) -> dict[str, Any]:
    payload = _index_payload(root)
    rows = list(payload["urls"])
    return {"urls": [*rows, dict(rows[0])]}


@pytest.mark.parametrize("job_id", REGISTRY_PUBLISH_JOBS)
def test_preflight_refuses_a_duplicate_row(
    jobs: dict[str, Any], tmp_path: Path, job_id: str
) -> None:
    """A third row duplicating an expected one would vanish in a mapping conversion."""
    code = _inline_python(jobs[job_id], PREFLIGHT_STEP)
    workdir = tmp_path / f"dup-{job_id}"
    workdir.mkdir()
    _promotion_fixture(workdir)
    done = _run_validator(
        code,
        workdir,
        {
            "INDEX_JSON_URL": "https://stub/index",
            "GITHUB_OUTPUT": str(workdir / "gh_output"),
            "STUB_PLAN": json.dumps([{"kind": "json", "payload": _duplicate_payload(workdir)}]),
        },
        prelude=NETWORK_STUB,
    )
    assert done.returncode != 0, done.stdout


@pytest.mark.parametrize("job_id", REGISTRY_PUBLISH_JOBS)
@pytest.mark.parametrize(
    "broken",
    [
        pytest.param("digest", id="malformed_digest"),
        pytest.param("url", id="missing_url"),
        pytest.param("row", id="row_not_an_object"),
    ],
)
def test_preflight_refuses_a_malformed_row(
    jobs: dict[str, Any], tmp_path: Path, job_id: str, broken: str
) -> None:
    """Each row must be a well-formed object with a canonical digest and an https url."""
    code = _inline_python(jobs[job_id], PREFLIGHT_STEP)
    workdir = tmp_path / f"malformed-{job_id}-{broken}"
    workdir.mkdir()
    _promotion_fixture(workdir)
    payload = _index_payload(workdir)
    if broken == "digest":
        payload["urls"][0]["digests"]["sha256"] = "NOTAHASH"
    elif broken == "url":
        payload["urls"][0]["url"] = ""
    else:
        payload["urls"][0] = "not-an-object"
    done = _run_validator(
        code,
        workdir,
        {
            "INDEX_JSON_URL": "https://stub/index",
            "GITHUB_OUTPUT": str(workdir / "gh_output"),
            "STUB_PLAN": json.dumps([{"kind": "json", "payload": payload}]),
        },
        prelude=NETWORK_STUB,
    )
    assert done.returncode != 0, done.stdout


@pytest.mark.parametrize("job_id", REGISTRY_VERIFY_JOBS)
@pytest.mark.parametrize(
    ("case", "refusal"),
    [
        pytest.param("duplicate", "more than once", id="duplicate_row"),
        pytest.param("extra", "unexpected filename", id="extra_row"),
        pytest.param("wrong_hash", "different digest", id="wrong_digest"),
        pytest.param("malformed", "malformed digest", id="malformed_digest"),
    ],
)
def test_poll_fails_immediately_on_an_unhealable_state(
    jobs: dict[str, Any], tmp_path: Path, job_id: str, case: str, refusal: str
) -> None:
    """An immutable wrong state must fail at once, and for the reason that state actually is.

    The download bytes are supplied correctly, so a copy that collapses a duplicate row would
    otherwise complete successfully. Each case also names the refusal it expects, so a mutant
    cannot pass by failing somewhere else.
    """
    code = _inline_python(jobs[job_id], POLL_STEP)
    workdir = tmp_path / f"pollbad-{job_id}-{case}"
    workdir.mkdir()
    _promotion_fixture(workdir)
    if case == "duplicate":
        payload = _duplicate_payload(workdir)
    elif case == "malformed":
        payload = _index_payload(workdir)
        payload["urls"][0]["digests"]["sha256"] = "NOTAHASH"
    else:
        payload = _index_payload(workdir, mutate=case)
    plan: list[dict[str, Any]] = [{"kind": "json", "payload": payload}]
    for index in range(2):
        plan.append({"kind": "raw", "text": f"payload-{index}"})
    done = _run_validator(
        code,
        workdir,
        {
            "INDEX_JSON_URL": "https://stub/index",
            "STUB_PLAN": json.dumps(plan),
            "STUB_CLOCK_STEP": "0",
        },
        prelude=NETWORK_STUB,
    )
    output = done.stdout + done.stderr
    assert done.returncode != 0, (job_id, case, done.stdout)
    assert refusal in output, (job_id, case, output[-400:])
    assert "timed out" not in output, "an unhealable state must not wait"


@pytest.mark.parametrize("job_id", REGISTRY_VERIFY_JOBS)
def test_poll_retries_a_correct_subset_until_the_deadline(
    jobs: dict[str, Any], tmp_path: Path, job_id: str
) -> None:
    """A correct but incomplete index is ordinary propagation and is retried, then reported."""
    code = _inline_python(jobs[job_id], POLL_STEP)
    workdir = tmp_path / f"pollsubset-{job_id}"
    workdir.mkdir()
    _promotion_fixture(workdir)
    done = _run_validator(
        code,
        workdir,
        {
            "INDEX_JSON_URL": "https://stub/index",
            "STUB_PLAN": json.dumps(
                [{"kind": "json", "payload": _index_payload(workdir, mutate="partial")}]
            ),
            "STUB_CLOCK_STEP": "1000",
        },
        prelude=NETWORK_STUB,
    )
    assert done.returncode != 0
    assert "timed out waiting for the index" in done.stdout + done.stderr
    assert "still missing" in done.stdout + done.stderr


# ---- The curated release body ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "green"),
    [
        pytest.param("## 9.9.9\n\nnotes\n", True, id="exact"),
        pytest.param("## 9.9.9\n\nnotes", True, id="one_terminal_lf_removed"),
        pytest.param("## 9.9.9\n\nnotes\n\n", False, id="extra_terminal_lf"),
        pytest.param("## 9.9.9\r\n\r\nnotes\n", False, id="internal_crlf"),
        pytest.param("## 9.9.9\n\nnotes  \n", False, id="trailing_whitespace_changed"),
        pytest.param("## 9.9.9\n\nother\n", False, id="different_text"),
    ],
)
def test_the_release_body_comparison_is_exact_apart_from_one_terminal_lf(
    jobs: dict[str, Any], tmp_path: Path, body: str, green: bool
) -> None:
    """Only the documented single terminal line feed may differ; nothing else is normalized."""
    for label, code, expect_draft in _release_sources(jobs):
        workdir = tmp_path / f"body-{label}-{abs(hash(body))}"
        workdir.mkdir(parents=True, exist_ok=True)
        _release_fixture(workdir, is_draft=expect_draft == "true", body=body)
        done = _run_validator(
            code,
            workdir,
            _identity_environment(RELEASE_STATE=str(workdir / "state"), EXPECT_DRAFT=expect_draft),
        )
        if green:
            assert done.returncode == 0, (label, done.stdout + done.stderr)
        else:
            assert done.returncode != 0, (label, done.stdout)


@pytest.mark.parametrize("target", ["wheel", "sdist"])
def test_each_release_contract_copy_refuses_a_tampered_distribution(
    jobs: dict[str, Any], tmp_path: Path, target: str
) -> None:
    """Removing either byte comparison from both validators must fail the suite."""
    for label, code, expect_draft in _release_sources(jobs):
        workdir = tmp_path / f"tamper-{label}-{target}"
        workdir.mkdir(parents=True, exist_ok=True)
        _release_fixture(workdir, is_draft=expect_draft == "true")
        name = (
            f"metrifid-{FIXTURE_VERSION}-py3-none-any.whl"
            if target == "wheel"
            else f"metrifid-{FIXTURE_VERSION}.tar.gz"
        )
        (workdir / "state" / name).write_bytes(b"tampered payload")
        done = _run_validator(
            code,
            workdir,
            _identity_environment(RELEASE_STATE=str(workdir / "state"), EXPECT_DRAFT=expect_draft),
        )
        assert done.returncode != 0, (label, target, done.stdout)


# ---- The serialized public manifest ------------------------------------------------------------------------


def _stub_distributions(site: Path, versions: dict[str, str]) -> Path:
    """A private site directory whose metadata answers `importlib.metadata.version` exactly."""
    site.mkdir(parents=True)
    for name, version in sorted(versions.items()):
        info = site / f"{name}-{version}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n", encoding="utf-8"
        )
    return site


def _manifest_fixture(workdir: Path) -> tuple[list[dict[str, Any]], str]:
    """Build the manifest builder's inputs and return the exact rows and tree it must emit."""
    (workdir / "dist").mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    # `sorted(Path("dist").iterdir())` orders the canonical pair wheel-first, because `-` precedes
    # `.` in the two names that follow the shared `metrifid-<version>` stem.
    for index, name in enumerate(RELEASE_ASSET_NAMES[:2]):
        payload = f"payload-{index}".encode()
        (workdir / "dist" / name).write_bytes(payload)
        rows.append(
            {
                "filename": name,
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    (workdir / ".github/workflows").mkdir(parents=True)
    (workdir / ".github/workflows/publish.yml").write_bytes(FIXTURE_WORKFLOW)
    _junit_fixture(workdir / "junit.xml", ACCEPTED_SKIPS)
    (workdir / "suite-runtime.json").write_text(json.dumps(FIXTURE_SUITE_RUNTIME), encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=workdir, check=True)
    subprocess.run(["git", "add", "-A"], cwd=workdir, check=True)
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "fixture"],
        cwd=workdir,
        check=True,
    )
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=workdir,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    assert re.fullmatch(r"[0-9a-f]{40}", tree), tree
    return rows, tree


def test_the_serialized_manifest_holds_exactly_the_expected_values(
    jobs: dict[str, Any], tmp_path: Path
) -> None:
    """The emitted JSON is compared value by value, not merely key by key.

    A key added after the literal, a wrong identity, a duplicated or renamed artifact row, a changed
    size or digest, and any future registry or release outcome all fail here.
    """
    code = _inline_python(jobs["build_and_validate"], MANIFEST_STEP)
    workdir = tmp_path / "manifest"
    workdir.mkdir()
    rows, tree = _manifest_fixture(workdir)
    site = _stub_distributions(
        tmp_path / "manifest-stubs",
        {
            "hatchling": STUB_BACKEND_VERSION,
            "build": STUB_FRONTEND_VERSION,
            "twine": STUB_TWINE_VERSION,
        },
    )
    done = _run_validator(
        code,
        workdir,
        {
            "RELEASE_VERSION": FIXTURE_VERSION,
            "PLANNED_TAG": FIXTURE_VERSION,
            "RELEASE_MODE": STAGED_RELEASE,
            "RUN_REPOSITORY": FIXTURE_REPOSITORY,
            "RUN_COMMIT": FIXTURE_COMMIT,
            "RUNNER_LABEL": "Linux-X64",
            "JUNIT_REPORT": str(workdir / "junit.xml"),
            "SUITE_RUNTIME": str(workdir / "suite-runtime.json"),
            "PYTHONPATH": str(site),
        },
    )
    assert done.returncode == 0, done.stdout + done.stderr
    manifest = json.loads((workdir / "dist-promotion-manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == MANIFEST_TOP_LEVEL, sorted(manifest)
    assert manifest["schema_version"] == 4
    assert manifest["project"] == "metrifid"
    assert manifest["version"] == FIXTURE_VERSION
    assert manifest["tag"] == FIXTURE_VERSION
    assert manifest["mode"] == STAGED_RELEASE
    assert manifest["commit"] == FIXTURE_COMMIT
    assert manifest["repository"] == FIXTURE_REPOSITORY
    assert manifest["tree"] == tree
    assert manifest["workflow_sha256"] == FIXTURE_WORKFLOW_SHA
    assert manifest["build"] == {
        "backend": "hatchling",
        "backend_version": STUB_BACKEND_VERSION,
        "frontend_version": STUB_FRONTEND_VERSION,
        "twine_version": STUB_TWINE_VERSION,
        "isolated": False,
        "runner": "Linux-X64",
    }
    assert manifest["suite_runtime"] == FIXTURE_SUITE_RUNTIME
    assert manifest["artifacts"] == rows
    assert len({row["filename"] for row in manifest["artifacts"]}) == 2
    assert manifest["outcomes"] == {
        "observed_test_cases": len(ACCEPTED_SKIPS) + 1,
        "pip_check": "passed",
        "help_surfaces": "passed",
        "demo": "passed",
        "mjb_characterization": "passed",
    }


# ---- Guards restored from the parent contract ----------------------------------------------------------------


def test_the_release_identity_inputs_are_required(document: dict[Any, Any]) -> None:
    """A release cannot be dispatched without naming its version, commit and tag."""
    inputs = _triggers(document)["workflow_dispatch"]["inputs"]
    for name in ("release_version", "expected_commit", "planned_tag"):
        assert inputs[name]["required"] is True, name


def test_the_version_grammar_admits_only_a_canonical_stable_release(jobs: dict[str, Any]) -> None:
    """No leading `v`, no leading-zero segment, no dev, local, pre-release or post-release suffix."""
    body = _job_text(jobs["build_and_validate"])
    match = re.search(r'canonical = re\.compile\(r"([^"]+)"\)', body)
    assert match is not None, "the version grammar must be stated once, as a compiled pattern"
    grammar = re.compile(match.group(1))
    for good in ("0.7.0", "1.2.3", "10.0.11"):
        assert grammar.fullmatch(good) is not None, good
    for bad in (
        "v1.2.3",
        "01.2.3",
        "1.02.3",
        "0.7.0.dev0",
        "1.2.3rc1",
        "1.2.3.post1",
        "1.2.3+local",
        "1",
        "",
    ):
        assert grammar.fullmatch(bad) is None, bad


def test_pytest_check_is_supplied_where_product_tests_import_it() -> None:
    """The requirement is derived from the repository, not from a remembered number."""
    tests_root = REPOSITORY / "tests"
    if not tests_root.is_dir():
        pytest.skip("the product test tree is not present in this checkout")
    importers = [
        path
        for path in sorted(tests_root.rglob("*.py"))
        if re.search(
            r"^\s*(?:import pytest_check|from pytest_check)", path.read_text(encoding="utf-8"), re.M
        )
    ]
    assert importers, "expected at least one product test module to import pytest_check"
    jobs = _jobs(_document())
    step = _step_named(jobs["build_and_validate"], SUITE_STEP)
    assert "pytest-check" in str(step["run"])
    contract_job = _job_text(jobs["workflow_contract"])
    assert "pytest-check" not in contract_job, "the contract-only job imports only pytest and yaml"


def test_the_parity_wheel_is_never_selectable(jobs: dict[str, Any]) -> None:
    """A wheel rebuilt from the sdist is evidence; no upload or release step may pick it up."""
    build = _job_text(jobs["build_and_validate"])
    renames = [
        line
        for line in _logical_lines(build)
        if "find parity" in line and "-exec mv" in line and "parity-rebuild.notdist" in line
    ]
    assert len(renames) == 1, "the parity wheel must actually be renamed out of the wheel namespace"
    assert "find parity -maxdepth 1 -name '*.whl' | wc -l | tr -d ' ')\" = \"0\"" in build
    uploads = [
        step
        for step in _steps(jobs["build_and_validate"])
        if "actions/upload-artifact" in str(step.get("uses", ""))
    ]
    assert "parity" not in str(uploads[0]["with"]["path"])
    for job_id in _external_job_ids(jobs):
        assert "parity" not in _job_text(jobs[job_id]), job_id


def test_sdist_backslash_and_top_level_safety_are_enforced(jobs: dict[str, Any]) -> None:
    """A backslash path or an ambiguous root is refused before anything is extracted."""
    body = _job_text(jobs["build_and_validate"])
    assert body.index("sdist member uses a backslash path") < body.index("extractall")
    assert body.index("ambiguous sdist top-level layout") < body.index("extractall")


def test_sdist_metadata_and_source_correspondence_are_proved(jobs: dict[str, Any]) -> None:
    """PKG-INFO and every source member are checked before the sdist is used."""
    body = _job_text(jobs["build_and_validate"])
    for rule in (
        "sdist PKG-INFO does not report the release version",
        "sdist PKG-INFO does not declare Apache-2.0",
        "sdist source member differs",
    ):
        assert rule in body, rule


def test_the_native_mujoco_identity_is_asserted(jobs: dict[str, Any]) -> None:
    """The package version, the native string and the native integer must all agree."""
    body = _job_text(jobs["build_and_validate"])
    assert "native MuJoCo string is" in body
    assert "native MuJoCo integer disagrees" in body
    assert "suite runtime has MuJoCo" in body


def test_no_registry_preflight_imports_an_unused_module(jobs: dict[str, Any]) -> None:
    """Dead imports are removed along with the code that once needed them."""
    for job_id in REGISTRY_PUBLISH_JOBS:
        code = _inline_python(jobs[job_id], PREFLIGHT_STEP)
        assert "import hashlib" not in code, job_id


@pytest.mark.parametrize("target", ["wheel", "sdist"])
def test_each_release_contract_copy_refuses_bytes_that_match_the_recorded_digest(
    jobs: dict[str, Any], tmp_path: Path, target: str
) -> None:
    """Only the byte comparison can catch this, so removing it from both copies must fail here."""
    for label, code, expect_draft in _release_sources(jobs):
        workdir = tmp_path / f"bytedrift-{label}-{target}"
        workdir.mkdir(parents=True, exist_ok=True)
        _release_fixture(workdir, is_draft=expect_draft == "true", byte_only_drift=target)
        done = _run_validator(
            code,
            workdir,
            _identity_environment(RELEASE_STATE=str(workdir / "state"), EXPECT_DRAFT=expect_draft),
        )
        assert done.returncode != 0, (label, target, done.stdout)


@pytest.mark.parametrize("target", ["wheel", "sdist"])
def test_each_release_contract_copy_refuses_a_digest_that_no_longer_describes_the_bytes(
    jobs: dict[str, Any], tmp_path: Path, target: str
) -> None:
    """Only the hash comparison can catch this, so removing it from both copies must fail here."""
    for label, code, expect_draft in _release_sources(jobs):
        workdir = tmp_path / f"hashdrift-{label}-{target}"
        workdir.mkdir(parents=True, exist_ok=True)
        _release_fixture(workdir, is_draft=expect_draft == "true", hash_only_drift=target)
        done = _run_validator(
            code,
            workdir,
            _identity_environment(RELEASE_STATE=str(workdir / "state"), EXPECT_DRAFT=expect_draft),
        )
        assert done.returncode != 0, (label, target, done.stdout)


def _runtime_stub(
    site: Path,
    *,
    package_version: str = PRIMARY_MUJOCO,
    native_string: str = PRIMARY_MUJOCO,
    native_integer: int = PRIMARY_MUJOCO_NATIVE,
) -> Path:
    """A private site directory whose `mujoco` module and metadata answer with fixture values."""
    _stub_distributions(
        site,
        {
            "pytest": STUB_PYTEST_VERSION,
            "numpy": STUB_NUMPY_VERSION,
            "mujoco": package_version,
        },
    )
    (site / "mujoco.py").write_text(
        f'def mj_versionString() -> str:\n    return "{native_string}"\n\n\n'
        f"def mj_version() -> int:\n    return {native_integer}\n",
        encoding="utf-8",
    )
    return site


def test_the_runtime_identity_block_records_the_exact_native_identity(
    jobs: dict[str, Any], tmp_path: Path
) -> None:
    """The real runtime-identity block runs against a stubbed MuJoCo and records exact values."""
    code = _inline_python(jobs["build_and_validate"], SUITE_STEP)
    workdir = tmp_path / "runtime-green"
    workdir.mkdir()
    site = _runtime_stub(workdir / "stubs")
    target = workdir / "suite-runtime.json"
    done = _run_validator(
        code,
        workdir,
        {
            "PRIMARY_MUJOCO": PRIMARY_MUJOCO,
            "SUITE_RUNTIME": str(target),
            "PYTHONPATH": str(site),
        },
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "python": sys.version.split()[0],
        "pytest": STUB_PYTEST_VERSION,
        "numpy": STUB_NUMPY_VERSION,
        "mujoco": PRIMARY_MUJOCO,
        "mujoco_native_string": PRIMARY_MUJOCO,
        "mujoco_native_integer": PRIMARY_MUJOCO_NATIVE,
    }


@pytest.mark.parametrize(
    ("overrides", "refusal"),
    [
        pytest.param(
            {"package_version": "3.11.0"}, "suite runtime has MuJoCo", id="wrong_package_version"
        ),
        pytest.param(
            {"native_string": "3.11.0"}, "native MuJoCo string is", id="wrong_native_string"
        ),
        pytest.param(
            {"native_integer": 3_011_000},
            "native MuJoCo integer disagrees",
            id="wrong_native_integer",
        ),
    ],
)
def test_the_runtime_identity_block_refuses_each_wrong_native_field(
    jobs: dict[str, Any], tmp_path: Path, overrides: dict[str, Any], refusal: str
) -> None:
    """Each field has its own refusal, so comparing the wrong one no longer passes unnoticed."""
    code = _inline_python(jobs["build_and_validate"], SUITE_STEP)
    workdir = tmp_path / f"runtime-{'-'.join(sorted(overrides))}"
    workdir.mkdir()
    site = _runtime_stub(workdir / "stubs", **overrides)
    target = workdir / "suite-runtime.json"
    done = _run_validator(
        code,
        workdir,
        {
            "PRIMARY_MUJOCO": PRIMARY_MUJOCO,
            "SUITE_RUNTIME": str(target),
            "PYTHONPATH": str(site),
        },
    )
    assert done.returncode != 0, done.stdout
    assert refusal in done.stderr, done.stderr
    assert not target.exists()


def test_the_native_mujoco_comparisons_are_live_inequalities(jobs: dict[str, Any]) -> None:
    """A refusal message in a dead branch is not a check, and neither is an inverted operator.

    The runtime-identity block is parsed, and each MuJoCo field must be the left side of exactly one
    live `!=` comparison against the value it is supposed to contradict.
    """
    code = _inline_python(jobs["build_and_validate"], SUITE_STEP)
    compared: dict[str, ast.Compare] = {}
    for node in ast.walk(ast.parse(code)):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        left = node.test.left
        if not isinstance(left, ast.Subscript) or not isinstance(left.value, ast.Name):
            continue
        if left.value.id != "runtime" or not isinstance(left.slice, ast.Constant):
            continue
        field = left.slice.value
        assert isinstance(field, str)
        assert field not in compared, f"{field} is compared more than once"
        compared[field] = node.test
    for field in ("mujoco", "mujoco_native_string", "mujoco_native_integer"):
        assert field in compared, field
        assert [type(op) for op in compared[field].ops] == [ast.NotEq], field
    for field in ("mujoco", "mujoco_native_string"):
        comparator = compared[field].comparators[0]
        assert isinstance(comparator, ast.Name), field
        assert comparator.id == "expected", field
    native = compared["mujoco_native_integer"].comparators[0]
    constants = {
        node.value
        for node in ast.walk(native)
        if isinstance(node, ast.Constant) and isinstance(node.value, int)
    }
    assert {1_000_000, 1_000} <= constants, sorted(constants)


# ---- Registry-verification journeys keep their output outside the model root ------------------------
#
# Metrifid refuses an output directory inside a declared model root with
# OUTPUT_PATH_INVALID / output_inside_model_root, and exits 64. A verification journey that writes
# its models and its receipt into one directory therefore cannot pass, and that refusal is the
# product behaving correctly. Both registry-verification jobs embed their own copy of the journey,
# so each copy is executed here rather than read, with the console script replaced by a fake that
# inspects the argument vector the journey actually builds.

JOURNEY_STEPS = {
    "testpypi_verify": "Install the downloaded wheel and run the public journeys",
    "pypi_verify": "Install the public wheel and run the public journeys",
}

# Stands in for the installed console script. It fails the journey unless the invocation really is
# `metrifid certify`, the two input arguments name files the journey created, and the output
# argument resolves outside the parent directory of each input. Only then does it publish a receipt,
# so the journey's own nonempty-receipt assertion still has to run.
JOURNEY_FAKE = """
import json as _json
import pathlib as _pathlib
import subprocess as _subprocess
import types as _types

_LOG = _pathlib.Path(os.environ["JOURNEY_ARGV_LOG"])


def _fake_run(argv, *args, **kwargs):
    _LOG.write_text(_json.dumps(list(argv)) + "\\n", encoding="utf-8")
    if list(argv[:2]) != ["metrifid", "certify"]:
        raise SystemExit(f"journey did not invoke `metrifid certify`: {list(argv)}")
    if "--output" not in argv:
        raise SystemExit(f"journey passed no --output: {list(argv)}")
    marker = argv.index("--output")
    inputs = [_pathlib.Path(value) for value in argv[2:marker]]
    if len(inputs) != 2:
        raise SystemExit(f"journey passed {len(inputs)} model arguments, expected 2: {list(argv)}")
    for value in inputs:
        if not value.is_file():
            raise SystemExit(f"journey passed a model argument that is not a file: {value}")
    output = _pathlib.Path(argv[marker + 1]).resolve()
    for value in inputs:
        root = value.resolve().parent
        if output == root or root in output.parents:
            raise SystemExit(
                f"the certify output {output} is inside the model root {root}; Metrifid refuses "
                "this with OUTPUT_PATH_INVALID / output_inside_model_root and exits 64"
            )
    output.mkdir(parents=True, exist_ok=True)
    (output / "certification.json").write_text("{}", encoding="utf-8")
    return _types.SimpleNamespace(returncode=0, stdout="", stderr="")


_subprocess.run = _fake_run
"""


def _run_journey(job: dict[str, Any], step_name: str, workdir: Path, *, source: str | None = None):
    """Execute one embedded journey against the fake console script."""
    log = workdir.parent / f"{workdir.name}-argv.json"
    completed = _run_validator(
        _inline_python(job, step_name) if source is None else source,
        workdir,
        {"JOURNEY_ARGV_LOG": str(log)},
        prelude=JOURNEY_FAKE,
    )
    return completed, log


@pytest.mark.parametrize("job_id", REGISTRY_VERIFY_JOBS)
def test_the_registry_journey_publishes_outside_the_model_root(
    jobs: dict[str, Any], job_id: str, tmp_path: Path
) -> None:
    """Each shipped journey runs, and the receipt it asks for lands outside both input roots."""
    workdir = tmp_path / "journey"
    workdir.mkdir()
    completed, log = _run_journey(jobs[job_id], JOURNEY_STEPS[job_id], workdir)
    assert completed.returncode == 0, completed.stderr
    assert "certify journey published" in completed.stdout, completed.stdout
    argv = json.loads(log.read_text(encoding="utf-8"))
    assert argv[:2] == ["metrifid", "certify"], argv


@pytest.mark.parametrize("job_id", REGISTRY_VERIFY_JOBS)
def test_moving_the_registry_journey_output_back_inside_the_model_root_fails(
    jobs: dict[str, Any], job_id: str, tmp_path: Path
) -> None:
    """The regression this contract exists for: the pre-correction shape must not pass."""
    source = _inline_python(jobs[job_id], JOURNEY_STEPS[job_id])
    reverted = source.replace(
        'receipt = pathlib.Path("receipt")', 'receipt = models / "receipt"', 1
    )
    assert reverted != source, "the mutation must actually change the journey"
    workdir = tmp_path / "reverted"
    workdir.mkdir()
    completed, _ = _run_journey(jobs[job_id], JOURNEY_STEPS[job_id], workdir, source=reverted)
    assert completed.returncode != 0
    assert "output_inside_model_root" in completed.stderr, completed.stderr
