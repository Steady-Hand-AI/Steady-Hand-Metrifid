"""Exact public namespace, typing marker, and metadata tests."""

from __future__ import annotations

import importlib.metadata
import importlib.resources
import json
import os
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

import pytest
import pytest_check as check
from packaging.markers import Marker
from packaging.requirements import Requirement

import metrifid
from metrifid.version import __version__ as CURRENT_VERSION

EXPECTED_PUBLIC = [
    "__version__",
    "Binary64",
    "ExactRational",
    "canonical_json_bytes",
    "canonical_sha256",
    "strict_json_loads",
    "ComparisonStatus",
    "EngineThreadpoolState",
    "LimitationCode",
    "OperationalExitCode",
    "ReasonCode",
    "ReasonRecord",
    "STATUS_PRECEDENCE",
    "REASON_REGISTRY",
    "ComparisonConfig",
    "ComparisonContractIdentity",
    "ComparisonReceipt",
    "finalize_receipt",
    "validate_receipt",
    "OperationalStage",
    "OperationalReasonCode",
    "OperationalFailure",
    "OperationalToolObservation",
    "InputDigestCode",
    "InputDigest",
    "write_state_artifact",
    "write_actions_artifact",
]


def test_public_exports_include_supported_names_without_private_leaks() -> None:
    """Expose every supported public name and keep the advertised surface curated.

    Submodules are deliberately reachable as normal package attributes: Python binds a submodule on
    its parent package whenever the submodule is imported, and suppressing that broke the ordinary
    ``import metrifid.errors`` idiom. The curated surface is expressed by ``__all__`` and
    ``__dir__`` instead, and those must not advertise implementation modules.
    """
    assert set(EXPECTED_PUBLIC).issubset(metrifid.__all__)
    assert all(hasattr(metrifid, name) for name in EXPECTED_PUBLIC)
    assert not any(name.startswith("_") and name != "__version__" for name in metrifid.__all__)
    implementation_modules = (
        "json_values",
        "errors",
        "operational",
        "schemas",
        "version",
        "workload_writers",
    )
    assert not any(name in metrifid.__all__ for name in implementation_modules)
    assert not any(name in metrifid.__dir__() for name in implementation_modules)


def test_submodules_behave_like_normal_python_package_attributes() -> None:
    """Support the ordinary ``import package.module`` then ``package.module`` idiom."""
    import metrifid.errors
    import metrifid.schemas

    assert metrifid.errors is not None
    assert metrifid.schemas is not None
    assert metrifid.errors is sys.modules["metrifid.errors"]
    assert metrifid.schemas is sys.modules["metrifid.schemas"]


def _specifier_parts(value: str) -> tuple[str, ...]:
    """Split one comma-separated specifier into sorted, order-independent parts."""
    return tuple(sorted(part.strip() for part in value.split(",") if part.strip()))


def _normalize_requirement(requirement: str) -> tuple[str, tuple[str, ...]]:
    """Return one requirement as a distribution name and its order-independent specifiers."""
    token = requirement.split(";")[0].strip()
    positions = [index for index in (token.find(c) for c in "<>=!~") if index >= 0]
    cut = min(positions) if positions else len(token)
    name = token[:cut].strip().lower().replace("_", "-")
    return name, _specifier_parts(token[cut:].strip())


# The supported platform classes this project publishes for. Upstream MuJoCo ships Darwin x86_64
# wheels only below 3.11, so exactly one of these four resolves under a ceiling.
_INTEL_MACOS: dict[str, str] = {"platform_system": "Darwin", "platform_machine": "x86_64"}
_SUPPORTED_ENVIRONMENTS: dict[str, dict[str, str]] = {
    "intel macOS": _INTEL_MACOS,
    "Apple silicon macOS": {"platform_system": "Darwin", "platform_machine": "arm64"},
    "Linux x86_64": {"platform_system": "Linux", "platform_machine": "x86_64"},
    "Linux aarch64": {"platform_system": "Linux", "platform_machine": "aarch64"},
}


def _runtime_requirements(requirements: Iterable[str]) -> list[str]:
    """Return the runtime requirements, excluding every optional extra."""
    return [requirement for requirement in requirements if "extra ==" not in requirement]


def _requirement_marker(requirement: str) -> Marker | None:
    """Return the PEP 508 environment marker of one requirement, if it carries one."""
    return Requirement(requirement).marker


def _active_mujoco_specifiers(
    requirements: Iterable[str], environment: Mapping[str, str]
) -> tuple[str, ...] | None:
    """Return the specifiers of the single MuJoCo requirement active in one environment.

    ``None`` means the requirement set is not well formed for that environment: either no MuJoCo
    requirement applies, or more than one does. Both are contract failures, and reporting them the
    same way lets the mutation controls below assert against real behaviour rather than against a
    string edit.
    """
    active = [
        requirement
        for requirement in _runtime_requirements(requirements)
        if _normalize_requirement(requirement)[0] == "mujoco"
        and ((marker := _requirement_marker(requirement)) is None or marker.evaluate(environment))
    ]
    if len(active) != 1:
        return None
    return tuple(sorted(_normalize_requirement(active[0])[1]))


def _mujoco_resolution_problems(requirements: Iterable[str]) -> list[str]:
    """Return one problem per supported platform whose MuJoCo resolution is wrong.

    This is the whole contract in one function so it can be run against the real published
    metadata and against deliberately broken variants of it. A control that only proved a string
    edit changed a string would prove nothing about resolution.
    """
    requirements = list(requirements)
    problems: list[str] = []
    for label, environment in _SUPPORTED_ENVIRONMENTS.items():
        active = _active_mujoco_specifiers(requirements, environment)
        if active is None:
            problems.append(f"{label}: exactly one MuJoCo requirement must apply, none or many do")
            continue
        expected = ("<3.11", ">=3.9") if environment is _INTEL_MACOS else (">=3.9",)
        if active != expected:
            problems.append(f"{label}: MuJoCo resolves to {active}, expected {expected}")
    return problems


def test_intel_macos_resolves_mujoco_under_the_published_ceiling() -> None:
    """Darwin x86_64 must activate exactly >=3.9,<3.11 and nothing else."""
    import importlib.metadata as metadata

    requirements = metadata.distribution("metrifid").requires or []
    assert _active_mujoco_specifiers(requirements, _INTEL_MACOS) == ("<3.11", ">=3.9")


def test_every_other_supported_platform_resolves_mujoco_without_a_ceiling() -> None:
    """Linux and Apple silicon must activate exactly >=3.9, with no upper bound."""
    import importlib.metadata as metadata

    requirements = metadata.distribution("metrifid").requires or []
    for label, environment in _SUPPORTED_ENVIRONMENTS.items():
        if environment is _INTEL_MACOS:
            continue
        assert _active_mujoco_specifiers(requirements, environment) == (">=3.9",), label


def test_the_two_mujoco_markers_are_exclusive_and_exhaustive() -> None:
    """Exactly one MuJoCo requirement applies on every supported platform class."""
    import importlib.metadata as metadata

    requirements = metadata.distribution("metrifid").requires or []
    assert _mujoco_resolution_problems(requirements) == []


@pytest.mark.parametrize(
    ("mutation", "description"),
    [
        (
            lambda text: text.replace(
                "platform_system != 'Darwin' or platform_machine != 'x86_64'",
                "platform_system != 'Darwin' and platform_machine != 'x86_64'",
            ),
            "the complement uses and instead of or",
        ),
        (lambda text: text.replace("<3.11,", ""), "the Intel ceiling is removed"),
        (lambda text: text.replace("<3.11", "<4.0"), "the Intel ceiling is raised past 3.11"),
        (lambda text: text.replace("<3.11", "<3.12"), "the Intel ceiling is raised to 3.12"),
        (
            lambda text: text.replace(
                "mujoco>=3.9; platform_system !=", "mujoco<3.11,>=3.9; platform_system !="
            ),
            "the ceiling leaks onto every other platform",
        ),
        (
            lambda text: text.replace(
                "; platform_system == 'Darwin' and platform_machine == 'x86_64'", ""
            ),
            "the Intel requirement loses its marker and applies everywhere",
        ),
    ],
    ids=[
        "and-instead-of-or",
        "ceiling-removed",
        "ceiling-raised-to-4",
        "ceiling-raised-to-312",
        "ceiling-leaks",
        "marker-dropped",
    ],
)
def test_breaking_the_mujoco_markers_is_detected(
    mutation: Callable[[str], str], description: str
) -> None:
    """Each way of breaking the declared resolution must be caught by the same contract.

    The mutation is applied to the real published requirements and the result is fed back through
    the very function the positive tests use, so a passing control means the contract rejects the
    broken metadata, not merely that a substring changed.
    """
    import importlib.metadata as metadata

    real = _runtime_requirements(metadata.distribution("metrifid").requires or [])
    assert _mujoco_resolution_problems(real) == [], "the shipped metadata must be clean first"
    mutated = [mutation(requirement) for requirement in real]
    assert mutated != real, f"the mutation did not change anything: {description}"
    assert _mujoco_resolution_problems(mutated), description


def test_version_matches_installed_metadata() -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises version matches installed metadata; the observable command or import
    contract is pinned without relying on repository layout.
    """
    assert metrifid.__version__ == CURRENT_VERSION
    assert importlib.metadata.version("metrifid") == metrifid.__version__


def test_py_typed_is_installed() -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises py typed is installed; the observable command or import contract is
    pinned without relying on repository layout.
    """
    marker = importlib.resources.files("metrifid").joinpath("py.typed")
    assert marker.is_file()


def test_separate_consumer_strict_type_check(tmp_path: Path) -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises separate consumer strict type check; the observable command or
    import contract is pinned without relying on repository layout.
    """
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "from metrifid import (\n"
        "    Binary64, ComparisonConfig, ComparisonStatus, ExactRational,\n"
        "    EngineThreadpoolState, LimitationCode, OperationalFailure,\n"
        "    canonical_json_bytes, canonical_sha256, strict_json_loads,\n"
        ")\n"
        "binary: Binary64 = Binary64.from_float(1.0)\n"
        "rational: ExactRational = ExactRational.from_decimal_token('0.01')\n"
        "status: ComparisonStatus = ComparisonStatus.COVERAGE_INSUFFICIENT\n"
        "threadpool: EngineThreadpoolState = EngineThreadpoolState.DISABLED\n"
        "limitation: LimitationCode = LimitationCode.DECLARED_WORKLOAD_ONLY\n"
        "payload: bytes = canonical_json_bytes({'x': binary.to_primitive()})\n"
        "digest: str = canonical_sha256({'r': rational.to_primitive()})\n"
        "loaded: object = strict_json_loads(payload)\n"
        "config_type: type[ComparisonConfig] = ComparisonConfig\n"
        "failure_type: type[OperationalFailure] = OperationalFailure\n"
        "assert status and threadpool and limitation and digest and loaded and config_type and failure_type\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", str(consumer)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_the_installed_commands_and_runtime_dependencies_are_declared() -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises the installed commands and runtime dependencies are declared; the
    observable command or import contract is pinned without relying on repository layout.
    """
    import importlib.metadata as metadata

    distribution = metadata.distribution("metrifid")
    scripts = {
        entry.name: entry.value
        for entry in distribution.entry_points
        if entry.group == "console_scripts"
    }
    assert scripts == {"metrifid": "metrifid.cli:main"}
    runtime = _runtime_requirements(distribution.requires or [])
    assert {_normalize_requirement(requirement)[0] for requirement in runtime} == {
        "mujoco",
        "numpy",
    }
    assert [r for r in runtime if _normalize_requirement(r)[0] == "numpy"] == ["numpy>=1.26"]
    # MuJoCo is declared as two complementary environment-marked requirements; exactly which one
    # applies is the subject of the platform-resolution contracts below.
    assert len([r for r in runtime if _normalize_requirement(r)[0] == "mujoco"]) == 2


def test_the_declared_python_support_has_no_upper_bound() -> None:
    """Declare Python >=3.11 with no ceiling and classify 3.11 through 3.14."""
    import importlib.metadata as metadata

    meta = metadata.distribution("metrifid").metadata
    requires_python = str(meta["Requires-Python"]).replace(" ", "")
    assert _specifier_parts(requires_python) == (">=3.11",)
    classifiers = set(meta.get_all("Classifier") or [])
    for minor in ("3.11", "3.12", "3.13", "3.14"):
        assert f"Programming Language :: Python :: {minor}" in classifiers


def test_only_the_intel_macos_mujoco_requirement_declares_a_ceiling() -> None:
    """Keep every runtime dependency minimum-only except the one reproduced platform exception.

    Upstream publishes no Darwin x86_64 MuJoCo wheel at or above 3.11, so an ordinary install
    there resolves to a source archive that cannot build. That one requirement carries a ceiling;
    nothing else may, and the ceiling may not apply anywhere else.
    """
    import importlib.metadata as metadata

    for requirement in _runtime_requirements(metadata.distribution("metrifid").requires or []):
        name, specifiers = _normalize_requirement(requirement)
        bounded = [s for s in specifiers if not s.startswith(">=")]
        if not bounded:
            continue
        assert name == "mujoco", (name, specifiers)
        assert bounded == ["<3.11"], specifiers
        marker = _requirement_marker(requirement)
        assert marker is not None, requirement
        assert marker.evaluate(_INTEL_MACOS), requirement
        for label, environment in _SUPPORTED_ENVIRONMENTS.items():
            if environment is _INTEL_MACOS:
                continue
            assert not marker.evaluate(environment), (label, requirement)


def test_the_development_extra_declares_no_ceiling_and_no_numpy() -> None:
    """Keep the development extra on minimum-only specifiers and free of a NumPy duplicate."""
    import importlib.metadata as metadata

    dev = [
        requirement
        for requirement in metadata.distribution("metrifid").requires or []
        if "extra ==" in requirement and "dev" in requirement
    ]
    assert dev, "the dev extra is not declared"
    for requirement in dev:
        name, specifiers = _normalize_requirement(requirement.split(";")[0])
        assert name != "numpy", "the dev extra duplicates the NumPy runtime dependency"
        for specifier in specifiers:
            assert specifier.startswith(">="), (name, specifier)


def test_the_public_workload_writers_remain_importable() -> None:
    """Protect the supported user interface from accidental drift.

    This scenario exercises the public workload writers remain importable; the observable
    command or import contract is pinned without relying on repository layout.
    """
    from metrifid import write_actions_artifact, write_state_artifact

    assert callable(write_state_artifact)
    assert callable(write_actions_artifact)
    assert {"write_state_artifact", "write_actions_artifact"} <= set(metrifid.__all__)


# --- Workload Qualification supported submodule surface ----------------------------------------
#
# `metrifid.workload_qualification.__all__` is a compatibility commitment, so it is pinned exactly
# rather than by subset. Everything here runs against the installed distribution.

WORKLOAD_QUALIFICATION_PUBLIC = [
    "QualificationExitCode",
    "QualificationResult",
    "QualificationStatus",
    "WorkloadQualificationOperationError",
    "load_and_validate_workload_qualification_receipt",
    "qualify_configuration_file",
]

WORKLOAD_QUALIFICATION_WITHDRAWN = (
    "CellOutcome",
    "MAX_PROBE_GROUPS",
    "MAX_SUBSETS",
    "MAX_VARIANTS",
    "MAX_WORKLOADS",
    "MIN_PROBE_GROUPS",
    "MIN_VARIANTS",
    "MIN_WORKLOADS",
    "ProbeGroup",
    "ProbeGroupStatus",
    "ProbeVariant",
    "QUALIFICATION_LIMITATIONS",
    "QualificationConfig",
    "QualificationLimitationCode",
    "REQUIRED_BUDGET",
    "WorkloadCandidate",
    "planned_comparisons",
    "qualification_exit_code",
    "validate_qualification_receipt",
)


def test_the_workload_qualification_surface_is_exactly_the_six_supported_names() -> None:
    """Pin the supported submodule surface exactly, in its declared order."""
    import metrifid.workload_qualification as qualification

    check.equal(
        list(qualification.__all__),
        WORKLOAD_QUALIFICATION_PUBLIC,
        "the workload_qualification __all__ is not the exact frozen six-name list",
    )
    for name in WORKLOAD_QUALIFICATION_PUBLIC:
        check.is_true(
            hasattr(qualification, name),
            f"the supported name {name!r} does not resolve on the installed package",
        )


def test_a_star_import_exposes_exactly_the_supported_names() -> None:
    """`from ... import *` must bind the six supported names and nothing else."""
    namespace: dict[str, object] = {}
    exec("from metrifid.workload_qualification import *", namespace)
    bound = sorted(name for name in namespace if not name.startswith("__"))
    check.equal(
        bound,
        sorted(WORKLOAD_QUALIFICATION_PUBLIC),
        "a star import bound a different set of names than the supported surface",
    )


def test_the_withdrawn_names_are_not_public_attributes() -> None:
    """Names withdrawn from the public surface must not resolve on the package."""
    import metrifid.workload_qualification as qualification

    for name in WORKLOAD_QUALIFICATION_WITHDRAWN:
        check.is_false(
            hasattr(qualification, name),
            f"{name!r} is still reachable as a public attribute after being withdrawn",
        )
        check.is_not_in(
            name,
            qualification.__all__,
            f"{name!r} is still advertised in __all__ after being withdrawn",
        )


def test_importing_the_submodule_has_no_native_or_filesystem_side_effect(tmp_path: Path) -> None:
    """Importing the package must not load MuJoCo or perform filesystem or network work.

    A child process installs an audit hook before the import and records the events that would mean
    real work: opening a file for writing, creating or removing a path, or touching a socket.
    Reading the module's own source is what the import system does for every module and is not
    counted. This observes the real import rather than trusting the module source.
    """
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import json, sys\n"
        "side_effects = []\n"
        "_MUTATING = (\n"
        "    'os.mkdir', 'os.rmdir', 'os.remove', 'os.rename', 'os.symlink', 'os.link',\n"
        "    'os.truncate', 'shutil.copyfile', 'shutil.move', 'tempfile.mkstemp',\n"
        ")\n"
        "def _hook(name, args):\n"
        "    if name == 'open':\n"
        "        mode = args[1] if len(args) > 1 else None\n"
        "        if isinstance(mode, str) and any(flag in mode for flag in 'wxa+'):\n"
        "            side_effects.append([name, str(args[0])])\n"
        "    elif name in _MUTATING or name.startswith('socket.'):\n"
        "        side_effects.append([name, str(args[0]) if args else ''])\n"
        "sys.addaudithook(_hook)\n"
        "before = set(sys.modules)\n"
        "import metrifid.workload_qualification as q\n"
        "after = set(sys.modules)\n"
        "print(json.dumps({\n"
        "    'mujoco': any(m == 'mujoco' or m.startswith('mujoco.') for m in after),\n"
        "    'numpy': any(m == 'numpy' or m.startswith('numpy.') for m in after - before),\n"
        "    'side_effects': side_effects[:20],\n"
        "    'names': list(q.__all__),\n"
        "}))\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [sys.executable, str(probe)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    observed = json.loads(result.stdout)
    check.is_false(observed["mujoco"], "importing the submodule eagerly imported MuJoCo")
    check.is_false(observed["numpy"], "importing the submodule eagerly imported NumPy")
    check.equal(
        observed["side_effects"],
        [],
        f"importing the submodule performed filesystem or network work: {observed['side_effects']}",
    )
    check.equal(
        observed["names"],
        WORKLOAD_QUALIFICATION_PUBLIC,
        "the child process observed a different supported surface",
    )


def test_a_separate_strict_mypy_consumer_uses_all_six_supported_names(tmp_path: Path) -> None:
    """A consumer outside the repository, without PYTHONPATH, type-checks against the wheel."""
    consumer = tmp_path / "consumer.py"
    consumer.write_text(
        "from pathlib import Path\n"
        "from metrifid.workload_qualification import (\n"
        "    QualificationExitCode,\n"
        "    QualificationResult,\n"
        "    QualificationStatus,\n"
        "    WorkloadQualificationOperationError,\n"
        "    load_and_validate_workload_qualification_receipt,\n"
        "    qualify_configuration_file,\n"
        ")\n"
        "\n"
        "def run(configuration: Path) -> tuple[QualificationStatus, int]:\n"
        '    """Run one campaign and return its completed status and exit code."""\n'
        "    try:\n"
        "        result: QualificationResult = qualify_configuration_file(configuration)\n"
        "    except WorkloadQualificationOperationError:\n"
        "        return QualificationStatus.UNRESOLVED, int(QualificationExitCode.UNRESOLVED)\n"
        "    return result.status, result.exit_code\n"
        "\n"
        "def reload(receipt: Path) -> str:\n"
        '    """Validate one published receipt and return its recorded status."""\n'
        "    document = load_and_validate_workload_qualification_receipt(receipt)\n"
        "    return str(document['status'])\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-m", "mypy", "--strict", str(consumer)],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=environment,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_result_reports_the_exit_code_its_registry_declares_for_every_status() -> None:
    """A real result's `exit_code` must agree with the registry, for every completed status.

    Comparing the two registries to each other proves nothing: they would move together. What has to
    hold is that the value `QualificationResult.exit_code` actually returns is the one
    `QualificationExitCode` declares for that result's status, so a real result object is built for
    each status and its property is read.
    """
    from metrifid.workload_qualification import (
        QualificationExitCode,
        QualificationResult,
        QualificationStatus,
    )

    statuses = list(QualificationStatus)
    assert statuses, "the status registry is empty, so there is nothing to check"

    for status in statuses:
        check.is_true(
            hasattr(QualificationExitCode, status.name),
            f"the exit-code registry has no member for status {status.name}",
        )
        declared = int(getattr(QualificationExitCode, status.name))
        result = QualificationResult(
            status=status,
            receipt={},
            receipt_sha256="0" * 64,
            qualification_json=Path("workload_qualification.json"),
            qualification_markdown=Path("workload_qualification.md"),
        )
        check.equal(
            result.exit_code,
            declared,
            f"a result whose status is {status.name} reports exit code {result.exit_code}, but "
            f"QualificationExitCode declares {declared}",
        )
        check.is_(
            result.status,
            status,
            f"a result built with status {status.name} did not report that status back",
        )
        check.is_in(
            declared,
            (0, 20, 30),
            f"status {status.name} maps to an exit code outside the frozen set",
        )
