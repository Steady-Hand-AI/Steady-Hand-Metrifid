"""Exact public namespace, typing marker, and metadata tests."""

from __future__ import annotations

import importlib.metadata
import importlib.resources
import subprocess
import sys
from pathlib import Path

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
    runtime = {
        _normalize_requirement(requirement)
        for requirement in (distribution.requires or [])
        if "extra ==" not in requirement
    }
    assert runtime == {("mujoco", ("==3.10.0.*",)), ("numpy", (">=1.26",))}


def test_the_declared_python_support_has_no_upper_bound() -> None:
    """Declare Python >=3.11 with no ceiling and classify 3.11 through 3.14."""
    import importlib.metadata as metadata

    meta = metadata.distribution("metrifid").metadata
    requires_python = str(meta["Requires-Python"]).replace(" ", "")
    assert _specifier_parts(requires_python) == (">=3.11",)
    classifiers = set(meta.get_all("Classifier") or [])
    for minor in ("3.11", "3.12", "3.13", "3.14"):
        assert f"Programming Language :: Python :: {minor}" in classifiers


def test_no_runtime_dependency_declares_an_upper_bound() -> None:
    """Keep every runtime dependency free of a ceiling except the exact MuJoCo engine family."""
    import importlib.metadata as metadata

    for requirement in metadata.distribution("metrifid").requires or []:
        if "extra ==" in requirement:
            continue
        name, specifiers = _normalize_requirement(requirement)
        for specifier in specifiers:
            if name == "mujoco":
                assert specifier == "==3.10.0.*"
            else:
                assert specifier.startswith(">="), (name, specifier)


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
