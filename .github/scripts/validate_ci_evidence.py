"""Decide whether one CI run produced complete, self-consistent release evidence.

This is the single decision authority for the release summary. The workflow only collects
artifacts and calls this module; every accept or reject judgement lives here, so a lane cannot be
waved through by a shell expression that swallowed its own exit code.

Three ideas do the real work.

The first is that a job result is not evidence. ``needs.<job>.result`` does report ``skipped``
honestly, but the status a conditionally skipped job reports to branch protection is a success, so
a required check must never be skippable and the summary must judge the results it is handed rather
than the fact that it was reached. Every required result is therefore checked explicitly, and every
expected lane is enumerated by name from the artifact it uploaded.

The second is that evidence must be bound to the lane that claims it. A smoke lane must not be
counted as a complete suite, and a document saying "Linux x86_64, Python 3.12" does not prove the
macOS arm64 Python 3.14 boundary no matter how well formed it is. Each lane has one exact
expectation here, and identity is compared against that expectation rather than merely being
present.

The third is that the run must say what it judged. A pull request builds a synthetic merge commit,
so the checked-out subject and the head a reviewer reads are different objects. The build receipt
records both and is checked against the trusted values of the current run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ElementTree
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

SUPPORT_TIER = "ADMITTED_CAPABILITY_COMPATIBLE_PROFILE"
OBJECT_ID = re.compile(r"[0-9a-f]{40}")

# One canonical numeric component: no leading zero, ASCII digits only.
_COMPONENT = r"(?:0|[1-9][0-9]*)"
_TRIPLET = rf"({_COMPONENT})\.({_COMPONENT})\.({_COMPONENT})"
# A local-version identifier is nonempty and alphanumeric; separators never repeat or trail.
_LOCAL = r"[0-9A-Za-z]+(?:[._-][0-9A-Za-z]+)*"
_POST = rf"(?:\.post{_COMPONENT})?"

# Four field-specific grammars. They look similar on purpose but they are not the same contract: a
# suffix admitted on the package field is not admitted on the base, native, or banner fields. Every
# one is applied with `fullmatch`, because `$` also matches before a terminal newline and these
# fields are exact strings, not lines.
MUJOCO_PACKAGE_VERSION = re.compile(rf"{_TRIPLET}{_POST}(?:\+{_LOCAL})?")
MUJOCO_BASE_VERSION = re.compile(_TRIPLET)
NUMPY_RELEASE_VERSION = re.compile(rf"{_TRIPLET}{_POST}(?:\+{_LOCAL})?")
# A `sys.version` continuation opens with one character that is neither Unicode whitespace nor an
# ASCII control or DEL, and carries no NUL anywhere; after that it is free-form interpreter text
# and may span lines.
_CONTINUATION = r"[^\s\x00-\x1f\x7f][^\x00]*"
# A CPython banner is the triplet alone, or the triplet, one ASCII space, and one continuation.
# No other separator opens a continuation, and a lone trailing space is not one.
PYTHON_VERSION_BANNER = re.compile(rf"{_TRIPLET}(?: {_CONTINUATION})?")

# No MuJoCo component may reach 1000: the native encoding packs each into three decimal digits.
MAX_COMPONENT = 999

# Floors the project already declares. They bound what a resolver-latest lane may report without
# pinning it to a version that will be stale the next time the resolver runs.
MINIMUM_MUJOCO = (3, 9)
MINIMUM_NUMPY = (1, 26)


@dataclass(frozen=True)
class Expectation:
    """Exactly what one evidence directory must prove about the runtime it ran on."""

    tier: str
    platform_system: str
    platform_machine: str
    python_major_minor: str
    # ``None`` means the lane deliberately resolves the newest supported release. That value is
    # required to be present and self-consistent, never pinned here: hard-coding a future version
    # would make a resolver lane look fixed and quietly stop testing what it exists to test.
    mujoco_version: str | None = None
    numpy_version: str | None = None


# The two default complete suites and the four boundary smokes, with the exact coordinate each one
# exists to prove. FULL_LANES and SMOKE_LANES are derived so a lane cannot drift between them.
LANE_EXPECTATIONS: Mapping[str, Expectation] = {
    "linux_x64_py312_full": Expectation("full", "Linux", "x86_64", "3.12"),
    "linux_x64_py311_numpy_min": Expectation(
        "full", "Linux", "x86_64", "3.11", mujoco_version="3.9.0", numpy_version="1.26.4"
    ),
    "linux_x64_py313": Expectation("smoke", "Linux", "x86_64", "3.13"),
    "linux_x64_py314": Expectation("smoke", "Linux", "x86_64", "3.14"),
    "macos_arm64_py314": Expectation("smoke", "Darwin", "arm64", "3.14"),
    "macos_x64_py312": Expectation("smoke", "Darwin", "x86_64", "3.12", mujoco_version="3.10.0"),
}
FULL_LANES: tuple[str, ...] = tuple(
    name for name, want in LANE_EXPECTATIONS.items() if want.tier == "full"
)
SMOKE_LANES: tuple[str, ...] = tuple(
    name for name, want in LANE_EXPECTATIONS.items() if want.tier == "smoke"
)

RETAINED_EXPECTATIONS: Mapping[str, Expectation] = {
    "prior_validated": Expectation("focused", "Linux", "x86_64", "3.12", mujoco_version="3.10.0"),
    "current_validated": Expectation("focused", "Linux", "x86_64", "3.12", mujoco_version="3.11.0"),
}
SDIST_EXPECTATION = Expectation("focused", "Linux", "x86_64", "3.12")
BUILD_EXPECTATION = Expectation("quality", "Linux", "x86_64", "3.11")

# Jobs whose failure must fail the release summary. ``build_and_quality`` is included directly:
# without it there is no original artifact and no review subject to bind anything to. The optional
# diagnostic job is deliberately absent.
REQUIRED_UPSTREAM: tuple[str, ...] = (
    "build_and_quality",
    "matrix_lane",
    "minimum_dependency_lane",
    "retained_compatibility_lane",
    "sdist_install_lane",
    "distribution_equivalence",
    "test_composite_action",
)

REQUIRED_IDENTITY_FIELDS: tuple[str, ...] = (
    "native_version_integer",
    "native_version_string",
    "numpy_version",
    "package_base_version",
    "package_version",
    "platform_machine",
    "platform_system",
    "python_major_minor",
    "python_version",
    "support_tier",
)

CHECKOUT_IDENTITY_FIELDS: tuple[str, ...] = (
    "checkout_sha",
    "checkout_tree",
    "event_name",
    "event_sha",
    "pull_request_head_sha",
    "pull_request_head_tree",
    "schema_version",
)
CHECKOUT_SCHEMA_VERSION = 1

# The only events this workflow accepts. A run reported under any other name is not a run this
# release contract has ever reasoned about, however well formed the rest of its evidence looks.
SUPPORTED_EVENTS: tuple[str, ...] = ("push", "pull_request", "workflow_dispatch")

PUBLIC_COMMANDS: tuple[str, ...] = (
    "certify",
    "review-model",
    "compare",
    "audit-timestep",
    "qualify-workload",
    "review-runtime",
    "run-runtime-review",
)

# Written by each lane immediately before it invokes one command's help surface.
COMMAND_MARKER_PREFIX = "### metrifid"

# Diagnostic evidence is optional by construction. The summary downloads it so that ignoring it is
# demonstrated rather than assumed, and never lets it stand in for a required lane.
DIAGNOSTIC_PREFIX = "diagnostic-"


class EvidenceError(AssertionError):
    """One CI run failed to prove something the release boundary requires."""


@dataclass(frozen=True)
class LaneReport:
    """What one lane proved, reduced to the facts the summary decides on."""

    lane_id: str
    tier: str
    installed_sha256: str
    test_count: int
    skipped: frozenset[str] = field(default_factory=frozenset)
    cases: frozenset[str] = field(default_factory=frozenset)


def _require(condition: bool, message: str) -> None:
    """Raise the one error type this module reports, so callers can catch exactly it."""
    if not condition:
        raise EvidenceError(message)


def _read_text(path: Path, description: str) -> str:
    """Read one required evidence file, naming it when it is absent."""
    _require(path.is_file(), f"missing {description}: {path}")
    return path.read_text(encoding="utf-8").strip()


def _read_json(path: Path, description: str) -> dict[str, object]:
    """Read one required evidence document."""
    text = _read_text(path, description)
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as error:
        raise EvidenceError(f"unreadable {description}: {path}: {error}") from error
    _require(isinstance(loaded, dict), f"{description} must be a JSON object: {path}")
    assert isinstance(loaded, dict)
    return loaded


def _exact_int(value: object, description: str) -> int:
    """Require a real integer. ``True`` equals ``1`` in Python, so a type check is the only check."""
    _require(type(value) is int, f"{description} must be an int, got {type(value).__name__}")
    assert isinstance(value, int)
    return value


def _triplet(
    pattern: re.Pattern[str], value: object, description: str, syntax: str
) -> tuple[int, int, int]:
    """Require one field's own syntax and return its numeric triplet."""
    text = _nonempty_string(value, description)
    match = pattern.fullmatch(text)
    _require(match is not None, f"{description} is not {syntax}: {text!r}")
    assert match is not None
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _mujoco_package_triplet(value: object, description: str) -> tuple[int, int, int]:
    """Parse an admitted MuJoCo package version, which may carry a post or local suffix."""
    triplet = _triplet(
        MUJOCO_PACKAGE_VERSION, value, description, "a canonical stable MuJoCo package version"
    )
    for part in triplet:
        _require(
            part <= MAX_COMPONENT,
            f"{description} has a component of {part}, which the native encoding cannot represent",
        )
    return triplet


def _mujoco_base_triplet(value: object, description: str) -> tuple[int, int, int]:
    """Parse a MuJoCo base or native string, which admits no suffix at all."""
    return _triplet(
        MUJOCO_BASE_VERSION, value, description, "an exact MAJOR.MINOR.PATCH MuJoCo version"
    )


def _numpy_release_triplet(value: object, description: str) -> tuple[int, int, int]:
    """Parse a canonical stable NumPy release."""
    return _triplet(NUMPY_RELEASE_VERSION, value, description, "a canonical stable NumPy release")


def _python_banner_triplet(value: object, description: str) -> tuple[int, int, int]:
    """Parse the leading triplet of a real CPython version banner."""
    triplet = _triplet(
        PYTHON_VERSION_BANNER,
        value,
        description,
        "a Python version banner with a canonical triplet",
    )
    for part in triplet:
        _require(part <= MAX_COMPONENT, f"{description} has an implausible component {part}")
    return triplet


def _at_least(triplet: tuple[int, int, int], floor: tuple[int, int], description: str) -> None:
    """Require a version to be at or above a declared project floor."""
    _require(
        triplet[:2] >= floor,
        f"{description} {'.'.join(str(part) for part in triplet)} is below the declared floor "
        f"{floor[0]}.{floor[1]}",
    )


def _nonempty_string(value: object, description: str) -> str:
    """Require one evidence field to be a real string rather than null or a number."""
    _require(isinstance(value, str), f"{description} must be a string, got {type(value).__name__}")
    assert isinstance(value, str)
    _require(bool(value.strip()), f"{description} must not be empty")
    return value


def validate_upstream_results(results: Mapping[str, str]) -> None:
    """Reject any required job that did not actually succeed.

    ``skipped`` and ``cancelled`` are rejected as firmly as ``failure``. The externally reported
    status of a conditionally skipped job is a success, so the summary decides on the results it is
    handed and never treats being reached as evidence that anything ran.
    """
    for job in REQUIRED_UPSTREAM:
        _require(job in results, f"required upstream job not reported: {job}")
        result = results[job]
        _require(result == "success", f"required upstream job {job} did not succeed: {result!r}")


def _junit_cases(path: Path, description: str) -> tuple[frozenset[str], frozenset[str]]:
    """Return every test identity and the skipped subset, refusing any failure or error."""
    _require(path.is_file(), f"missing {description}: {path}")
    tree = ElementTree.parse(path)
    cases: set[str] = set()
    skipped: set[str] = set()
    for case in tree.iter("testcase"):
        identity = f"{case.attrib.get('classname', '')}::{case.attrib.get('name', '')}"
        _require(identity not in cases, f"duplicate test identity in {description}: {identity}")
        cases.add(identity)
        _require(case.find("failure") is None, f"failed test in {description}: {identity}")
        _require(case.find("error") is None, f"errored test in {description}: {identity}")
        if case.find("skipped") is not None:
            skipped.add(identity)
    _require(bool(cases), f"{description} recorded no test at all: {path}")
    return frozenset(cases), frozenset(skipped)


def _validate_mujoco_identity(
    identity: Mapping[str, object], subject: str, want: Expectation
) -> None:
    """Require the three MuJoCo fields to describe one coherent runtime.

    The package field may carry a post or local suffix; the base and native strings may not. Both
    are compared as exact canonical strings, not only as parsed integers, so a suffixed base string
    cannot slip through on the strength of its numbers.
    """
    package = _mujoco_package_triplet(identity["package_version"], f"{subject} package_version")
    canonical = ".".join(str(part) for part in package)
    for name in ("package_base_version", "native_version_string"):
        _mujoco_base_triplet(identity[name], f"{subject} {name}")
        _require(
            identity[name] == canonical,
            f"{subject} {name} is {identity[name]!r}, not the canonical {canonical!r} of its "
            f"package version",
        )
    encoded = _exact_int(identity["native_version_integer"], f"{subject} native_version_integer")
    expected = package[0] * 1_000_000 + package[1] * 1_000 + package[2]
    _require(
        encoded == expected,
        f"{subject} native_version_integer is {encoded}, not {expected} for {canonical}",
    )
    if want.mujoco_version is not None:
        _require(
            identity["package_version"] == want.mujoco_version,
            f"{subject} ran MuJoCo {identity['package_version']!r}, not {want.mujoco_version!r}",
        )
    else:
        # A resolver-latest lane must report a real admitted runtime, but pinning it here would
        # freeze the very thing it exists to keep moving.
        _at_least(package, MINIMUM_MUJOCO, f"{subject} resolver-latest MuJoCo")


def _validate_python_identity(
    identity: Mapping[str, object], subject: str, want: Expectation
) -> None:
    """Require the Python banner, the recorded minor, and the lane expectation to agree."""
    triplet = _python_banner_triplet(identity["python_version"], f"{subject} python_version")
    minor = f"{triplet[0]}.{triplet[1]}"
    recorded = _nonempty_string(identity["python_major_minor"], f"{subject} python_major_minor")
    _require(
        minor == recorded,
        f"{subject} python_version reports {minor} but python_major_minor says {recorded}",
    )
    _require(
        recorded == want.python_major_minor,
        f"{subject} ran Python {recorded!r}, not {want.python_major_minor!r}",
    )


def _validate_identity(directory: Path, subject: str, want: Expectation) -> dict[str, object]:
    """Require the evidence to describe the exact runtime the subject claims to be."""
    identity = _read_json(directory / "runtime_identity.json", f"{subject} runtime identity")
    for name in REQUIRED_IDENTITY_FIELDS:
        _require(name in identity, f"{subject} runtime identity omits {name}")
    _require(
        _nonempty_string(identity["platform_system"], f"{subject} platform_system")
        == want.platform_system,
        f"{subject} ran on {identity['platform_system']!r}, not {want.platform_system!r}",
    )
    _require(
        _nonempty_string(identity["platform_machine"], f"{subject} platform_machine")
        == want.platform_machine,
        f"{subject} ran on {identity['platform_machine']!r}, not {want.platform_machine!r}",
    )
    _validate_python_identity(identity, subject, want)
    _validate_mujoco_identity(identity, subject, want)
    numpy = _numpy_release_triplet(identity["numpy_version"], f"{subject} numpy_version")
    if want.numpy_version is not None:
        _require(
            identity["numpy_version"] == want.numpy_version,
            f"{subject} ran NumPy {identity['numpy_version']!r}, not {want.numpy_version!r}",
        )
    else:
        _at_least(numpy, MINIMUM_NUMPY, f"{subject} resolver-latest NumPy")
    _require(
        identity["support_tier"] == SUPPORT_TIER,
        f"{subject} runtime is not an admitted profile: {identity}",
    )
    return identity


def _validate_command_coverage(directory: Path, subject: str) -> None:
    """Every lane must exercise all seven public command surfaces.

    The lane writes one ``### metrifid <command> --help`` marker immediately before each
    invocation. Searching for the bare command name would prove nothing: ``metrifid --help`` prints
    the whole subcommand list, so every name appears from a single invocation.
    """
    help_log = _read_text(directory / "cli_help.log", f"{subject} command help log")
    for command in PUBLIC_COMMANDS:
        marker = f"{COMMAND_MARKER_PREFIX} {command} --help"
        _require(marker in help_log, f"{subject} did not exercise the {command} command")


def _validate_original_binding(
    directory: Path, subject: str, manifest: Mapping[str, object], kind: str
) -> str:
    """Bind one lane to the original publication-candidate artifact it claims to have installed."""
    entry = manifest.get(kind)
    _require(isinstance(entry, dict), f"original manifest has no {kind} entry")
    assert isinstance(entry, dict)
    expected = entry.get("sha256")
    installed = _read_json(directory / "installed_artifact.json", f"{subject} installed artifact")
    _require(
        installed.get("kind") == kind,
        f"{subject} installed a {installed.get('kind')!r} where the summary expects {kind!r}",
    )
    actual = installed.get("sha256")
    _require(
        isinstance(actual, str) and actual == expected,
        f"{subject} did not install the original {kind}: {actual!r} != {expected!r}",
    )
    assert isinstance(actual, str)
    return actual


def _validate_lane(directory: Path, lane_id: str, manifest: Mapping[str, object]) -> LaneReport:
    """Validate one normal lane against its exact declared boundary."""
    want = LANE_EXPECTATIONS[lane_id]
    recorded_id = _read_text(directory / "lane_id.txt", f"{lane_id} lane id")
    _require(recorded_id == lane_id, f"lane directory {lane_id} records lane id {recorded_id!r}")
    recorded_tier = _read_text(directory / "tier.txt", f"{lane_id} tier")
    _require(
        recorded_tier == want.tier,
        f"lane {lane_id} declares tier {recorded_tier!r} where the summary requires {want.tier!r}",
    )
    digest = _validate_original_binding(directory, lane_id, manifest, "wheel")
    _validate_identity(directory, lane_id, want)
    _validate_command_coverage(directory, lane_id)
    if want.tier == "full":
        cases, skipped = _junit_cases(directory / "full.xml", f"{lane_id} complete suite")
        _require(
            not (directory / "focused.xml").exists(),
            f"complete lane {lane_id} also uploaded a focused report",
        )
        return LaneReport(lane_id, want.tier, digest, len(cases), skipped, cases)
    cases, skipped = _junit_cases(directory / "focused.xml", f"{lane_id} focused suite")
    # A smoke lane that uploaded a complete report is claiming a boundary it did not pay for.
    _require(
        not (directory / "full.xml").exists(),
        f"smoke lane {lane_id} uploaded a complete-suite report",
    )
    return LaneReport(lane_id, want.tier, digest, len(cases), skipped, frozenset())


def _validate_complete_inventory(reports: Mapping[str, LaneReport]) -> None:
    """Compare the complete test inventory only between the complete lanes."""
    full = [reports[lane] for lane in FULL_LANES]
    _require(len(full) == 2, f"expected exactly two complete lanes, found {len(full)}")
    first, second = full[0], full[1]
    _require(
        first.cases == second.cases,
        "the two complete lanes disagree on which tests exist: "
        f"{sorted(first.cases ^ second.cases)[:5]}",
    )
    _require(
        first.skipped == second.skipped,
        "the two complete lanes disagree on which tests were skipped: "
        f"{sorted(first.skipped ^ second.skipped)[:5]}",
    )


def _validate_retained(root: Path, manifest: Mapping[str, object]) -> None:
    """Require each retained exact profile to have validated its own exact tuple."""
    for role, want in RETAINED_EXPECTATIONS.items():
        directory = root / f"retained-{role}"
        _require(directory.is_dir(), f"missing retained evidence for {role}")
        _validate_original_binding(directory, f"retained-{role}", manifest, "wheel")
        _validate_identity(directory, f"retained-{role}", want)
        _validate_command_coverage(directory, f"retained-{role}")
        # Focused, and named focused: the retained lanes deliberately do not run the whole suite.
        _junit_cases(directory / "focused.xml", f"retained-{role} focused suite")
        result = _read_json(
            directory / "compatibility" / "compatibility_validation.json",
            f"retained-{role} compatibility validation",
        )
        _require(result.get("passed") is True, f"retained {role} validation did not pass")
        _require(result.get("profile_role") == role, f"retained {role} reports another role")
        _require(
            result.get("expected_mujoco_package_version") == want.mujoco_version,
            f"retained {role} validated the wrong MuJoCo version",
        )
        retained = result.get("retained_exact_profile_validation")
        _require(isinstance(retained, dict), f"retained {role} omits its exact profile validation")
        assert isinstance(retained, dict)
        _require(
            retained.get("validation_tier") == "VALIDATED_EXACT_PROFILE",
            f"retained {role} is not a validated exact profile",
        )
        exact_tuple = retained.get("exact_profile_tuple")
        _require(isinstance(exact_tuple, dict), f"retained {role} omits its exact profile tuple")
        encoded = json.dumps(
            exact_tuple, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        _require(
            hashlib.sha256(encoded).hexdigest() == retained.get("exact_profile_tuple_sha256"),
            f"retained {role} exact profile tuple does not match its recorded digest",
        )


def _validate_sdist_lane(root: Path, manifest: Mapping[str, object]) -> None:
    """Require the direct-sdist lane to have installed the original sdist and run focused suites."""
    directory = root / "sdist-install-evidence"
    _require(directory.is_dir(), "missing direct-sdist evidence")
    _validate_original_binding(directory, "sdist_install_lane", manifest, "sdist")
    _validate_identity(directory, "sdist_install_lane", SDIST_EXPECTATION)
    _validate_command_coverage(directory, "sdist_install_lane")
    _junit_cases(directory / "focused.xml", "sdist_install_lane focused suite")
    _require(
        not (directory / "full.xml").exists(),
        "the direct-sdist lane uploaded a complete-suite report",
    )


def validate_review_subject(
    directory: Path, event_name: str, event_sha: str, pull_request_head_sha: str
) -> dict[str, object]:
    """Validate the build receipt that says which subject this run judged.

    A pull request is built from a synthetic merge commit, so the checked-out object and the head a
    reviewer reads are different. Both are recorded, and both are checked against the trusted values
    of the current run rather than against anything the receipt asserts about itself.
    """
    _require(
        event_name in SUPPORTED_EVENTS,
        f"this run reports an unsupported event: {event_name!r}",
    )
    receipt = _read_json(directory / "checkout_identity.json", "review subject receipt")
    _require(
        tuple(sorted(receipt)) == CHECKOUT_IDENTITY_FIELDS,
        f"review subject receipt has the wrong key set: {sorted(receipt)}",
    )
    # ``True == 1`` in Python, so the type check has to come first or a boolean walks straight in.
    schema = _exact_int(receipt["schema_version"], "review subject schema_version")
    _require(
        schema == CHECKOUT_SCHEMA_VERSION,
        f"unsupported review subject schema: {receipt['schema_version']!r}",
    )
    _require(
        receipt["event_name"] in SUPPORTED_EVENTS,
        f"the receipt records an unsupported event: {receipt['event_name']!r}",
    )
    _require(
        receipt["event_name"] == event_name,
        f"receipt event {receipt['event_name']!r} is not this run's {event_name!r}",
    )
    for name in ("event_sha", "checkout_sha", "checkout_tree"):
        value = _nonempty_string(receipt[name], f"review subject {name}")
        _require(
            OBJECT_ID.fullmatch(value) is not None, f"review subject {name} is not an object id"
        )
    _require(
        receipt["event_sha"] == event_sha,
        f"receipt event sha {receipt['event_sha']!r} is not this run's {event_sha!r}",
    )
    # The workflow asserts this in-job too; re-deciding it here keeps the summary self-contained.
    _require(
        receipt["checkout_sha"] == receipt["event_sha"],
        "the checked-out commit is not the commit this event dispatched",
    )
    head_sha = receipt["pull_request_head_sha"]
    head_tree = receipt["pull_request_head_tree"]
    if event_name == "pull_request":
        pairs: tuple[tuple[str, object], ...] = (
            ("head sha", head_sha),
            ("head tree", head_tree),
        )
        for label, candidate in pairs:
            text = _nonempty_string(candidate, f"review subject pull request {label}")
            _require(
                OBJECT_ID.fullmatch(text) is not None,
                f"review subject pull request {label} is not an object id",
            )
        _require(
            head_sha == pull_request_head_sha,
            f"receipt head {head_sha!r} is not this run's head {pull_request_head_sha!r}",
        )
    else:
        _require(
            head_sha is None and head_tree is None,
            f"a {event_name} run must record null pull-request identity, got {head_sha!r}",
        )
    return dict(receipt)


def diagnostic_names(names: Iterable[str]) -> list[str]:
    """Return the optional diagnostic artifacts, which never gate default success.

    The summary downloads these deliberately so that ignoring them is demonstrated rather than
    assumed: a failed diagnostic cell is present in the evidence tree and still cannot affect the
    decision, because nothing below ever reads it.
    """
    return sorted(name for name in names if name.startswith(DIAGNOSTIC_PREFIX))


def validate(
    root: Path,
    upstream: Mapping[str, str],
    event_name: str,
    event_sha: str,
    pull_request_head_sha: str,
) -> dict[str, object]:
    """Validate one complete CI evidence tree and return the accepted summary."""
    validate_upstream_results(upstream)
    original = root / "original-artifacts"
    manifest = _read_json(original / "original_manifest.json", "original manifest")
    for kind in ("wheel", "sdist"):
        entry = manifest.get(kind)
        _require(isinstance(entry, dict), f"original manifest has no {kind} entry")
        assert isinstance(entry, dict)
        for name in ("filename", "size_bytes", "sha256"):
            _require(name in entry, f"original manifest {kind} entry omits {name}")

    quality = original / "quality-evidence"
    _validate_identity(quality, "build_and_quality", BUILD_EXPECTATION)
    subject = validate_review_subject(quality, event_name, event_sha, pull_request_head_sha)

    reports = {
        lane_id: _validate_lane(root / f"lane-{lane_id}", lane_id, manifest)
        for lane_id in LANE_EXPECTATIONS
    }
    present = {path.name.removeprefix("lane-") for path in root.glob("lane-*") if path.is_dir()}
    expected = set(LANE_EXPECTATIONS)
    _require(
        present == expected,
        f"normal lane set is wrong: unexpected {sorted(present - expected)}, "
        f"missing {sorted(expected - present)}",
    )
    _validate_complete_inventory(reports)
    _validate_retained(root, manifest)
    _validate_sdist_lane(root, manifest)

    digests = {report.installed_sha256 for report in reports.values()}
    _require(len(digests) == 1, f"lanes installed more than one wheel: {sorted(digests)}")
    return {
        "accepted_subject": subject,
        "full_lanes": list(FULL_LANES),
        "smoke_lanes": list(SMOKE_LANES),
        "original_wheel_sha256": digests.pop(),
        "complete_suite_test_count": reports[FULL_LANES[0]].test_count,
        "ignored_diagnostic_artifacts": diagnostic_names(
            path.name for path in root.glob("*") if path.is_dir()
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the downloaded evidence tree named on the command line."""
    parser = argparse.ArgumentParser(description="Decide one CI run's release evidence.")
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument(
        "--upstream-results",
        required=True,
        help="JSON object mapping each required job id to its needs result",
    )
    parser.add_argument("--event-name", required=True)
    parser.add_argument("--event-sha", required=True)
    parser.add_argument("--pull-request-head-sha", default="")
    arguments = parser.parse_args(argv)
    try:
        decoded = json.loads(arguments.upstream_results)
    except json.JSONDecodeError as error:
        sys.stderr.write(f"upstream results are not JSON: {error}\n")
        return 2
    if not isinstance(decoded, dict):
        sys.stderr.write("upstream results must be a JSON object\n")
        return 2
    upstream = {str(key): str(value) for key, value in decoded.items()}
    try:
        summary = validate(
            arguments.evidence_root,
            upstream,
            arguments.event_name,
            arguments.event_sha,
            arguments.pull_request_head_sha,
        )
    except EvidenceError as error:
        sys.stderr.write(f"RELEASE EVIDENCE REJECTED: {error}\n")
        return 1
    sys.stdout.write(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    sys.stdout.write("PASS_METRIFID_RELEASE_MATRIX\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
