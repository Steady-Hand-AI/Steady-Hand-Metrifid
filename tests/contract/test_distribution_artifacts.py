"""The two published artifacts must carry the same package and the same promises.

A user may install the wheel from PyPI or build from the source distribution. Those two routes have
to deliver the same importable package, the same console entry point, and the same declared
metadata. These tests build both from the candidate source and compare them semantically — not by
archive bytes, which legitimately differ in timestamps and generator strings.

Every build here is isolated PEP 517, which is what a real consumer's `pip install` performs.
"""

from __future__ import annotations

import ast
import hashlib
import re
import subprocess
import sys
import tarfile
import zipfile
from email.message import Message
from email.parser import BytesParser
from pathlib import Path

import pytest
from packaging.requirements import Requirement

_REQUIRED_SDIST_MEMBERS = (
    "README.md",
    "CHANGELOG.md",
    "CONTRIBUTING.md",
    "DCO",
    "LICENSE",
    "NOTICE",
    "SECURITY.md",
    "CODE_OF_CONDUCT.md",
    ".pre-commit-config.yaml",
    ".github/quality-constraints.txt",
    "pyproject.toml",
    "tools/mjb_characterization.py",
    "tools/native_upgrade_profile_worker.py",
)
_COMPARED_METADATA_FIELDS = ("Name", "Version", "Summary", "Requires-Python")
# The supported platform classes the project publishes for. Exactly one of them resolves MuJoCo
# under a ceiling, because upstream publishes no Darwin x86_64 wheel at or above 3.11.
_SUPPORTED_PLATFORMS = {
    "intel macOS": {"platform_system": "Darwin", "platform_machine": "x86_64"},
    "Apple silicon macOS": {"platform_system": "Darwin", "platform_machine": "arm64"},
    "Linux x86_64": {"platform_system": "Linux", "platform_machine": "x86_64"},
    "Linux aarch64": {"platform_system": "Linux", "platform_machine": "aarch64"},
}
_FROZEN_WORKER_RESOURCE = "metrifid/runtime_review/native_evidence_worker.py.txt"
_FROZEN_WORKER_SHA256 = "b00e509a344593806c088c4e49783ed71bacd815466d74bce9e27c931535b4ff"


def _repository_root() -> Path:
    """Return the source checkout to build from."""
    return Path(__file__).resolve().parents[2]


def _authoritative_source_version() -> str:
    """Read the package version from its single source-controlled assignment."""
    version_path = _repository_root() / "src" / "metrifid" / "version.py"
    module = ast.parse(version_path.read_text(encoding="utf-8"), filename=str(version_path))
    values: list[str] = []
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in statement.targets
        ):
            continue
        value = ast.literal_eval(statement.value)
        assert isinstance(value, str)
        values.append(value)
    assert len(values) == 1, values
    return values[0]


def _build(source: Path, outdir: Path, *, wheel_only: bool = False) -> None:
    """Build distributions from one source tree through isolated PEP 517.

    A failed build fails this gate. Treating a nonzero exit as a skip made every backend, metadata,
    packaging-membership and dependency-provisioning failure look like a green run, because a
    skipped test still exits zero. An environment that cannot provision the declared build
    requirements has not satisfied this contract; it has failed to run it.
    """
    command = [sys.executable, "-m", "build"]
    if wheel_only:
        command.append("--wheel")
    command += ["--outdir", str(outdir), str(source)]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=3600)
    if completed.returncode != 0:
        pytest.fail(
            f"isolated PEP 517 build of {source} exited {completed.returncode}\n"
            f"--- stdout ---\n{completed.stdout[-2000:]}\n"
            f"--- stderr ---\n{completed.stderr[-2000:]}",
            pytrace=False,
        )


@pytest.fixture(scope="module")
def distributions(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Build the direct wheel and sdist, then rebuild a wheel from the extracted sdist."""
    root: Path = tmp_path_factory.mktemp("distributions").resolve()
    direct = root / "direct"
    _build(_repository_root(), direct)
    wheels = sorted(direct.glob("*.whl"))
    sdists = sorted(direct.glob("*.tar.gz"))
    assert len(wheels) == 1, wheels
    assert len(sdists) == 1, sdists

    extracted = root / "extracted"
    extracted.mkdir()
    with tarfile.open(sdists[0]) as archive:
        archive.extractall(extracted, filter="data")
    unpacked = next(path for path in extracted.iterdir() if path.is_dir())

    rebuilt_dir = root / "from_sdist"
    _build(unpacked, rebuilt_dir, wheel_only=True)
    rebuilt = sorted(rebuilt_dir.glob("*.whl"))
    assert len(rebuilt) == 1, rebuilt
    return {"wheel": wheels[0], "sdist": sdists[0], "rebuilt": rebuilt[0], "unpacked": unpacked}


def _package_members(wheel: Path) -> dict[str, bytes]:
    """Return every importable package member of one wheel."""
    with zipfile.ZipFile(wheel) as archive:
        return {
            name: archive.read(name)
            for name in archive.namelist()
            if name.startswith("metrifid/") and not name.endswith("/")
        }


def _dist_info(wheel: Path, suffix: str) -> bytes:
    """Return one `.dist-info` member of a wheel by its file name."""
    with zipfile.ZipFile(wheel) as archive:
        name = next(n for n in archive.namelist() if n.endswith(f".dist-info/{suffix}"))
        return archive.read(name)


def _metadata(wheel: Path) -> Message:
    """Parse the Core Metadata of one wheel."""
    return BytesParser().parsebytes(_dist_info(wheel, "METADATA"))


def _sdist_metadata(sdist: Path) -> Message:
    """Parse the Core Metadata the sdist itself carries, from its single root `PKG-INFO`."""
    with tarfile.open(sdist) as archive:
        names = [
            name
            for name in archive.getnames()
            if name.endswith("/PKG-INFO") and name.count("/") == 1
        ]
        assert len(names) == 1, names
        extracted = archive.extractfile(names[0])
        assert extracted is not None
        return BytesParser().parsebytes(extracted.read())


def _legal_file_bytes(wheel: Path, filename: str) -> bytes:
    """Return one PEP 639 legal file carried inside a wheel."""
    with zipfile.ZipFile(wheel) as archive:
        name = next(
            n for n in archive.namelist() if "/licenses/" in n and n.endswith(f"/{filename}")
        )
        return archive.read(name)


def test_sdist_contains_every_documented_repository_path(distributions: dict[str, Path]) -> None:
    """Ship everything the contributor and security documentation references."""
    with tarfile.open(distributions["sdist"]) as archive:
        names = {n.split("/", 1)[1] for n in archive.getnames() if "/" in n}
    missing = [member for member in _REQUIRED_SDIST_MEMBERS if member not in names]
    assert missing == [], missing
    for prefix in ("src/", "tests/", "docs/", "examples/"):
        assert any(name.startswith(prefix) for name in names), prefix


def test_direct_wheel_package_bytes_match_the_candidate_source(
    distributions: dict[str, Path],
) -> None:
    """Publish exactly the source package bytes, with nothing added or rewritten."""
    source = _repository_root() / "src"
    expected = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in sorted((source / "metrifid").rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    expected[_FROZEN_WORKER_RESOURCE] = (
        _repository_root() / "tools" / "native_upgrade_profile_worker.py"
    ).read_bytes()
    assert _package_members(distributions["wheel"]) == expected


def test_distribution_carries_the_frozen_evidence_worker(
    distributions: dict[str, Path],
) -> None:
    """Bind both wheel routes to the exact authoritative standalone worker source."""
    authoritative = (_repository_root() / "tools" / "native_upgrade_profile_worker.py").read_bytes()
    assert hashlib.sha256(authoritative).hexdigest() == _FROZEN_WORKER_SHA256
    assert _package_members(distributions["wheel"])[_FROZEN_WORKER_RESOURCE] == authoritative
    assert _package_members(distributions["rebuilt"])[_FROZEN_WORKER_RESOURCE] == authoritative
    extracted_worker = distributions["unpacked"] / "tools" / "native_upgrade_profile_worker.py"
    assert extracted_worker.read_bytes() == authoritative


def test_direct_and_rebuilt_wheels_carry_identical_package_members(
    distributions: dict[str, Path],
) -> None:
    """Install the same importable package whichever artifact a user starts from."""
    direct = _package_members(distributions["wheel"])
    rebuilt = _package_members(distributions["rebuilt"])
    assert sorted(direct) == sorted(rebuilt)
    assert direct == rebuilt


def test_direct_and_rebuilt_wheels_share_one_console_entry_point(
    distributions: dict[str, Path],
) -> None:
    """Expose the same `metrifid` command from both artifacts."""
    direct = _dist_info(distributions["wheel"], "entry_points.txt")
    rebuilt = _dist_info(distributions["rebuilt"], "entry_points.txt")
    assert direct == rebuilt
    assert b"metrifid = metrifid.cli:main" in direct


def test_direct_and_rebuilt_wheels_declare_semantically_equal_metadata(
    distributions: dict[str, Path],
) -> None:
    """Promise the same name, version, summary, Python range, dependencies, and license."""
    direct, rebuilt = _metadata(distributions["wheel"]), _metadata(distributions["rebuilt"])
    for field in _COMPARED_METADATA_FIELDS:
        assert direct.get(field) == rebuilt.get(field), field
    assert sorted(direct.get_all("Requires-Dist") or []) == sorted(
        rebuilt.get_all("Requires-Dist") or []
    )
    license_direct = direct.get("License-Expression") or direct.get("License")
    license_rebuilt = rebuilt.get("License-Expression") or rebuilt.get("License")
    assert license_direct == license_rebuilt == "Apache-2.0"


def test_direct_and_rebuilt_wheels_carry_identical_license_and_typing_markers(
    distributions: dict[str, Path],
) -> None:
    """Carry the same license text and the same PEP 561 marker in both artifacts."""
    for filename in ("LICENSE", "NOTICE"):
        assert _legal_file_bytes(distributions["wheel"], filename) == _legal_file_bytes(
            distributions["rebuilt"], filename
        )
    direct = _package_members(distributions["wheel"])
    rebuilt = _package_members(distributions["rebuilt"])
    assert direct["metrifid/py.typed"] == rebuilt["metrifid/py.typed"]


def test_distribution_metadata_identifies_license_and_owner(
    distributions: dict[str, Path],
) -> None:
    """Expose Apache-2.0 and project attribution in both wheels."""
    for key in ("wheel", "rebuilt"):
        metadata = _metadata(distributions[key])
        assert metadata.get("License-Expression") == "Apache-2.0"
        assert metadata.get("Author") == "Volodymyr Barylyak"
        assert metadata.get("Maintainer") == "Volodymyr Barylyak"
        assert sorted(metadata.get_all("License-File") or []) == ["LICENSE", "NOTICE"]
        assert "License :: OSI Approved :: Apache Software License" not in (
            metadata.get_all("Classifier") or []
        )


def test_artifacts_report_the_authoritative_source_version(
    distributions: dict[str, Path],
) -> None:
    """Match the one source-controlled version across every wheel and sdist route.

    The grammar admits a canonical release form and a development form, because the same identity
    contract has to hold on both sides of a release. What it does not admit is a second source of
    truth: every route below is compared to the one assignment in `version.py`, not to each other,
    and the sdist is read through its own metadata rather than through a name that could agree with
    a stale document.
    """
    version = _authoritative_source_version()
    assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:\.dev[0-9]+)?", version)
    assert _metadata(distributions["wheel"]).get("Version") == version
    assert _metadata(distributions["rebuilt"]).get("Version") == version
    # The sdist's own metadata bytes, not its filename and not the wheel rebuilt from it: those
    # would agree with a stale PKG-INFO instead of contradicting it.
    assert _sdist_metadata(distributions["sdist"]).get("Version") == version
    assert distributions["wheel"].name == f"metrifid-{version}-py3-none-any.whl"
    assert distributions["sdist"].name == f"metrifid-{version}.tar.gz"
    assert distributions["rebuilt"].name == f"metrifid-{version}-py3-none-any.whl"


def _declared_mujoco_resolution(metadata: Message) -> dict[str, tuple[str, ...] | None]:
    """How each supported platform class resolves MuJoCo under one distribution's metadata.

    ``None`` for a platform means the requirement set is not well formed there: either no MuJoCo
    requirement applies or more than one does. Both are contract failures.
    """
    runtime = [
        value
        for value in (metadata.get_all("Requires-Dist") or [])
        if "extra ==" not in value and Requirement(value).name == "mujoco"
    ]
    resolution: dict[str, tuple[str, ...] | None] = {}
    for label, environment in _SUPPORTED_PLATFORMS.items():
        active = [
            requirement
            for requirement in (Requirement(value) for value in runtime)
            if requirement.marker is None or requirement.marker.evaluate(environment)
        ]
        resolution[label] = (
            tuple(sorted(str(specifier) for specifier in active[0].specifier))
            if len(active) == 1
            else None
        )
    return resolution


def test_every_distribution_form_declares_the_same_platform_resolution(
    distributions: dict[str, Path],
) -> None:
    """The wheel, the sdist's own metadata and the sdist-rebuilt wheel must agree exactly.

    The Intel macOS bound exists because upstream publishes no Darwin x86_64 MuJoCo wheel at or
    above 3.11. A bound that reached the wheel but not the sdist, or that survived the direct
    build but not the rebuild from source, would leave one published install path still broken.
    Marker mutation controls live with the metadata contract; this asserts only that every
    published form carries the same declared semantics and the authoritative version.
    """
    version = _authoritative_source_version()
    expected = {
        "intel macOS": ("<3.11", ">=3.9"),
        "Apple silicon macOS": (">=3.9",),
        "Linux x86_64": (">=3.9",),
        "Linux aarch64": (">=3.9",),
    }
    observed = {
        "wheel": _metadata(distributions["wheel"]),
        "sdist": _sdist_metadata(distributions["sdist"]),
        "rebuilt": _metadata(distributions["rebuilt"]),
    }
    for form, metadata in observed.items():
        assert metadata.get("Version") == version, form
        assert _declared_mujoco_resolution(metadata) == expected, form
        numpy = [
            value
            for value in (metadata.get_all("Requires-Dist") or [])
            if "extra ==" not in value and Requirement(value).name == "numpy"
        ]
        assert numpy == ["numpy>=1.26"], (form, numpy)
