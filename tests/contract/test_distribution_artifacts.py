"""The two published artifacts must carry the same package and the same promises.

A user may install the wheel from PyPI or build from the source distribution. Those two routes have
to deliver the same importable package, the same console entry point, and the same declared
metadata. These tests build both from the candidate source and compare them semantically — not by
archive bytes, which legitimately differ in timestamps and generator strings.

Every build here is isolated PEP 517, which is what a real consumer's `pip install` performs.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tarfile
import zipfile
from email.message import Message
from email.parser import BytesParser
from pathlib import Path

import pytest

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
)
_COMPARED_METADATA_FIELDS = ("Name", "Version", "Summary", "Requires-Python")


def _repository_root() -> Path:
    """Return the source checkout to build from."""
    return Path(__file__).resolve().parents[2]


def _build(source: Path, outdir: Path, *, wheel_only: bool = False) -> None:
    """Build distributions from one source tree through isolated PEP 517."""
    command = [sys.executable, "-m", "build"]
    if wheel_only:
        command.append("--wheel")
    command += ["--outdir", str(outdir), str(source)]
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=3600)
    if completed.returncode != 0:  # pragma: no cover - surfaced as a skip or failure below
        pytest.skip(
            f"isolated build unavailable: {completed.stdout[-2000:]}{completed.stderr[-2000:]}"
        )


@pytest.fixture(scope="module")
def distributions(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Build the direct wheel and sdist, then rebuild a wheel from the extracted sdist."""
    if shutil.which(sys.executable) is None:  # pragma: no cover - environment guard
        pytest.skip("no interpreter available for isolated builds")
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
    assert _package_members(distributions["wheel"]) == expected


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


def test_both_artifacts_report_the_numeric_release_version(
    distributions: dict[str, Path],
) -> None:
    """Use one three-integer version across the wheel, sdist, and rebuilt wheel."""
    version = _metadata(distributions["wheel"]).get("Version")
    assert version == "0.2.1"
    assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version)
    assert version == _metadata(distributions["rebuilt"]).get("Version")
    assert distributions["wheel"].name == f"metrifid-{version}-py3-none-any.whl"
    assert distributions["sdist"].name == f"metrifid-{version}.tar.gz"
    assert distributions["rebuilt"].name == f"metrifid-{version}-py3-none-any.whl"
