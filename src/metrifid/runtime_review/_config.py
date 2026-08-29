"""Strict configuration admission for one exact Native Runtime Review campaign."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final, Self, TypeAlias, cast

from .._json_admission import (
    CONFIG_JSON_LIMITS,
    bounded_strict_json_loads,
    read_bounded_regular_file,
)
from .._schema_primitives import (
    _bounded_int,
    _fields,
    _name,
    _nonempty_string,
    _object,
    _sequence,
    _string,
)
from ..json_values import CanonicalValue, canonical_sha256, require_sha256
from ._paths import (
    admit_relative_portable_path,
    ensure_nonoverlapping,
    resolve_existing_directory,
    resolve_new_output_path,
)

CONFIG_SCHEMA: Final[str] = "metrifid.runtime_review_config"
CONFIG_SCHEMA_VERSION: Final[int] = 1
CONFIG_SCHEMA_VERSION_V2: Final[int] = 2
REQUIRED_HORIZON: Final[str] = "1"
BASELINE_PROFILE_ID: Final[str] = "A_3.10.0"
BASELINE_MUJOCO_VERSION: Final[str] = "3.10.0"
CANDIDATE_PROFILE_ID: Final[str] = "B_3.11.0"
CANDIDATE_MUJOCO_VERSION: Final[str] = "3.11.0"
PROFILE_ROLES: Final[tuple[str, str]] = ("baseline", "candidate")
STEP_DTS: Final[tuple[str, str, str]] = ("0.004", "0.002", "0.001")
REPEAT_IDS: Final[tuple[int, int]] = (0, 1)
_MINIMUM_PROFILE_VERSION: Final[tuple[int, int, int]] = (3, 9, 0)
_STABLE_PACKAGE_VERSION: Final[re.Pattern[str]] = re.compile(
    r"\A(?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)"
    r"(?:\.post(?:0|[1-9][0-9]*))?"
    r"(?:\+[0-9A-Za-z]+(?:[._-][0-9A-Za-z]+)*)?\Z"
)
_NATIVE_VERSION: Final[re.Pattern[str]] = re.compile(
    r"\A(?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)\Z"
)
_PROFILE_IDENTITY_LIMIT: Final[int] = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class RuntimeProfileConfig:
    """One exact native-runtime profile supported by the first referee."""

    profile_id: str
    mujoco_version: str

    @classmethod
    def from_primitive(cls, value: object, field: str) -> Self:
        """Decode one profile object with no optional or unknown fields."""
        obj = _object(value, field)
        _fields(obj, {"profile_id", "mujoco_version"}, field)
        return cls(
            _nonempty_string(obj["profile_id"], f"{field}.profile_id"),
            _nonempty_string(obj["mujoco_version"], f"{field}.mujoco_version"),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit the exact declared native-runtime profile."""
        return {"profile_id": self.profile_id, "mujoco_version": self.mujoco_version}


@dataclass(frozen=True, slots=True)
class RuntimeProfileConfigV2:
    """One semantic role bound to an exact measured native-profile identity."""

    profile_role: str
    package_version: str
    native_version: str
    native_version_integer: int
    profile_identity_sha256: str
    identity_file: str

    def __post_init__(self) -> None:
        """Require a coherent stable package/native tuple and portable identity locator."""
        if self.profile_role not in PROFILE_ROLES:
            raise ValueError("profile_role must be baseline or candidate")
        package = _stable_version_triplet(self.package_version, package=True)
        native = _stable_version_triplet(self.native_version, package=False)
        if package < _MINIMUM_PROFILE_VERSION:
            raise ValueError("package_version must be a stable MuJoCo version at or above 3.9.0")
        if package != native:
            raise ValueError("package_version base triplet must equal native_version")
        expected_integer = package[0] * 1_000_000 + package[1] * 1_000 + package[2]
        if (
            type(self.native_version_integer) is not int
            or self.native_version_integer != expected_integer
        ):
            raise ValueError("native_version_integer must encode the exact native_version triplet")
        require_sha256(self.profile_identity_sha256, "profile_identity_sha256")
        admitted = admit_relative_portable_path(self.identity_file, "identity_file")
        expected_locator = f"profile_identities/{self.profile_role}.json"
        if admitted != expected_locator:
            raise ValueError(f"identity_file for {self.profile_role} must be {expected_locator!r}")

    @classmethod
    def from_primitive(cls, value: object, field: str) -> Self:
        """Decode one role-based profile declaration with an exact field set."""
        obj = _object(value, field)
        _fields(
            obj,
            {
                "profile_role",
                "package_version",
                "native_version",
                "native_version_integer",
                "profile_identity_sha256",
                "identity_file",
            },
            field,
        )
        return cls(
            profile_role=_nonempty_string(obj["profile_role"], f"{field}.profile_role"),
            package_version=_nonempty_string(obj["package_version"], f"{field}.package_version"),
            native_version=_nonempty_string(obj["native_version"], f"{field}.native_version"),
            native_version_integer=_bounded_int(
                obj["native_version_integer"],
                f"{field}.native_version_integer",
                1,
                999_999_999,
            ),
            profile_identity_sha256=require_sha256(
                obj["profile_identity_sha256"], f"{field}.profile_identity_sha256"
            ),
            identity_file=admit_relative_portable_path(
                obj["identity_file"], f"{field}.identity_file"
            ),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit the exact role and measured profile declaration."""
        return {
            "profile_role": self.profile_role,
            "package_version": self.package_version,
            "native_version": self.native_version,
            "native_version_integer": self.native_version_integer,
            "profile_identity_sha256": self.profile_identity_sha256,
            "identity_file": self.identity_file,
        }


@dataclass(frozen=True, slots=True)
class ExpectedSubjectConfig:
    """Expected source-closure identity shared by every admitted evidence cell."""

    fixture_id: str
    source_closure_sha256: str
    fixture_manifest_sha256: str

    def __post_init__(self) -> None:
        """Validate the semantic fixture label and both lowercase digests."""
        _name(self.fixture_id, "expected_subject.fixture_id")
        require_sha256(self.source_closure_sha256, "expected_subject.source_closure_sha256")
        require_sha256(self.fixture_manifest_sha256, "expected_subject.fixture_manifest_sha256")

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode the expected subject with an exact field set."""
        obj = _object(value, "expected_subject")
        _fields(
            obj,
            {"fixture_id", "source_closure_sha256", "fixture_manifest_sha256"},
            "expected_subject",
        )
        return cls(
            _name(obj["fixture_id"], "expected_subject.fixture_id"),
            require_sha256(obj["source_closure_sha256"], "expected_subject.source_closure_sha256"),
            require_sha256(
                obj["fixture_manifest_sha256"], "expected_subject.fixture_manifest_sha256"
            ),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit the expected subject identities."""
        return {
            "fixture_id": self.fixture_id,
            "source_closure_sha256": self.source_closure_sha256,
            "fixture_manifest_sha256": self.fixture_manifest_sha256,
        }


@dataclass(frozen=True, slots=True)
class ExpectedWorkloadConfig:
    """Expected workload identities shared by every admitted evidence cell."""

    semantic_sha256: str
    initial_state_semantic_sha256: str
    action_program_semantic_sha256: str

    def __post_init__(self) -> None:
        """Validate all three exact lowercase workload digests."""
        require_sha256(self.semantic_sha256, "expected_workload.semantic_sha256")
        require_sha256(
            self.initial_state_semantic_sha256,
            "expected_workload.initial_state_semantic_sha256",
        )
        require_sha256(
            self.action_program_semantic_sha256,
            "expected_workload.action_program_semantic_sha256",
        )

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode the expected workload with an exact field set."""
        obj = _object(value, "expected_workload")
        _fields(
            obj,
            {
                "semantic_sha256",
                "initial_state_semantic_sha256",
                "action_program_semantic_sha256",
            },
            "expected_workload",
        )
        return cls(
            require_sha256(obj["semantic_sha256"], "expected_workload.semantic_sha256"),
            require_sha256(
                obj["initial_state_semantic_sha256"],
                "expected_workload.initial_state_semantic_sha256",
            ),
            require_sha256(
                obj["action_program_semantic_sha256"],
                "expected_workload.action_program_semantic_sha256",
            ),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit the expected workload identities."""
        return {
            "semantic_sha256": self.semantic_sha256,
            "initial_state_semantic_sha256": self.initial_state_semantic_sha256,
            "action_program_semantic_sha256": self.action_program_semantic_sha256,
        }


@dataclass(frozen=True, slots=True)
class RuntimeReviewCellConfig:
    """One declared role, step, repeat, and evidence-directory slot."""

    profile_role: str
    step_dt: str
    repeat_id: int
    directory: str

    def __post_init__(self) -> None:
        """Validate the closed slot vocabulary and portable directory spelling."""
        if self.profile_role not in PROFILE_ROLES:
            raise ValueError("profile_role must be baseline or candidate")
        if self.step_dt not in STEP_DTS:
            raise ValueError(f"step_dt must be one of {list(STEP_DTS)}")
        if type(self.repeat_id) is not int or self.repeat_id not in REPEAT_IDS:
            raise ValueError(f"repeat_id must be one of {list(REPEAT_IDS)}")
        admit_relative_portable_path(self.directory, "cell.directory")

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode one exact campaign slot."""
        obj = _object(value, "cell")
        _fields(obj, {"profile_role", "step_dt", "repeat_id", "directory"}, "cell")
        return cls(
            _nonempty_string(obj["profile_role"], "cell.profile_role"),
            _nonempty_string(obj["step_dt"], "cell.step_dt"),
            _bounded_int(obj["repeat_id"], "cell.repeat_id", 0, 1),
            admit_relative_portable_path(obj["directory"], "cell.directory"),
        )

    @property
    def slot(self) -> tuple[str, str, int]:
        """Return the semantic role/step/repeat key for this cell."""
        return (self.profile_role, self.step_dt, self.repeat_id)

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit one canonical campaign-cell declaration."""
        return {
            "profile_role": self.profile_role,
            "step_dt": self.step_dt,
            "repeat_id": self.repeat_id,
            "directory": self.directory,
        }


@dataclass(frozen=True, slots=True)
class RuntimeReviewConfig:
    """The complete strict configuration for one exact twelve-cell review."""

    schema: str
    schema_version: int
    baseline_profile: RuntimeProfileConfig
    candidate_profile: RuntimeProfileConfig
    expected_subject: ExpectedSubjectConfig
    expected_workload: ExpectedWorkloadConfig
    required_horizon: str
    step_dts: tuple[str, ...]
    repeat_ids: tuple[int, ...]
    cells: tuple[RuntimeReviewCellConfig, ...]
    output_dir: str

    def __post_init__(self) -> None:
        """Enforce the frozen profiles, exact shape, and canonical cell ordering."""
        if self.schema != CONFIG_SCHEMA:
            raise ValueError(f"schema must be {CONFIG_SCHEMA}")
        if type(self.schema_version) is not int or self.schema_version != CONFIG_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be the integer {CONFIG_SCHEMA_VERSION}")
        if not isinstance(self.baseline_profile, RuntimeProfileConfig):
            raise TypeError("baseline_profile must be a RuntimeProfileConfig")
        if not isinstance(self.candidate_profile, RuntimeProfileConfig):
            raise TypeError("candidate_profile must be a RuntimeProfileConfig")
        _require_supported_profile(
            self.baseline_profile,
            BASELINE_PROFILE_ID,
            BASELINE_MUJOCO_VERSION,
            "baseline_profile",
        )
        _require_supported_profile(
            self.candidate_profile,
            CANDIDATE_PROFILE_ID,
            CANDIDATE_MUJOCO_VERSION,
            "candidate_profile",
        )
        if not isinstance(self.expected_subject, ExpectedSubjectConfig):
            raise TypeError("expected_subject must be an ExpectedSubjectConfig")
        if not isinstance(self.expected_workload, ExpectedWorkloadConfig):
            raise TypeError("expected_workload must be an ExpectedWorkloadConfig")
        if self.required_horizon != REQUIRED_HORIZON:
            raise ValueError(f"required_horizon must be the exact token {REQUIRED_HORIZON!r}")
        _require_exact_members(self.step_dts, STEP_DTS, "step_dts")
        _require_exact_members(self.repeat_ids, REPEAT_IDS, "repeat_ids")
        if type(self.cells) is not tuple or any(
            not isinstance(cell, RuntimeReviewCellConfig) for cell in self.cells
        ):
            raise TypeError("cells must be a tuple of RuntimeReviewCellConfig values")
        canonical = _canonical_cells(self.cells)
        object.__setattr__(self, "step_dts", STEP_DTS)
        object.__setattr__(self, "repeat_ids", REPEAT_IDS)
        object.__setattr__(self, "cells", canonical)
        admit_relative_portable_path(self.output_dir, "output_dir")

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode the exact top-level configuration and reject every alternate shape."""
        obj = _object(value, "RuntimeReviewConfig")
        _fields(
            obj,
            {
                "schema",
                "schema_version",
                "baseline_profile",
                "candidate_profile",
                "expected_subject",
                "expected_workload",
                "required_horizon",
                "step_dts",
                "repeat_ids",
                "cells",
                "output_dir",
            },
            "RuntimeReviewConfig",
        )
        return cls(
            _string(obj["schema"], "schema"),
            _bounded_int(obj["schema_version"], "schema_version", 1, 1),
            RuntimeProfileConfig.from_primitive(obj["baseline_profile"], "baseline_profile"),
            RuntimeProfileConfig.from_primitive(obj["candidate_profile"], "candidate_profile"),
            ExpectedSubjectConfig.from_primitive(obj["expected_subject"]),
            ExpectedWorkloadConfig.from_primitive(obj["expected_workload"]),
            _string(obj["required_horizon"], "required_horizon"),
            tuple(
                _string(item, "step_dts item") for item in _sequence(obj["step_dts"], "step_dts")
            ),
            tuple(
                _bounded_int(item, "repeat_ids item", 0, 1)
                for item in _sequence(obj["repeat_ids"], "repeat_ids")
            ),
            tuple(
                RuntimeReviewCellConfig.from_primitive(item)
                for item in _sequence(obj["cells"], "cells")
            ),
            admit_relative_portable_path(obj["output_dir"], "output_dir"),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit canonical semantics with the campaign cells in frozen slot order."""
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "baseline_profile": self.baseline_profile.to_primitive(),
            "candidate_profile": self.candidate_profile.to_primitive(),
            "expected_subject": self.expected_subject.to_primitive(),
            "expected_workload": self.expected_workload.to_primitive(),
            "required_horizon": self.required_horizon,
            "step_dts": list(self.step_dts),
            "repeat_ids": list(self.repeat_ids),
            "cells": [cell.to_primitive() for cell in self.cells],
            "output_dir": self.output_dir,
        }


@dataclass(frozen=True, slots=True)
class RuntimeReviewConfigV2:
    """The role-based configuration for one exact twelve-cell review."""

    schema: str
    schema_version: int
    baseline_profile: RuntimeProfileConfigV2
    candidate_profile: RuntimeProfileConfigV2
    expected_subject: ExpectedSubjectConfig
    expected_workload: ExpectedWorkloadConfig
    required_horizon: str
    step_dts: tuple[str, ...]
    repeat_ids: tuple[int, ...]
    cells: tuple[RuntimeReviewCellConfig, ...]
    output_dir: str

    def __post_init__(self) -> None:
        """Enforce semantic roles, exact science shape, and canonical cell ordering."""
        if self.schema != CONFIG_SCHEMA:
            raise ValueError(f"schema must be {CONFIG_SCHEMA}")
        if type(self.schema_version) is not int or self.schema_version != CONFIG_SCHEMA_VERSION_V2:
            raise ValueError(f"schema_version must be the integer {CONFIG_SCHEMA_VERSION_V2}")
        if not isinstance(self.baseline_profile, RuntimeProfileConfigV2):
            raise TypeError("baseline_profile must be a RuntimeProfileConfigV2")
        if not isinstance(self.candidate_profile, RuntimeProfileConfigV2):
            raise TypeError("candidate_profile must be a RuntimeProfileConfigV2")
        if self.baseline_profile.profile_role != "baseline":
            raise ValueError("baseline_profile.profile_role must be baseline")
        if self.candidate_profile.profile_role != "candidate":
            raise ValueError("candidate_profile.profile_role must be candidate")
        if self.baseline_profile.identity_file == self.candidate_profile.identity_file:
            raise ValueError("profile identity files must be distinct")
        if not isinstance(self.expected_subject, ExpectedSubjectConfig):
            raise TypeError("expected_subject must be an ExpectedSubjectConfig")
        if not isinstance(self.expected_workload, ExpectedWorkloadConfig):
            raise TypeError("expected_workload must be an ExpectedWorkloadConfig")
        if self.required_horizon != REQUIRED_HORIZON:
            raise ValueError(f"required_horizon must be the exact token {REQUIRED_HORIZON!r}")
        _require_exact_members(self.step_dts, STEP_DTS, "step_dts")
        _require_exact_members(self.repeat_ids, REPEAT_IDS, "repeat_ids")
        if type(self.cells) is not tuple or any(
            not isinstance(cell, RuntimeReviewCellConfig) for cell in self.cells
        ):
            raise TypeError("cells must be a tuple of RuntimeReviewCellConfig values")
        object.__setattr__(self, "step_dts", STEP_DTS)
        object.__setattr__(self, "repeat_ids", REPEAT_IDS)
        object.__setattr__(self, "cells", _canonical_cells(self.cells))
        admit_relative_portable_path(self.output_dir, "output_dir")

    @classmethod
    def from_primitive(cls, value: object) -> Self:
        """Decode the exact role-based configuration and reject alternate shapes."""
        obj = _object(value, "RuntimeReviewConfigV2")
        _fields(
            obj,
            {
                "schema",
                "schema_version",
                "baseline_profile",
                "candidate_profile",
                "expected_subject",
                "expected_workload",
                "required_horizon",
                "step_dts",
                "repeat_ids",
                "cells",
                "output_dir",
            },
            "RuntimeReviewConfigV2",
        )
        return cls(
            _string(obj["schema"], "schema"),
            _bounded_int(obj["schema_version"], "schema_version", 2, 2),
            RuntimeProfileConfigV2.from_primitive(obj["baseline_profile"], "baseline_profile"),
            RuntimeProfileConfigV2.from_primitive(obj["candidate_profile"], "candidate_profile"),
            ExpectedSubjectConfig.from_primitive(obj["expected_subject"]),
            ExpectedWorkloadConfig.from_primitive(obj["expected_workload"]),
            _string(obj["required_horizon"], "required_horizon"),
            tuple(
                _string(item, "step_dts item") for item in _sequence(obj["step_dts"], "step_dts")
            ),
            tuple(
                _bounded_int(item, "repeat_ids item", 0, 1)
                for item in _sequence(obj["repeat_ids"], "repeat_ids")
            ),
            tuple(
                RuntimeReviewCellConfig.from_primitive(item)
                for item in _sequence(obj["cells"], "cells")
            ),
            admit_relative_portable_path(obj["output_dir"], "output_dir"),
        )

    def to_primitive(self) -> dict[str, CanonicalValue]:
        """Emit canonical role-based semantics in frozen campaign order."""
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "baseline_profile": self.baseline_profile.to_primitive(),
            "candidate_profile": self.candidate_profile.to_primitive(),
            "expected_subject": self.expected_subject.to_primitive(),
            "expected_workload": self.expected_workload.to_primitive(),
            "required_horizon": self.required_horizon,
            "step_dts": list(self.step_dts),
            "repeat_ids": list(self.repeat_ids),
            "cells": [cell.to_primitive() for cell in self.cells],
            "output_dir": self.output_dir,
        }


@dataclass(frozen=True, slots=True)
class AdmittedRuntimeReviewConfiguration:
    """A strict configuration plus byte identity and resolved filesystem bindings."""

    config: RuntimeReviewConfig
    path: Path
    base_dir: Path
    raw_bytes: bytes
    raw_sha256: str
    semantic_sha256: str
    cell_directories: tuple[Path, ...]
    output_dir: Path

    def __post_init__(self) -> None:
        """Require all retained identities and canonical path bindings to agree."""
        if not isinstance(self.config, RuntimeReviewConfig):
            raise TypeError("config must be a RuntimeReviewConfig")
        if len(self.cell_directories) != len(self.config.cells):
            raise ValueError("cell_directories must align with every canonical configuration cell")
        if hashlib.sha256(self.raw_bytes).hexdigest() != self.raw_sha256:
            raise ValueError("raw_sha256 does not match raw_bytes")
        if canonical_sha256(self.config.to_primitive()) != self.semantic_sha256:
            raise ValueError("semantic_sha256 does not match canonical configuration semantics")


@dataclass(frozen=True, slots=True)
class AdmittedRuntimeReviewConfigurationV2:
    """A role-based configuration plus exact filesystem and identity-file bindings."""

    config: RuntimeReviewConfigV2
    path: Path
    base_dir: Path
    raw_bytes: bytes
    raw_sha256: str
    semantic_sha256: str
    cell_directories: tuple[Path, ...]
    profile_identity_paths: tuple[Path, Path]
    profile_identity_file_sha256: tuple[str, str]
    output_dir: Path

    def __post_init__(self) -> None:
        """Require configuration bytes and every ordered path binding to agree."""
        if not isinstance(self.config, RuntimeReviewConfigV2):
            raise TypeError("config must be a RuntimeReviewConfigV2")
        if len(self.cell_directories) != len(self.config.cells):
            raise ValueError("cell_directories must align with every canonical configuration cell")
        if type(self.profile_identity_paths) is not tuple or len(self.profile_identity_paths) != 2:
            raise ValueError("profile_identity_paths must bind baseline then candidate")
        expected_identity_paths = tuple(
            self.base_dir / profile.identity_file
            for profile in (self.config.baseline_profile, self.config.candidate_profile)
        )
        if self.profile_identity_paths != expected_identity_paths:
            raise ValueError("profile_identity_paths do not bind the declared identity files")
        if (
            type(self.profile_identity_file_sha256) is not tuple
            or len(self.profile_identity_file_sha256) != 2
        ):
            raise ValueError("profile_identity_file_sha256 must bind baseline then candidate")
        for role, digest in zip(PROFILE_ROLES, self.profile_identity_file_sha256, strict=True):
            require_sha256(digest, f"{role} profile identity raw SHA-256")
        if hashlib.sha256(self.raw_bytes).hexdigest() != self.raw_sha256:
            raise ValueError("raw_sha256 does not match raw_bytes")
        if canonical_sha256(self.config.to_primitive()) != self.semantic_sha256:
            raise ValueError("semantic_sha256 does not match canonical configuration semantics")

    def profile_identity_path(self, role: str) -> Path:
        """Return the exact admitted profile-identity file for one semantic role."""
        if role not in PROFILE_ROLES:
            raise ValueError("role must be baseline or candidate")
        return self.profile_identity_paths[PROFILE_ROLES.index(role)]

    def profile_identity_file_hash(self, role: str) -> str:
        """Return the admitted raw-file digest for one semantic role identity."""
        if role not in PROFILE_ROLES:
            raise ValueError("role must be baseline or candidate")
        return self.profile_identity_file_sha256[PROFILE_ROLES.index(role)]


AdmittedRuntimeReviewConfigurationAny: TypeAlias = (
    AdmittedRuntimeReviewConfiguration | AdmittedRuntimeReviewConfigurationV2
)


def load_runtime_review_configuration(
    path: str | Path,
) -> AdmittedRuntimeReviewConfigurationAny:
    """Read and explicitly dispatch one legacy or role-based review configuration."""
    target = Path(path).absolute()
    raw = read_bounded_regular_file(target, CONFIG_JSON_LIMITS.max_bytes)
    primitive = bounded_strict_json_loads(raw, CONFIG_JSON_LIMITS)
    obj = _object(primitive, "RuntimeReviewConfig")
    if obj.get("schema") != CONFIG_SCHEMA:
        raise ValueError(f"schema must be {CONFIG_SCHEMA}")
    version = obj.get("schema_version")
    if type(version) is not int:
        raise TypeError("schema_version must be an integer and not a boolean")
    if version == CONFIG_SCHEMA_VERSION:
        return _admit_runtime_review_configuration_v1(target, raw, primitive)
    if version == CONFIG_SCHEMA_VERSION_V2:
        return _admit_runtime_review_configuration_v2(target, raw, primitive)
    raise ValueError(f"unsupported runtime-review configuration schema_version: {version}")


def _admit_runtime_review_configuration_v1(
    target: Path, raw: bytes, primitive: CanonicalValue
) -> AdmittedRuntimeReviewConfiguration:
    """Admit schema-version-1 bytes through the immutable legacy implementation."""
    config = RuntimeReviewConfig.from_primitive(primitive)
    base = target.parent.resolve(strict=True)
    directories = tuple(
        resolve_existing_directory(base, cell.directory, f"cells[{index}].directory")
        for index, cell in enumerate(config.cells)
    )
    output = resolve_new_output_path(base, config.output_dir)
    protected = {
        f"cell {cell.profile_role}/{cell.step_dt}/repeat_{cell.repeat_id}": directory
        for cell, directory in zip(config.cells, directories, strict=True)
    }
    ensure_nonoverlapping(output, protected)
    return AdmittedRuntimeReviewConfiguration(
        config=config,
        path=target,
        base_dir=base,
        raw_bytes=raw,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        semantic_sha256=canonical_sha256(config.to_primitive()),
        cell_directories=directories,
        output_dir=output,
    )


def _admit_runtime_review_configuration_v2(
    target: Path, raw: bytes, primitive: CanonicalValue
) -> AdmittedRuntimeReviewConfigurationV2:
    """Admit role-based bytes and resolve both exact profile-identity files."""
    config = RuntimeReviewConfigV2.from_primitive(primitive)
    base = target.parent.resolve(strict=True)
    directories = tuple(
        resolve_existing_directory(base, cell.directory, f"cells[{index}].directory")
        for index, cell in enumerate(config.cells)
    )
    profiles = (config.baseline_profile, config.candidate_profile)
    identity_bindings = cast(
        tuple[tuple[Path, str], tuple[Path, str]],
        tuple(
            _resolve_existing_profile_identity(base, profile.identity_file, profile.profile_role)
            for profile in profiles
        ),
    )
    identity_paths = cast(tuple[Path, Path], tuple(item[0] for item in identity_bindings))
    identity_hashes = cast(tuple[str, str], tuple(item[1] for item in identity_bindings))
    output = resolve_new_output_path(base, config.output_dir)
    protected = {
        **{
            f"cell {cell.profile_role}/{cell.step_dt}/repeat_{cell.repeat_id}": directory
            for cell, directory in zip(config.cells, directories, strict=True)
        },
        **{
            f"profile identity {profile.profile_role}": identity_path
            for profile, identity_path in zip(profiles, identity_paths, strict=True)
        },
    }
    ensure_nonoverlapping(output, protected)
    return AdmittedRuntimeReviewConfigurationV2(
        config=config,
        path=target,
        base_dir=base,
        raw_bytes=raw,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
        semantic_sha256=canonical_sha256(config.to_primitive()),
        cell_directories=directories,
        profile_identity_paths=identity_paths,
        profile_identity_file_sha256=identity_hashes,
        output_dir=output,
    )


def _resolve_existing_profile_identity(base: Path, relative: str, role: str) -> tuple[Path, str]:
    """Resolve and hash one confined regular identity without following declared links."""
    admitted = admit_relative_portable_path(relative, f"{role}_profile.identity_file")
    locator = PurePosixPath(admitted)
    parent = resolve_existing_directory(
        base,
        locator.parent.as_posix(),
        f"{role}_profile.identity_file parent",
    )
    target = parent / locator.name
    raw = read_bounded_regular_file(target, _PROFILE_IDENTITY_LIMIT)
    return target, hashlib.sha256(raw).hexdigest()


def _stable_version_triplet(value: str, *, package: bool) -> tuple[int, int, int]:
    """Parse one bounded stable package token or exact native triplet."""
    if type(value) is not str:
        raise TypeError("runtime profile version must be a string")
    pattern = _STABLE_PACKAGE_VERSION if package else _NATIVE_VERSION
    match = pattern.fullmatch(value)
    if match is None:
        field = "package_version" if package else "native_version"
        raise ValueError(f"{field} must use the bounded stable version grammar")
    triplet = tuple(int(match.group(name)) for name in ("major", "minor", "patch"))
    if any(component >= 1000 for component in triplet):
        raise ValueError("runtime profile version components must be below 1000")
    return cast(tuple[int, int, int], triplet)


def _require_supported_profile(
    profile: RuntimeProfileConfig,
    profile_id: str,
    mujoco_version: str,
    field: str,
) -> None:
    """Require one profile to equal the first product's exact supported identity."""
    if profile.profile_id != profile_id or profile.mujoco_version != mujoco_version:
        raise ValueError(
            f"{field} must be profile {profile_id!r} with native MuJoCo {mujoco_version!r}"
        )


def _require_exact_members(
    actual: tuple[object, ...], expected: tuple[object, ...], field: str
) -> None:
    """Require one tuple to contain every frozen member exactly once in any input order."""
    if type(actual) is not tuple or len(actual) != len(expected) or set(actual) != set(expected):
        raise ValueError(f"{field} must contain exactly {list(expected)} once each")


def _canonical_cells(
    cells: tuple[RuntimeReviewCellConfig, ...],
) -> tuple[RuntimeReviewCellConfig, ...]:
    """Require the complete unique campaign shape and return frozen slot order."""
    expected_slots = tuple(
        (role, step_dt, repeat_id)
        for role in PROFILE_ROLES
        for step_dt in STEP_DTS
        for repeat_id in REPEAT_IDS
    )
    by_slot: dict[tuple[str, str, int], RuntimeReviewCellConfig] = {}
    for cell in cells:
        if cell.slot in by_slot:
            raise ValueError(f"cells contains duplicate slot {cell.slot!r}")
        by_slot[cell.slot] = cell
    missing = [slot for slot in expected_slots if slot not in by_slot]
    extra = [slot for slot in by_slot if slot not in expected_slots]
    if missing or extra or len(cells) != len(expected_slots):
        raise ValueError(f"cells must contain the exact twelve campaign slots; missing={missing!r}")
    directories = [cell.directory for cell in cells]
    if len(set(directories)) != len(directories):
        raise ValueError("every campaign cell must declare a distinct directory")
    return tuple(by_slot[slot] for slot in expected_slots)
