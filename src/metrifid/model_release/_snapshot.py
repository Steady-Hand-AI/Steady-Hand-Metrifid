"""Complete bounded public-field facts and typed semantic facts from one private MJB.

Decision-bearing field discovery is closed by a versioned, independently measured catalog.  The
full readable surface is classified before comparable values are selected, so a new member or
changed kind cannot be silently omitted.  The complete MJB digest remains the artifact identity;
these facts explain and classify that artifact without making a dynamic-behavior claim.
"""

from __future__ import annotations

import hashlib
import struct
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, cast

from .._model_closure import ModelAdmissionRefusal
from ..certify._artifact import ArtifactSubject, load_subject_model
from ..json_values import Binary64, CanonicalValue, canonical_sha256
from ._policy import PolicyObjectType
from ._public_field_registry_catalog import (
    PUBLIC_FIELD_REGISTRY_SCHEMA,
    PUBLIC_FIELD_REGISTRY_SCHEMA_VERSION,
    characterized_registry,
)

if TYPE_CHECKING:
    import mujoco  # type: ignore[import-untyped]


_HISTORICAL_ENTRY: Final = characterized_registry("3.10.0")
if _HISTORICAL_ENTRY is None:  # pragma: no cover - construction-time invariant
    raise RuntimeError("the accepted historical public-field registry is missing")
PUBLIC_FIELD_REGISTRY_COUNT: Final = _HISTORICAL_ENTRY.comparable_registry_count
PUBLIC_FIELD_REGISTRY_SHA256: Final = _HISTORICAL_ENTRY.comparable_registry_sha256
MAX_SEMANTIC_OBJECTS: Final = 100_000

_CONTAINER_PATHS: Final = (
    "opt",
    "stat",
    "vis",
    "vis.global_",
    "vis.headlight",
    "vis.map",
    "vis.quality",
    "vis.rgba",
    "vis.scale",
)


class SnapshotRefusal(ValueError):
    """A deterministic compiled-snapshot refusal suitable for operational conversion."""

    def __init__(self, role: str, issue: str, **evidence: CanonicalValue) -> None:
        """Capture a role-local issue and canonical evidence."""
        if role not in {"baseline", "candidate"}:
            raise ValueError("snapshot refusal role must be baseline or candidate")
        self.role = role
        self.issue = issue
        self.evidence: dict[str, CanonicalValue] = {"issue": issue, **evidence}
        super().__init__(f"{role}:{issue}")


@dataclass(frozen=True, slots=True)
class PublicFieldFact:
    """Exact identity and shape facts for one public compiled-model field."""

    path: str
    kind: str
    dtype: str | None
    shape: tuple[int, ...] | None
    value_sha256: str

    def detail_primitive(self) -> dict[str, CanonicalValue]:
        """Return non-value metadata for a compiled-field change row."""
        return {
            "kind": self.kind,
            "dtype": self.dtype,
            "shape": None if self.shape is None else list(self.shape),
        }


@dataclass(frozen=True, slots=True)
class SemanticObjectFact:
    """A named semantic object and its closed policy-facing field values."""

    object_type: PolicyObjectType
    object_name: str
    fields: Mapping[str, CanonicalValue]

    @property
    def object_sha256(self) -> str:
        """Return the exact identity used by presence ADD/REMOVE changes."""
        return canonical_sha256(cast("CanonicalValue", dict(self.fields)))


@dataclass(frozen=True, slots=True)
class CompiledModelSnapshot:
    """One role's decision-bearing public and semantic compiled facts."""

    public_fields: Mapping[str, PublicFieldFact]
    semantic_objects: Mapping[tuple[PolicyObjectType, str], SemanticObjectFact]
    coverage_issues: tuple[str, ...]
    registry_sha256: str


@dataclass(frozen=True, slots=True)
class PublicFieldSurfaceMeasurement:
    """One deterministic classification of every readable owned public-model member."""

    full_public_surface_sha256: str
    full_public_surface_count: int
    comparable_registry_sha256: str
    comparable_registry_count: int
    comparable_rows: tuple[tuple[str, str, object], ...]
    opaque_member_paths: tuple[str, ...]

    def identity_primitive(self) -> dict[str, CanonicalValue]:
        """Return the native-value-free identity used by clean-process measurements."""
        return {
            "full_public_surface_sha256": self.full_public_surface_sha256,
            "full_public_surface_count": self.full_public_surface_count,
            "comparable_registry_sha256": self.comparable_registry_sha256,
            "comparable_registry_count": self.comparable_registry_count,
            "opaque_member_paths": list(self.opaque_member_paths),
        }


def build_compiled_model_snapshot(
    subject: ArtifactSubject,
    role: str,
    runtime_base_version: str,
    runtime_evidence: Mapping[str, CanonicalValue],
) -> CompiledModelSnapshot:
    """Load one retained private MJB and capture bounded facts from its exact measured bytes.

    The subject is reached only through its retained descriptor, so these decision-bearing facts
    describe the same object the embedded Certify receipt identifies, and a same-user process
    holding no descriptor of its own has no name it could point somewhere else.
    """
    try:
        model = load_subject_model(subject)
    except ModelAdmissionRefusal:
        raise
    except Exception as exc:
        raise SnapshotRefusal(
            role,
            "private_mjb_reload_failed",
            exception_type=type(exc).__name__,
        ) from exc
    try:
        public_fields, measurement = _public_field_facts(
            model,
            role,
            runtime_base_version,
            runtime_evidence,
        )
        semantic_objects, coverage = _semantic_object_facts(model, role)
        _validate_reference_graph(model, role)
        return CompiledModelSnapshot(
            MappingProxyType(public_fields),
            MappingProxyType(semantic_objects),
            tuple(sorted(coverage)),
            measurement.comparable_registry_sha256,
        )
    finally:
        del model


def _public_field_facts(
    model: mujoco.MjModel,
    role: str,
    runtime_base_version: str,
    runtime_evidence: Mapping[str, CanonicalValue],
) -> tuple[dict[str, PublicFieldFact], PublicFieldSurfaceMeasurement]:
    """Admit a characterized full surface before hashing comparable public values."""
    measurement = measure_public_field_surface(model)
    observed_evidence: dict[str, CanonicalValue] = {
        **runtime_evidence,
        "runtime_base_version": runtime_base_version,
        "observed_full_public_surface_sha256": measurement.full_public_surface_sha256,
        "observed_full_public_surface_count": measurement.full_public_surface_count,
        "observed_registry_sha256": measurement.comparable_registry_sha256,
        "observed_registry_count": measurement.comparable_registry_count,
    }
    if measurement.opaque_member_paths:
        raise SnapshotRefusal(
            role,
            "opaque_public_model_members",
            **observed_evidence,
            opaque_member_paths=list(measurement.opaque_member_paths),
            exact_reason=(
                "review-model cannot call a typed projection complete while readable opaque "
                "public model data has no admitted value representation"
            ),
            claim_risk="decision-bearing model data could be silently omitted",
            remediation="characterize and catalog the observed public model surface",
        )
    expected = characterized_registry(runtime_base_version)
    if expected is None:
        raise SnapshotRefusal(
            role,
            "uncharacterized_public_model_surface",
            **observed_evidence,
            exact_reason=(
                "review-model cannot call an older typed public-field projection complete for "
                "this uncharacterized runtime base version"
            ),
            claim_risk="new decision-bearing model data could be silently omitted",
            remediation="characterize this runtime surface twice in independent clean processes",
        )
    if (
        measurement.full_public_surface_count != expected.full_public_surface_count
        or measurement.full_public_surface_sha256 != expected.full_public_surface_sha256
    ):
        raise SnapshotRefusal(
            role,
            "public_model_surface_mismatch",
            **observed_evidence,
            expected_full_public_surface_sha256=expected.full_public_surface_sha256,
            expected_full_public_surface_count=expected.full_public_surface_count,
            exact_reason="the readable public model member names or stable kinds changed",
            claim_risk="the cataloged typed model projection is no longer complete",
            remediation="remeasure and review the changed public model surface before cataloging it",
        )
    if (
        measurement.comparable_registry_count != expected.comparable_registry_count
        or measurement.comparable_registry_sha256 != expected.comparable_registry_sha256
    ):
        raise SnapshotRefusal(
            role,
            "public_field_registry_mismatch",
            **runtime_evidence,
            runtime_base_version=runtime_base_version,
            observed_field_count=measurement.comparable_registry_count,
            expected_field_count=expected.comparable_registry_count,
            observed_registry_sha256=measurement.comparable_registry_sha256,
            expected_registry_sha256=expected.comparable_registry_sha256,
            exact_reason="the comparable typed public-field registry changed",
            claim_risk="the decision-bearing compiled-field projection is no longer complete",
            remediation="remeasure and review the changed comparable registry before cataloging it",
        )
    return (
        {
            path: _public_field_fact(path, kind, value)
            for path, kind, value in measurement.comparable_rows
        },
        measurement,
    )


def measure_public_field_surface(model: mujoco.MjModel) -> PublicFieldSurfaceMeasurement:
    """Fingerprint all readable public members and retain admitted comparable values."""
    holders = _holders(model)
    owned_container_ids = {id(holder) for holder in holders.values()}
    surface_rows: list[tuple[str, str]] = []
    comparable_rows: list[tuple[str, str, object]] = []
    opaque_paths: list[str] = []
    for prefix, holder in holders.items():
        for name in sorted(member for member in dir(holder) if not member.startswith("_")):
            path = f"{prefix}.{name}" if prefix else name
            try:
                value = getattr(holder, name)
            except Exception:
                continue
            surface_kind, comparable_kind = _public_member_kinds(value, owned_container_ids)
            surface_rows.append((path, surface_kind))
            if comparable_kind is not None:
                comparable_rows.append((path, comparable_kind, value))
            elif surface_kind.startswith("opaque_data:"):
                opaque_paths.append(path)
    surface_rows.sort()
    comparable_rows.sort(key=lambda row: row[0])
    surface_payload = "\n".join(f"{path}\0{kind}" for path, kind in surface_rows).encode(
        "utf-8", errors="strict"
    )
    registry_payload = "\n".join(f"{path}\0{kind}" for path, kind, _ in comparable_rows).encode(
        "utf-8", errors="strict"
    )
    return PublicFieldSurfaceMeasurement(
        hashlib.sha256(surface_payload).hexdigest(),
        len(surface_rows),
        hashlib.sha256(registry_payload).hexdigest(),
        len(comparable_rows),
        tuple(comparable_rows),
        tuple(sorted(opaque_paths)),
    )


def _holders(model: mujoco.MjModel) -> dict[str, object]:
    """Return every owned holder covered by the versioned public surface catalog."""
    holders: dict[str, object] = {"": model}
    for path in _CONTAINER_PATHS:
        holder: object = model
        try:
            for component in path.split("."):
                holder = getattr(holder, component)
        except Exception:
            continue
        else:
            holders[path] = holder
    return holders


def _public_member_kinds(
    value: object,
    owned_container_ids: set[int],
) -> tuple[str, str | None]:
    """Classify one readable member for full-surface and comparable-value hashing."""
    import numpy as np

    if id(value) in owned_container_ids:
        return "owned_container", None
    if isinstance(value, np.ndarray):
        return "comparable_ndarray", "ndarray"
    if isinstance(value, np.generic):
        return "comparable_numpy_scalar", "numpy_scalar"
    if type(value) in {bool, int, float, bytes, str}:
        scalar_kind = type(value).__name__
        return f"comparable_{scalar_kind}", scalar_kind
    if callable(value):
        return "callable", None
    value_type = type(value)
    return f"opaque_data:{value_type.__module__}.{value_type.__qualname__}", None


def _public_field_fact(path: str, kind: str, value: object) -> PublicFieldFact:
    """Build exact type/shape/hash facts for one admitted public value."""
    import numpy as np

    if isinstance(value, np.ndarray):
        contiguous = np.ascontiguousarray(value)
        digest = hashlib.sha256()
        digest.update(f"array:{contiguous.dtype.str}:{tuple(contiguous.shape)}:".encode())
        digest.update(contiguous.tobytes(order="C"))
        return PublicFieldFact(
            path,
            kind,
            str(contiguous.dtype),
            tuple(int(size) for size in contiguous.shape),
            digest.hexdigest(),
        )
    scalar = value.item() if isinstance(value, np.generic) else value
    return PublicFieldFact(
        path, kind, None, None, hashlib.sha256(_scalar_payload(scalar)).hexdigest()
    )


def _scalar_payload(value: object) -> bytes:
    """Encode one admitted scalar without normalizing any floating-point bits."""
    if isinstance(value, bool):
        return b"bool:" + (b"1" if value else b"0")
    if isinstance(value, int):
        return b"int:" + str(value).encode()
    if isinstance(value, float):
        return b"float:" + struct.pack("<d", value)
    if isinstance(value, bytes):
        return b"bytes:" + value
    if isinstance(value, str):
        return b"str:" + value.encode("utf-8", errors="strict")
    raise TypeError("public scalar is outside the frozen registry")


def _semantic_object_facts(
    model: mujoco.MjModel, role: str
) -> tuple[dict[tuple[PolicyObjectType, str], SemanticObjectFact], list[str]]:
    """Capture the small named semantic surface used by maintainer policies."""
    import mujoco

    counts = {
        PolicyObjectType.BODY: int(model.nbody) - 1,
        PolicyObjectType.JOINT: int(model.njnt),
        PolicyObjectType.GEOM: int(model.ngeom),
        PolicyObjectType.MESH: int(model.nmesh),
        PolicyObjectType.ACTUATOR: int(model.nu),
    }
    if sum(counts.values()) > MAX_SEMANTIC_OBJECTS:
        raise SnapshotRefusal(
            role,
            "semantic_object_budget_exceeded",
            object_count=sum(counts.values()),
            limit=MAX_SEMANTIC_OBJECTS,
        )
    object_types = {
        PolicyObjectType.BODY: mujoco.mjtObj.mjOBJ_BODY,
        PolicyObjectType.JOINT: mujoco.mjtObj.mjOBJ_JOINT,
        PolicyObjectType.GEOM: mujoco.mjtObj.mjOBJ_GEOM,
        PolicyObjectType.MESH: mujoco.mjtObj.mjOBJ_MESH,
        PolicyObjectType.ACTUATOR: mujoco.mjtObj.mjOBJ_ACTUATOR,
    }
    objects: dict[tuple[PolicyObjectType, str], SemanticObjectFact] = {}
    coverage: list[str] = []
    actuator_semantics = _compiled_actuator_semantics(model, role)
    for object_type, count in counts.items():
        start = 1 if object_type is PolicyObjectType.BODY else 0
        for object_id in range(start, start + count):
            name = mujoco.mj_id2name(model, object_types[object_type], object_id)
            if name is None or not name:
                coverage.append(f"{object_type.value}:{object_id}:unnamed")
                continue
            if "*" in name:
                coverage.append(f"{object_type.value}:{object_id}:reserved_wildcard_character")
                continue
            if len(name.encode("utf-8", errors="strict")) > 256:
                coverage.append(f"{object_type.value}:{object_id}:overlong_name")
                continue
            key = (object_type, name)
            if key in objects:
                raise SnapshotRefusal(
                    role,
                    "duplicate_semantic_object_name",
                    object_type=object_type.value,
                    object_name=name,
                )
            fields = _semantic_fields(model, object_type, object_id, actuator_semantics)
            objects[key] = SemanticObjectFact(object_type, name, MappingProxyType(fields))
    return objects, coverage


def _semantic_fields(
    model: mujoco.MjModel,
    object_type: PolicyObjectType,
    object_id: int,
    actuator_semantics: Mapping[int, dict[str, CanonicalValue]],
) -> dict[str, CanonicalValue]:
    """Return exactly the policy-facing fields for one named object."""
    if object_type is PolicyObjectType.BODY:
        parent_id = int(model.body_parentid[object_id])
        return {
            "parent": _body_name(model, parent_id),
            "mass": _binary64(float(model.body_mass[object_id])),
            "inertia": _binary64_vector(model.body_inertia[object_id]),
        }
    if object_type is PolicyObjectType.JOINT:
        return {
            "body": _body_name(model, int(model.jnt_bodyid[object_id])),
            "type": int(model.jnt_type[object_id]),
            "limited": bool(model.jnt_limited[object_id]),
            "range": _binary64_vector(model.jnt_range[object_id]),
        }
    if object_type is PolicyObjectType.GEOM:
        mesh_name: str | None = None
        if int(model.geom_type[object_id]) == 7:  # mjGEOM_MESH in MuJoCo 3.10.0
            data_id = int(model.geom_dataid[object_id])
            mesh_name = _object_name(model, "mesh", data_id)
        return {
            "body": _body_name(model, int(model.geom_bodyid[object_id])),
            "mesh": mesh_name,
        }
    if object_type is PolicyObjectType.MESH:
        return {"compiled_geometry_sha256": _mesh_geometry_sha256(model, object_id)}
    if object_type is PolicyObjectType.ACTUATOR:
        return dict(actuator_semantics[object_id])
    raise AssertionError("unreachable semantic object type")


def _object_name(model: mujoco.MjModel, kind: str, object_id: int) -> str | None:
    """Resolve one named object reference without accepting a numeric fallback."""
    import mujoco

    if object_id < 0:
        return None
    registry = {
        "body": mujoco.mjtObj.mjOBJ_BODY,
        "mesh": mujoco.mjtObj.mjOBJ_MESH,
    }
    return cast(str | None, mujoco.mj_id2name(model, registry[kind], object_id))


def _compiled_actuator_semantics(
    model: mujoco.MjModel, role: str
) -> dict[int, dict[str, CanonicalValue]]:
    """Resolve all transmission IDs to unique typed names in O(objects + actuators)."""
    import mujoco

    object_specs = {
        "JOINT": (mujoco.mjtObj.mjOBJ_JOINT, int(model.njnt)),
        "SITE": (mujoco.mjtObj.mjOBJ_SITE, int(model.nsite)),
        "TENDON": (mujoco.mjtObj.mjOBJ_TENDON, int(model.ntendon)),
        "BODY": (mujoco.mjtObj.mjOBJ_BODY, int(model.nbody)),
    }
    registries: dict[str, tuple[list[str | None], Counter[str]]] = {}
    for token, (object_type, count) in object_specs.items():
        names = [mujoco.mj_id2name(model, object_type, index) for index in range(count)]
        registries[token] = (names, Counter(name for name in names if name))
    transmissions = {
        int(mujoco.mjtTrn.mjTRN_JOINT): ("JOINT", ("JOINT",)),
        int(mujoco.mjtTrn.mjTRN_JOINTINPARENT): ("JOINTINPARENT", ("JOINT",)),
        int(mujoco.mjtTrn.mjTRN_SLIDERCRANK): ("SLIDERCRANK", ("SITE", "SITE")),
        int(mujoco.mjtTrn.mjTRN_TENDON): ("TENDON", ("TENDON",)),
        int(mujoco.mjtTrn.mjTRN_SITE): ("SITE", ("SITE", "OPTIONAL_SITE")),
        int(mujoco.mjtTrn.mjTRN_BODY): ("BODY", ("BODY",)),
    }
    semantics: dict[int, dict[str, CanonicalValue]] = {}
    for actuator_id in range(int(model.nu)):
        transmission = int(model.actuator_trntype[actuator_id])
        try:
            transmission_type, target_types = transmissions[transmission]
        except KeyError:
            raise SnapshotRefusal(
                role,
                "unsupported_actuator_transmission",
                actuator_index=actuator_id,
                transmission=transmission,
            ) from None
        raw_targets = [int(value) for value in model.actuator_trnid[actuator_id]]
        references: list[CanonicalValue] = []
        for slot, target_type in enumerate(target_types):
            target_id = raw_targets[slot]
            if target_type == "OPTIONAL_SITE" and target_id < 0:
                continue
            token = "SITE" if target_type == "OPTIONAL_SITE" else target_type
            references.append(
                _resolve_compiled_target(
                    registries,
                    token,
                    target_id,
                    role,
                    actuator_id,
                )
            )
        semantics[actuator_id] = {
            "transmission": transmission,
            "targets": {
                "transmission_type": transmission_type,
                "references": references,
            },
        }
    return semantics


def _resolve_compiled_target(
    registries: Mapping[str, tuple[list[str | None], Counter[str]]],
    object_type: str,
    target_id: int,
    role: str,
    actuator_id: int,
) -> dict[str, CanonicalValue]:
    """Return one in-domain uniquely named target reference or refuse the snapshot."""
    names, counts = registries[object_type]
    if target_id < 0 or target_id >= len(names):
        raise SnapshotRefusal(
            role,
            "actuator_target_out_of_range",
            actuator_index=actuator_id,
            target_index=target_id,
            target_object_type=object_type,
        )
    name = names[target_id]
    if not name:
        raise SnapshotRefusal(
            role,
            "actuator_target_name_missing",
            actuator_index=actuator_id,
            target_index=target_id,
            target_object_type=object_type,
        )
    if counts[name] != 1:
        raise SnapshotRefusal(
            role,
            "actuator_target_name_ambiguous",
            actuator_index=actuator_id,
            target_name=name,
            target_object_type=object_type,
        )
    return {"object_type": object_type, "name": name}


def _body_name(model: mujoco.MjModel, body_id: int) -> str:
    """Return a body reference, using the frozen world token for body zero."""
    if body_id == 0:
        return "world"
    name = _object_name(model, "body", body_id)
    return name if name else f"@unnamed-body-{body_id}"


def _mesh_geometry_sha256(model: mujoco.MjModel, mesh_id: int) -> str:
    """Identify compiled mesh geometry slices, independent of source asset encoding."""
    digest = hashlib.sha256()
    digest.update(b"metrifid.compiled_mesh_geometry.v1\0")
    _hash_array(digest, "mesh_pos", model.mesh_pos[mesh_id])
    _hash_array(digest, "mesh_quat", model.mesh_quat[mesh_id])
    _hash_array(digest, "mesh_scale", model.mesh_scale[mesh_id])
    slices = (
        ("mesh_vert", model.mesh_vert, model.mesh_vertadr, model.mesh_vertnum),
        ("mesh_normal", model.mesh_normal, model.mesh_normaladr, model.mesh_normalnum),
        ("mesh_texcoord", model.mesh_texcoord, model.mesh_texcoordadr, model.mesh_texcoordnum),
        ("mesh_face", model.mesh_face, model.mesh_faceadr, model.mesh_facenum),
        ("mesh_facenormal", model.mesh_facenormal, model.mesh_faceadr, model.mesh_facenum),
        ("mesh_facetexcoord", model.mesh_facetexcoord, model.mesh_faceadr, model.mesh_facenum),
    )
    for label, array, addresses, counts in slices:
        start = int(addresses[mesh_id])
        count = int(counts[mesh_id])
        _hash_array(digest, label, array[start : start + count])
    return digest.hexdigest()


def _hash_array(digest: Any, label: str, value: object) -> None:
    """Extend one semantic digest with an exact labeled NumPy array."""
    import numpy as np

    contiguous = np.ascontiguousarray(value)
    digest.update(f"{label}:{contiguous.dtype.str}:{tuple(contiguous.shape)}:".encode())
    digest.update(contiguous.tobytes(order="C"))


def _binary64(value: float) -> dict[str, CanonicalValue]:
    """Return the repository's exact tagged Binary64 representation."""
    return Binary64.from_float(value).to_primitive()


def _binary64_vector(value: Any) -> list[CanonicalValue]:
    """Return a fixed vector without losing any binary floating-point bits."""
    return [_binary64(float(item)) for item in value]


def _validate_reference_graph(model: mujoco.MjModel, role: str) -> None:
    """Recheck the rooted body graph and direct body/mesh reference domains."""
    nbody = int(model.nbody)
    parents = [int(value) for value in model.body_parentid]
    if len(parents) != nbody or not parents or parents[0] != 0:
        raise SnapshotRefusal(role, "body_parent_shape_or_root_invalid")
    for body_id in range(1, nbody):
        parent = parents[body_id]
        if parent < 0 or parent >= body_id:
            raise SnapshotRefusal(
                role,
                "body_graph_not_rooted_acyclic",
                body_id=body_id,
                parent_id=parent,
            )
    for label, values in (
        ("joint", model.jnt_bodyid),
        ("geom", model.geom_bodyid),
    ):
        for object_id, raw in enumerate(values):
            body_id = int(raw)
            if body_id < 0 or body_id >= nbody:
                raise SnapshotRefusal(
                    role,
                    "body_reference_out_of_range",
                    object_type=label,
                    object_id=object_id,
                    body_id=body_id,
                )
    for geom_id, raw in enumerate(model.geom_dataid):
        if int(model.geom_type[geom_id]) != 7:
            continue
        mesh_id = int(raw)
        if mesh_id < 0 or mesh_id >= int(model.nmesh):
            raise SnapshotRefusal(
                role,
                "mesh_reference_out_of_range",
                geom_id=geom_id,
                mesh_id=mesh_id,
            )


__all__ = [
    "PUBLIC_FIELD_REGISTRY_COUNT",
    "PUBLIC_FIELD_REGISTRY_SCHEMA",
    "PUBLIC_FIELD_REGISTRY_SCHEMA_VERSION",
    "PUBLIC_FIELD_REGISTRY_SHA256",
    "CompiledModelSnapshot",
    "PublicFieldFact",
    "SemanticObjectFact",
    "SnapshotRefusal",
    "build_compiled_model_snapshot",
]
