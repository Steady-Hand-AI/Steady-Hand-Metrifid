"""Pure validation for completed model admission semantic-identity evidence."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from ._model_closure import AlignedActuator, AlignedJoint, ModelAdmissionRefusal
from .json_values import (
    CanonicalValue,
    FrozenCanonicalObject,
    canonical_json_bytes,
    canonical_sha256,
    freeze_canonical,
    thaw_canonical,
)
from .schemas import ActuatorAliasPair, AliasArtifact, JointAliasPair

if TYPE_CHECKING:
    from ._model_admission import (
        ActuatorDescriptor,
        CompiledModelIdentity,
        JointDescriptor,
    )
    from ._model_identity import SemanticAlignment


def validate_unique_named_actuators(actuators: Sequence[ActuatorDescriptor]) -> None:
    """Require all nonnull actuator semantic names to be unique."""
    names = tuple(item.name for item in actuators if item.name is not None)
    if len(names) != len(set(names)):
        raise ValueError("nonnull actuator names must be unique")


def validate_represented_joint_targets(
    joints: Sequence[JointDescriptor],
    actuators: Sequence[ActuatorDescriptor],
) -> None:
    """Require each represented JOINT target to name a represented compiled joint."""
    joint_names = {item.name for item in joints}
    if any(
        target.object_type == "JOINT" and target.name not in joint_names
        for actuator in actuators
        for target in actuator.targets
    ):
        raise ValueError("actuator joint target must name a compiled joint")


def _binding_primitive(value: FrozenCanonicalObject) -> dict[str, CanonicalValue]:
    """Thaw one frozen alias binding into a concrete canonical object."""
    primitive = thaw_canonical(value)
    if type(primitive) is not dict:
        raise TypeError("alias binding must be an object")
    return primitive


def _parse_binding(
    value: FrozenCanonicalObject,
) -> tuple[str, JointAliasPair | ActuatorAliasPair, FrozenCanonicalObject]:
    """Parse value into the parse binding representation used by model identity validation, rejecting invalid input with ValueError."""
    primitive = _binding_primitive(value)
    kind = primitive.get("kind")
    if kind == "JOINT":
        expected = {"kind", "canonical_name", "baseline_name", "candidate_name"}
        if set(primitive) != expected:
            raise ValueError("JOINT alias binding fields do not match the frozen schema")
        joint_pair = JointAliasPair.from_primitive(
            {key: val for key, val in primitive.items() if key != "kind"}
        )
        normalized: dict[str, CanonicalValue] = {"kind": "JOINT", **joint_pair.to_primitive()}
        return "JOINT", joint_pair, cast(FrozenCanonicalObject, freeze_canonical(normalized))
    if kind == "ACTUATOR":
        expected = {"kind", "canonical_name", "baseline", "candidate"}
        if set(primitive) != expected:
            raise ValueError("ACTUATOR alias binding fields do not match the frozen schema")
        actuator_pair = ActuatorAliasPair.from_primitive(
            {key: val for key, val in primitive.items() if key != "kind"}
        )
        normalized = {"kind": "ACTUATOR", **actuator_pair.to_primitive()}
        return "ACTUATOR", actuator_pair, cast(FrozenCanonicalObject, freeze_canonical(normalized))
    raise ValueError("unknown alias binding kind")


def strict_alias_bindings_from_primitive(value: object) -> tuple[FrozenCanonicalObject, ...]:
    """Parse raw completed bindings through the exact alias-pair schemas."""
    if type(value) is not list:
        raise TypeError("alias_bindings must be an array")
    parsed: list[FrozenCanonicalObject] = []
    for item in cast(list[object], value):
        if type(item) is not dict:
            raise TypeError("alias binding must be an object")
        frozen = cast(FrozenCanonicalObject, freeze_canonical(cast(CanonicalValue, item)))
        _, _, normalized = _parse_binding(frozen)
        parsed.append(normalized)
    return tuple(parsed)


def parse_alias_bindings(
    bindings: Sequence[FrozenCanonicalObject],
) -> tuple[tuple[JointAliasPair, ...], tuple[ActuatorAliasPair, ...]]:
    """Return strict typed bindings and enforce local one-use invariants."""
    joints: list[JointAliasPair] = []
    actuators: list[ActuatorAliasPair] = []
    for binding in bindings:
        kind, pair, normalized = _parse_binding(binding)
        if canonical_json_bytes(thaw_canonical(binding)) != canonical_json_bytes(
            thaw_canonical(normalized)
        ):
            raise ValueError("alias binding is not in exact canonical schema form")
        if kind == "JOINT":
            joints.append(cast(JointAliasPair, pair))
        else:
            actuators.append(cast(ActuatorAliasPair, pair))
    _validate_binding_uniqueness(joints, actuators)
    return (
        tuple(sorted(joints, key=lambda item: item.canonical_name)),
        tuple(sorted(actuators, key=lambda item: item.canonical_name)),
    )


def _unique(values: Sequence[object], field: str) -> None:
    """Require a sequence of semantic identifiers to contain no duplicates."""
    if len(values) != len(set(values)):
        raise ValueError(f"{field} must be unique")


def _validate_binding_uniqueness(
    joints: Sequence[JointAliasPair],
    actuators: Sequence[ActuatorAliasPair],
) -> None:
    """Require every alias endpoint and canonical name to be used at most once."""
    _unique([item.canonical_name for item in joints], "joint alias canonical names")
    _unique([item.baseline_name for item in joints], "baseline joint alias endpoints")
    _unique([item.candidate_name for item in joints], "candidate joint alias endpoints")
    _unique([item.canonical_name for item in actuators], "actuator alias canonical names")
    _unique(
        [canonical_json_bytes(item.baseline.to_primitive()) for item in actuators],
        "baseline actuator alias endpoints",
    )
    _unique(
        [canonical_json_bytes(item.candidate.to_primitive()) for item in actuators],
        "candidate actuator alias endpoints",
    )


def _nonoverlapping(slices: Sequence[tuple[int, int]], field: str) -> None:
    """Require positive state slices to be pairwise nonoverlapping."""
    cursor = 0
    for start, width in sorted(slices):
        if start < cursor:
            raise ValueError(f"{field} must be nonoverlapping")
        cursor = start + width


def validate_semantic_alignment_local(
    joints: Sequence[AlignedJoint],
    actuators: Sequence[AlignedActuator],
    aliases_raw_sha256: str | None,
    aliases_semantic_sha256: str | None,
    alias_bindings: Sequence[FrozenCanonicalObject],
) -> None:
    """Enforce strict local slice, address, activation, and alias invariants."""
    for role in ("baseline", "candidate"):
        _unique([getattr(item, f"{role}_qpos") for item in joints], f"{role} qpos slices")
        _unique([getattr(item, f"{role}_qvel") for item in joints], f"{role} qvel slices")
        _unique(
            [getattr(item, f"{role}_control_address") for item in actuators],
            f"{role} actuator control addresses",
        )
        activation = [
            (cast(int, getattr(item, f"{role}_activation_address")), item.activation_width)
            for item in actuators
            if item.activation_width > 0
        ]
        _nonoverlapping(activation, f"{role} activation slices")
        _unique(activation, f"{role} activation slices")
    parse_alias_bindings(alias_bindings)
    if (aliases_raw_sha256 is None) != (aliases_semantic_sha256 is None):
        raise ValueError("alias hashes must both be present or absent")
    if alias_bindings and aliases_raw_sha256 is None:
        raise ValueError("absent alias hashes require an empty binding set")


def _exact_coverage(expected: Sequence[object], actual: Sequence[object], field: str) -> None:
    """Require aligned addresses to cover the compiled address set exactly once."""
    if (
        len(actual) != len(expected)
        or len(actual) != len(set(actual))
        or set(actual) != set(expected)
    ):
        raise ValueError(f"{field} must be an exact one-to-one cover")


def _validate_pair_bijection(
    baseline: CompiledModelIdentity,
    candidate: CompiledModelIdentity,
    alignment: SemanticAlignment,
) -> None:
    """Require alignment to form a complete one-to-one mapping across compiled roles."""
    for role, compiled in (("baseline", baseline), ("candidate", candidate)):
        _exact_coverage(
            [(item.qpos_address, item.qpos_width) for item in compiled.joints],
            [getattr(item, f"{role}_qpos") for item in alignment.joints],
            f"{role} joint qpos coverage",
        )
        _exact_coverage(
            [(item.qvel_address, item.qvel_width) for item in compiled.joints],
            [getattr(item, f"{role}_qvel") for item in alignment.joints],
            f"{role} joint qvel coverage",
        )
        _exact_coverage(
            [item.control_address for item in compiled.actuators],
            [getattr(item, f"{role}_control_address") for item in alignment.actuators],
            f"{role} actuator control coverage",
        )
        _exact_coverage(
            [
                (cast(int, item.activation_address), item.activation_width)
                for item in compiled.actuators
                if item.activation_width > 0
            ],
            [
                (cast(int, getattr(item, f"{role}_activation_address")), item.activation_width)
                for item in alignment.actuators
                if item.activation_width > 0
            ],
            f"{role} actuator activation coverage",
        )


def _reconstructed_aliases(
    baseline_closure_sha256: str,
    candidate_closure_sha256: str,
    alignment: SemanticAlignment,
) -> AliasArtifact | None:
    """Rebuild the alias artifact implied by completed canonical bindings."""
    joints, actuators = parse_alias_bindings(alignment.alias_bindings)
    if alignment.aliases_raw_sha256 is None:
        if joints or actuators:
            raise ValueError("alias bindings require alias hashes")
        return None
    artifact = AliasArtifact(
        "metrifid.aliases",
        1,
        baseline_closure_sha256,
        candidate_closure_sha256,
        joints,
        actuators,
    )
    semantic_hash = canonical_sha256(artifact.to_primitive())
    if semantic_hash != alignment.aliases_semantic_sha256:
        raise ValueError("aliases_semantic_sha256 does not match reconstructed aliases")
    return artifact


def validate_model_pair_semantics(
    baseline_closure_sha256: str,
    candidate_closure_sha256: str,
    baseline: CompiledModelIdentity,
    candidate: CompiledModelIdentity,
    alignment: SemanticAlignment,
) -> None:
    """Require exact bijection and equality with generated production alignment."""
    _validate_pair_bijection(baseline, candidate, alignment)
    aliases = _reconstructed_aliases(
        baseline_closure_sha256,
        candidate_closure_sha256,
        alignment,
    )
    from ._model_identity import align_compiled_models

    try:
        expected = align_compiled_models(
            baseline,
            candidate,
            aliases,
            aliases_raw_sha256=alignment.aliases_raw_sha256,
            aliases_semantic_sha256=alignment.aliases_semantic_sha256,
        )
    except ModelAdmissionRefusal as exc:
        raise ValueError("completed alignment cannot be regenerated") from exc
    if expected.to_primitive() != alignment.to_primitive():
        raise ValueError("completed alignment differs from generated semantic alignment")
