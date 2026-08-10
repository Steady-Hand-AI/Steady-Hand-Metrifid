"""Collect model dependencies scenarios that pin decisions and evidence boundaries."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import mujoco
import pytest

from metrifid import _model_closure as closure
from metrifid import _model_dependencies as dep
from metrifid import _model_dependencies as dependencies
from metrifid._model_closure import ModelAdmissionRefusal as _ModelAdmissionRefusal
from metrifid._model_closure import create_model_closure_snapshot as _snapshot
from metrifid._model_dependencies import _member_map
from metrifid._model_dependencies import discover_snapshot_dependencies as _discover
from metrifid.operational import OperationalReasonCode
from metrifid.schemas import ModelClosureIdentity, ModelClosureMember

OBJ = """v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 2 3
f 1 2 4
f 1 3 4
f 2 3 4
"""


def _write_model_root(root: Path, xml: str = "<mujoco/>") -> Path:
    """Write write model root data into the isolated test workspace.

    The model dependencies scenario observes real bytes and filesystem effects for model
    dependencies.
    """
    root.mkdir(parents=True)
    (root / "model.xml").write_text(xml, encoding="utf-8")
    return root.resolve()


def _refusal_reason(
    exc: pytest.ExceptionInfo[closure.ModelAdmissionRefusal],
) -> OperationalReasonCode:
    """Extract the refusal code from dependency discovery failure."""
    return exc.value.reason


def _write_asset_member(root, relative, data=b"binary-asset"):
    """Write write asset member data into the isolated test workspace.

    The model dependencies scenario observes real bytes and filesystem effects for model
    dependencies.
    """
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        target.write_text(data, encoding="utf-8")
    else:
        target.write_bytes(data)
    return target


def _write_asset_model(root, body, compiler='<compiler meshdir="meshes"/>', include=None):
    """Write a model entrypoint with selected compiler, include, and asset XML."""
    parts = ['<mujoco model="t">', compiler]
    if include is not None:
        _write_asset_member(root, "included.xml", f"<mujoco>{include}</mujoco>")
        parts.append('<include file="included.xml"/>')
    parts.append("<asset>")
    parts.append(body)
    parts.append("</asset>")
    parts.append(
        '<worldbody><body name="b"><joint name="j" type="hinge"/>'
        '<geom name="g" type="sphere" size="0.1"/></body></worldbody>'
    )
    parts.append("</mujoco>")
    _write_asset_member(root, "model.xml", "\n".join(parts))
    return "model.xml"


def _deps(root, entrypoint="model.xml"):
    """Construct the deps fixture used by model dependencies scenarios.

    Deterministic setup isolates model dependencies without bypassing the contract boundary
    under assertion.
    """
    snap = _snapshot(root, entrypoint, "baseline")
    try:
        return set(_discover(snap, "baseline"))
    finally:
        snap.close()


def _minimal_binary_stl():
    # Minimal binary STL: 80-byte header, uint32 triangle count, one triangle.
    """Encode one triangle as a valid binary STL dependency."""
    import struct

    header = b"\0" * 80
    tri = struct.pack("<12fH", 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0)
    return header + struct.pack("<I", 1) + tri


def _minimal_png_bytes():
    """Decode a tiny valid PNG dependency for discovery tests."""
    import base64

    return base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAIAQAAAADXcRDDAAAAEklEQVR4nGP4"
        "z8Dwn4GBgYEBADgIAv1z0nQyAAAAAElFTkSuQmCC"
    )


MODEL_BODY_XML = (
    '<worldbody><body name="b"><joint name="j" type="hinge"/>'
    '<geom name="g" type="sphere" size="0.1"/></body></worldbody>'
)


def _write_composite_member(root: Path, relative: str, data: bytes | str) -> Path:
    """Write write composite member data into the isolated test workspace.

    The model dependencies scenario observes real bytes and filesystem effects for model
    dependencies.
    """
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        target.write_text(data, encoding="utf-8")
    else:
        target.write_bytes(data)
    return target


def _composite(root: Path, entrypoint: str = "model.xml"):
    """Expand the composite document of one measured snapshot."""
    snap = _snapshot(root, entrypoint, "baseline")
    try:
        composite = dep._CompositeModel(snap, "baseline", _member_map(snap))
        composite.expand()
        return composite
    finally:
        snap.close()


def _expand_refusal(root: Path, entrypoint: str = "model.xml"):
    """Construct the expand refusal fixture used by model dependencies scenarios.

    Deterministic setup isolates model dependencies without bypassing the contract boundary
    under assertion.
    """
    snap = _snapshot(root, entrypoint, "baseline")
    try:
        with pytest.raises(_ModelAdmissionRefusal) as exc:
            dep._CompositeModel(snap, "baseline", _member_map(snap)).expand()
        return exc.value
    finally:
        snap.close()


def test_nested_include_and_mesh_dependency_set_is_deterministic(tmp_path: Path) -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises nested include and mesh dependency set is deterministic; incomplete,
    ambiguous, or unbound model components cannot support certification.
    """
    root = tmp_path / "root"
    (root / "parts").mkdir(parents=True)
    (root / "model.xml").write_text(
        '<mujoco><compiler meshdir="parts"/><asset><mesh name="m" file="tetra.obj"/></asset>'
        '<include file="parts/body.xml"/></mujoco>',
        encoding="utf-8",
    )
    (root / "parts" / "body.xml").write_text(
        '<mujoco><worldbody><body><joint name="j"/><geom type="mesh" mesh="m" mass="1"/>'
        "</body></worldbody></mujoco>",
        encoding="utf-8",
    )
    (root / "parts" / "tetra.obj").write_text(OBJ, encoding="ascii")
    with closure.create_model_closure_snapshot(root.resolve(), "model.xml", "baseline") as snapshot:
        assert dependencies.discover_snapshot_dependencies(snapshot, "baseline") == (
            "model.xml",
            "parts/body.xml",
            "parts/tetra.obj",
        )


def test_external_dependency_and_relative_escape_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises external dependency and relative escape refuse; incomplete,
    ambiguous, or unbound model components cannot support certification.
    """
    outside = tmp_path / "outside.xml"
    outside.write_text("<mujoco/>", encoding="utf-8")
    root = _write_model_root(
        tmp_path / "root",
        f'<mujoco><include file="{outside.as_posix()}"/></mujoco>',
    )
    with closure.create_model_closure_snapshot(root, "model.xml", "candidate") as snapshot:
        with pytest.raises(closure.ModelAdmissionRefusal) as exc:
            dependencies.discover_snapshot_dependencies(snapshot, "candidate")
        assert _refusal_reason(exc) is OperationalReasonCode.MODEL_CLOSURE_PATH_ESCAPE
        # The public reason code is unchanged. Since 0.1.0a14 the composite-MJCF expansion reaches
        # the escaping include first and names it directly, which is strictly better evidence than
        # naming the dependency MuJoCo derived from it.
        assert exc.value.evidence["stage"] == "composite_mjcf_expansion"
        assert exc.value.evidence["offending_include"] == outside.as_posix()
        assert exc.value.evidence["snapshot_entrypoint"] == "model.xml"

        escaped = snapshot.snapshot_root.parent / "relative.xml"
        escaped.write_text("<mujoco/>", encoding="utf-8")
        monkeypatch.setattr(
            dependencies.mujoco,
            "mju_getXMLDependencies",
            lambda _: [str(snapshot.snapshot_entrypoint), "../relative.xml"],
        )
        with pytest.raises(closure.ModelAdmissionRefusal) as relative:
            dependencies.discover_snapshot_dependencies(snapshot, "candidate")
        assert _refusal_reason(relative) is OperationalReasonCode.MODEL_CLOSURE_PATH_ESCAPE


def test_discovery_failures_are_role_specific_and_malformed_results_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises discovery failures are role specific and malformed results refuse;
    incomplete, ambiguous, or unbound model components cannot support certification.
    """
    root = _write_model_root(tmp_path / "root")
    with closure.create_model_closure_snapshot(root, "model.xml", "baseline") as snapshot:

        def fail(_: str) -> list[str]:
            """Raise MuJoCo's dependency-enumeration failure for this branch."""
            raise mujoco.FatalError("broken dependency")

        monkeypatch.setattr(dependencies.mujoco, "mju_getXMLDependencies", fail)
        with pytest.raises(closure.ModelAdmissionRefusal) as baseline:
            dependencies.discover_snapshot_dependencies(snapshot, "baseline")
        assert _refusal_reason(baseline) is OperationalReasonCode.BASELINE_MODEL_COMPILE_ERROR
        assert baseline.value.evidence["exception_type"] == "FatalError"

        with pytest.raises(closure.ModelAdmissionRefusal) as candidate:
            dependencies.discover_snapshot_dependencies(snapshot, "candidate")
        assert _refusal_reason(candidate) is OperationalReasonCode.CANDIDATE_MODEL_COMPILE_ERROR

        monkeypatch.setattr(dependencies.mujoco, "mju_getXMLDependencies", lambda _: ("x",))
        with pytest.raises(closure.ModelAdmissionRefusal) as malformed:
            dependencies.discover_snapshot_dependencies(snapshot, "baseline")
        assert _refusal_reason(malformed) is OperationalReasonCode.BASELINE_MODEL_COMPILE_ERROR

        monkeypatch.setattr(dependencies.mujoco, "mju_getXMLDependencies", lambda _: [3])
        with pytest.raises(closure.ModelAdmissionRefusal):
            dependencies.discover_snapshot_dependencies(snapshot, "baseline")


def test_missing_invalid_and_mutated_measured_members_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises missing invalid and mutated measured members refuse; incomplete,
    ambiguous, or unbound model components cannot support certification.
    """
    root = _write_model_root(tmp_path / "root")
    with closure.create_model_closure_snapshot(root, "model.xml", "baseline") as snapshot:
        extra = snapshot.snapshot_root / "extra.xml"
        extra.write_text("<mujoco/>", encoding="utf-8")
        monkeypatch.setattr(
            dependencies.mujoco,
            "mju_getXMLDependencies",
            lambda _: [str(snapshot.snapshot_entrypoint), str(extra)],
        )
        with pytest.raises(closure.ModelAdmissionRefusal) as absent:
            dependencies.discover_snapshot_dependencies(snapshot, "baseline")
        assert _refusal_reason(absent) is OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID
        assert absent.value.evidence["issue"] == "dependency_not_in_measured_closure"

        missing = snapshot.snapshot_root / "missing.xml"
        monkeypatch.setattr(
            dependencies.mujoco,
            "mju_getXMLDependencies",
            lambda _: [str(snapshot.snapshot_entrypoint), str(missing)],
        )
        with pytest.raises(closure.ModelAdmissionRefusal) as missing_error:
            dependencies.discover_snapshot_dependencies(snapshot, "baseline")
        assert _refusal_reason(missing_error) is OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID

        snapshot.snapshot_entrypoint.write_text("<mujoco model='changed'/>", encoding="utf-8")
        monkeypatch.setattr(
            dependencies.mujoco,
            "mju_getXMLDependencies",
            lambda _: [str(snapshot.snapshot_entrypoint)],
        )
        with pytest.raises(closure.ModelAdmissionRefusal) as changed:
            dependencies.discover_snapshot_dependencies(snapshot, "baseline")
        assert _refusal_reason(changed) is OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID
        assert changed.value.evidence["issue"] == "dependency_does_not_match_measured_member"


def test_symlink_nonregular_and_entrypoint_omission_refuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises symlink nonregular and entrypoint omission refuse; incomplete,
    ambiguous, or unbound model components cannot support certification.
    """
    root = tmp_path / "root"
    root.mkdir()
    (root / "model.xml").write_text("<mujoco/>", encoding="utf-8")
    (root / "other.xml").write_text("<mujoco/>", encoding="utf-8")
    with closure.create_model_closure_snapshot(
        root.resolve(), "model.xml", "candidate"
    ) as snapshot:
        link = snapshot.snapshot_root / "link.xml"
        link.symlink_to(snapshot.snapshot_entrypoint)
        monkeypatch.setattr(dependencies.mujoco, "mju_getXMLDependencies", lambda _: [str(link)])
        with pytest.raises(closure.ModelAdmissionRefusal) as symlink:
            dependencies.discover_snapshot_dependencies(snapshot, "candidate")
        assert _refusal_reason(symlink) is OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID

        other = snapshot.snapshot_root / "other.xml"
        monkeypatch.setattr(dependencies.mujoco, "mju_getXMLDependencies", lambda _: [str(other)])
        with pytest.raises(closure.ModelAdmissionRefusal) as omitted:
            dependencies.discover_snapshot_dependencies(snapshot, "candidate")
        assert _refusal_reason(omitted) is OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID
        assert omitted.value.evidence["issue"] == "entrypoint_missing_from_dependency_set"

        directory = snapshot.snapshot_root / "dir"
        directory.mkdir()
        fake_member = ModelClosureMember(
            "dir",
            0,
            hashlib.sha256(b"").hexdigest(),
        )
        original = snapshot.identity
        snapshot.identity = cast(
            ModelClosureIdentity,
            SimpleNamespace(members=(*original.members, fake_member), member_count=3),
        )
        monkeypatch.setattr(
            dependencies.mujoco, "mju_getXMLDependencies", lambda _: [str(directory)]
        )
        with pytest.raises(closure.ModelAdmissionRefusal) as nonregular:
            dependencies.discover_snapshot_dependencies(snapshot, "candidate")
        assert _refusal_reason(nonregular) is OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID
        snapshot.identity = original


def test_internal_validation_branches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep model equivalence claims scoped to complete compiled identities.

    This scenario exercises internal validation branches; incomplete, ambiguous, or unbound
    model components cannot support certification.
    """
    root = _write_model_root(tmp_path / "root")
    with closure.create_model_closure_snapshot(root, "model.xml", "baseline") as snapshot:
        assert dependencies._candidate_path(
            snapshot, str(snapshot.snapshot_entrypoint)
        ).is_absolute()
        assert dependencies._candidate_path(snapshot, "child.xml") == (
            snapshot.snapshot_entrypoint.parent / "child.xml"
        )
        for invalid in ("", "bad\x00path"):
            with pytest.raises(ValueError):
                dependencies._candidate_path(snapshot, invalid)
        with pytest.raises(ValueError):
            dependencies.discover_snapshot_dependencies(snapshot, "comparison")
        with pytest.raises(ValueError):
            dependencies._compile_error_reason("comparison")

        duplicate = SimpleNamespace(
            identity=SimpleNamespace(
                members=(snapshot.identity.members[0], snapshot.identity.members[0]),
                member_count=2,
            )
        )
        with pytest.raises(ValueError):
            dependencies._member_map(duplicate)  # type: ignore[arg-type]

        monkeypatch.setattr(
            dependencies,
            "_raw_dependencies",
            lambda *_: [str(snapshot.snapshot_entrypoint), str(snapshot.snapshot_entrypoint)],
        )
        assert dependencies.discover_snapshot_dependencies(snapshot, "baseline") == ("model.xml",)


def test_meshdir_hfield_parent_token_resolves_into_root(tmp_path):
    """1. compiler meshdir + hfield file='../height_field.png' -> contained root/height_field.png."""
    _write_asset_member(tmp_path, "height_field.png", _minimal_png_bytes())
    _write_asset_member(tmp_path, "meshes/keep.STL", _minimal_binary_stl())
    entry = _write_asset_model(
        tmp_path,
        '<hfield name="h" size="1 1 0.1 0.1" file="../height_field.png"/>'
        '<mesh name="m" file="keep.STL"/>',
    )
    found = _deps(tmp_path, entry)
    assert "height_field.png" in found
    assert "meshes/keep.STL" in found


def test_meshdir_ordinary_mesh_still_resolves(tmp_path):
    """2. compiler meshdir + ordinary mesh file remains correct."""
    _write_asset_member(tmp_path, "meshes/part.STL", _minimal_binary_stl())
    entry = _write_asset_model(tmp_path, '<mesh name="m" file="part.STL"/>')
    assert "meshes/part.STL" in _deps(tmp_path, entry)


def test_texturedir_texture_resolves(tmp_path):
    """3. compiler texturedir + texture file resolves to the declared contained member."""
    _write_asset_member(tmp_path, "tex/skin.png", _minimal_png_bytes())
    entry = _write_asset_model(
        tmp_path,
        '<texture name="t" type="2d" file="../tex/skin.png"/>',
        compiler='<compiler texturedir="meshes"/>',
    )
    assert "tex/skin.png" in _deps(tmp_path, entry)


def test_assetdir_supplies_both_directories(tmp_path):
    """4. compiler assetdir supplies mesh/hfield and texture directories."""
    _write_asset_member(tmp_path, "shared/part.STL", _minimal_binary_stl())
    _write_asset_member(tmp_path, "shared/skin.png", _minimal_png_bytes())
    entry = _write_asset_model(
        tmp_path,
        '<mesh name="m" file="part.STL"/><texture name="t" type="2d" file="skin.png"/>',
        compiler='<compiler assetdir="shared"/>',
    )
    found = _deps(tmp_path, entry)
    assert {"shared/part.STL", "shared/skin.png"} <= found


def test_explicit_directories_override_assetdir(tmp_path):
    """5. explicit meshdir/texturedir override assetdir."""
    _write_asset_member(tmp_path, "m/part.STL", _minimal_binary_stl())
    _write_asset_member(tmp_path, "t/skin.png", _minimal_png_bytes())
    _write_asset_member(tmp_path, "shared/part.STL", _minimal_binary_stl())
    entry = _write_asset_model(
        tmp_path,
        '<mesh name="m" file="part.STL"/><texture name="t" type="2d" file="skin.png"/>',
        compiler='<compiler assetdir="shared" meshdir="m" texturedir="t"/>',
    )
    found = _deps(tmp_path, entry)
    assert {"m/part.STL", "t/skin.png"} <= found
    assert "shared/part.STL" not in found


def test_strippath_strips_the_asset_token(tmp_path):
    """6. strippath=true strips the asset file token before joining."""
    from metrifid._model_dependencies import _compiler_relative_join

    base = tmp_path
    joined = _compiler_relative_join(base, "meshes", "ignored/deeper/part.STL", True)
    assert joined == base / "meshes" / "part.STL"
    kept = _compiler_relative_join(base, "meshes", "ignored/deeper/part.STL", False)
    assert kept == base / "meshes" / "ignored" / "deeper" / "part.STL"


def test_compiler_element_in_included_xml_is_honored(tmp_path):
    """7. a compiler element supplied by an included XML file is honored."""
    _write_asset_member(tmp_path, "height_field.png", _minimal_png_bytes())
    _write_asset_member(tmp_path, "meshes/keep.STL", _minimal_binary_stl())
    _write_asset_member(tmp_path, "assets.xml", '<mujoco><compiler meshdir="meshes"/></mujoco>')
    _write_asset_member(
        tmp_path,
        "model.xml",
        '<mujoco model="t"><include file="assets.xml"/>'
        '<asset><hfield name="h" size="1 1 0.1 0.1" file="../height_field.png"/>'
        '<mesh name="m" file="keep.STL"/></asset>'
        '<worldbody><body name="b"><joint name="j" type="hinge"/>'
        '<geom name="g" type="sphere" size="0.1"/></body></worldbody></mujoco>',
    )
    found = _deps(tmp_path, "model.xml")
    assert "height_field.png" in found


def test_8b_unbound_asset_refuses_without_search(tmp_path):
    """8/17. an asset that binds to no contained member refuses; no tree search may rescue it."""
    from metrifid._model_dependencies import _compiler_directory_binding, _member_map

    _write_asset_member(tmp_path, "meshes/keep.STL", _minimal_binary_stl())
    _write_asset_member(tmp_path, "deep/nested/secret.png", _minimal_png_bytes())
    _write_asset_member(
        tmp_path,
        "model.xml",
        '<mujoco model="t"><compiler meshdir="meshes"/>'
        '<asset><mesh name="m" file="keep.STL"/></asset>'
        '<worldbody><body name="b"><joint name="j" type="hinge"/>'
        '<geom name="g" type="sphere" size="0.1"/></body></worldbody></mujoco>',
    )
    snap = _snapshot(tmp_path, "model.xml", "baseline")
    try:
        members = _member_map(snap)
        # A path naming a file that exists in the tree but is not a declared asset must not bind.
        bogus = snap.snapshot_root / "secret.png"
        assert _compiler_directory_binding(snap, "baseline", members, bogus) is None
    finally:
        snap.close()


def test_multiple_justified_members_refuses(tmp_path):
    """9. two contained members justified by one reported dependency refuses."""
    _write_asset_member(tmp_path, "left/shared.STL", _minimal_binary_stl())
    _write_asset_member(tmp_path, "right/shared.STL", _minimal_binary_stl())
    _write_asset_member(
        tmp_path,
        "model.xml",
        '<mujoco model="t"><compiler meshdir="left" strippath="true"/>'
        '<asset><mesh name="a" file="left/shared.STL"/>'
        '<mesh name="b" file="right/shared.STL"/></asset>'
        '<worldbody><body name="b"><joint name="j" type="hinge"/>'
        '<geom name="g" type="sphere" size="0.1"/></body></worldbody></mujoco>',
    )
    snap = _snapshot(tmp_path, "model.xml", "baseline")
    try:
        members = _member_map(snap)
        composite = dep._CompositeModel(snap, "baseline", members)
        composite.expand()
        # Both declarations strip to the same basename, so both justify the same reported path.
        # Two justified members must never silently pick one.
        original = dep._CompositeModel.directory
        try:
            dep._CompositeModel.directory = lambda self, kind: "left"
            left = dep._compiler_directory_binding(
                snap, "baseline", members, snap.snapshot_root / "shared.STL"
            )
            dep._CompositeModel.directory = lambda self, kind: "right"
            right = dep._compiler_directory_binding(
                snap, "baseline", members, snap.snapshot_root / "shared.STL"
            )
        finally:
            dep._CompositeModel.directory = original
        assert left is not None
        assert right is not None
        assert left[1] != right[1]
    finally:
        snap.close()
