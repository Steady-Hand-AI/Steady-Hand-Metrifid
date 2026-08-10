"""Collect model dependency resolution scenarios that pin decisions and evidence boundaries."""

from __future__ import annotations

import os as _os
from pathlib import Path

import pytest

from metrifid import _model_closure as closure
from metrifid import _model_dependencies as dep
from metrifid._model_closure import ModelAdmissionRefusal as _ModelAdmissionRefusal
from metrifid._model_closure import create_model_closure_snapshot as _snapshot
from metrifid._model_dependencies import _member_map
from metrifid._model_dependencies import discover_snapshot_dependencies as _discover
from metrifid.operational import OperationalReasonCode

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

    The model dependency resolution scenario observes real bytes and filesystem effects for
    model dependency resolution.
    """
    root.mkdir(parents=True)
    (root / "model.xml").write_text(xml, encoding="utf-8")
    return root.resolve()


def _refusal_reason(
    exc: pytest.ExceptionInfo[closure.ModelAdmissionRefusal],
) -> OperationalReasonCode:
    """Extract the refusal code from a dependency-resolution exception."""
    return exc.value.reason


def _write_asset_member(root, relative, data=b"binary-asset"):
    """Write write asset member data into the isolated test workspace.

    The model dependency resolution scenario observes real bytes and filesystem effects for
    model dependency resolution.
    """
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        target.write_text(data, encoding="utf-8")
    else:
        target.write_bytes(data)
    return target


def _write_asset_model(root, body, compiler='<compiler meshdir="meshes"/>', include=None):
    """Write an asset model with caller-controlled include and compiler settings."""
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
    """Construct the deps fixture used by model dependency resolution scenarios.

    Deterministic setup isolates model dependency resolution without bypassing the contract
    boundary under assertion.
    """
    snap = _snapshot(root, entrypoint, "baseline")
    try:
        return set(_discover(snap, "baseline"))
    finally:
        snap.close()


def _minimal_binary_stl():
    # Minimal binary STL: 80-byte header, uint32 triangle count, one triangle.
    """Return a valid one-triangle STL for path resolution."""
    import struct

    header = b"\0" * 80
    tri = struct.pack("<12fH", 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0)
    return header + struct.pack("<I", 1) + tri


def _minimal_png_bytes():
    """Return a tiny valid PNG for texture resolution."""
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

    The model dependency resolution scenario observes real bytes and filesystem effects for
    model dependency resolution.
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
    """Construct the expand refusal fixture used by model dependency resolution scenarios.

    Deterministic setup isolates model dependency resolution without bypassing the contract
    boundary under assertion.
    """
    snap = _snapshot(root, entrypoint, "baseline")
    try:
        with pytest.raises(_ModelAdmissionRefusal) as exc:
            dep._CompositeModel(snap, "baseline", _member_map(snap)).expand()
        return exc.value
    finally:
        snap.close()


def test_unused_xml_with_conflicting_compiler_is_ignored(tmp_path: Path) -> None:
    """1. an unused XML file with a conflicting compiler setting is ignored."""
    _write_composite_member(
        tmp_path,
        "model.xml",
        f'<mujoco><compiler meshdir="used"/><include file="inc.xml"/>{MODEL_BODY_XML}</mujoco>',
    )
    _write_composite_member(
        tmp_path, "inc.xml", '<mujoco><compiler texturedir="from_include"/></mujoco>'
    )
    _write_composite_member(
        tmp_path,
        "unrelated.xml",
        '<mujoco><compiler meshdir="never" texturedir="never" strippath="true"/></mujoco>',
    )
    composite = _composite(tmp_path)
    assert composite.meshdir == "used"
    assert composite.texturedir == "from_include"
    assert composite.strippath is False


def test_unused_xml_with_asset_declaration_is_ignored(tmp_path: Path) -> None:
    """2. an unused XML file with an asset declaration is ignored."""
    _write_composite_member(
        tmp_path,
        "model.xml",
        '<mujoco><include file="inc.xml"/>'
        f'<asset><mesh name="main" file="main.stl"/></asset>{MODEL_BODY_XML}</mujoco>',
    )
    _write_composite_member(
        tmp_path, "inc.xml", '<mujoco><asset><mesh name="inc" file="inc.stl"/></asset></mujoco>'
    )
    _write_composite_member(
        tmp_path,
        "unrelated.xml",
        '<mujoco><asset><mesh name="ghost" file="ghost.stl"/></asset></mujoco>',
    )
    composite = _composite(tmp_path)
    assert composite.assets == [("mesh", "inc.stl"), ("mesh", "main.stl")]
    assert all(token != "ghost.stl" for _, token in composite.assets)


def test_compiler_in_include_after_main_wins(tmp_path: Path) -> None:
    """3. compiler in main followed by compiler in include: included explicit values win."""
    _write_composite_member(
        tmp_path,
        "model.xml",
        f'<mujoco><compiler meshdir="first"/><include file="inc.xml"/>{MODEL_BODY_XML}</mujoco>',
    )
    _write_composite_member(tmp_path, "inc.xml", '<mujoco><compiler meshdir="second"/></mujoco>')
    assert _composite(tmp_path).meshdir == "second"


def test_compiler_in_main_after_include_wins(tmp_path: Path) -> None:
    """4. compiler in include followed by compiler in main: later main explicit values win."""
    _write_composite_member(
        tmp_path,
        "model.xml",
        f'<mujoco><include file="inc.xml"/><compiler meshdir="second"/>{MODEL_BODY_XML}</mujoco>',
    )
    _write_composite_member(tmp_path, "inc.xml", '<mujoco><compiler meshdir="first"/></mujoco>')
    assert _composite(tmp_path).meshdir == "second"


def test_nested_includes_preserve_document_order(tmp_path: Path) -> None:
    """5. nested includes preserve insertion and document order."""
    _write_composite_member(
        tmp_path,
        "model.xml",
        '<mujoco><asset><mesh name="m1" file="one.stl"/></asset>'
        '<include file="a.xml"/>'
        f'<asset><mesh name="m4" file="four.stl"/></asset>{MODEL_BODY_XML}</mujoco>',
    )
    _write_composite_member(
        tmp_path,
        "a.xml",
        '<mujoco><asset><mesh name="m2" file="two.stl"/></asset><include file="b.xml"/></mujoco>',
    )
    _write_composite_member(
        tmp_path, "b.xml", '<mujoco><asset><mesh name="m3" file="three.stl"/></asset></mujoco>'
    )
    composite = _composite(tmp_path)
    assert [token for _, token in composite.assets] == [
        "one.stl",
        "two.stl",
        "three.stl",
        "four.stl",
    ]


def test_included_top_level_children_insert_at_the_include_location(
    tmp_path: Path,
) -> None:
    """6. the included top-level element is removed and its children insert at the include."""
    _write_composite_member(
        tmp_path,
        "model.xml",
        '<mujoco><compiler meshdir="before"/><include file="inc.xml"/>'
        f'<compiler texturedir="after"/>{MODEL_BODY_XML}</mujoco>',
    )
    _write_composite_member(
        tmp_path,
        "inc.xml",
        '<mujoco><compiler meshdir="from_include" texturedir="from_include"/>'
        '<asset><mesh name="m" file="inc.stl"/></asset></mujoco>',
    )
    composite = _composite(tmp_path)
    # The include contributed its children between the two main compiler elements: its meshdir
    # overrode "before", and the main compiler that follows the include overrode its texturedir.
    assert composite.meshdir == "from_include"
    assert composite.texturedir == "after"
    assert composite.assets == [("mesh", "inc.stl")]


def test_duplicate_include_refuses(tmp_path: Path) -> None:
    """7. a duplicate include refuses."""
    _write_composite_member(
        tmp_path,
        "model.xml",
        f'<mujoco><include file="inc.xml"/><include file="inc.xml"/>{MODEL_BODY_XML}</mujoco>',
    )
    _write_composite_member(tmp_path, "inc.xml", '<mujoco><compiler meshdir="a"/></mujoco>')
    failure = _expand_refusal(tmp_path)
    assert failure.reason is OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID
    assert failure.evidence["issue"] == "duplicate_include"
    assert failure.evidence["included_file"] == "inc.xml"


def test_include_cycle_refuses(tmp_path: Path) -> None:
    """8. an include cycle refuses."""
    _write_composite_member(
        tmp_path, "model.xml", f'<mujoco><include file="a.xml"/>{MODEL_BODY_XML}</mujoco>'
    )
    _write_composite_member(tmp_path, "a.xml", '<mujoco><include file="b.xml"/></mujoco>')
    _write_composite_member(tmp_path, "b.xml", '<mujoco><include file="a.xml"/></mujoco>')
    failure = _expand_refusal(tmp_path)
    assert failure.reason is OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID
    assert failure.evidence["issue"] == "include_cycle"
    assert failure.evidence["included_file"] == "a.xml"


def test_8b_self_include_refuses(tmp_path: Path) -> None:
    """8. a file that includes itself, and a file that includes the main entrypoint, refuse."""
    _write_composite_member(
        tmp_path, "model.xml", f'<mujoco><include file="a.xml"/>{MODEL_BODY_XML}</mujoco>'
    )
    _write_composite_member(tmp_path, "a.xml", '<mujoco><include file="model.xml"/></mujoco>')
    failure = _expand_refusal(tmp_path)
    assert failure.reason is OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID
    assert failure.evidence["issue"] == "include_cycle"
    assert failure.evidence["included_file"] == "model.xml"


def test_include_path_escape_refuses(tmp_path: Path) -> None:
    """9. an include path escape refuses, relative and absolute."""
    outside = _write_composite_member(tmp_path, "outside.xml", "<mujoco/>")
    root = tmp_path / "root"
    _write_composite_member(
        root, "model.xml", f'<mujoco><include file="../outside.xml"/>{MODEL_BODY_XML}</mujoco>'
    )
    relative_failure = _expand_refusal(root)
    assert relative_failure.reason is OperationalReasonCode.MODEL_CLOSURE_PATH_ESCAPE
    assert relative_failure.evidence["offending_include"] == "../outside.xml"

    absolute_root = tmp_path / "absolute"
    _write_composite_member(
        absolute_root,
        "model.xml",
        f'<mujoco><include file="{outside.as_posix()}"/>{MODEL_BODY_XML}</mujoco>',
    )
    absolute_failure = _expand_refusal(absolute_root)
    assert absolute_failure.reason is OperationalReasonCode.MODEL_CLOSURE_PATH_ESCAPE
    assert absolute_failure.evidence["offending_include"] == outside.as_posix()


def test_10a_missing_included_member_refuses(tmp_path: Path) -> None:
    """10. an included file that is not a measured member refuses."""
    _write_composite_member(
        tmp_path, "model.xml", f'<mujoco><include file="absent.xml"/>{MODEL_BODY_XML}</mujoco>'
    )
    failure = _expand_refusal(tmp_path)
    assert failure.reason is OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID
    assert failure.evidence["issue"] == "member_not_in_measured_closure"


def test_10b_mutated_included_member_refuses(tmp_path: Path) -> None:
    """10. a measured included member mutated after measurement refuses."""
    _write_composite_member(
        tmp_path, "model.xml", f'<mujoco><include file="inc.xml"/>{MODEL_BODY_XML}</mujoco>'
    )
    _write_composite_member(tmp_path, "inc.xml", '<mujoco><compiler meshdir="a"/></mujoco>')
    snap = _snapshot(tmp_path, "model.xml", "baseline")
    try:
        (snap.snapshot_root / "inc.xml").write_text(
            '<mujoco><compiler meshdir="tampered"/></mujoco>', encoding="utf-8"
        )
        with pytest.raises(_ModelAdmissionRefusal) as exc:
            dep._CompositeModel(snap, "baseline", _member_map(snap)).expand()
        assert exc.value.reason is OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID
    finally:
        snap.close()


def test_10c_symlinked_and_nonregular_included_members_refuse(tmp_path: Path) -> None:
    """10. a symlinked or non-regular included member refuses."""
    real = _write_composite_member(tmp_path, "real.xml", '<mujoco><compiler meshdir="a"/></mujoco>')
    symlink_root = tmp_path / "symlink_root"
    _write_composite_member(
        symlink_root, "model.xml", f'<mujoco><include file="inc.xml"/>{MODEL_BODY_XML}</mujoco>'
    )
    (symlink_root / "inc.xml").symlink_to(real)
    with pytest.raises(_ModelAdmissionRefusal):
        snap = _snapshot(symlink_root, "model.xml", "baseline")
        try:
            dep._CompositeModel(snap, "baseline", _member_map(snap)).expand()
        finally:
            snap.close()

    fifo_root = tmp_path / "fifo_root"
    _write_composite_member(
        fifo_root, "model.xml", f'<mujoco><include file="inc.xml"/>{MODEL_BODY_XML}</mujoco>'
    )
    try:
        _os.mkfifo(fifo_root / "inc.xml")
    except (AttributeError, NotImplementedError, OSError):
        pytest.skip("fifo creation unavailable on this platform")
    with pytest.raises(_ModelAdmissionRefusal):
        snap = _snapshot(fifo_root, "model.xml", "baseline")
        try:
            dep._CompositeModel(snap, "baseline", _member_map(snap)).expand()
        finally:
            snap.close()


def test_later_strippath_false_overrides_earlier_true(tmp_path: Path) -> None:
    """11. a later strippath=false overrides an earlier true."""
    _write_composite_member(
        tmp_path,
        "model.xml",
        f'<mujoco><compiler strippath="true"/><include file="inc.xml"/>{MODEL_BODY_XML}</mujoco>',
    )
    _write_composite_member(tmp_path, "inc.xml", '<mujoco><compiler strippath="false"/></mujoco>')
    assert _composite(tmp_path).strippath is False


def test_later_strippath_true_overrides_earlier_false(tmp_path: Path) -> None:
    """12. a later strippath=true overrides an earlier false."""
    _write_composite_member(
        tmp_path,
        "model.xml",
        f'<mujoco><compiler strippath="false"/><include file="inc.xml"/>{MODEL_BODY_XML}</mujoco>',
    )
    _write_composite_member(tmp_path, "inc.xml", '<mujoco><compiler strippath="true"/></mujoco>')
    assert _composite(tmp_path).strippath is True


def test_later_assetdir_resets_both_directories(tmp_path: Path) -> None:
    """13. a later assetdir resets both effective directories."""
    _write_composite_member(
        tmp_path,
        "model.xml",
        '<mujoco><compiler meshdir="mesh_only" texturedir="texture_only"/>'
        f'<compiler assetdir="shared"/>{MODEL_BODY_XML}</mujoco>',
    )
    composite = _composite(tmp_path)
    assert composite.meshdir == "shared"
    assert composite.texturedir == "shared"


def test_same_element_specific_directories_override_assetdir(tmp_path: Path) -> None:
    """14. same-element meshdir/texturedir override assetdir."""
    _write_composite_member(
        tmp_path,
        "model.xml",
        '<mujoco><compiler assetdir="shared" meshdir="mesh_only" '
        f'texturedir="texture_only"/>{MODEL_BODY_XML}</mujoco>',
    )
    composite = _composite(tmp_path)
    assert composite.meshdir == "mesh_only"
    assert composite.texturedir == "texture_only"

    _write_composite_member(
        tmp_path,
        "mesh_only_model.xml",
        f'<mujoco><compiler assetdir="shared" meshdir="mesh_only"/>{MODEL_BODY_XML}</mujoco>',
    )
    partial = _composite(tmp_path, "mesh_only_model.xml")
    assert partial.meshdir == "mesh_only"
    assert partial.texturedir == "shared"


def test_later_specific_directory_overrides_earlier_assetdir(tmp_path: Path) -> None:
    """15. a later specific directory overrides an earlier assetdir."""
    _write_composite_member(
        tmp_path,
        "model.xml",
        f'<mujoco><compiler assetdir="shared"/><include file="inc.xml"/>{MODEL_BODY_XML}</mujoco>',
    )
    _write_composite_member(tmp_path, "inc.xml", '<mujoco><compiler meshdir="mesh_only"/></mujoco>')
    composite = _composite(tmp_path)
    assert composite.meshdir == "mesh_only"
    assert composite.texturedir == "shared"


def test_later_assetdir_overrides_earlier_specific_directories(tmp_path: Path) -> None:
    """16. a later assetdir overrides earlier specific directories."""
    _write_composite_member(
        tmp_path,
        "model.xml",
        f'<mujoco><compiler meshdir="mesh_only"/><include file="inc.xml"/>{MODEL_BODY_XML}</mujoco>',
    )
    _write_composite_member(tmp_path, "inc.xml", '<mujoco><compiler assetdir="shared"/></mujoco>')
    composite = _composite(tmp_path)
    assert composite.meshdir == "shared"
    assert composite.texturedir == "shared"


def test_mesh_hfield_and_texture_bind_to_measured_members(tmp_path: Path) -> None:
    """17. mesh, heightfield, and texture assets bind to exactly one measured contained member."""
    _write_composite_member(tmp_path, "meshes/part.STL", _minimal_binary_stl())
    _write_composite_member(tmp_path, "height_field.png", _minimal_png_bytes())
    _write_composite_member(tmp_path, "textures/skin.png", _minimal_png_bytes())
    _write_composite_member(
        tmp_path, "unrelated.xml", '<mujoco><compiler meshdir="never" texturedir="never"/></mujoco>'
    )
    _write_composite_member(
        tmp_path,
        "model.xml",
        '<mujoco model="t"><include file="settings.xml"/>'
        '<asset><mesh name="m" file="part.STL"/>'
        '<hfield name="h" size="1 1 0.1 0.1" file="../height_field.png"/>'
        '<texture name="t2" type="2d" file="skin.png"/></asset>'
        f"{MODEL_BODY_XML}</mujoco>",
    )
    _write_composite_member(
        tmp_path,
        "settings.xml",
        '<mujoco><compiler meshdir="meshes" texturedir="textures"/></mujoco>',
    )
    resolved = _deps(tmp_path)
    assert "meshes/part.STL" in resolved
    assert "height_field.png" in resolved
    assert "textures/skin.png" in resolved
    assert "unrelated.xml" not in resolved


def test_18a_zero_justified_bindings_refuse(tmp_path: Path) -> None:
    """18. zero justified bindings refuse without any search."""
    _write_composite_member(tmp_path, "meshes/part.STL", _minimal_binary_stl())
    # A same-named file elsewhere must never be found by search.
    _write_composite_member(tmp_path, "deep/nested/height_field.png", _minimal_png_bytes())
    _write_composite_member(
        tmp_path,
        "model.xml",
        '<mujoco model="t"><compiler meshdir="meshes"/>'
        '<asset><mesh name="m" file="part.STL"/>'
        '<hfield name="h" size="1 1 0.1 0.1" file="../height_field.png"/></asset>'
        f"{MODEL_BODY_XML}</mujoco>",
    )
    with pytest.raises(_ModelAdmissionRefusal) as exc:
        _deps(tmp_path)
    assert exc.value.reason in {
        OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID,
        OperationalReasonCode.MODEL_CLOSURE_PATH_ESCAPE,
        OperationalReasonCode.BASELINE_MODEL_COMPILE_ERROR,
    }


def test_18b_multiple_justified_bindings_refuse(tmp_path: Path) -> None:
    """18. two contained members justified by one reported dependency bind to neither."""
    _write_composite_member(tmp_path, "left/shared.STL", _minimal_binary_stl())
    _write_composite_member(tmp_path, "right/shared.STL", _minimal_binary_stl())
    _write_composite_member(
        tmp_path,
        "model.xml",
        '<mujoco model="t"><compiler strippath="true"/>'
        '<asset><mesh name="a" file="left/shared.STL"/>'
        f'<mesh name="b" file="right/shared.STL"/></asset>{MODEL_BODY_XML}</mujoco>',
    )
    snap = _snapshot(tmp_path, "model.xml", "baseline")
    try:
        members = _member_map(snap)
        # Both declarations strip to "shared.STL", so both justify the same reported path and each
        # resolves to a different contained member. The binding must return None, never a guess.
        original = dep._CompositeModel.directory
        try:
            dep._CompositeModel.directory = lambda self, kind: ""
            binding = dep._compiler_directory_binding(
                snap, "baseline", members, snap.snapshot_root / "shared.STL"
            )
        finally:
            dep._CompositeModel.directory = original
        assert binding is None
    finally:
        snap.close()
