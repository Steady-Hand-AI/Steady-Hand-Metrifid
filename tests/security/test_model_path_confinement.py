"""Collect model path confinement scenarios that pin decisions and evidence boundaries."""

from __future__ import annotations

import base64 as _b64
import hashlib as _hashlib
import os
import os as _os
import struct as _struct
import subprocess as _subprocess
import sys as _sys
import textwrap as _textwrap
import time as _time
from pathlib import Path

import pytest

from metrifid import _model_closure as closure
from metrifid import _model_dependencies as _dep
from metrifid._model_closure import ModelAdmissionRefusal
from metrifid._model_closure import create_model_closure_snapshot as _snapshot
from metrifid._model_dependencies import _member_map as _members
from metrifid._model_dependencies import (
    discover_snapshot_dependencies as _discover,
)
from metrifid.operational import OperationalReasonCode


def _create_model_root(tmp_path: Path) -> Path:
    """Create an absolute model root containing one minimal XML entrypoint."""
    root = tmp_path / "root"
    root.mkdir(parents=True)
    (root / "model.xml").write_text("<mujoco/>", encoding="utf-8")
    return root.resolve()


def _refusal_reason(
    exc: pytest.ExceptionInfo[closure.ModelAdmissionRefusal],
) -> OperationalReasonCode:
    """Extract the typed admission reason from a confinement refusal."""
    return exc.value.reason


OBJ = """v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
f 1 2 3
f 1 2 4
f 1 3 4
f 2 3 4
"""


def _write_model_member(root, relative, data=b"x"):
    """Write text or binary fixture data beneath a model root."""
    target = Path(root) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        target.write_text(data, encoding="utf-8")
    else:
        target.write_bytes(data)
    return target


def _minimal_binary_stl():
    """Encode one valid binary STL triangle for mesh-path tests."""
    return (
        b"\0" * 80
        + _struct.pack("<I", 1)
        + _struct.pack("<12fH", 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0)
    )


def _minimal_png_bytes():
    """Decode a minimal PNG payload for texture-path tests."""
    return _b64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAIAQAAAADXcRDDAAAAEklEQVR4nGP4"
        "z8Dwn4GBgYEBADgIAv1z0nQyAAAAAElFTkSuQmCC"
    )


def _write_asset_model(root, assets, compiler='<compiler meshdir="meshes"/>'):
    """Write a compilable MuJoCo model with caller-selected asset references."""
    _write_model_member(
        root,
        "model.xml",
        f'<mujoco model="t">{compiler}<asset>{assets}</asset>'
        '<worldbody><body name="b"><joint name="j" type="hinge"/>'
        '<geom name="g" type="sphere" size="0.1"/></body></worldbody></mujoco>',
    )
    return "model.xml"


def _refusal_reason(exc):
    """Return the refusal code emitted while resolving model dependencies."""
    return exc.value.reason


_COMPOSITE_BODY = (
    '<worldbody><body name="b"><joint name="j" type="hinge"/>'
    '<geom name="g" type="sphere" size="0.1"/></body></worldbody>'
)


def _expand_composite_model(root, entrypoint="model.xml"):
    """Expand model dependencies from an immutable closure snapshot."""
    with _snapshot(root, entrypoint, "baseline") as snapshot:
        composite = _dep._CompositeModel(snapshot, "baseline", _members(snapshot))
        composite.expand()
        return composite


_SWAP_BUDGET_SECONDS = 2.0


_SWAP_CHILD = _textwrap.dedent(
    """
    import os, socket, sys
    from pathlib import Path
    from metrifid import _model_closure as closure

    root = Path(sys.argv[1]).resolve()
    swap = sys.argv[2]
    target = root / "model.xml"


    def perform_swap():
        target.unlink()
        if swap == "fifo":
            os.mkfifo(target)
        elif swap == "symlink":
            target.symlink_to(root / "elsewhere.xml")
        elif swap == "directory":
            target.mkdir()
        elif swap == "regular":
            target.write_bytes(b"<mujoco/>")
        elif swap == "socket":
            # AF_UNIX paths are capped near 104 bytes, so bind by relative name.
            os.chdir(root)
            socket.socket(socket.AF_UNIX).bind("model.xml")
        else:
            raise SystemExit("unknown swap")


    # The defect lives in the window between full-root enumeration and the member open. The swap is
    # therefore landed immediately after the real enumeration returns and before any member is read.
    # Nothing in the read path, the refusal, or the identity checks is stubbed, and no writer is ever
    # opened on the FIFO.
    real_enumerate = closure._enumerate_members
    swapped = []


    def enumerate_then_swap(*args, **kwargs):
        members = real_enumerate(*args, **kwargs)
        if not swapped:
            assert any(m.relative_path == "model.xml" for m in members), members
            perform_swap()
            swapped.append(True)
        return members


    closure._enumerate_members = enumerate_then_swap
    try:
        closure.measure_model_closure(root, "model.xml", "baseline")
    except closure.ModelAdmissionRefusal as exc:
        print("REFUSED", exc.reason.value, exc.evidence.get("issue"))
        raise SystemExit(0)
    print("NO_REFUSAL")
    raise SystemExit(3)
    """
)


def _swap_child(tmp_path, swap: str, name: str):
    """Run one post-enumeration swap in a child process under a hard wall-clock bound."""
    root = tmp_path / name
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.xml").write_bytes(b"<mujoco/>")
    (root / "elsewhere.xml").write_bytes(b"<mujoco/><!-- external target -->")
    script = root / "child.py"
    script.write_text(_SWAP_CHILD, encoding="utf-8")

    env = dict(_os.environ)
    env["PYTHONPATH"] = str(Path(closure.__file__).resolve().parents[1])
    start = _time.monotonic()
    try:
        completed = _subprocess.run(
            [_sys.executable, str(script), str(root), swap],
            capture_output=True,
            text=True,
            timeout=_SWAP_BUDGET_SECONDS,
            env=env,
        )
    except _subprocess.TimeoutExpired:
        elapsed = _time.monotonic() - start
        raise AssertionError(
            f"the {swap} swap did not complete within {_SWAP_BUDGET_SECONDS}s "
            f"(waited {elapsed:.2f}s); the reader blocked instead of failing closed"
        ) from None
    return completed, _time.monotonic() - start


def test_root_must_be_absolute_existing_real_directory(tmp_path: Path) -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises root must be absolute existing real directory; the assertions bind
    admission to exact model bytes, resource boundaries, or an explicit refusal reason.
    """
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        closure.measure_model_closure(Path("relative"), "model.xml", "baseline")
    assert _refusal_reason(exc) is OperationalReasonCode.MODEL_ROOT_INVALID
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        closure.measure_model_closure((tmp_path / "missing").resolve(), "model.xml", "baseline")
    assert _refusal_reason(exc) is OperationalReasonCode.MODEL_ROOT_INVALID
    regular = tmp_path / "file"
    regular.write_text("x", encoding="utf-8")
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        closure.measure_model_closure(regular.resolve(), "model.xml", "baseline")
    assert _refusal_reason(exc) is OperationalReasonCode.MODEL_ROOT_INVALID
    link = tmp_path / "root-link"
    link.symlink_to(_create_model_root(tmp_path / "target"), target_is_directory=True)
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        closure.measure_model_closure(link.absolute(), "model.xml", "baseline")
    assert _refusal_reason(exc) is OperationalReasonCode.MODEL_ROOT_INVALID


@pytest.mark.parametrize(
    "entrypoint",
    [
        "/model.xml",
        "../model.xml",
        "a/../model.xml",
        ".",
        "",
        "a\\model.xml",
        "model.xml\x00tail",
        "model.json",
        "a//model.xml",
    ],
)
def test_entrypoint_path_escape_and_shape_refuse(tmp_path: Path, entrypoint: str) -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises entrypoint path escape and shape refuse; the assertions bind
    admission to exact model bytes, resource boundaries, or an explicit refusal reason.
    """
    root = _create_model_root(tmp_path)
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        closure.measure_model_closure(root, entrypoint, "baseline")
    assert exc.value.reason in {
        OperationalReasonCode.MODEL_CLOSURE_PATH_ESCAPE,
        OperationalReasonCode.MODEL_ENTRYPOINT_INVALID,
    }


def test_non_string_entrypoint_and_missing_entrypoint_refuse(tmp_path: Path) -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises non string entrypoint and missing entrypoint refuse; the assertions
    bind admission to exact model bytes, resource boundaries, or an explicit refusal reason.
    """
    root = _create_model_root(tmp_path)
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        closure.measure_model_closure(root, 3, "baseline")  # type: ignore[arg-type]
    assert _refusal_reason(exc) is OperationalReasonCode.MODEL_ENTRYPOINT_INVALID
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        closure.measure_model_closure(root, "missing.xml", "baseline")
    assert _refusal_reason(exc) is OperationalReasonCode.MODEL_ENTRYPOINT_INVALID


def test_member_symlink_and_special_file_refuse(tmp_path: Path) -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises member symlink and special file refuse; the assertions bind
    admission to exact model bytes, resource boundaries, or an explicit refusal reason.
    """
    root = _create_model_root(tmp_path)
    target = tmp_path / "outside"
    target.write_text("secret", encoding="utf-8")
    (root / "link").symlink_to(target)
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        closure.measure_model_closure(root, "model.xml", "baseline")
    assert _refusal_reason(exc) is OperationalReasonCode.MODEL_CLOSURE_SYMLINK_REFUSED
    (root / "link").unlink()
    fifo = root / "pipe"
    os.mkfifo(fifo)
    try:
        with pytest.raises(closure.ModelAdmissionRefusal) as exc:
            closure.measure_model_closure(root, "model.xml", "baseline")
        assert _refusal_reason(exc) is OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID
    finally:
        fifo.unlink()


def test_surrogate_component_refuses_as_non_utf8() -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises surrogate component refuses as non utf8; the assertions bind
    admission to exact model bytes, resource boundaries, or an explicit refusal reason.
    """
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        closure._validate_component("\ud800", "baseline", "bad")
    assert _refusal_reason(exc) is OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID


def test_absolute_include_and_asset_escape_cannot_be_hidden(tmp_path: Path) -> None:
    # Closure admission remains rooted even when source XML contains an external reference;
    # MuJoCo compilation later refuses or resolves only the immutable snapshot.
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises absolute include and asset escape cannot be hidden; the assertions
    bind admission to exact model bytes, resource boundaries, or an explicit refusal reason.
    """
    root = _create_model_root(tmp_path)
    outside = tmp_path / "outside.xml"
    outside.write_text("<mujoco/>", encoding="utf-8")
    (root / "model.xml").write_text(
        f'<mujoco><include file="{outside.as_posix()}"/></mujoco>', encoding="utf-8"
    )
    identity = closure.measure_model_closure(root, "model.xml", "baseline")
    assert [member.path for member in identity.members] == ["model.xml"]


def test_external_include_changes_dynamics_but_product_refuses_before_identity(
    tmp_path: Path,
) -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises external include changes dynamics but product refuses before
    identity; the assertions bind admission to exact model bytes, resource boundaries, or an
    explicit refusal reason.
    """
    import mujoco

    from metrifid import _model_identity as identity

    external = tmp_path / "external.xml"
    root = tmp_path / "root"
    root.mkdir()
    root_xml = f'<mujoco><include file="{external.as_posix()}"/></mujoco>'
    (root / "model.xml").write_text(root_xml, encoding="utf-8")
    root_sha = _hashlib.sha256((root / "model.xml").read_bytes()).hexdigest()

    def write_external(damping: str) -> None:
        """Write write external data into the isolated test workspace.

        The model path confinement scenario observes real bytes and filesystem effects for
        external include changes dynamics but product refuses before identity.
        """
        external.write_text(
            '<mujoco><worldbody><body><joint name="j" damping="'
            + damping
            + '"/><geom size=".1" mass="1"/></body></worldbody>'
            '<actuator><motor name="a" joint="j"/></actuator></mujoco>',
            encoding="utf-8",
        )

    write_external("0.1")
    first = mujoco.MjModel.from_xml_path(str(root / "model.xml"))
    with pytest.raises(closure.ModelAdmissionRefusal) as first_refusal:
        identity.build_model_pair_identity(root.resolve(), "model.xml", root.resolve(), "model.xml")
    assert _refusal_reason(first_refusal) is OperationalReasonCode.MODEL_CLOSURE_PATH_ESCAPE

    write_external("9.0")
    second = mujoco.MjModel.from_xml_path(str(root / "model.xml"))
    with pytest.raises(closure.ModelAdmissionRefusal) as second_refusal:
        identity.build_model_pair_identity(root.resolve(), "model.xml", root.resolve(), "model.xml")
    assert _refusal_reason(second_refusal) is OperationalReasonCode.MODEL_CLOSURE_PATH_ESCAPE
    assert float(first.dof_damping[0]) == pytest.approx(0.1)
    assert float(second.dof_damping[0]) == pytest.approx(9.0)
    assert _hashlib.sha256((root / "model.xml").read_bytes()).hexdigest() == root_sha


@pytest.mark.parametrize("mode", ["asset", "meshdir", "assetdir"])
def test_external_mesh_and_compiler_directories_refuse(tmp_path: Path, mode: str) -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises external mesh and compiler directories refuse; the assertions bind
    admission to exact model bytes, resource boundaries, or an explicit refusal reason.
    """
    from metrifid import _model_identity as identity

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "mesh.obj").write_text(OBJ, encoding="ascii")
    root = tmp_path / "root"
    root.mkdir()
    if mode == "asset":
        compiler = ""
        file_value = (outside / "mesh.obj").as_posix()
    else:
        compiler = f'<compiler {mode}="{outside.as_posix()}"/>'
        file_value = "mesh.obj"
    (root / "model.xml").write_text(
        f'<mujoco>{compiler}<asset><mesh name="mesh" file="{file_value}"/></asset>'
        '<worldbody><body><joint name="j"/><geom type="mesh" mesh="mesh" mass="1"/>'
        '</body></worldbody><actuator><motor name="a" joint="j"/></actuator></mujoco>',
        encoding="utf-8",
    )
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        identity.build_model_pair_identity(root.resolve(), "model.xml", root.resolve(), "model.xml")
    assert _refusal_reason(exc) is OperationalReasonCode.MODEL_CLOSURE_PATH_ESCAPE


def test_relative_include_escape_never_returns_pair_identity(tmp_path: Path) -> None:
    """Prevent unmeasured model state from supporting an equivalence claim.

    This scenario exercises relative include escape never returns pair identity; the assertions
    bind admission to exact model bytes, resource boundaries, or an explicit refusal reason.
    """
    from metrifid import _model_identity as identity

    root = tmp_path / "root"
    root.mkdir()
    (tmp_path / "outside.xml").write_text("<mujoco/>", encoding="utf-8")
    (root / "model.xml").write_text(
        '<mujoco><include file="../outside.xml"/></mujoco>', encoding="utf-8"
    )
    with pytest.raises(closure.ModelAdmissionRefusal) as exc:
        identity.build_model_pair_identity(root.resolve(), "model.xml", root.resolve(), "model.xml")
    assert _refusal_reason(exc) in {
        OperationalReasonCode.MODEL_CLOSURE_PATH_ESCAPE,
        OperationalReasonCode.BASELINE_MODEL_COMPILE_ERROR,
    }


def test_absolute_external_asset_filename_refuses(tmp_path):
    """10. an absolute asset filename outside the root refuses."""
    outside = tmp_path / "outside"
    outside.mkdir()
    external = _write_model_member(outside, "external.png", _minimal_png_bytes())
    root = tmp_path / "root"
    _write_model_member(root, "meshes/keep.STL", _minimal_binary_stl())
    entry = _write_asset_model(
        root,
        f'<hfield name="h" size="1 1 0.1 0.1" file="{external.as_posix()}"/>'
        '<mesh name="m" file="keep.STL"/>',
    )
    with _snapshot(root, entry, "baseline") as snap:
        with pytest.raises(ModelAdmissionRefusal) as exc:
            _discover(snap, "baseline")
        assert _refusal_reason(exc) in {
            OperationalReasonCode.MODEL_CLOSURE_PATH_ESCAPE,
            OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID,
            OperationalReasonCode.BASELINE_MODEL_COMPILE_ERROR,
        }


def test_absolute_external_compiler_directory_refuses(tmp_path):
    """11. an absolute compiler directory outside the root refuses."""
    outside = tmp_path / "outside"
    _write_model_member(outside, "part.STL", _minimal_binary_stl())
    root = tmp_path / "root"
    _write_model_member(root, "meshes/keep.STL", _minimal_binary_stl())
    entry = _write_asset_model(
        root,
        '<mesh name="m" file="part.STL"/>',
        compiler=f'<compiler meshdir="{outside.as_posix()}"/>',
    )
    with _snapshot(root, entry, "baseline") as snap:
        with pytest.raises(ModelAdmissionRefusal):
            _discover(snap, "baseline")


def test_relative_compiler_directory_escape_refuses(tmp_path):
    """12. a relative compiler directory that escapes the root refuses."""
    _write_model_member(tmp_path, "outside/part.STL", _minimal_binary_stl())
    root = tmp_path / "root"
    _write_model_member(root, "meshes/keep.STL", _minimal_binary_stl())
    entry = _write_asset_model(
        root, '<mesh name="m" file="part.STL"/>', compiler='<compiler meshdir="../outside"/>'
    )
    with _snapshot(root, entry, "baseline") as snap:
        with pytest.raises(ModelAdmissionRefusal):
            _discover(snap, "baseline")


def test_asset_escape_after_joining_refuses(tmp_path):
    """13. an asset token that escapes after compiler-directory joining refuses."""
    _write_model_member(tmp_path, "escaped.png", _minimal_png_bytes())
    root = tmp_path / "root"
    _write_model_member(root, "meshes/keep.STL", _minimal_binary_stl())
    entry = _write_asset_model(
        root,
        '<hfield name="h" size="1 1 0.1 0.1" file="../../escaped.png"/>'
        '<mesh name="m" file="keep.STL"/>',
    )
    with _snapshot(root, entry, "baseline") as snap:
        with pytest.raises(ModelAdmissionRefusal):
            _discover(snap, "baseline")


_ORIGINAL_MODEL = b"<mujoco/>"
_REPLACEMENT_MODEL = b"<mujocoX>"
assert len(_ORIGINAL_MODEL) == len(_REPLACEMENT_MODEL)


def _replacement_test_root(tmp_path: Path, name: str) -> Path:
    """Create a named model root containing the pre-replacement bytes."""
    root = tmp_path / name
    root.mkdir()
    (root / "model.xml").write_bytes(_ORIGINAL_MODEL)
    return root


def _measure_after_enumeration_replacement(root: Path, replacement: bytes):
    """Replace an enumerated member before open and return its measured evidence."""
    real_enumerate = closure._enumerate_members
    observed: dict[str, bool] = {}

    def enumerate_then_replace(*args, **kwargs):
        """Replace the regular file immediately after real closure enumeration."""
        members = real_enumerate(*args, **kwargs)
        if not observed:
            target = root / "model.xml"
            before = target.lstat()
            target.write_bytes(replacement)
            after = target.lstat()
            observed.update(
                inode_reused=before.st_ino == after.st_ino,
                identity_tuple_equal=(before.st_dev, before.st_ino, before.st_mode, before.st_size)
                == (after.st_dev, after.st_ino, after.st_mode, after.st_size),
            )
        return members

    closure._enumerate_members = enumerate_then_replace
    try:
        identity = closure.measure_model_closure(root, "model.xml", "baseline")
    finally:
        closure._enumerate_members = real_enumerate
    member = next(item for item in identity.members if item.path == "model.xml")
    return identity, member, observed


def _snapshot_after_enumeration_replacement(root: Path, replacement: bytes):
    """Snapshot replacement bytes introduced after closure enumeration."""
    real_enumerate = closure._enumerate_members
    replaced: list[bool] = []

    def enumerate_then_replace(*args, **kwargs):
        """Replace the member before the snapshot reader opens it."""
        members = real_enumerate(*args, **kwargs)
        if not replaced:
            target = root / "model.xml"
            target.write_bytes(replacement)
            replaced.append(True)
        return members

    closure._enumerate_members = enumerate_then_replace
    try:
        return _snapshot(root, "model.xml", "baseline")
    finally:
        closure._enumerate_members = real_enumerate


def test_byte_identical_replacement_binds_the_bytes_present(tmp_path: Path) -> None:
    """Bind byte-identical replacement content without relying on inode identity."""
    root = _replacement_test_root(tmp_path, "identical_root")
    _, member, observed = _measure_after_enumeration_replacement(root, _ORIGINAL_MODEL)
    present = (root / "model.xml").read_bytes()
    assert member.sha256 == _hashlib.sha256(present).hexdigest()
    assert member.sha256 == _hashlib.sha256(_ORIGINAL_MODEL).hexdigest()
    assert member.size_bytes == len(_ORIGINAL_MODEL)
    assert observed, "replacement hook did not execute"


def test_same_size_replacement_binds_the_replacement_bytes(tmp_path: Path) -> None:
    """Bind a same-size replacement to its digest and immutable snapshot bytes."""
    root = _replacement_test_root(tmp_path, "different_root")
    _, member, observed = _measure_after_enumeration_replacement(root, _REPLACEMENT_MODEL)
    original_sha = _hashlib.sha256(_ORIGINAL_MODEL).hexdigest()
    replacement_sha = _hashlib.sha256(_REPLACEMENT_MODEL).hexdigest()
    assert original_sha != replacement_sha
    assert member.sha256 == replacement_sha
    assert member.sha256 != original_sha
    assert member.sha256 == _hashlib.sha256((root / "model.xml").read_bytes()).hexdigest()
    assert observed, "replacement hook did not execute"

    with _snapshot_after_enumeration_replacement(
        _replacement_test_root(tmp_path, "different_snapshot"), _REPLACEMENT_MODEL
    ) as snapshot:
        snapshot_bytes = (snapshot.snapshot_root / "model.xml").read_bytes()
        snapshot_member = next(
            item for item in snapshot.identity.members if item.path == "model.xml"
        )
        assert snapshot_bytes == _REPLACEMENT_MODEL
        assert snapshot_member.sha256 == replacement_sha


def test_persistent_replacement_matches_snapshot_at_final_verification(tmp_path: Path) -> None:
    """Accept final verification while live bytes still match the private snapshot."""
    root = _replacement_test_root(tmp_path, "persistent_root")
    with _snapshot_after_enumeration_replacement(root, _REPLACEMENT_MODEL) as snapshot:
        member = next(item for item in snapshot.identity.members if item.path == "model.xml")
        assert member.sha256 == _hashlib.sha256(_REPLACEMENT_MODEL).hexdigest()
        assert (snapshot.snapshot_root / "model.xml").read_bytes() == _REPLACEMENT_MODEL
        assert (root / "model.xml").read_bytes() == _REPLACEMENT_MODEL
        closure.verify_model_closure_unchanged(snapshot, "baseline")


def test_restoring_original_bytes_after_snapshot_is_refused(tmp_path: Path) -> None:
    """Refuse final verification when live bytes diverge from the snapshot."""
    root = _replacement_test_root(tmp_path, "restore_root")
    with _snapshot_after_enumeration_replacement(root, _REPLACEMENT_MODEL) as snapshot:
        member = next(item for item in snapshot.identity.members if item.path == "model.xml")
        assert member.sha256 == _hashlib.sha256(_REPLACEMENT_MODEL).hexdigest()
        (root / "model.xml").write_bytes(_ORIGINAL_MODEL)
        with pytest.raises(ModelAdmissionRefusal) as exc:
            closure.verify_model_closure_unchanged(snapshot, "baseline")
        assert exc.value.reason is OperationalReasonCode.MODEL_CLOSURE_MUTATED
        assert (
            exc.value.evidence["expected_closure_sha256"]
            != exc.value.evidence["actual_closure_sha256"]
        )
