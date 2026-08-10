"""Collect model path races scenarios that pin decisions and evidence boundaries."""

from __future__ import annotations

import base64 as _b64
import hashlib as _hashlib
import os as _os
import socket as _socket
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
from metrifid._model_dependencies import (
    _compiler_directory_binding as _binding,
)
from metrifid._model_dependencies import _member_map as _members
from metrifid._model_dependencies import (
    discover_snapshot_dependencies as _discover,
)
from metrifid.operational import OperationalReasonCode


def _create_model_root(tmp_path: Path) -> Path:
    """Create the ordinary model root used before each race mutation."""
    root = tmp_path / "root"
    root.mkdir(parents=True)
    (root / "model.xml").write_text("<mujoco/>", encoding="utf-8")
    return root.resolve()


def _refusal_reason(
    exc: pytest.ExceptionInfo[closure.ModelAdmissionRefusal],
) -> OperationalReasonCode:
    """Extract the admission code produced by a raced closure read."""
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
    """Place text or bytes at a selected path under the race fixture root."""
    target = Path(root) / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        target.write_text(data, encoding="utf-8")
    else:
        target.write_bytes(data)
    return target


def _minimal_binary_stl():
    """Return one valid STL triangle for raced mesh dependency tests."""
    return (
        b"\0" * 80
        + _struct.pack("<I", 1)
        + _struct.pack("<12fH", 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0)
    )


def _minimal_png_bytes():
    """Return a valid tiny PNG for raced texture dependency tests."""
    return _b64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAIAQAAAADXcRDDAAAAEklEQVR4nGP4"
        "z8Dwn4GBgYEBADgIAv1z0nQyAAAAAElFTkSuQmCC"
    )


def _write_asset_model(root, assets, compiler='<compiler meshdir="meshes"/>'):
    """Create a MuJoCo asset model whose dependencies can be raced."""
    _write_model_member(
        root,
        "model.xml",
        f'<mujoco model="t">{compiler}<asset>{assets}</asset>'
        '<worldbody><body name="b"><joint name="j" type="hinge"/>'
        '<geom name="g" type="sphere" size="0.1"/></body></worldbody></mujoco>',
    )
    return "model.xml"


def _refusal_reason(exc):
    """Return the dependency refusal code observed after a filesystem race."""
    return exc.value.reason


_COMPOSITE_BODY = (
    '<worldbody><body name="b"><joint name="j" type="hinge"/>'
    '<geom name="g" type="sphere" size="0.1"/></body></worldbody>'
)


def _expand_composite_model(root, entrypoint="model.xml"):
    """Expand a snapshotted composite model while race hooks are active."""
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


def test_symlink_member_refuses(tmp_path):
    """14. a symlinked asset member refuses."""
    root = tmp_path / "root"
    _write_model_member(root, "meshes/keep.STL", _minimal_binary_stl())
    real = _write_model_member(tmp_path, "real.png", _minimal_png_bytes())
    entry = _write_asset_model(
        root,
        '<hfield name="h" size="1 1 0.1 0.1" file="../height_field.png"/>'
        '<mesh name="m" file="keep.STL"/>',
    )
    link = root / "height_field.png"
    link.symlink_to(real)
    with pytest.raises(ModelAdmissionRefusal):
        with _snapshot(root, entry, "baseline") as snap:
            _discover(snap, "baseline")


def test_nonregular_member_refuses(tmp_path):
    """15. a non-regular asset member refuses."""
    root = tmp_path / "root"
    _write_model_member(root, "meshes/keep.STL", _minimal_binary_stl())
    entry = _write_asset_model(root, '<mesh name="m" file="keep.STL"/>')
    fifo = root / "height_field.png"
    try:
        _os.mkfifo(fifo)
    except (AttributeError, NotImplementedError, OSError):
        pytest.skip("fifo creation unavailable on this platform")
    with pytest.raises(ModelAdmissionRefusal):
        with _snapshot(root, entry, "baseline") as snap:
            _discover(snap, "baseline")


def test_measured_member_mutation_refuses(tmp_path):
    """16. a mutated measured member with a changed size or hash refuses."""
    root = tmp_path / "root"
    _write_model_member(root, "height_field.png", _minimal_png_bytes())
    _write_model_member(root, "meshes/keep.STL", _minimal_binary_stl())
    entry = _write_asset_model(
        root,
        '<hfield name="h" size="1 1 0.1 0.1" file="../height_field.png"/>'
        '<mesh name="m" file="keep.STL"/>',
    )
    with _snapshot(root, entry, "baseline") as snap:
        (snap.snapshot_root / "height_field.png").write_bytes(_minimal_png_bytes() + b"tampered")
        with pytest.raises(ModelAdmissionRefusal) as exc:
            _discover(snap, "baseline")
        assert _refusal_reason(exc) is OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID


def test_no_basename_or_recursive_fallback(tmp_path):
    """17. no recursive filename or basename fallback can bind an undeclared file."""
    root = tmp_path / "root"
    _write_model_member(root, "meshes/keep.STL", _minimal_binary_stl())
    _write_model_member(root, "deep/nested/height_field.png", _minimal_png_bytes())
    entry = _write_asset_model(root, '<mesh name="m" file="keep.STL"/>')
    with _snapshot(root, entry, "baseline") as snap:
        members = _members(snap)
        # A same-named file elsewhere in the tree must never justify an unbound dependency.
        assert (
            _binding(snap, "baseline", members, snap.snapshot_root / ".." / "height_field.png")
            is None
        )
        assert _binding(snap, "baseline", members, snap.snapshot_root / "height_field.png") is None


def test_unused_xml_cannot_introduce_an_external_dependency(tmp_path):
    """An unrelated XML naming an external asset must not reach resolution at all."""
    outside = tmp_path / "outside"
    outside.mkdir()
    _write_model_member(outside, "external.STL", _minimal_binary_stl())
    root = tmp_path / "root"
    _write_model_member(root, "meshes/keep.STL", _minimal_binary_stl())
    _write_model_member(
        root,
        "model.xml",
        f'<mujoco><compiler meshdir="meshes"/>'
        f'<asset><mesh name="m" file="keep.STL"/></asset>{_COMPOSITE_BODY}</mujoco>',
    )
    _write_model_member(
        root,
        "unrelated.xml",
        f'<mujoco><compiler meshdir="{outside.as_posix()}"/>'
        '<asset><mesh name="x" file="external.STL"/></asset></mujoco>',
    )
    composite = _expand_composite_model(root)
    assert composite.meshdir == "meshes"
    assert composite.assets == [("mesh", "keep.STL")]


def test_include_escape_cannot_hide_an_external_file(tmp_path):
    """An include that leaves the measured root refuses, however it is spelled."""
    _write_model_member(tmp_path, "outside.xml", '<mujoco><compiler meshdir="a"/></mujoco>')
    root = tmp_path / "root"
    _write_model_member(root, "meshes/keep.STL", _minimal_binary_stl())
    for token in ("../outside.xml", "./../outside.xml", "sub/../../outside.xml"):
        _write_model_member(
            root, "model.xml", f'<mujoco><include file="{token}"/>{_COMPOSITE_BODY}</mujoco>'
        )
        with pytest.raises(ModelAdmissionRefusal) as exc:
            _expand_composite_model(root)
        assert _refusal_reason(exc) is OperationalReasonCode.MODEL_CLOSURE_PATH_ESCAPE


def test_absolute_external_include_refuses(tmp_path):
    """An absolute include outside the measured root refuses."""
    external = _write_model_member(
        tmp_path, "outside.xml", '<mujoco><compiler meshdir="a"/></mujoco>'
    )
    root = tmp_path / "root"
    _write_model_member(root, "meshes/keep.STL", _minimal_binary_stl())
    _write_model_member(
        root,
        "model.xml",
        f'<mujoco><include file="{external.as_posix()}"/>{_COMPOSITE_BODY}</mujoco>',
    )
    with pytest.raises(ModelAdmissionRefusal) as exc:
        _expand_composite_model(root)
    assert _refusal_reason(exc) is OperationalReasonCode.MODEL_CLOSURE_PATH_ESCAPE


def test_strippath_cannot_be_used_to_escape(tmp_path):
    """Strippath removes directory components; it never grants an external binding."""
    outside = tmp_path / "outside"
    outside.mkdir()
    _write_model_member(outside, "shared.STL", _minimal_binary_stl())
    root = tmp_path / "root"
    _write_model_member(root, "meshes/keep.STL", _minimal_binary_stl())
    _write_model_member(
        root,
        "model.xml",
        f'<mujoco><compiler meshdir="{outside.as_posix()}" strippath="true"/>'
        f'<asset><mesh name="m" file="../../shared.STL"/></asset>{_COMPOSITE_BODY}</mujoco>',
    )
    with _snapshot(root, "model.xml", "baseline") as snapshot:
        members = _members(snapshot)
        binding = _dep._compiler_directory_binding(
            snapshot, "baseline", members, snapshot.snapshot_root / "shared.STL"
        )
        assert binding is None


def test_mutated_included_member_cannot_change_resolution(tmp_path):
    """Rewriting an included file after measurement refuses instead of taking effect."""
    root = tmp_path / "root"
    _write_model_member(root, "meshes/keep.STL", _minimal_binary_stl())
    _write_model_member(
        root,
        "model.xml",
        f'<mujoco><include file="settings.xml"/>'
        f'<asset><mesh name="m" file="keep.STL"/></asset>{_COMPOSITE_BODY}</mujoco>',
    )
    _write_model_member(root, "settings.xml", '<mujoco><compiler meshdir="meshes"/></mujoco>')
    with _snapshot(root, "model.xml", "baseline") as snapshot:
        (snapshot.snapshot_root / "settings.xml").write_text(
            '<mujoco><compiler meshdir="/etc"/></mujoco>', encoding="utf-8"
        )
        with pytest.raises(ModelAdmissionRefusal) as exc:
            _dep._CompositeModel(snapshot, "baseline", _members(snapshot)).expand()
        assert _refusal_reason(exc) is OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID


def test_nonregular_member_refuses_without_blocking(tmp_path):
    """A special file swapped in after measurement fails closed and never blocks on open."""
    root = tmp_path / "root"
    _write_model_member(root, "meshes/keep.STL", _minimal_binary_stl())
    _write_model_member(
        root, "model.xml", f'<mujoco><include file="settings.xml"/>{_COMPOSITE_BODY}</mujoco>'
    )
    _write_model_member(root, "settings.xml", '<mujoco><compiler meshdir="meshes"/></mujoco>')
    with _snapshot(root, "model.xml", "baseline") as snapshot:
        target = snapshot.snapshot_root / "settings.xml"
        target.unlink()
        try:
            _os.mkfifo(target)
        except (AttributeError, NotImplementedError, OSError):
            pytest.skip("fifo creation unavailable on this platform")
        with pytest.raises(ModelAdmissionRefusal) as exc:
            _dep._CompositeModel(snapshot, "baseline", _members(snapshot)).expand()
        assert _refusal_reason(exc) is OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID


def test_symlinked_included_member_refuses(tmp_path):
    """A symlink swapped in after measurement is never followed."""
    real = _write_model_member(tmp_path, "real.xml", '<mujoco><compiler meshdir="/etc"/></mujoco>')
    root = tmp_path / "root"
    _write_model_member(root, "meshes/keep.STL", _minimal_binary_stl())
    _write_model_member(
        root, "model.xml", f'<mujoco><include file="settings.xml"/>{_COMPOSITE_BODY}</mujoco>'
    )
    _write_model_member(root, "settings.xml", '<mujoco><compiler meshdir="meshes"/></mujoco>')
    with _snapshot(root, "model.xml", "baseline") as snapshot:
        target = snapshot.snapshot_root / "settings.xml"
        target.unlink()
        target.symlink_to(real)
        with pytest.raises(ModelAdmissionRefusal) as exc:
            _dep._CompositeModel(snapshot, "baseline", _members(snapshot)).expand()
        assert _refusal_reason(exc) is OperationalReasonCode.MODEL_CLOSURE_MEMBER_INVALID


def test_fifo_swap_after_enumeration_refuses_within_the_bound(tmp_path):
    """A real FIFO swapped in after enumeration refuses promptly, with no writer."""
    completed, elapsed = _swap_child(tmp_path, "fifo", "fifo_root")
    assert elapsed < _SWAP_BUDGET_SECONDS
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("REFUSED MODEL_CLOSURE_MUTATED"), completed.stdout
    assert "member_changed_before_open" in completed.stdout


@pytest.mark.parametrize("swap", ["symlink", "directory"])
def test_adjacent_swaps_after_enumeration_refuse_within_the_bound(tmp_path, swap):
    """A symlink and a directory swapped in after enumeration refuse promptly.

    Both are file kinds the reader must never follow or read, so the refusal is about the kind of
    object. A regular-file replacement is a different question, answered by content: see the
    content-binding tests, which pin the measured hash to the exact bytes read.
    """
    completed, elapsed = _swap_child(tmp_path, swap, f"{swap}_root")
    assert elapsed < _SWAP_BUDGET_SECONDS
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("REFUSED MODEL_CLOSURE_MUTATED"), completed.stdout


def test_unix_socket_swap_after_enumeration_refuses_within_the_bound(tmp_path):
    """A Unix socket swapped in after enumeration refuses promptly, where supported."""
    if not hasattr(_socket, "AF_UNIX"):
        pytest.skip("AF_UNIX sockets are unavailable on this platform")
    completed, elapsed = _swap_child(tmp_path, "socket", "socket_root")
    if completed.returncode != 0 and "PermissionError" in completed.stderr:
        pytest.skip("Unix-domain socket creation is denied by the execution sandbox")
    assert elapsed < _SWAP_BUDGET_SECONDS
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.startswith("REFUSED MODEL_CLOSURE_MUTATED"), completed.stdout


def test_swapped_member_content_is_never_read(tmp_path):
    """No swap case may follow the replacement or read the external bytes."""
    root = tmp_path / "no_follow_root"
    root.mkdir()
    (root / "model.xml").write_bytes(b"<mujoco/>")
    external = tmp_path / "external.xml"
    external.write_bytes(b"<mujoco/><!-- must never be read -->")
    members = closure._enumerate_members(root, "baseline", closure.MAX_MODEL_CLOSURE_BYTES)
    member = next(m for m in members if m.relative_path == "model.xml")
    (root / "model.xml").unlink()
    (root / "model.xml").symlink_to(external)
    with pytest.raises(ModelAdmissionRefusal) as exc:
        closure._read_member(member, "baseline")
    assert _refusal_reason(exc) is OperationalReasonCode.MODEL_CLOSURE_MUTATED
    assert exc.value.evidence["issue"] == "member_changed_before_open"


def test_model_closure_refuses_replaced_intermediate_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refuse a child-directory replacement before any outside bytes are read."""
    root = tmp_path / "intermediate_root"
    outside = tmp_path / "outside"
    (root / "assets").mkdir(parents=True)
    outside.mkdir()
    (root / "model.xml").write_bytes(b"<mujoco/>")
    (root / "assets" / "inside.bin").write_bytes(b"inside")
    (outside / "inside.bin").write_bytes(b"outside secret")
    real_entry = closure._enumerated_entry
    swapped = False

    def queue_then_swap(root_arg, role, entry, pending):
        """Replace the just-queued child name with an external symlink."""
        nonlocal swapped
        result = real_entry(root_arg, role, entry, pending)
        if entry.name == "assets" and result is None and not swapped:
            (root / "assets").rename(root / "admitted_assets")
            (root / "assets").symlink_to(outside, target_is_directory=True)
            swapped = True
        return result

    monkeypatch.setattr(closure, "_enumerated_entry", queue_then_swap)
    with pytest.raises(ModelAdmissionRefusal) as exc:
        closure._measure(root, "model.xml", "baseline", closure.MAX_MODEL_CLOSURE_BYTES)
    assert swapped
    assert exc.value.reason is OperationalReasonCode.MODEL_CLOSURE_MUTATED
    assert exc.value.evidence["path"] == "assets"


def test_snapshot_verification_refuses_replaced_public_root(tmp_path: Path) -> None:
    """Require the public root name to keep identifying the retained root object."""
    root = tmp_path / "public_root"
    root.mkdir()
    (root / "model.xml").write_bytes(b"<mujoco/>")
    with closure.create_model_closure_snapshot(root, "model.xml", "baseline") as snapshot:
        root.rename(tmp_path / "detached_root")
        root.mkdir()
        (root / "model.xml").write_bytes(b"<mujoco/>")
        with pytest.raises(ModelAdmissionRefusal) as exc:
            closure.verify_model_closure_unchanged(snapshot, "baseline")
    assert exc.value.reason is OperationalReasonCode.MODEL_CLOSURE_MUTATED
    assert exc.value.evidence["issue"] == "root_changed_after_open"


def test_unchanged_regular_member_is_byte_identical(tmp_path):
    """An unchanged regular member keeps its bytes, member hash, and closure identity."""
    root = tmp_path / "stable_root"
    root.mkdir()
    payload = b'<mujoco model="stable"><worldbody/></mujoco>'
    (root / "model.xml").write_bytes(payload)
    (root / "extra.bin").write_bytes(bytes(range(256)) * 4)

    members = closure._enumerate_members(root, "baseline", closure.MAX_MODEL_CLOSURE_BYTES)
    member = next(m for m in members if m.relative_path == "model.xml")
    assert closure._read_member(member, "baseline") == payload

    first = closure.measure_model_closure(root, "model.xml", "baseline")
    second = closure.measure_model_closure(root, "model.xml", "baseline")
    assert first == second
    assert first.sha256() == second.sha256()
    by_path = {m.path: m for m in first.members}
    assert by_path["model.xml"].sha256 == _hashlib.sha256(payload).hexdigest()
    assert by_path["model.xml"].size_bytes == len(payload)
    assert set(by_path) == {"model.xml", "extra.bin"}


def test_nonblocking_open_does_not_change_regular_file_content(tmp_path):
    """O_NONBLOCK must not alter normal regular-file content or identity."""
    root = tmp_path / "nonblock_root"
    root.mkdir()
    payload = bytes(range(256)) * 1024  # larger than one read chunk boundary is not required,
    (root / "model.xml").write_bytes(payload)  # but a non-trivial size exercises the read loop
    members = closure._enumerate_members(root, "baseline", closure.MAX_MODEL_CLOSURE_BYTES)
    member = next(m for m in members if m.relative_path == "model.xml")
    content = closure._read_member(member, "baseline")
    assert content == payload
    assert _hashlib.sha256(content).hexdigest() == _hashlib.sha256(payload).hexdigest()
