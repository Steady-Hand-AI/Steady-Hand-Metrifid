#!/usr/bin/env python3
"""Characterize complete MuJoCo MJB identity on the active runtime.

The harness compiles synthetic MJCF fixtures, serializes every byte returned by
``mj_saveModel``, and checks determinism, relocation, neutral-source refactors,
physical perturbations, and known complete-MJB over-sensitivity. Supplying
``--model`` adds repeated-compilation and relocation checks for one real MJCF
entrypoint.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import textwrap
from collections.abc import Sequence

import mujoco
import numpy as np

ASSETS_XML = textwrap.dedent(
    """\
    <mujoco model="probe_assets">
      <default>
        <geom friction="0.7 0.05 0.01"/>
      </default>
    </mujoco>
    """
)

MODEL_XML = textwrap.dedent(
    """\
    <mujoco model="probe">
      <compiler angle="radian" meshdir="{meshdir}" autolimits="true"/>
      <include file="assets.xml"/>
      <option timestep="0.002" integrator="implicitfast"/>
      <asset>
        <mesh name="part" file="{meshfile}" scale="1 1 1"/>
      </asset>
      <worldbody>
        <geom name="floor" type="plane" size="5 5 0.1"/>
        <body name="link1" pos="0 0 1">
          <joint name="j1" type="hinge" axis="0 1 0" damping="0.11" armature="0.01"/>
          <geom name="g1" type="capsule" fromto="0 0 0 0 0 -0.4" size="0.05" mass="0.734"/>
          <body name="link2" pos="0 0 -0.4">
            <joint name="j2" type="hinge" axis="0 1 0" damping="0.07"/>
            <geom name="g2" type="mesh" mesh="part" mass="0.2"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <position name="a1" joint="j1" kp="40" ctrlrange="-1 1"/>
        <position name="a2" joint="j2" kp="40" ctrlrange="-1 1"/>
      </actuator>
      <keyframe>
        <key name="home"
             qpos="0.0
                   0.0"/>
      </keyframe>
    </mujoco>
    """
)

DEFAULTS_XML = textwrap.dedent(
    """\
    <mujoco model="probe">
      <compiler angle="radian" meshdir="{meshdir}" autolimits="true"/>
      <include file="assets.xml"/>
      <option timestep="0.002" integrator="implicitfast"/>
      <default>
        <joint type="hinge" axis="0 1 0" damping="0.11" armature="0.01"/>
        <position kp="40" ctrlrange="-1 1"/>
      </default>
      <asset>
        <mesh name="part" file="{meshfile}" scale="1 1 1"/>
      </asset>
      <worldbody>
        <geom name="floor" type="plane" size="5 5 0.1"/>
        <body name="link1" pos="0 0 1">
          <joint name="j1"/>
          <geom name="g1" type="capsule" fromto="0 0 0 0 0 -0.4" size="0.05" mass="0.734"/>
          <body name="link2" pos="0 0 -0.4">
            <joint name="j2" damping="0.07" armature="0"/>
            <geom name="g2" type="mesh" mesh="part" mass="0.2"/>
          </body>
        </body>
      </worldbody>
      <actuator>
        <position name="a1" joint="j1"/>
        <position name="a2" joint="j2"/>
      </actuator>
      <keyframe>
        <key name="home" qpos="0.0 0.0"/>
      </keyframe>
    </mujoco>
    """
)


class Matrix:
    """Collect and render deterministic characterization checks."""

    def __init__(self) -> None:
        """Create an empty result matrix."""
        self.rows: list[tuple[str, str, bool, str]] = []

    def add(self, group: str, name: str, passed: bool, note: str = "") -> None:
        """Append one named check and its optional diagnostic note."""
        self.rows.append((group, name, bool(passed), note))

    def render(self) -> None:
        """Print checks grouped by characterization category."""
        width = max(len(row[1]) for row in self.rows) + 2
        current: str | None = None
        for group, name, passed, note in self.rows:
            if group != current:
                print(f"\n[{group}]")
                current = group
            flag = "PASS" if passed else "FAIL"
            suffix = f"   {note}" if note else ""
            print(f"  {flag}  {name:<{width}}{suffix}")


def mjb_bytes(xml_path: str) -> bytes:
    """Compile one MJCF entrypoint and return the complete serialized MJB bytes."""
    model = mujoco.MjModel.from_xml_path(xml_path)
    size = mujoco.mj_sizeModel(model)
    buffer = np.zeros(size, dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    return buffer.tobytes()


def mjb_digest(xml_path: str) -> str:
    """Return the SHA-256 digest of the complete serialized MJB artifact."""
    return hashlib.sha256(mjb_bytes(xml_path)).hexdigest()


def first_difference(left: bytes, right: bytes) -> tuple[int | None, int]:
    """Return the first differing byte offset and total differing byte count."""
    if left == right:
        return None, 0
    common = min(len(left), len(right))
    offsets = [index for index in range(common) if left[index] != right[index]]
    return (offsets[0] if offsets else common), len(offsets) + abs(len(left) - len(right))


def tetrahedron_stl(apex: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> bytes:
    """Build a small deterministic binary STL fixture with a configurable apex."""
    vertices = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), apex]
    triangles = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    output = b"\x00" * 80 + struct.pack("<I", len(triangles))
    for triangle in triangles:
        output += struct.pack("<3f", 0.0, 0.0, 1.0)
        for index in triangle:
            output += struct.pack("<3f", *vertices[index])
        output += struct.pack("<H", 0)
    return output


def build_tree(
    root: str,
    model_text: str | None = None,
    meshdir: str = "meshes",
    meshfile: str = "part.stl",
    stl: bytes | None = None,
    assets_text: str = ASSETS_XML,
) -> str:
    """Write one self-contained MJCF fixture tree and return its entrypoint path."""
    os.makedirs(os.path.join(root, meshdir), exist_ok=True)
    with open(os.path.join(root, meshdir, meshfile), "wb") as stream:
        stream.write(stl if stl is not None else tetrahedron_stl())
    with open(os.path.join(root, "assets.xml"), "w", encoding="utf-8") as stream:
        stream.write(assets_text)
    path = os.path.join(root, "model.xml")
    text = model_text if model_text is not None else MODEL_XML
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(text.format(meshdir=meshdir, meshfile=meshfile))
    return path


def run_determinism(matrix: Matrix, temporary_root: str) -> None:
    """Check repeated, fresh-process, round-trip, relocation, and size stability."""
    root = os.path.join(temporary_root, "det")
    path = build_tree(root)
    digests = {mjb_digest(path) for _ in range(10)}
    matrix.add("determinism", "10 compiles, one process", len(digests) == 1)

    code = (
        "import hashlib,numpy as np,mujoco;"
        f"m=mujoco.MjModel.from_xml_path({path!r});"
        "n=mujoco.mj_sizeModel(m);b=np.zeros(n,dtype=np.uint8);"
        "mujoco.mj_saveModel(m,None,b);"
        "print(hashlib.sha256(b.tobytes()).hexdigest())"
    )
    fresh: set[str] = set()
    child_ok = True
    for _ in range(5):
        process = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        child_ok = child_ok and process.returncode == 0
        fresh.add(process.stdout.strip())
    matrix.add("determinism", "5 fresh processes", child_ok and fresh == digests)

    serialized = mjb_bytes(path)
    roundtrip_path = os.path.join(temporary_root, "roundtrip.mjb")
    with open(roundtrip_path, "wb") as stream:
        stream.write(serialized)
    reloaded = mujoco.MjModel.from_binary_path(roundtrip_path)
    buffer = np.zeros(mujoco.mj_sizeModel(reloaded), dtype=np.uint8)
    mujoco.mj_saveModel(reloaded, None, buffer)
    matrix.add("determinism", "save -> load -> save round trip", buffer.tobytes() == serialized)

    relocated_root = os.path.join(temporary_root, "a", "much", "deeper", "relocated_directory")
    os.makedirs(os.path.dirname(relocated_root), exist_ok=True)
    shutil.copytree(root, relocated_root)
    relocated = mjb_bytes(os.path.join(relocated_root, "model.xml"))
    offset, count = first_difference(serialized, relocated)
    note = "" if relocated == serialized else f"first diff at byte {offset}, {count} bytes differ"
    matrix.add("determinism", "relocated to a different absolute path", relocated == serialized, note)
    matrix.add("determinism", "MJB size is stable", len(relocated) == len(serialized), f"{len(serialized)} bytes")


def run_neutral_refactors(matrix: Matrix, temporary_root: str) -> None:
    """Check source refactors that must compile to the same complete MJB artifact."""
    base = mjb_digest(build_tree(os.path.join(temporary_root, "neutral_base")))
    wrapped = MODEL_XML.replace(
        '<geom name="g1" type="capsule" fromto="0 0 0 0 0 -0.4" size="0.05" mass="0.734"/>',
        '<geom name="g1"\n              type="capsule"\n              fromto="0 0 0 0 0 -0.4"'
        '\n              size="0.05"\n              mass="0.734"/>',
    )
    cases = {
        "attribute line wrapping": wrapped,
        "keyframe qpos reflow": MODEL_XML.replace(
            'qpos="0.0\n                   0.0"', 'qpos="0.0 0.0"'
        ),
        "comment added": MODEL_XML.replace("<worldbody>", "<!-- reviewed -->\n      <worldbody>"),
        "0.734 -> 7.34e-1": MODEL_XML.replace('mass="0.734"', 'mass="7.34e-1"'),
        "0.734 -> 0.7340000": MODEL_XML.replace('mass="0.734"', 'mass="0.7340000"'),
        "0.002 -> 2e-3": MODEL_XML.replace('timestep="0.002"', 'timestep="2e-3"'),
        "attributes hoisted into <default>": DEFAULTS_XML,
    }
    for label, text in cases.items():
        token = label.replace(" ", "_").replace("<", "").replace(">", "")
        digest = mjb_digest(build_tree(os.path.join(temporary_root, f"neutral_{token}"), model_text=text))
        matrix.add("behavior-neutral refactors", label, digest == base)
    renamed = mjb_digest(build_tree(os.path.join(temporary_root, "neutral_meshdir"), meshdir="assets"))
    matrix.add("behavior-neutral refactors", "meshdir renamed", renamed == base)


def run_sensitivity(matrix: Matrix, temporary_root: str) -> None:
    """Check small physical or solver changes that must alter complete MJB identity."""
    base = mjb_digest(build_tree(os.path.join(temporary_root, "sensitivity_base")))
    edits = {
        "body mass +0.0136%": ('mass="0.734"', 'mass="0.7341"'),
        "joint damping": ('damping="0.11"', 'damping="0.1100001"'),
        "joint armature": ('armature="0.01"', 'armature="0.0100001"'),
        "actuator gain": ('name="a1" joint="j1" kp="40"', 'name="a1" joint="j1" kp="40.000001"'),
        "actuator ctrlrange": (
            'name="a1" joint="j1" kp="40" ctrlrange="-1 1"',
            'name="a1" joint="j1" kp="40" ctrlrange="-1 0.999999"',
        ),
        "timestep": ('timestep="0.002"', 'timestep="0.0020001"'),
        "integrator": ('integrator="implicitfast"', 'integrator="Euler"'),
        "solver iterations": (
            '<option timestep="0.002" integrator="implicitfast"/>',
            '<option timestep="0.002" integrator="implicitfast" iterations="42"/>',
        ),
        "keyframe qpos value": ('key name="home"', 'key name="home" ctrl="0.0000001 0.0"'),
        "rounded numeric": ('mass="0.734"', 'mass="0.7340001"'),
    }
    for label, (old, new) in edits.items():
        text = MODEL_XML.replace(old, new)
        if text == MODEL_XML:
            raise AssertionError(f"edit {label!r} did not apply")
        token = re_safe_token(label)
        digest = mjb_digest(build_tree(os.path.join(temporary_root, f"sensitivity_{token}"), model_text=text))
        matrix.add("physical sensitivity", label, digest != base)
    friction = mjb_digest(
        build_tree(
            os.path.join(temporary_root, "sensitivity_friction"),
            assets_text=ASSETS_XML.replace("0.7 0.05 0.01", "0.7000001 0.05 0.01"),
        )
    )
    matrix.add("physical sensitivity", "geom friction via include", friction != base)
    mesh = mjb_digest(
        build_tree(
            os.path.join(temporary_root, "sensitivity_mesh"),
            stl=tetrahedron_stl((0.0, 0.0, 1.0001)),
        )
    )
    matrix.add("physical sensitivity", "mesh vertex moved", mesh != base)


def re_safe_token(value: str) -> str:
    """Convert a human-readable case label into a deterministic directory token."""
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def run_oversensitivity(matrix: Matrix, temporary_root: str) -> None:
    """Document non-dynamic edits that intentionally break exact complete-MJB identity."""
    base = mjb_digest(build_tree(os.path.join(temporary_root, "oversensitivity_base")))
    cases = {
        "rgba visual only": MODEL_XML.replace(
            'size="0.05" mass="0.734"', 'size="0.05" mass="0.734" rgba="1 0 0 1"'
        ),
        "model name": MODEL_XML.replace('model="probe"', 'model="probe_v2"'),
        "joint renamed": MODEL_XML.replace('name="j1"', 'name="j1_joint"').replace(
            'joint="j1"', 'joint="j1_joint"'
        ),
    }
    for label, text in cases.items():
        digest = mjb_digest(
            build_tree(os.path.join(temporary_root, f"oversensitivity_{re_safe_token(label)}"), model_text=text)
        )
        matrix.add("known over-sensitivity", label, digest != base)
    renamed = mjb_digest(
        build_tree(os.path.join(temporary_root, "oversensitivity_meshfile"), meshfile="renamed_part.stl")
    )
    matrix.add("known over-sensitivity", "mesh filename changed", renamed != base)


def run_real_model(matrix: Matrix, temporary_root: str, model_path: str) -> None:
    """Check repeated compilation and relocation for one real model closure."""
    digests = {mjb_digest(model_path) for _ in range(5)}
    matrix.add("real model closure", f"5 compiles stable: {os.path.basename(model_path)}", len(digests) == 1)
    source_root = os.path.dirname(os.path.abspath(model_path))
    relocated_root = os.path.join(temporary_root, "real", "relocated_model_root")
    os.makedirs(os.path.dirname(relocated_root), exist_ok=True)
    shutil.copytree(source_root, relocated_root)
    relocated_path = os.path.join(relocated_root, os.path.basename(model_path))
    original, relocated = mjb_bytes(model_path), mjb_bytes(relocated_path)
    offset, count = first_difference(original, relocated)
    note = f"{len(original)} bytes" if original == relocated else f"first diff at byte {offset}, {count} differ"
    matrix.add("real model closure", "relocated to a different absolute path", original == relocated, note)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the optional real-model entrypoint argument."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", help="path to a real MJCF entrypoint to include")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the complete characterization matrix and return a process status."""
    args = parse_args(argv)
    print(f"mujoco python : {mujoco.__version__}")
    print(f"platform      : {sys.platform} / {os.uname().machine}")
    print(f"numpy         : {np.__version__}")
    matrix = Matrix()
    temporary_root = tempfile.mkdtemp(prefix="mjb_characterization_")
    try:
        run_determinism(matrix, temporary_root)
        run_neutral_refactors(matrix, temporary_root)
        run_sensitivity(matrix, temporary_root)
        run_oversensitivity(matrix, temporary_root)
        if args.model:
            run_real_model(matrix, temporary_root, args.model)
    finally:
        matrix.render()
        shutil.rmtree(temporary_root, ignore_errors=True)
    failures = [row[1] for row in matrix.rows if not row[2]]
    if failures:
        print(f"\n{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print(f"\nall {len(matrix.rows)} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
