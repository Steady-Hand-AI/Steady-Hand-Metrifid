"""Fixed no-wall/no-stop planar pushing profile, expressed in SI units."""

from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._request import RepairRequest

FAMILY = "planar_push_timed15_v1"
PARAMETER_DIMENSION = 15
BLOCK_SIZE_M = (0.08, 0.04, 0.02)
BLOCK_CENTER_BOUNDS_M = ((-0.18, 0.18), (-0.18, 0.18))
FLOOR_HALF_SIZES_M = (0.2, 0.2, 0.01)
SUPPORT_BOUNDS_M = tuple((-extent, extent) for extent in FLOOR_HALF_SIZES_M[:2])
RAIL_Y_LOWER_MIN_M = -0.18
INITIAL_FINGER_Y_M = -0.04
NOMINAL_DT_S = 0.002
FINER_DT_S = 0.001
SETTLE_SECONDS = 0.4
EPISODE_SECONDS = 4.0
HOLD_SECONDS = 0.2
HOLD_LIMITS = (0.005, math.pi / 60, 0.01, 0.1)


def setup_steps() -> int:
    """Return the nominal settling integration count."""
    return round(SETTLE_SECONDS / NOMINAL_DT_S)


def pair_steps() -> int:
    """Return the potential integration work of one full timestep pair."""
    return sum(round(EPISODE_SECONDS / dt) for dt in (NOMINAL_DT_S, FINER_DT_S))


def minimum_step_budget() -> int:
    """Return setup, one screening pair and one confirmation pair."""
    return setup_steps() + 2 * pair_steps()


def resolve_profile(request: RepairRequest) -> dict[str, Any]:
    """Resolve only the admitted task's rail and target into fixed physics."""
    rails = [[-0.18, 0.18], [request.task.rail_y_lower_m, 0.18]]
    return {
        "family": FAMILY,
        "scene": {
            "gravity": [0, 0, -9.81],
            "floor": {"surface_z": 0, "half_sizes_m": list(FLOOR_HALF_SIZES_M)},
            "block": {
                "size_m": list(BLOCK_SIZE_M),
                "mass_kg": 0.1,
                "initial_center_m": [-0.06, -0.04, 0.0105],
                "initial_yaw_rad": 0,
            },
            "finger": {
                "radius_m": 0.008,
                "height_m": 0.03,
                "center_z_m": 0.02,
                "initial_xy_m": [-0.14, INITIAL_FINGER_Y_M],
                "mass_kg": 0.02,
                "rail_limits_xy_m": rails,
            },
            "contact": {
                "condim": 3,
                "sliding_friction": {"block_floor": 0.6, "finger_block": 0.8},
                "solref": [0.02, 1.0],
                "solimp": [0.9, 0.95, 0.001, 0.5, 2],
                "penetration_gate_m": 0.001,
            },
            "runtime": {
                "nominal_dt_s": NOMINAL_DT_S,
                "finer_dt_s": FINER_DT_S,
                "integrator": "implicitfast",
                "solver": "Newton",
                "solver_iterations": 100,
                "solver_tolerance": 1e-10,
                "cone": "elliptic",
                "control_period_s": 0.02,
                "settle_seconds": SETTLE_SECONDS,
                "episode_seconds": EPISODE_SECONDS,
            },
            "target": {
                "xy_m": list(request.task.target_xy_m),
                "yaw_rad": request.task.target_yaw_rad,
                "position_tolerance_m": HOLD_LIMITS[0],
                "yaw_tolerance_rad": HOLD_LIMITS[1],
                "linear_speed_max_m_s": HOLD_LIMITS[2],
                "angular_speed_max_rad_s": HOLD_LIMITS[3],
                "stable_hold_s": HOLD_SECONDS,
            },
            "whole_episode_gates": {
                "lift_above_settled_com_m": 0.005,
                "tilt_max_rad": math.pi / 36,
                "block_xy_com_bounds_m": [list(x) for x in BLOCK_CENTER_BOUNDS_M],
                "block_speed_max_m_s": 0.5,
                "finger_speed_max_m_s": 0.5,
            },
        },
        "controller": {
            "parameter_dimension": PARAMETER_DIMENSION,
            "position_gain_N_m": 30.0,
            "velocity_gain_N_s_m": 1.0,
            "force_norm_bound_N": 2.0,
            "reference_speed_bound_m_s": 0.12,
            "reference_bounds_xy_m": [x.copy() for x in rails],
        },
    }


def _numbers(values: Iterable[float]) -> str:
    return " ".join(format(float(value), ".17g") for value in values)


def _body(world: ET.Element, scene: dict[str, Any], role: str) -> None:
    properties = scene[role]
    if role == "block":
        yaw = properties["initial_yaw_rad"]
        body = ET.SubElement(
            world,
            "body",
            name="block_body",
            pos=_numbers(properties["initial_center_m"]),
            quat=_numbers([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]),
        )
        ET.SubElement(body, "freejoint", name="block_free")
        ET.SubElement(
            body,
            "geom",
            name="block",
            type="box",
            size=_numbers(value / 2 for value in properties["size_m"]),
            mass=str(properties["mass_kg"]),
        )
    else:
        body = ET.SubElement(
            world,
            "body",
            name="finger_body",
            pos=_numbers([*properties["initial_xy_m"], properties["center_z_m"]]),
        )
        for axis, vector in enumerate(([1, 0, 0], [0, 1, 0])):
            bounds = [
                value - properties["initial_xy_m"][axis]
                for value in properties["rail_limits_xy_m"][axis]
            ]
            ET.SubElement(
                body,
                "joint",
                name="finger_" + "xy"[axis],
                type="slide",
                axis=_numbers(vector),
                limited="true",
                range=_numbers(bounds),
                damping="0",
                armature="0",
            )
        ET.SubElement(
            body,
            "geom",
            name="finger",
            type="cylinder",
            size=_numbers([properties["radius_m"], properties["height_m"] / 2]),
            mass=str(properties["mass_kg"]),
        )


def scene_xml(profile: dict[str, Any], dt_s: float) -> str:
    """Generate the supported scene; never load caller-supplied XML."""
    scene = profile["scene"]
    runtime = scene["runtime"]
    root = ET.Element("mujoco", model="metrifid_planar_push")
    ET.SubElement(root, "compiler", angle="radian")
    option = ET.SubElement(
        root,
        "option",
        timestep=str(dt_s),
        gravity=_numbers(scene["gravity"]),
        integrator=runtime["integrator"],
        solver=runtime["solver"],
        iterations=str(runtime["solver_iterations"]),
        tolerance=str(runtime["solver_tolerance"]),
        cone=runtime["cone"],
    )
    ET.SubElement(option, "flag", warmstart="enable")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", contype="0", conaffinity="0", condim="3")
    world = ET.SubElement(root, "worldbody")
    floor = scene["floor"]
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="box",
        size=_numbers(floor["half_sizes_m"]),
        pos=_numbers([0, 0, floor["surface_z"] - floor["half_sizes_m"][2]]),
    )
    _body(world, scene, "block")
    _body(world, scene, "finger")
    contact = ET.SubElement(root, "contact")
    for first, second, group in (
        ("block", "floor", "block_floor"),
        ("finger", "block", "finger_block"),
    ):
        friction = scene["contact"]["sliding_friction"][group]
        ET.SubElement(
            contact,
            "pair",
            name=f"pair_{first}_{second}",
            geom1=first,
            geom2=second,
            condim="3",
            friction=_numbers([friction, friction, 0, 0, 0]),
            solref=_numbers(scene["contact"]["solref"]),
            solimp=_numbers(scene["contact"]["solimp"]),
        )
    actuator = ET.SubElement(root, "actuator")
    for axis in "xy":
        ET.SubElement(
            actuator,
            "motor",
            name="motor_" + axis,
            joint="finger_" + axis,
            gear="1",
            ctrllimited="true",
            ctrlrange="-2 2",
        )
    return ET.tostring(root, encoding="unicode")


def generate_expected(request: RepairRequest) -> tuple[dict[str, Any], str, str, dict[str, object]]:
    """Return independently generated profile, scenes and timestep identities."""
    profile = resolve_profile(request)
    nominal, finer = (scene_xml(profile, dt) for dt in (NOMINAL_DT_S, FINER_DT_S))
    identities: dict[str, object] = {
        "nominal_dt_s": NOMINAL_DT_S,
        "finer_dt_s": FINER_DT_S,
        "nominal_scene_sha256": hashlib.sha256(nominal.encode()).hexdigest(),
        "finer_scene_sha256": hashlib.sha256(finer.encode()).hexdigest(),
    }
    return profile, nominal, finer, identities
