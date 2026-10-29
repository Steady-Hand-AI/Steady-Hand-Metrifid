"""Timed object-relative strokes with fixed four-face routing.

Block pose is observed at stroke boundaries; finger feedback controls a
continuous bounded reference at 50 Hz within the requested rails.
"""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
CONFIG: dict[str, Any] = {
    "arrival_tolerance_m": 0.0035,
    "transit_clearance_m": 0.015,
    "control_period_s": 0.02,
    "reference_speed_bound_m_s": 0.12,
    "tangent_extent_fraction": 0.95,
}


def decode_parameters(parameters: Any) -> Array:
    """Decode closed-box geometric coordinates into three local strokes."""
    values = np.asarray(parameters, dtype=float).reshape(-1)[:12].reshape(3, 4).copy()
    values[:, 0] %= 1.0
    values[:, 1:] = np.clip(values[:, 1:], 0.0, 1.0)
    result = values.copy()
    result[:, 0] *= 2.0 * math.pi
    result[:, 1] = 2.0 * values[:, 1] - 1.0
    result[:, 2:] = 0.12 * (2.0 * values[:, 2:] - 1.0)
    norms = np.linalg.norm(result[:, 2:], axis=1)
    for index in range(3):
        if norms[index] > 0.12:
            result[index, 2:] *= 0.12 / norms[index]
    return result


def decode_schedule(parameters: Any) -> dict[str, Any]:
    """Scale durations together, then floor cumulative deadlines once at 50 Hz."""
    values = np.asarray(parameters, dtype=float).reshape(-1)
    timing = np.clip(values[12:15], 0.0, 1.0)
    raw = 0.20 + 2.0 * timing
    scale = min(1.0, 3.74 / float(raw.sum()))
    scaled = raw * scale
    cumulative = np.cumsum(scaled)
    deadlines = np.minimum(187, np.floor((cumulative + 1e-10) / 0.02).astype(int))
    ticks = np.concatenate(([0], deadlines))
    if np.any(np.diff(ticks) <= 0):
        raise ValueError("Decoded stroke boundaries must be strictly increasing")
    boundaries = ticks * 0.02
    starts, ends = boundaries[:-1], boundaries[1:]
    return {
        "boundaries_s": boundaries.tolist(),
        "decoded_boundaries_s": boundaries.tolist(),
        "stroke_windows_s": np.column_stack((starts, ends)).tolist(),
        "motion_deadlines_s": ends.tolist(),
        "scheduled_starts_s": starts.tolist(),
        "stationary_tail_start_s": float(ends[-1]),
        "timing_parameters_normalized": timing.tolist(),
    }


def _static_clear(
    a: Any, b: Any, scene: Any, finger_radius: Any, clearance: Any, bounds: Any
) -> bool:
    return bool(
        not (
            np.any(a < bounds[:, 0] - 1e-10)
            or np.any(a > bounds[:, 1] + 1e-10)
            or np.any(b < bounds[:, 0] - 1e-10)
            or np.any(b > bounds[:, 1] + 1e-10)
        )
    )


def plan_route(
    finger_xy: Any,
    reference_xy: Any,
    block_xy: Any,
    block_yaw: Any,
    angle: Any,
    tangential_offset: Any,
    scene: Any,
    config: Any = None,
) -> Any:
    """Plan the shorter collision-admitted four-face perimeter route."""
    return _plan_route_four_faces(
        finger_xy, reference_xy, block_xy, block_yaw, angle, tangential_offset, scene, config
    )


def _norm(v: Any) -> float:
    return math.hypot(float(v[0]), float(v[1]))


def _box_intersection(a: Any, b: Any, lower: Any, upper: Any, epsilon: Any = 1e-10) -> bool:
    """Whether a segment enters the open rectangle; boundary tangency is legal."""
    lo, hi = (0.0, 1.0)
    for axis in range(2):
        low, high = (float(lower[axis]) + epsilon, float(upper[axis]) - epsilon)
        delta = float(b[axis] - a[axis])
        if abs(delta) < 1e-15:
            if a[axis] <= low or a[axis] >= high:
                return False
        else:
            first, last = ((low - a[axis]) / delta, (high - a[axis]) / delta)
            if first > last:
                first, last = (last, first)
            lo, hi = (max(lo, first), min(hi, last))
            if lo > hi:
                return False
    return hi >= 0.0 and lo <= 1.0 and (lo <= hi)


def _perimeter_coordinate(point: Any, half: Any) -> float:
    hx, hy = (float(half[0]), float(half[1]))
    x, y = (float(point[0]), float(point[1]))
    if abs(y + hy) < 1e-08:
        return x + hx
    if abs(x - hx) < 1e-08:
        return 2 * hx + y + hy
    if abs(y - hy) < 1e-08:
        return 4 * hx + 2 * hy - x - hx
    return (4 * hx + 4 * hy - (y + hy)) % (4 * hx + 4 * hy)


def _perimeter_points(start: Any, end: Any, half: Any, direction: Any) -> Any:
    hx, hy = (float(half[0]), float(half[1]))
    vertices = [np.array([-hx, -hy]), np.array([hx, -hy]), np.array([hx, hy]), np.array([-hx, hy])]
    perimeter = 4 * (hx + hy)
    s0, s1 = (_perimeter_coordinate(start, half), _perimeter_coordinate(end, half))
    total = (s1 - s0 if direction == 1 else s0 - s1) % perimeter
    points: list[Any] = []
    for vertex in vertices:
        sv = _perimeter_coordinate(vertex, half)
        distance = (sv - s0 if direction == 1 else s0 - sv) % perimeter
        if 1e-09 < distance < total - 1e-09:
            points.append((distance, vertex))
    return [start] + [p for _, p in sorted(points, key=lambda x: x[0])] + [end]


def approach_geometry(
    half: Any,
    angle: Any,
    tangential_offset: Any,
    finger_radius: Any,
    clearance: Any,
    tangent_fraction: Any = 0.95,
) -> Any:
    """Ray selects any box side; offset covers its tangential surface region."""
    direction = np.array([math.cos(angle), math.sin(angle)])
    ratios = np.abs(direction) / half
    axis = int(np.argmax(ratios))
    tangent = 1 - axis
    sign = 1.0 if direction[axis] >= 0.0 else -1.0
    point = direction / ratios[axis]
    extent = tangent_fraction * half[tangent]
    point[tangent] = np.clip(point[tangent] + tangential_offset * extent, -extent, extent)
    point[axis] = sign * half[axis]
    normal = np.zeros(2)
    normal[axis] = sign
    return (point + finger_radius * normal, point + (finger_radius + clearance) * normal, normal)


def _outward_retraction_clear(
    start: Any, end: Any, physical_half: Any, penetration_limit: Any
) -> bool:
    """Check a connector against the physical-radius block rectangle.

    A starting solver overlap up to the existing physical penetration gate may
    leave through a nearby face only when its signed normal distance increases
    monotonically. An outside start never gains an exception to cross the box.
    """
    if not _box_intersection(start, end, -physical_half, physical_half):
        return True
    if not np.all(np.abs(start) < physical_half):
        return False
    delta = end - start
    for axis in range(2):
        sign = 1.0 if start[axis] >= 0.0 else -1.0
        depth = physical_half[axis] - sign * start[axis]
        outward = sign * delta[axis]
        if depth <= penetration_limit + 1e-10 and outward > 1e-12 and (outward >= depth - 1e-10):
            return True
    return False


def _common_route_reasons(
    ref: Any,
    finger: Any,
    outer: Any,
    contact: Any,
    physical: Any,
    bounds: Any,
    world: Any,
    radius: Any,
    clearance: Any,
    scene: Any,
) -> list[str]:
    common_reasons: list[str] = []
    if not _static_clear(ref, finger, scene, radius, 0.0, bounds):
        common_reasons.append("reference_connector_static_or_rail")
    if not _static_clear(world(outer), world(contact), scene, radius, 0.0, bounds):
        common_reasons.append("approach_static_or_rail")
    if _box_intersection(outer, contact, -physical, physical):
        common_reasons.append("approach_crosses_block")
    if not _static_clear(world(outer), world(outer), scene, radius, clearance, bounds):
        common_reasons.append("outer_waypoint_collision")
    return common_reasons


def _checked_route(
    perimeter: Any,
    inside_inflation: Any,
    finger: Any,
    ref: Any,
    contact_world: Any,
    initial_reasons: Any,
    clear_transit: Any,
    face: Any,
) -> dict[str, Any]:
    points = perimeter if inside_inflation else [finger, *perimeter]
    shortened = [points[0]]
    index = 0
    while index < len(points) - 1:
        next_index = index + 1
        for candidate in range(len(points) - 1, index, -1):
            if clear_transit(points[index], points[candidate]):
                next_index = candidate
                break
        shortened.append(points[next_index])
        index = next_index
    route = ([finger] if inside_inflation else []) + shortened + [contact_world]
    if _norm(ref - finger) > 1e-10:
        route = [ref, *route]
    else:
        route[0] = ref
    compact = [route[0]]
    for point in route[1:]:
        if _norm(point - compact[-1]) > 1e-10:
            compact.append(point)
    if len(compact) == 1:
        compact.append(contact_world)
    reasons = list(initial_reasons)
    for a, b in pairwise(shortened):
        if not clear_transit(a, b):
            reasons.append("perimeter_or_connector_collision")
            break
    length = sum((_norm(b - a) for a, b in pairwise(compact)))
    row = {
        "feasible": not reasons,
        "length_m": length,
        "reasons": sorted(set(reasons)),
        "exit_face": face,
        "waypoints_xy_m": [p.tolist() for p in compact],
    }
    return row


def _route_options(candidates: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for name, rows in candidates.items():
        valid = [(index, row) for index, row in rows if row["feasible"]]
        _, best = min(valid or rows, key=lambda item: (item[1]["length_m"], item[0]))
        options[name] = dict(best)
        if not valid:
            options[name]["reasons"] = sorted(
                {reason for _, row in rows for reason in row["reasons"]}
            )
    return options


def _plan_route_four_faces(
    finger_xy: Any,
    reference_xy: Any,
    block_xy: Any,
    block_yaw: Any,
    angle: Any,
    tangential_offset: Any,
    scene: Any,
    config: Any,
) -> Any:
    """Check four outward face projections and both full perimeter routes."""
    cfg = dict(CONFIG)
    finger, ref, center = (
        np.asarray(value, dtype=float) for value in (finger_xy, reference_xy, block_xy)
    )
    half = np.asarray(scene["block"]["size_m"][:2], dtype=float) * 0.5
    radius = float(scene["finger"]["radius_m"])
    clearance = float(cfg["transit_clearance_m"])
    bounds = np.asarray(scene["finger"]["rail_limits_xy_m"], dtype=float)
    rotation = np.array(
        [[math.cos(block_yaw), -math.sin(block_yaw)], [math.sin(block_yaw), math.cos(block_yaw)]]
    )
    local_start = rotation.T @ (finger - center)
    physical, inflated = (half + radius, half + radius + clearance)
    penetration_limit = float(scene["contact"]["penetration_gate_m"])
    inside_inflation = bool(np.all(np.abs(local_start) < inflated - 1e-10))
    contact, outer, normal = approach_geometry(
        half, angle, tangential_offset, radius, clearance, cfg["tangent_extent_fraction"]
    )

    def world(local: Any) -> Any:
        return center + rotation @ local

    def clear_transit(a: Any, b: Any) -> Any:
        return not _box_intersection(
            rotation.T @ (a - center), rotation.T @ (b - center), -inflated, inflated
        ) and _static_clear(a, b, scene, radius, clearance, bounds)

    common_reasons = _common_route_reasons(
        ref, finger, outer, contact, physical, bounds, world, radius, clearance, scene
    )
    candidates: dict[str, list[tuple[int, dict[str, Any]]]] = {
        "counterclockwise": [],
        "clockwise": [],
    }
    exits: dict[str, Any] = {}
    for exit_index, (face, axis, sign) in enumerate(
        (("-X", 0, -1.0), ("+X", 0, 1.0), ("-Y", 1, -1.0), ("+Y", 1, 1.0))
    ):
        anchor = np.clip(local_start, -inflated, inflated)
        anchor[axis] = sign * inflated[axis]
        initial_reasons = list(common_reasons)
        if not _static_clear(finger, world(anchor), scene, radius, 0.0, bounds):
            initial_reasons.append("retract_static_or_rail")
        if not _outward_retraction_clear(local_start, anchor, physical, penetration_limit):
            initial_reasons.append("retract_crosses_block")
        exits[face] = {
            "anchor_local_xy_m": anchor.tolist(),
            "anchor_xy_m": world(anchor).tolist(),
            "options": {},
        }
        for name, direction in (("counterclockwise", 1), ("clockwise", -1)):
            perimeter = [world(p) for p in _perimeter_points(anchor, outer, inflated, direction)]
            row = _checked_route(
                perimeter,
                inside_inflation,
                finger,
                ref,
                world(contact),
                initial_reasons,
                clear_transit,
                face,
            )
            candidates[name].append((exit_index, row))
            exits[face]["options"][name] = {
                key: row[key] for key in ("feasible", "length_m", "reasons")
            }
    options = _route_options(candidates)
    feasible = [(row["length_m"], name) for name, row in options.items() if row["feasible"]]
    selected = min(feasible)[1] if feasible else None
    return {
        "routing_version": "four_face_outward_exits_v1",
        "selected_direction": selected,
        "options": options,
        "selected_exit_face": None if selected is None else options[selected]["exit_face"],
        "exit_options": exits,
        "waypoints_xy_m": [] if selected is None else options[selected]["waypoints_xy_m"],
        "approach_xy_m": world(contact).tolist(),
        "outer_approach_xy_m": world(outer).tolist(),
        "approach_normal_world": (rotation @ normal).tolist(),
        "block_xy_m": center.tolist(),
        "block_yaw_rad": float(block_yaw),
    }


class StrokeController:
    """Online timed three-stroke controller with four-face routing and PD30 feedback."""

    def __init__(self, profile: dict[str, Any], parameters: Any) -> None:
        self.protocol = profile
        self.scene = profile["scene"]
        self.params = np.asarray(parameters, dtype=float).reshape(-1).copy()
        self.config = dict(CONFIG)
        self.decoded = decode_parameters(self.params)
        self.period = float(self.config["control_period_s"])
        self.speed = float(self.config["reference_speed_bound_m_s"])
        self.schedule = decode_schedule(self.params)
        self.windows = np.asarray(self.schedule["stroke_windows_s"], dtype=float)
        self.deadlines = np.asarray(self.schedule["motion_deadlines_s"], dtype=float)
        self.reset(self.scene["finger"]["initial_xy_m"])

    def reset(self, initial_finger_xy: Any) -> None:
        """Clear all controller history for a fresh trajectory."""
        self.reference = np.asarray(initial_finger_xy, dtype=float).copy()
        self.reference_velocity = np.zeros(2)
        self._last_t = 0.0
        self._stroke = -1
        self._mode = "idle"
        self._route: list[Any] = []
        self._waypoint = 0
        self._summaries: list[dict[str, Any]] = []
        self._push_target = self.reference.copy()
        self._push_direction = np.zeros(2)
        self._last_phase = "idle"
        self._frozen = False

    def _finish(self, t: Any) -> None:
        if self._stroke >= 0:
            row = self._summaries[-1]
            if row["arrival_status"] == "pending":
                row["arrival_status"] = "timeout"
                row["transit_time_s"] = max(
                    0.0,
                    min(float(t), row["absolute_motion_deadline_s"]) - row["actual_stroke_start_s"],
                )
            row["timed_out"] = row["arrival_status"] in ("timeout", "blocked")
            row["push_complete"] = (
                row["arrival_status"] == "arrived"
                and _norm(self._push_target - self.reference) < 1e-09
            )

    def _start(self, index: Any, t: Any, finger: Any, block_xy: Any, block_yaw: Any) -> None:
        self._finish(t)
        self._stroke = index
        values = self.decoded[index]
        reference_at_boundary = self.reference.tolist()
        route = plan_route(
            finger,
            self.reference,
            block_xy,
            block_yaw,
            values[0],
            values[1],
            self.scene,
            self.config,
        )
        self._route = [np.asarray(p, dtype=float) for p in route["waypoints_xy_m"]]
        self._waypoint = 1
        self._mode = "route" if route["selected_direction"] else "blocked"
        rotation = np.array(
            [
                [math.cos(block_yaw), -math.sin(block_yaw)],
                [math.sin(block_yaw), math.cos(block_yaw)],
            ]
        )
        self._desired_push = rotation @ values[2:]
        self._summaries.append(
            {
                "stroke_index": index,
                "nominal_window_s": self.windows[index].tolist(),
                "scheduled_stroke_start_s": float(self.schedule["scheduled_starts_s"][index]),
                "decoded_boundaries_s": self.schedule["decoded_boundaries_s"],
                "stationary_tail_start_s": self.schedule["stationary_tail_start_s"],
                "actual_stroke_start_s": float(t),
                "absolute_motion_deadline_s": float(self.deadlines[index]),
                "arrival_status": "pending" if self._mode == "route" else "blocked",
                "arrival_time_s": None,
                "transit_time_s": 0.0,
                "timed_out": self._mode == "blocked",
                "requested_push_length_m": _norm(self._desired_push),
                "capped_push_length_m": 0.0,
                "executed_push_length_m": 0.0,
                "push_complete": False,
                "finger_at_boundary_xy_m": finger.tolist(),
                "sampled_block_xy_m": list(map(float, block_xy)),
                "reference_at_boundary_xy_m": reference_at_boundary,
                "sampled_block_yaw_rad": float(block_yaw),
                **route,
            }
        )

    def _begin_push(self, t: Any) -> None:
        row = self._summaries[-1]
        remaining = max(0.0, float(self.deadlines[self._stroke]) - t)
        requested = _norm(self._desired_push)
        length = min(requested, self.speed * remaining)
        direction = self._desired_push / requested if requested > 1e-14 else np.zeros(2)
        bounds = np.asarray(self.protocol["controller"]["reference_bounds_xy_m"], dtype=float)
        for axis in range(2):
            if direction[axis] > 1e-14:
                length = min(
                    length, max(0.0, (bounds[axis, 1] - self.reference[axis]) / direction[axis])
                )
            elif direction[axis] < -1e-14:
                length = min(
                    length, max(0.0, (bounds[axis, 0] - self.reference[axis]) / direction[axis])
                )
        self._push_target = self.reference + length * direction
        self._push_direction = direction
        self._mode = "push"
        row.update(
            arrival_status="arrived",
            arrival_time_s=float(t),
            transit_time_s=float(t - row["actual_stroke_start_s"]),
            capped_push_length_m=float(length),
            push_onset_reference_xy_m=self.reference.tolist(),
            push_target_reference_xy_m=self._push_target.tolist(),
            remaining_slot_at_push_s=float(remaining),
        )

    def _advance_route(self, t: float, finger: Array) -> None:
        tolerance = float(self.config["arrival_tolerance_m"])
        if self._mode == "route":
            approach = np.asarray(self._summaries[-1]["approach_xy_m"])
            if self._waypoint >= len(self._route) - 1 and _norm(finger - approach) <= tolerance:
                self._begin_push(t)
            else:
                while self._waypoint < len(self._route) - 1:
                    point = self._route[self._waypoint]
                    if _norm(self.reference - point) > 1e-09 or _norm(finger - point) > tolerance:
                        break
                    self._waypoint += 1
                if self._waypoint >= len(self._route) - 1 and _norm(finger - approach) <= tolerance:
                    self._begin_push(t)

    def update(self, t: Any, finger_xy: Any, block_xy: Any, block_yaw: Any) -> tuple[Array, Array]:
        """Advance the continuous reference from current finger and boundary block observations."""
        t = float(t)
        if t < self._last_t - 1e-09:
            raise ValueError("Controller time must be monotonic; reset between episodes")
        delta_t = max(0.0, t - self._last_t)
        previous_reference = self.reference.copy()
        self.reference += self.reference_velocity * delta_t
        if self._last_phase == "push" and self._summaries:
            self._summaries[-1]["executed_push_length_m"] += _norm(
                self.reference - previous_reference
            )
        self._last_t = t
        finger = np.asarray(finger_xy, dtype=float)
        self.reference_velocity.fill(0.0)
        if t >= self.deadlines[-1] - 1e-10:
            if not self._frozen:
                self._finish(t)
            self._frozen = True
            self._mode = self._last_phase = "hold"
            return (self.reference.copy(), self.reference_velocity.copy())
        index = min(2, int(np.searchsorted(self.windows[:, 1], t + 1e-10, side="right")))
        if index != self._stroke:
            self._start(index, t, finger, block_xy, block_yaw)
        if t >= self.deadlines[index] - 1e-10:
            self._finish(t)
            self._last_phase = "hold"
            return (self.reference.copy(), self.reference_velocity.copy())
        self._advance_route(t, finger)
        target = (
            self._push_target
            if self._mode == "push"
            else self._route[min(self._waypoint, len(self._route) - 1)]
            if self._mode == "route"
            else self.reference
        )
        displacement = target - self.reference
        distance = _norm(displacement)
        if distance > 1e-14:
            duration = min(self.period, max(0.0, float(self.deadlines[index]) - t))
            if duration >= self.period - 1e-10:
                self.reference_velocity = displacement * (
                    min(self.speed, distance / self.period) / distance
                )
        self._last_phase = self._mode
        return (self.reference.copy(), self.reference_velocity.copy())
