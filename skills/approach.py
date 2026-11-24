"""Approach skill: drive the base up to a named object, wherever it stands.

No map and no rooms. The skill finds the object with the cameras (turning on
the spot to search if needed), turns to face it, and drives straight towards
it until its near face is at the standoff ``next_action`` needs. Turns on the
real base overshoot (up to 2x on small angles), so every turn is followed by a new
measurement, and so is every drive leg.

Success means the next skill can work from here: the near face inside the window
for ``next_action``, seen by a camera (``verified``) or, when the object went out of
both cameras' view close to the standoff, by odometry (``verified: false``; pick and
place segment it again themselves).

The planner LLM sees only ``object_name`` and ``next_action``.
"""

import json
import logging
import math

from skill_based.clients.mcp import MCPClient
from skill_based.skills.pick import UR5_REACH_X
from skill_based.skills.place import CONTAINER_REACH_REAL_M, SURFACE_EDGE_INSET_M
from skill_based.utils import move_arm_to_look_forward, parse_seg_status, wait_until_still

logger = logging.getLogger(__name__)

# Base (base_footprint) to the object's near face after the approach. Pick and place
# segment from look_forward, whose arm camera sees the floor from about 0.8 m out.
STANDOFF_BY_NEXT_ACTION = {
    "pick": 0.85,
    "container_place": 0.65,
    "surface_place": 0.70,
    "floor_place": 0.85,
}

# Detections farther than this are another object across the room, not the target.
MAX_RANGE_M = 4.0
# Search: turn this far between looks, at most a full turn.
SEARCH_STEP_RAD = math.radians(60)
SEARCH_STEPS = 6
# Facing: turn again while the bearing is larger than this, at most this often.
BEARING_TOL_RAD = math.radians(6)
MAX_TURNS = 3
# Small turns on the spot overshoot 1.7-2x on the robot: command only part of the bearing.
TURN_GAIN = 0.55
# Drive: slowly, and short legs, so the object is measured again before it gets too close to see.
DRIVE_SPEED = 0.1
MAX_LEG_M = 0.8
MAX_LEGS = 4
# Drive for the standoff, accept a window: closer than this below it is the cameras' limit,
# the far end is where the next skill still reaches (its own reach rules).
NEAR_TOL_M = 0.05
_OBJECT_HALF_M = 0.10  # pick grasps about half the object behind its near face
_CONTAINER_HALF_M = 0.20  # container place drops into the middle
MAX_NEAR_BY_NEXT_ACTION = {
    "pick": UR5_REACH_X - _OBJECT_HALF_M,
    "container_place": CONTAINER_REACH_REAL_M - _CONTAINER_HALF_M,
    "surface_place": CONTAINER_REACH_REAL_M - SURFACE_EDGE_INSET_M,
    "floor_place": CONTAINER_REACH_REAL_M,  # the drop is in front of the near face
}
# A re-measurement this far from the odometry prediction is another object.
LOST_TOL_M = 0.30
# Lost from view this close to the standoff: drive the rest by odometry.
BLIND_MAX_M = 0.5
# Not seen at the start: back up this far once before searching (too close for both cameras).
BACKUP_LOOK_M = 0.4
# Closer than this the arm camera measures better than the front camera.
ARM_FIRST_BELOW_M = 1.5


def _ok(reply) -> bool:
    return str(reply).startswith("Successfully")


async def _measure(mcp: MCPClient, object_name: str, cameras: tuple[str, ...]) -> tuple[dict | None, int]:
    """Segment the object; return its bearing and distances in base_footprint, or None."""
    calls = 0
    for camera in cameras:
        try:
            raw = await mcp.call_tool_prefixed(
                "perception__segment_objects", {"prompt": object_name, "camera": camera, "timeout": 20}
            )
            calls += 1
            status = parse_seg_status(raw)
            logger.info(f"{camera} SAM3 '{object_name}' -> {status}")
            if status != "SUCCESS":
                continue
            raw = await mcp.call_tool_prefixed("perception__get_topdown_grasp_pose", {"object_name": object_name})
            calls += 1
            g = json.loads(raw) if isinstance(raw, str) else raw
            cx, cy = float(g["centroid_base_frame"]["x"]), float(g["centroid_base_frame"]["y"])
            dist = math.hypot(cx, cy)
            depth = cx - float(g["bbox_base_frame"]["x_min"])  # centroid behind the near face
        except Exception as e:
            calls += 1
            logger.warning(f"measure on {camera} failed: {e}")
            continue
        if dist > MAX_RANGE_M:
            logger.info(f"{camera}: '{object_name}' at {dist:.2f} m, beyond {MAX_RANGE_M} m: not the target")
            continue
        m = {"camera": camera, "dist": dist, "near": dist - max(depth, 0.0), "bearing": math.atan2(cy, cx)}
        logger.info(f"{camera}: near face {m['near']:.2f} m, bearing {math.degrees(m['bearing']):.1f} deg")
        return m, calls
    return None, calls


def _cameras(last: dict | None) -> tuple[str, ...]:
    if last is not None and last["dist"] < ARM_FIRST_BELOW_M:
        return ("arm", "front")
    return ("front", "arm")


async def _turn(mcp: MCPClient, angle: float) -> tuple[bool, str]:
    reply = await mcp.call_tool_prefixed("nav2__spin_robot", {"angle": round(angle, 4)})
    await wait_until_still(mcp)
    logger.info(f"turn {math.degrees(angle):+.0f} deg -> {'done' if _ok(reply) else 'FAILED'}")
    return _ok(reply), str(reply)


async def _face(mcp: MCPClient, object_name: str, m: dict) -> tuple[dict | None, int, str]:
    """Turn until the bearing is within tolerance, measuring after every turn."""
    calls = 0
    for _ in range(MAX_TURNS):
        if abs(m["bearing"]) <= BEARING_TOL_RAD:
            return m, calls, ""
        ok, reply = await _turn(mcp, m["bearing"] * TURN_GAIN)
        calls += 1
        if not ok:
            return None, calls, f"turn failed: {reply[:200]}"
        m, c = await _measure(mcp, object_name, _cameras(m))
        calls += c
        if m is None:
            return None, calls, f"lost '{object_name}' after turning towards it"
    if abs(m["bearing"]) <= BEARING_TOL_RAD:
        return m, calls, ""
    return None, calls, f"still {math.degrees(m['bearing']):.0f} deg off after {MAX_TURNS} turns"


def _in_window(near: float, next_action: str) -> bool:
    return STANDOFF_BY_NEXT_ACTION[next_action] - NEAR_TOL_M <= near <= MAX_NEAR_BY_NEXT_ACTION[next_action]


async def _drive(mcp: MCPClient, gap: float) -> tuple[bool, str, float]:
    """Drive towards (gap > 0) or back away from the object, one leg; returns the signed leg."""
    leg = max(-MAX_LEG_M, min(gap, MAX_LEG_M))
    tool = "nav2__drive_on_heading" if leg > 0 else "nav2__backup_robot"
    reply = await mcp.call_tool_prefixed(tool, {"distance": round(abs(leg), 3), "speed": DRIVE_SPEED})
    ok = _ok(reply)
    if ok:
        await wait_until_still(mcp)
    logger.info(f"{'drive' if leg > 0 else 'backup'} {abs(leg):.2f} m -> {'done' if ok else 'FAILED'}")
    return ok, f"{'drive' if leg > 0 else 'backup'} {abs(leg):.2f} m failed: {str(reply)[:200]}", leg


async def run(mcp: MCPClient, object_name: str, next_action: str) -> dict:
    """Drive up to ``object_name`` so that ``next_action`` can reach it.

    Returns ``{"success", "reason", "tool_calls_used"}``; on success also
    ``near_face_m``, ``bearing_deg`` and ``verified``.
    """
    tool_calls = 0

    def result(success: bool, reason: str, m: dict | None = None, verified: bool = True) -> dict:
        out = {"success": success, "reason": reason, "tool_calls_used": tool_calls}
        if success and m is not None:
            out["near_face_m"] = round(m["near"], 3)
            out["bearing_deg"] = round(math.degrees(m["bearing"]), 1)
            out["verified"] = verified
        return out

    if next_action not in STANDOFF_BY_NEXT_ACTION:
        return result(False, f"unknown next_action '{next_action}'; one of {sorted(STANDOFF_BY_NEXT_ACTION)}")
    if not object_name:
        return result(False, "object_name is empty")
    standoff = STANDOFF_BY_NEXT_ACTION[next_action]
    logger.info(f"approach '{object_name}' for {next_action}, standoff {standoff:.2f} m")

    # The arm tucked in look_forward: out of the base's way, and the arm camera looks ahead.
    arm = await move_arm_to_look_forward(mcp)
    tool_calls += 1
    if not arm.get("success"):
        return result(False, f"arm to look_forward failed: {arm.get('reason')}")

    # Find: look ahead, back up once and look again, then turn on the spot, up to a full turn.
    m, c = await _measure(mcp, object_name, ("front", "arm"))
    tool_calls += c
    if m is None:
        logger.info(f"'{object_name}' not seen: backing up {BACKUP_LOOK_M} m to look from farther away")
        ok, failure, _ = await _drive(mcp, -BACKUP_LOOK_M)
        tool_calls += 1
        if ok:
            m, c = await _measure(mcp, object_name, ("front", "arm"))
            tool_calls += c
        else:
            logger.info(f"{failure}; searching from here")
    step = 0
    while m is None and step < SEARCH_STEPS:
        step += 1
        logger.info(f"search {step}/{SEARCH_STEPS}")
        ok, reply = await _turn(mcp, SEARCH_STEP_RAD)
        tool_calls += 1
        if not ok:
            return result(False, f"search turn failed: {reply[:200]}")
        m, c = await _measure(mcp, object_name, ("front", "arm"))
        tool_calls += c
    if m is None:
        return result(False, f"'{object_name}' not found within {MAX_RANGE_M} m after a full turn")

    # Face it and drive in, measuring after every leg.
    for _ in range(MAX_LEGS):
        m, c, why = await _face(mcp, object_name, m)
        tool_calls += c
        if m is None:
            return result(False, why)
        if _in_window(m["near"], next_action):
            return result(True, f"at '{object_name}': near face {m['near']:.2f} m for {next_action}", m)
        ok, failure, leg = await _drive(mcp, m["near"] - standoff)
        tool_calls += 1
        if not ok:
            return result(False, failure)
        expected = m["near"] - leg
        seen, c = await _measure(mcp, object_name, _cameras(m))
        tool_calls += c
        if seen is not None and abs(seen["near"] - expected) > LOST_TOL_M:
            logger.info(f"{seen['camera']}: {seen['near']:.2f} m, expected {expected:.2f} m: another object")
            seen = None
        if seen is not None:
            m = seen
            continue
        # Not seen (too close for both cameras is normal at a low object): trust odometry if near enough.
        rest = expected - standoff
        logger.info(f"not seen after the drive; odometry puts the near face at {expected:.2f} m")
        if not _in_window(expected, next_action):
            if abs(rest) > BLIND_MAX_M:
                return result(False, f"lost '{object_name}' after driving {abs(leg):.2f} m, "
                                     f"{rest:.2f} m from the standoff")
            ok, failure, _ = await _drive(mcp, rest)
            tool_calls += 1
            if not ok:
                return result(False, failure)
            expected = standoff
        return result(True, f"at '{object_name}' by odometry: near face ~{expected:.2f} m for {next_action}; "
                            "not seen after the drive", {**m, "near": expected}, verified=False)
    m, c, why = await _face(mcp, object_name, m)
    tool_calls += c
    if m is not None and _in_window(m["near"], next_action):
        return result(True, f"at '{object_name}': near face {m['near']:.2f} m for {next_action}", m)
    near = f"{m['near']:.2f} m" if m else "unknown"
    return result(False, f"not at the standoff after {MAX_LEGS} drives (near face {near}, want {standoff:.2f}) {why}".strip())
