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

The planner LLM sees only ``object_name`` and ``next_action``. A failure says
what happened, the evidence and where the base is now (``reason``), its kind
(``failure``) and the log lines of the call (``steps``).
"""

import json
import logging
import math
from dataclasses import dataclass, field

from skill_based.clients.mcp import MCPClient
from skill_based.skills.pick import UR5_REACH_X
from skill_based.skills.place import CONTAINER_REACH_REAL_M, SURFACE_EDGE_INSET_M
from skill_based.utils import (
    SEG_ANSWERS,
    collect_steps,
    move_arm_to_look_forward,
    parse_seg_status,
    wait_until_still,
)

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


def _nav2_text(reply) -> str:
    """nav2 MCP reply on one line: its message and error code, not the JSON."""
    try:
        r = json.loads(reply)
        return f"{r.get('message', r)} ({r.get('error_code', '?')})"
    except (TypeError, ValueError, AttributeError):
        return " ".join(str(reply).split())[:150]


@dataclass
class _Tally:
    """What one approach call saw and did, for the result's reason."""

    looks: list[str] = field(default_factory=list)  # one segmentation status per look; TOO_FAR = beyond range
    forward_m: float = 0.0
    back_m: float = 0.0
    turns: int = 0
    stopped: str = ""  # a refused or aborted motion: nav2 does not say how far it got

    def motion(self) -> str:
        parts = []
        if self.forward_m:
            parts.append(f"drove {self.forward_m:.2f} m forward")
        if self.back_m:
            parts.append(f"backed up {self.back_m:.2f} m")
        if self.turns:
            parts.append(f"turned {self.turns} time{'s' if self.turns > 1 else ''}")
        if self.stopped:
            parts.append(f"{'then ' if parts else ''}stopped partway through a {self.stopped} "
                         "(nav2 does not report how far it got)")
        return f"The base {', '.join(parts)}." if parts else "The base did not move."

    def errors(self) -> list[str]:
        return [s for s in self.looks if s not in SEG_ANSWERS + ("TOO_FAR",)]


async def _measure(
    mcp: MCPClient, object_name: str, cameras: tuple[str, ...], tally: _Tally
) -> tuple[dict | None, int]:
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
            if status == "UNKNOWN":
                logger.warning(f"{camera} segment_objects reply: {str(raw)[:300]}")
            if status != "SUCCESS":
                tally.looks.append(status)
                continue
            raw = await mcp.call_tool_prefixed("perception__get_topdown_grasp_pose", {"object_name": object_name})
            calls += 1
            g = json.loads(raw) if isinstance(raw, str) else raw
            cx, cy = float(g["centroid_base_frame"]["x"]), float(g["centroid_base_frame"]["y"])
            dist = math.hypot(cx, cy)
            depth = cx - float(g["bbox_base_frame"]["x_min"])  # centroid behind the near face
        except Exception as e:
            calls += 1
            tally.looks.append(f"ERROR: {e}"[:80])
            logger.warning(f"measure on {camera} failed: {e}")
            continue
        if dist > MAX_RANGE_M:
            tally.looks.append("TOO_FAR")
            logger.info(f"{camera}: '{object_name}' at {dist:.2f} m, beyond {MAX_RANGE_M} m: not the target")
            continue
        tally.looks.append("SUCCESS")
        m = {"camera": camera, "dist": dist, "near": dist - max(depth, 0.0), "bearing": math.atan2(cy, cx)}
        logger.info(f"{camera}: near face {m['near']:.2f} m, bearing {math.degrees(m['bearing']):.1f} deg")
        return m, calls
    return None, calls


def _cameras(last: dict | None) -> tuple[str, ...]:
    if last is not None and last["dist"] < ARM_FIRST_BELOW_M:
        return ("arm", "front")
    return ("front", "arm")


def _where(m: dict) -> str:
    return f"{m['near']:.2f} m, {math.degrees(m['bearing']):+.0f} deg"


async def _turn(mcp: MCPClient, angle: float, tally: _Tally) -> tuple[bool, str]:
    reply = await mcp.call_tool_prefixed("nav2__spin_robot", {"angle": round(angle, 4)})
    await wait_until_still(mcp)
    ok = _ok(reply)
    tally.turns += ok
    if not ok:
        tally.stopped = f"{math.degrees(angle):+.0f} deg turn"
    logger.info(f"turn {math.degrees(angle):+.0f} deg -> {'done' if ok else 'FAILED: ' + _nav2_text(reply)}")
    return ok, _nav2_text(reply)


async def _face(
    mcp: MCPClient, object_name: str, m: dict, tally: _Tally
) -> tuple[dict | None, int, str, str]:
    """Turn until the bearing is within tolerance, measuring after every turn.

    Returns (measurement, calls, failure, reason); measurement None on failure.
    """
    calls = 0
    for _ in range(MAX_TURNS):
        if abs(m["bearing"]) <= BEARING_TOL_RAD:
            return m, calls, "", ""
        ok, reply = await _turn(mcp, m["bearing"] * TURN_GAIN, tally)
        calls += 1
        if not ok:
            return None, calls, "turn_failed", f"nav2 refused the turn towards '{object_name}' ({reply})."
        last = m
        m, c = await _measure(mcp, object_name, _cameras(last), tally)
        calls += c
        if m is None:
            return None, calls, "lost", (
                f"Saw '{object_name}' at {_where(last)}, but not after turning towards it.")
    if abs(m["bearing"]) <= BEARING_TOL_RAD:
        return m, calls, "", ""
    return None, calls, "not_facing", (
        f"'{object_name}' still {math.degrees(m['bearing']):.0f} deg off after {MAX_TURNS} turns "
        "(the base turns farther than commanded).")


def _in_window(near: float, next_action: str) -> bool:
    return STANDOFF_BY_NEXT_ACTION[next_action] - NEAR_TOL_M <= near <= MAX_NEAR_BY_NEXT_ACTION[next_action]


async def _drive(mcp: MCPClient, gap: float, tally: _Tally) -> tuple[bool, str, float]:
    """Drive towards (gap > 0) or back away from the object, one leg; returns the signed leg."""
    leg = max(-MAX_LEG_M, min(gap, MAX_LEG_M))
    what = "drive" if leg > 0 else "backup"
    tool = "nav2__drive_on_heading" if leg > 0 else "nav2__backup_robot"
    reply = await mcp.call_tool_prefixed(tool, {"distance": round(abs(leg), 3), "speed": DRIVE_SPEED})
    ok = _ok(reply)
    if ok:
        await wait_until_still(mcp)
        if leg > 0:
            tally.forward_m += leg
        else:
            tally.back_m -= leg
    else:
        tally.stopped = f"{abs(leg):.2f} m {what}"
    logger.info(f"{what} {abs(leg):.2f} m -> {'done' if ok else 'FAILED: ' + _nav2_text(reply)}")
    why = f"nav2 stopped the {what} ({_nav2_text(reply)}); usually an obstacle in the robot's path."
    return ok, why, leg


async def run(mcp: MCPClient, object_name: str, next_action: str) -> dict:
    """Drive up to ``object_name`` so that ``next_action`` can reach it.

    Returns ``{"success", "reason", "tool_calls_used"}``; on success also
    ``near_face_m``, ``bearing_deg`` and ``verified``, on failure ``failure``
    and ``steps``.
    """
    with collect_steps(logger) as steps:
        return await _run(mcp, object_name, next_action, steps)


async def _run(mcp: MCPClient, object_name: str, next_action: str, steps: list[str]) -> dict:
    tool_calls = 0
    tally = _Tally()

    def done(reason: str, m: dict, verified: bool = True) -> dict:
        return {"success": True, "reason": f"{reason} {tally.motion()}", "tool_calls_used": tool_calls,
                "near_face_m": round(m["near"], 3), "bearing_deg": round(math.degrees(m["bearing"]), 1),
                "verified": verified}

    def fail(failure: str, reason: str) -> dict:
        return {"success": False, "failure": failure, "reason": f"{reason} {tally.motion()}",
                "tool_calls_used": tool_calls, "steps": list(steps)}

    if next_action not in STANDOFF_BY_NEXT_ACTION:
        return fail("invalid_args", f"Unknown next_action '{next_action}'; one of {sorted(STANDOFF_BY_NEXT_ACTION)}.")
    if not object_name:
        return fail("invalid_args", "object_name is empty.")
    standoff = STANDOFF_BY_NEXT_ACTION[next_action]
    logger.info(f"approach '{object_name}' for {next_action}, standoff {standoff:.2f} m")

    # The arm tucked in look_forward: out of the base's way, and the arm camera looks ahead.
    arm = await move_arm_to_look_forward(mcp)
    tool_calls += 1
    if not arm.get("success"):
        return fail("arm_failed", f"Moving the arm to look_forward failed ({arm.get('reason')}). "
                                  "Usually the UR program is not running.")

    # Find: look ahead, back up once and look again, then turn on the spot, up to a full turn.
    m, c = await _measure(mcp, object_name, ("front", "arm"), tally)
    tool_calls += c
    if m is None:
        logger.info(f"'{object_name}' not seen: backing up {BACKUP_LOOK_M} m to look from farther away")
        ok, why, _ = await _drive(mcp, -BACKUP_LOOK_M, tally)
        tool_calls += 1
        if ok:
            m, c = await _measure(mcp, object_name, ("front", "arm"), tally)
            tool_calls += c
        else:
            logger.info(f"{why} Searching from here.")
    step = 0
    while m is None and step < SEARCH_STEPS:
        step += 1
        logger.info(f"search {step}/{SEARCH_STEPS}")
        ok, reply = await _turn(mcp, SEARCH_STEP_RAD, tally)
        tool_calls += 1
        if not ok:
            return fail("turn_failed", f"'{object_name}' not seen yet; nav2 refused search turn {step} ({reply}).")
        m, c = await _measure(mcp, object_name, ("front", "arm"), tally)
        tool_calls += c
    if m is None:
        errors = tally.errors()
        if len(errors) * 2 > len(tally.looks):
            return fail("perception_error", (
                f"Segmentation failed on {len(errors)} of {len(tally.looks)} looks ({errors[-1]}): "
                f"the perception service is not working, so the search says nothing about whether "
                f"'{object_name}' is here."))
        far = tally.looks.count("TOO_FAR")
        also = f" SAM3 found a '{object_name}' only beyond {MAX_RANGE_M} m ({far} times): another object." if far else ""
        return fail("not_found", (
            f"'{object_name}' not seen on {len(tally.looks)} looks with the front and arm cameras, "
            f"after backing up and over a full turn.{also} It is not within {MAX_RANGE_M} m, hidden, "
            "or too small to see from here (small objects only from ~1.7 m)."))

    # Face it and drive in, measuring after every leg.
    for _ in range(MAX_LEGS):
        m, c, failure, why = await _face(mcp, object_name, m, tally)
        tool_calls += c
        if m is None:
            return fail(failure, why)
        if _in_window(m["near"], next_action):
            return done(f"At '{object_name}' for {next_action}: near face {_where(m)}, seen by the {m['camera']} camera.", m)
        ok, why, leg = await _drive(mcp, m["near"] - standoff, tally)
        tool_calls += 1
        if not ok:
            return fail("drive_failed", f"'{object_name}' was at {_where(m)} before the drive; {why}")
        expected = m["near"] - leg
        seen, c = await _measure(mcp, object_name, _cameras(m), tally)
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
                return fail("lost", (
                    f"Lost '{object_name}' after driving {abs(leg):.2f} m: odometry puts it at {expected:.2f} m, "
                    f"{rest:.2f} m from the {standoff:.2f} m standoff, too far to finish blind."))
            ok, why, _ = await _drive(mcp, rest, tally)
            tool_calls += 1
            if not ok:
                return fail("drive_failed", why)
            expected = standoff
        return done(f"At '{object_name}' for {next_action} by odometry: near face ~{expected:.2f} m; "
                    "out of both cameras' view after the last drive (normal for low objects).",
                    {**m, "near": expected}, verified=False)
    m, c, failure, why = await _face(mcp, object_name, m, tally)
    tool_calls += c
    if m is not None and _in_window(m["near"], next_action):
        return done(f"At '{object_name}' for {next_action}: near face {_where(m)}, seen by the {m['camera']} camera.", m)
    near = _where(m) if m else "unknown"
    return fail("not_at_standoff", (
        f"Not at the {standoff:.2f} m standoff after {MAX_LEGS} drives (near face {near}). {why}").strip())
