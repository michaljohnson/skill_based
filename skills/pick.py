"""Pick skill — deterministic Python grasp primitive (real Summit XL).

The skill takes an object name and runs the canonical pipeline: arm to
look_forward, reset the planning scene (floor kept), segment on the arm
camera (front-cam fallback), top-down grasp pose, reach check, open
gripper, pre-grasp, descend, close gripper, check the hold from the
gripper result, lift, return to look_forward.

The deterministic implementation removes the LLM's turn-by-turn reasoning
and bakes the canonical sequence into Python with bounded retries. On
failure it returns a structured reason string; the planner does not
retry inside the skill but may choose to call pick again after the
navigator re-positions.
"""

import json
import logging
import math

from skill_based.clients.mcp import MCPClient
from skill_based.utils import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    HOLD_MAX,
    HOLD_MIN,
    gripper_command,
    gripper_position,
    holding,
    ik_joints as _ik,
    move_arm_to_look_forward,
    parse_joints as _parse_joints,
    parse_seg_status,
    plan_to_xyz as _plan_to_xyz,
    reset_scene,
)

logger = logging.getLogger(__name__)


# UR5 practical reach forward from base_footprint.
UR5_REACH_X = 1.10

# Pre-grasp clearance when the perception tool returns no pre_grasp_pose.
PRE_GRASP_CLEARANCE_M = 0.12

def _orientation_to_list(orientation: dict | list) -> list[float]:
    """Coerce a perception-MCP orientation (dict {x,y,z,w}) to [x, y, z, w]."""
    if isinstance(orientation, dict):
        return [
            float(orientation["x"]),
            float(orientation["y"]),
            float(orientation["z"]),
            float(orientation["w"]),
        ]
    return [float(v) for v in orientation]


def _mirror_about_tool_z(q: list[float]) -> list[float]:
    """The same top-down grasp turned 180° about the tool axis.

    The Robotiq 2F-140 is symmetric, so both grip the object the same way;
    the one closer to the current wrist saves a large wrist_3 turn.
    """
    x, y, z, w = q
    # q * (0, 0, 1, 0)
    return [y, -x, w, -z]


async def _segment_object(
    mcp: MCPClient,
    object_name: str,
    camera: str = "arm",
) -> tuple[str, int]:
    """Segment ``object_name`` on the requested camera, return (status, calls)."""
    try:
        raw = await mcp.call_tool_prefixed(
            "perception__segment_objects",
            {"prompt": object_name, "camera": camera, "timeout": 20},
        )
        return parse_seg_status(raw), 1
    except Exception as e:
        logger.warning(f"segment_objects({camera}) error: {e}")
        return "ERROR", 1


async def _grasp_pose(
    mcp: MCPClient, object_name: str
) -> tuple[dict | None, int]:
    """Read the top-down grasp pose from the most recent segmentation."""
    try:
        raw = await mcp.call_tool_prefixed(
            "perception__get_topdown_grasp_pose",
            {"object_name": object_name},
        )
        data = json.loads(raw) if isinstance(raw, str) else raw
        if "centroid_base_frame" not in data:
            logger.warning(f"grasp pose missing centroid: {data}")
            return None, 1
        return data, 1
    except Exception as e:
        logger.warning(f"get_topdown_grasp_pose error: {e}")
        return None, 1


async def _choose_orientation(
    mcp: MCPClient, x: float, y: float, z: float, q: list[float]
) -> tuple[list[float], int]:
    """Pick q or its mirror, whichever needs the smaller wrist_3 turn."""
    calls = 1
    current = _parse_joints(
        await mcp.call_tool_prefixed("moveit__get_current_joint_state", {"group": "arm"})
    )
    if current is None:
        return q, calls
    best, best_turn = q, None
    for cand in (q, _mirror_about_tool_z(q)):
        calls += 1
        joints = await _ik(mcp, x, y, z, cand)
        if joints is None:
            continue
        d = joints[5] - current[5]
        turn = abs(math.atan2(math.sin(d), math.cos(d)))
        if best_turn is None or turn < best_turn:
            best, best_turn = cand, turn
    return best, calls


async def run(mcp: MCPClient, object_name: str) -> dict:
    """Grasp ``object_name`` from the surface in front of the robot.

    Preconditions: robot already positioned so the object is within reach
    (navigator owns positioning). Arm in any state.

    Returns:
        ``{"success", "reason", "object_name", "tool_calls_used"}`` and on
        success ``held_object_height_m`` and ``pads_above_bottom_m`` (how far
        the pad middle sits above the object's lowest point, for place).
    """
    tool_calls = 0
    base_result = {"object_name": object_name}

    def fail(reason: str) -> dict:
        return {**base_result, "success": False, "reason": reason, "tool_calls_used": tool_calls}

    # Step 0 — the gripper must be open; a closed one may still hold something
    try:
        finger = await gripper_position(mcp)
    except Exception as e:
        finger = None
        logger.warning(f"gripper position read error: {e}")
    tool_calls += 1
    if finger is None:
        return fail("gripper position unknown (/joint_states without finger_joint)")
    if finger > HOLD_MIN:
        return fail(
            f"gripper not open (finger_joint {finger:.3f}): it may hold an object. "
            "Place or open it first."
        )

    # Step 1 — arm to look_forward (joint_state primary)
    arm = await move_arm_to_look_forward(mcp)
    tool_calls += 1
    if not arm.get("success"):
        return fail(f"arm reset failed: {arm.get('reason')}")

    # Step 2 — planning scene: drop stale objects, keep the floor
    tool_calls += 2
    try:
        await reset_scene(mcp)
    except Exception as e:
        return fail(f"planning scene reset failed, no floor in the scene: {e}")

    # Step 3 — segment on arm camera; front-cam fallback if missed
    status, calls = await _segment_object(mcp, object_name, camera="arm")
    tool_calls += calls
    logger.info(f"arm SAM3 -> {status}")
    if status != "SUCCESS":
        status, calls = await _segment_object(mcp, object_name, camera="front")
        tool_calls += calls
        logger.info(f"front SAM3 fallback -> {status}")
        if status != "SUCCESS":
            return fail(f"segmentation failed on both cameras for '{object_name}'")

    # Step 4 — top-down grasp pose
    grasp, calls = await _grasp_pose(mcp, object_name)
    tool_calls += calls
    if grasp is None:
        return fail("get_topdown_grasp_pose returned no centroid_base_frame")
    pose = grasp.get("grasp_pose")
    if not pose or "position" not in pose:
        return fail(
            "get_topdown_grasp_pose returned no grasp_pose "
            f"(possibly TF failure): keys={list(grasp.keys())}"
        )
    try:
        gx, gy, gz = (float(pose["position"][k]) for k in ("x", "y", "z"))
        pre = grasp.get("pre_grasp_pose")
        pre_z = float(pre["position"]["z"]) if pre else gz + PRE_GRASP_CLEARANCE_M
        orientation = _orientation_to_list(pose["orientation"])
    except (KeyError, TypeError, ValueError) as e:
        return fail(f"malformed grasp pose from perception: {e!r}")
    values = (gx, gy, gz, pre_z, *orientation)
    if len(orientation) != 4 or not all(math.isfinite(v) for v in values):
        return fail(f"malformed grasp pose from perception: {values}")

    # Heights in base_footprint (the camera-frame bounding box measures depth,
    # not height). None if missing: place must not guess a height.
    bbox = grasp.get("bbox_base_frame") or {}
    held_object_height_m = pads_above_bottom_m = None
    try:
        bottom, top = float(bbox["z_min"]), float(bbox["z_max"])
        pad_z = gz - float(grasp.get("gripper_offset_m", 0.188))
        if not all(math.isfinite(v) for v in (bottom, top, pad_z)):
            raise ValueError("non-finite value")
        held_object_height_m = max(0.0, top - bottom)
        pads_above_bottom_m = max(0.0, pad_z - bottom)
    except (KeyError, TypeError, ValueError) as e:
        logger.warning(f"no usable bbox_base_frame ({e!r}): held height unknown")
    logger.info(
        f"grasp x={gx:.3f} y={gy:.3f} z={gz:.3f} pre_z={pre_z:.3f} "
        f"height={held_object_height_m} pads_above_bottom={pads_above_bottom_m} "
        f"xy={grasp.get('grasp_xy_method')} z_rule={grasp.get('grasp_z_method')} "
        f"oriented={grasp.get('oriented')}"
    )

    # Step 5 — reach check
    if gx > UR5_REACH_X:
        return fail(
            f"grasp x={gx:.2f}m exceeds UR5 reach {UR5_REACH_X:.2f}m. "
            "Navigator must redeliver closer."
        )

    # Step 6 — open gripper
    try:
        opened = await gripper_command(mcp, GRIPPER_OPEN)
    except Exception as e:
        return fail(f"open_gripper error: {e}")
    tool_calls += 1
    # `not <=` also fails on a missing position (NaN)
    if not opened["position"] <= HOLD_MIN:
        return fail(f"gripper did not open (position {opened['position']:.3f}); arm not moved down")

    # Step 7 — orientation with the smaller wrist turn, then pre-grasp
    try:
        orientation, calls = await _choose_orientation(mcp, gx, gy, pre_z, orientation)
        tool_calls += calls
    except Exception as e:
        logger.warning(f"orientation choice failed, using the tool's: {e}")
    ok, info, calls = await _plan_to_xyz(mcp, gx, gy, pre_z, orientation)
    tool_calls += calls
    if not ok:
        return fail(f"pre-grasp plan failed: {info}")

    # Step 8 — descend to grasp z
    ok, info, calls = await _plan_to_xyz(mcp, gx, gy, gz, orientation)
    tool_calls += calls
    if not ok:
        back, back_info, calls = await _plan_to_xyz(mcp, gx, gy, pre_z, orientation)
        tool_calls += calls
        if not back:
            return fail(f"descent plan failed: {info}; retract failed too, arm near the grasp pose: {back_info}")
        return fail(f"descent plan failed: {info}")

    # Step 9 — close gripper; the hold is judged from where the fingers stalled
    try:
        closed = await gripper_command(mcp, GRIPPER_CLOSED)
    except Exception as e:
        logger.warning(f"close_gripper error: {e}")
        closed = None
    tool_calls += 1
    if closed is None or math.isnan(closed["position"]):
        # No usable result: only a fully closed gripper counts as empty
        try:
            finger = await gripper_position(mcp)
        except Exception as e:
            logger.warning(f"gripper position read error: {e}")
            finger = None
        tool_calls += 1
        if finger is None or not (math.isfinite(finger) and finger >= HOLD_MAX):
            return fail(
                f"gripper state unknown after the close (finger_joint {finger}); "
                "it may hold the object, arm left at the grasp pose"
            )
        closed = {"position": finger, "stalled": False, "status": None}
    logger.info(
        f"close -> position={closed['position']:.3f} stalled={closed['stalled']} "
        f"status={closed['status']}"
    )
    if not holding(closed):
        try:
            await gripper_command(mcp, GRIPPER_OPEN)
            await _plan_to_xyz(mcp, gx, gy, pre_z, orientation)
            await move_arm_to_look_forward(mcp)
        except Exception as e:
            logger.warning(f"recovery after empty grasp failed: {e}")
        tool_calls += 3
        return fail(
            f"empty grasp: gripper closed to {closed['position']:.3f} "
            f"(stalled={closed['stalled']}); object missed or slipped"
        )

    # Step 10 — lift back to the pre-grasp height
    ok, info, calls = await _plan_to_xyz(mcp, gx, gy, pre_z, orientation)
    tool_calls += calls
    if not ok:
        return fail(f"lift failed with the object held, arm left at the grasp pose: {info}")

    # Step 11 — return to look_forward (transit posture)
    arm = await move_arm_to_look_forward(mcp)
    tool_calls += 1
    if not arm.get("success"):
        logger.warning(f"post-lift look_forward failed: {arm.get('reason')}")

    return {
        **base_result,
        "success": True,
        "reason": f"grasped '{object_name}' (gripper stalled at {closed['position']:.3f})",
        "held_object_height_m": held_object_height_m,
        "pads_above_bottom_m": pads_above_bottom_m,
        "tool_calls_used": tool_calls,
    }
