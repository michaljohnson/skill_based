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
navigator re-positions. Every result says whether the gripper holds the
object (``holding``: true, false, or null when unknown); a failure adds its
kind (``failure``) and the log lines of the call (``steps``).
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
    SEG_ANSWERS,
    collect_steps,
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
    """Segment ``object_name`` on the requested camera, return (status, calls).

    SUCCESS_NO_CLOUD: SAM3 found it but the point cloud never arrived (camera hung).
    """
    try:
        raw = await mcp.call_tool_prefixed(
            "perception__segment_objects",
            {"prompt": object_name, "camera": camera, "timeout": 20},
        )
        status = parse_seg_status(raw)
        if status == "UNKNOWN":
            logger.warning(f"segment_objects({camera}) reply: {str(raw)[:300]}")
            if "reported SUCCESS" in str(raw):
                status = "SUCCESS_NO_CLOUD"
        return status, 1
    except Exception as e:
        logger.warning(f"segment_objects({camera}) error: {e}")
        return f"ERROR: {e}"[:80], 1


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
        ``{"success", "reason", "holding", "object_name", "tool_calls_used"}``;
        on success ``held_object_height_m`` and ``pads_above_bottom_m`` (how far
        the pad middle sits above the object's lowest point, for place), on
        failure ``failure`` and ``steps``.
    """
    with collect_steps(logger) as steps:
        return await _run(mcp, object_name, steps)


async def _run(mcp: MCPClient, object_name: str, steps: list[str]) -> dict:
    tool_calls = 0
    base_result = {"object_name": object_name}

    def fail(failure: str, reason: str, holding: bool | None = False) -> dict:
        return {**base_result, "success": False, "failure": failure, "reason": reason,
                "holding": holding, "tool_calls_used": tool_calls, "steps": list(steps)}

    # Step 0 — the gripper must be open; a closed one may still hold something
    try:
        finger = await gripper_position(mcp)
    except Exception as e:
        finger = None
        logger.warning(f"gripper position read error: {e}")
    tool_calls += 1
    if finger is None:
        return fail("gripper_failed", "Gripper position unknown (/joint_states without finger_joint); "
                                      "nothing moved.", holding=None)
    if finger > HOLD_MIN:
        held = HOLD_MIN <= finger <= HOLD_MAX
        return fail("gripper_not_open", (
            f"The gripper is not open (finger_joint {finger:.3f}): "
            f"{'it probably holds an object, place it first' if held else 'it is closed, open it first'}. "
            "Nothing moved."), holding=held)

    # Step 1 — arm to look_forward (joint_state primary)
    arm = await move_arm_to_look_forward(mcp)
    tool_calls += 1
    if not arm.get("success"):
        return fail("arm_failed", f"Moving the arm to look_forward failed ({arm.get('reason')}). "
                                  "Usually the UR program is not running.")

    # Step 2 — planning scene: drop stale objects, keep the floor
    tool_calls += 2
    try:
        await reset_scene(mcp)
    except Exception as e:
        return fail("scene_failed", f"Planning scene reset failed ({e}); without the floor box the arm "
                                    "is not planned safely. Arm at look_forward.")

    # Step 3 — segment on arm camera; front-cam fallback if missed
    status, calls = await _segment_object(mcp, object_name, camera="arm")
    tool_calls += calls
    arm_status = status
    logger.info(f"arm SAM3 -> {status}")
    if status != "SUCCESS":
        status, calls = await _segment_object(mcp, object_name, camera="front")
        tool_calls += calls
        logger.info(f"front SAM3 fallback -> {status}")
        if status != "SUCCESS":
            looks = (arm_status, status)
            if all(s in SEG_ANSWERS for s in looks):
                return fail("not_found", (
                    f"'{object_name}' not found by the arm or the front camera from here. Arm at look_forward. "
                    "Approach it first (next_action pick), or it is not here."))
            return fail("perception_error", (
                f"Segmentation of '{object_name}' did not work (arm: {arm_status}, front: {status}); "
                "SUCCESS_NO_CLOUD means SAM3 found it but the camera sent no point cloud (camera hung). "
                "Arm at look_forward."))

    # Step 4 — top-down grasp pose
    grasp, calls = await _grasp_pose(mcp, object_name)
    tool_calls += calls
    if grasp is None:
        return fail("perception_error", "Segmented, but get_topdown_grasp_pose gave no centroid "
                                        "(point cloud missing?). Arm at look_forward.")
    pose = grasp.get("grasp_pose")
    if not pose or "position" not in pose:
        return fail("perception_error", (
            f"get_topdown_grasp_pose returned no grasp_pose (possibly a TF failure; keys {list(grasp.keys())}). "
            "Arm at look_forward."))
    try:
        gx, gy, gz = (float(pose["position"][k]) for k in ("x", "y", "z"))
        pre = grasp.get("pre_grasp_pose")
        pre_z = float(pre["position"]["z"]) if pre else gz + PRE_GRASP_CLEARANCE_M
        orientation = _orientation_to_list(pose["orientation"])
    except (KeyError, TypeError, ValueError) as e:
        return fail("perception_error", f"Malformed grasp pose from perception ({e!r}). Arm at look_forward.")
    values = (gx, gy, gz, pre_z, *orientation)
    if len(orientation) != 4 or not all(math.isfinite(v) for v in values):
        return fail("perception_error", f"Malformed grasp pose from perception ({values}). Arm at look_forward.")

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
        return fail("out_of_reach", (
            f"'{object_name}' grasp point {gx:.2f} m ahead, beyond the arm's {UR5_REACH_X:.2f} m reach. "
            "Arm at look_forward. Approach it again (next_action pick)."))

    # Step 6 — open gripper
    try:
        opened = await gripper_command(mcp, GRIPPER_OPEN)
    except Exception as e:
        return fail("gripper_failed", f"Opening the gripper failed ({e}). Arm at look_forward.", holding=None)
    tool_calls += 1
    # `not <=` also fails on a missing position (NaN)
    if not opened["position"] <= HOLD_MIN:
        return fail("gripper_failed", (
            f"The gripper did not open (position {opened['position']:.3f}). Arm at look_forward."), holding=None)

    # Step 7 — orientation with the smaller wrist turn, then pre-grasp
    try:
        orientation, calls = await _choose_orientation(mcp, gx, gy, pre_z, orientation)
        tool_calls += calls
    except Exception as e:
        logger.warning(f"orientation choice failed, using the tool's: {e}")
    ok, info, calls = await _plan_to_xyz(mcp, gx, gy, pre_z, orientation)
    tool_calls += calls
    if not ok:
        return fail("motion_failed", (
            f"Moving above '{object_name}' failed at ({gx:.2f}, {gy:.2f}, {pre_z:.2f}) m: {info.rstrip('.')}. "
            "Arm at look_forward."))

    # Step 8 — descend to grasp z
    ok, info, calls = await _plan_to_xyz(mcp, gx, gy, gz, orientation)
    tool_calls += calls
    if not ok:
        back, back_info, calls = await _plan_to_xyz(mcp, gx, gy, pre_z, orientation)
        tool_calls += calls
        if not back:
            return fail("motion_failed", (
                f"Moving down to '{object_name}' failed ({info}), and the retract failed too ({back_info}). "
                "Arm near the grasp pose, gripper open."))
        return fail("motion_failed", (
            f"Moving down to '{object_name}' at z {gz:.2f} m failed: {info.rstrip('.')}. Arm above the object, gripper open."))

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
            return fail("gripper_failed", (
                f"Gripper state unknown after the close (finger_joint {finger}); it may hold "
                f"'{object_name}'. Arm at the grasp pose."), holding=None)
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
        return fail("grasp_empty", (
            f"Empty grasp: the gripper closed to {closed['position']:.3f} without stopping on "
            f"'{object_name}' (missed or slipped). Gripper opened again, arm back at look_forward. "
            "The object may have moved; pick again segments it anew."))

    # Step 10 — lift back to the pre-grasp height
    ok, info, calls = await _plan_to_xyz(mcp, gx, gy, pre_z, orientation)
    tool_calls += calls
    if not ok:
        return fail("motion_failed", (
            f"Holding '{object_name}', but the lift failed ({info}). Arm at the grasp pose."), holding=True)

    # Step 11 — return to look_forward (transit posture)
    arm = await move_arm_to_look_forward(mcp)
    tool_calls += 1
    if not arm.get("success"):
        logger.warning(f"post-lift look_forward failed: {arm.get('reason')}")

    return {
        **base_result,
        "success": True,
        "reason": (
            f"Holding '{object_name}': grasped at ({gx:.2f}, {gy:.2f}) m, the gripper stopped at "
            f"{closed['position']:.3f} on the object. "
            + ("Arm at look_forward." if arm.get("success") else "Arm still above the grasp point "
               f"(look_forward failed: {arm.get('reason')}).")
        ),
        "holding": True,
        "held_object_height_m": held_object_height_m,
        "pads_above_bottom_m": pads_above_bottom_m,
        "tool_calls_used": tool_calls,
    }
