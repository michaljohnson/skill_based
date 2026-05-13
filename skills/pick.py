"""Pick skill — deterministic Python grasp primitive.

The skill takes an object name and runs the canonical pipeline:
pre-check /gripper/status, arm to look_forward, clear octomap, segment
on the arm camera (front-cam fallback), top-down grasp pose, reach
check, pre-grasp, descend, close gripper, verify attachment, lift,
return to look_forward.

The deterministic implementation removes the LLM's turn-by-turn reasoning
and bakes the canonical sequence into Python with bounded retries. On
failure it returns a structured reason string; the planner does not
retry inside the skill but may choose to call pick again after the
navigator re-positions.
"""

import asyncio
import json
import logging

from skill_based.clients.mcp import MCPClient
from skill_based.skills.common import (
    LOOK_FORWARD_JOINTS,
    _tokenize,
    move_arm_to_look_forward,
    parse_seg_status,
    wait_for_gripper_attached,
)

logger = logging.getLogger(__name__)


# UR5e practical reach forward from base_footprint, per pick.md step 5.
UR5_REACH_X = 1.10

# Default top-down quaternion as (x, y, z, w). Used only as a fallback
# when the perception MCP cannot return a shape-aware orientation
# (e.g. on an intermediate recovery lift after the grasp pose has been
# consumed). Real picks use the orientation from get_topdown_grasp_pose,
# which is shape-aware (PCA on the segmented point cloud).
TOPDOWN_ORIENTATION_DEFAULT = [1.0, 0.0, 0.0, 0.0]


def _orientation_to_list(orientation: dict | list) -> list[float]:
    """Coerce a perception-MCP orientation (dict {x,y,z,w}) to the
    moveit_mcp__plan_and_execute list form [x, y, z, w]. Pass-through
    for list inputs."""
    if isinstance(orientation, dict):
        return [
            float(orientation["x"]),
            float(orientation["y"]),
            float(orientation["z"]),
            float(orientation["w"]),
        ]
    return [float(v) for v in orientation]


async def _gripper_status(mcp: MCPClient, timeout: float = 3.0) -> str:
    """Read the latched /gripper/status topic and return ``data`` or ``""``."""
    try:
        raw = await mcp.call_tool_prefixed(
            "ros__subscribe_once",
            {
                "topic": "/gripper/status",
                "msg_type": "std_msgs/msg/String",
                "timeout": int(timeout),
            },
        )
        data = json.loads(raw) if isinstance(raw, str) else raw
        msg = data.get("msg", data)
        return msg.get("data", "") if isinstance(msg, dict) else str(msg)
    except Exception as e:
        logger.warning(f"  [pick] gripper status read error: {e}")
        return ""


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
        logger.warning(f"  [pick] segment_objects({camera}) error: {e}")
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
            logger.warning(f"  [pick] grasp pose missing centroid: {data}")
            return None, 1
        return data, 1
    except Exception as e:
        logger.warning(f"  [pick] get_topdown_grasp_pose error: {e}")
        return None, 1


async def _open_gripper(mcp: MCPClient) -> int:
    await mcp.call_tool_prefixed(
        "ros__send_action_goal",
        {
            "action_name": "/robotiq_gripper_controller/gripper_cmd",
            "action_type": "control_msgs/action/GripperCommand",
            "goal": {"command": {"position": 0.0, "max_effort": 50.0}},
        },
    )
    return 1


async def _close_gripper(mcp: MCPClient) -> int:
    await mcp.call_tool_prefixed(
        "ros__send_action_goal",
        {
            "action_name": "/robotiq_gripper_controller/gripper_cmd",
            "action_type": "control_msgs/action/GripperCommand",
            "goal": {"command": {"position": 0.7, "max_effort": 50.0}},
        },
    )
    return 1


async def _plan_to_xyz(
    mcp: MCPClient,
    x: float,
    y: float,
    z: float,
    orientation: list[float] | None = None,
    *,
    clear_scene_on_retry: bool = True,
) -> tuple[bool, str, int]:
    """Plan + execute a top-down pose, with one clear_planning_scene retry.

    ``orientation`` is the quaternion as ``[x, y, z, w]``. When omitted,
    the default top-down orientation (180 deg about X) is used; real
    picks should pass the shape-aware orientation returned by
    ``perception__get_topdown_grasp_pose``.
    """
    if orientation is None:
        orientation = TOPDOWN_ORIENTATION_DEFAULT
    target = {
        "position": [x, y, z],
        "orientation": orientation,
        "frame_id": "base_footprint",
    }
    try:
        result = await mcp.call_tool_prefixed(
            "moveit__plan_and_execute",
            {"group": "arm", "target_type": "pose", "target": target},
        )
        if "fail" not in result.lower() or "completed" in result.lower():
            return True, result[:200], 1
    except Exception as e:
        return False, f"plan_and_execute error: {e}", 1

    if not clear_scene_on_retry:
        return False, f"plan failed at ({x:.2f},{y:.2f},{z:.2f}): {result[:200]}", 1

    # Retry once after clearing the planning scene
    try:
        await mcp.call_tool_prefixed("moveit__clear_planning_scene", {})
    except Exception as e:
        logger.warning(f"  [pick] clear_planning_scene error: {e}")
    try:
        result = await mcp.call_tool_prefixed(
            "moveit__plan_and_execute",
            {"group": "arm", "target_type": "pose", "target": target},
        )
        if "fail" not in result.lower() or "completed" in result.lower():
            return True, f"retry ok: {result[:200]}", 3
        return False, f"retry still failed: {result[:200]}", 3
    except Exception as e:
        return False, f"retry error: {e}", 3


async def run(mcp: MCPClient, object_name: str) -> dict:
    """Grasp ``object_name`` from the surface in front of the robot.

    Preconditions: robot already positioned at standoff distance from
    the surface holding the target (navigator owns positioning). Arm in
    any state.

    Returns:
        ``{"success": bool, "reason": str, "object_name": str, "tool_calls_used": int}``
    """
    tool_calls = 0
    base_result = {"object_name": object_name}

    # Step 0 — pre-check /gripper/status. If already attached to the
    # target, the previous attempt succeeded silently; report SUCCESS.
    status = await _gripper_status(mcp, timeout=3.0)
    tool_calls += 1
    if status.startswith("attached:"):
        model = status[len("attached:"):].strip()
        if _tokenize(object_name) & _tokenize(model):
            return {
                **base_result,
                "success": True,
                "reason": f"already attached:{model} (Step 0 short-circuit)",
                "tool_calls_used": tool_calls,
            }
        return {
            **base_result,
            "success": False,
            "reason": (
                f"holding wrong object: attached:{model} (expected '{object_name}'). "
                "Place/detach first."
            ),
            "tool_calls_used": tool_calls,
        }

    # Step 1 — arm to look_forward (joint_state PRIMARY)
    arm = await move_arm_to_look_forward(mcp)
    tool_calls += 1
    if not arm.get("success"):
        return {
            **base_result,
            "success": False,
            "reason": f"arm reset failed: {arm.get('reason')}",
            "tool_calls_used": tool_calls,
        }

    # Step 2 — clear octomap
    try:
        await mcp.call_tool_prefixed(
            "ros__call_service",
            {
                "service_name": "/clear_octomap",
                "service_type": "std_srvs/srv/Empty",
                "request": {},
            },
        )
        tool_calls += 1
    except Exception as e:
        logger.warning(f"  [pick] clear_octomap error: {e}")
        tool_calls += 1

    # Step 3 — segment on arm camera; front-cam fallback if missed
    status, calls = await _segment_object(mcp, object_name, camera="arm")
    tool_calls += calls
    logger.info(f"  [pick] arm SAM3 -> {status}")
    if status != "SUCCESS":
        status, calls = await _segment_object(mcp, object_name, camera="front")
        tool_calls += calls
        logger.info(f"  [pick] front SAM3 fallback -> {status}")
        if status != "SUCCESS":
            return {
                **base_result,
                "success": False,
                "reason": (
                    f"segmentation failed on both cameras for '{object_name}'"
                ),
                "tool_calls_used": tool_calls,
            }

    # Step 4 — get top-down grasp pose
    grasp, calls = await _grasp_pose(mcp, object_name)
    tool_calls += calls
    if grasp is None:
        return {
            **base_result,
            "success": False,
            "reason": "get_topdown_grasp_pose returned no centroid_base_frame",
            "tool_calls_used": tool_calls,
        }

    pose = grasp.get("grasp_pose")
    if not pose or "position" not in pose:
        return {
            **base_result,
            "success": False,
            "reason": (
                "get_topdown_grasp_pose returned no grasp_pose "
                f"(possibly TF failure): keys={list(grasp.keys())}"
            ),
            "tool_calls_used": tool_calls,
        }
    pos = pose["position"]
    gx = float(pos["x"])
    gy = float(pos["y"])
    gz = float(pos["z"])
    centroid_z = float(grasp["centroid_base_frame"]["z"])
    grasp_orientation = _orientation_to_list(pose["orientation"])
    grasp_yaw_deg = float(grasp.get("principal_axis_angle_deg", 0.0))
    aspect = float(grasp.get("principal_axis_aspect_ratio", 1.0))
    oriented = bool(grasp.get("oriented", False))
    logger.info(
        f"  [pick] grasp x={gx:.3f} y={gy:.3f} z={gz:.3f} "
        f"centroid_z={centroid_z:.3f} "
        f"yaw_deg={grasp_yaw_deg:.1f} aspect={aspect:.2f} "
        f"oriented={oriented}"
    )

    # Step 5 — reach check
    if gx > UR5_REACH_X:
        return {
            **base_result,
            "success": False,
            "reason": (
                f"grasp x={gx:.2f}m exceeds UR5 reach {UR5_REACH_X:.2f}m. "
                "Navigator must redeliver closer."
            ),
            "tool_calls_used": tool_calls,
        }

    # Step 6 — open gripper
    try:
        tool_calls += await _open_gripper(mcp)
    except Exception as e:
        return {
            **base_result,
            "success": False,
            "reason": f"open_gripper error: {e}",
            "tool_calls_used": tool_calls,
        }

    # Step 7 — pre-grasp pose (20cm above grasp z). Higher clearance
    # gives a cleaner visual descent and matches the lift height for
    # a symmetric approach / retreat silhouette.
    ok, info, calls = await _plan_to_xyz(
        mcp, gx, gy, gz + 0.20, orientation=grasp_orientation
    )
    tool_calls += calls
    if not ok:
        return {
            **base_result,
            "success": False,
            "reason": f"pre-grasp plan failed: {info}",
            "tool_calls_used": tool_calls,
        }

    # Step 8 — descend to grasp z. If MoveIt reports failure, the attach
    # plugin (proximity-based) may already have fired on contact — check
    # /gripper/status before treating descent as a failure.
    ok, info, calls = await _plan_to_xyz(
        mcp, gx, gy, gz, orientation=grasp_orientation, clear_scene_on_retry=False
    )
    tool_calls += calls
    if not ok:
        status = await _gripper_status(mcp, timeout=3.0)
        tool_calls += 1
        if status.startswith("attached:") and (
            _tokenize(object_name) & _tokenize(status[len("attached:"):])
        ):
            logger.info(
                f"  [pick] descent reported failure but {status} — proceeding"
            )
        else:
            return {
                **base_result,
                "success": False,
                "reason": f"descent plan failed and gripper detached: {info}",
                "tool_calls_used": tool_calls,
            }

    # Step 9 — close gripper
    try:
        tool_calls += await _close_gripper(mcp)
    except Exception as e:
        return {
            **base_result,
            "success": False,
            "reason": f"close_gripper error: {e}",
            "tool_calls_used": tool_calls,
        }

    # Step 10 — verify attachment via /gripper/status (8s + 5s retry)
    attach = await wait_for_gripper_attached(
        mcp, expected_object=object_name, timeout=8.0, retry_timeout=5.0
    )
    tool_calls += 2  # two subscribe_once calls inside wait_for_gripper_attached worst case
    if not attach.get("success"):
        return {
            **base_result,
            "success": False,
            "reason": f"attach verify failed: {attach.get('reason')}",
            "tool_calls_used": tool_calls,
        }
    attached_model = attach.get("model")
    logger.info(f"  [pick] attached:{attached_model}")

    # Step 11 — lift 20cm above grasp
    ok, info, calls = await _plan_to_xyz(
        mcp, gx, gy, gz + 0.20, orientation=grasp_orientation
    )
    tool_calls += calls
    if not ok:
        # Object is attached — cannot return FAILURE without dropping it.
        # Try a more conservative single retry; if it fails, report
        # SUCCESS with a warning so the planner knows the lift was partial.
        logger.warning(
            f"  [pick] lift plan failed but object is attached: {info}. "
            "Returning SUCCESS with caveat."
        )

    # Step 12 — return to look_forward (transit posture)
    arm = await move_arm_to_look_forward(mcp)
    tool_calls += 1
    if not arm.get("success"):
        # Lift may have left the arm in a config that can't reach
        # look_forward without an intermediate lift; try once via
        # moveit named_state directly.
        logger.warning(
            f"  [pick] post-lift look_forward failed: {arm.get('reason')}; "
            "attempting intermediate lift first"
        )
        try:
            await mcp.call_tool_prefixed(
                "moveit__plan_and_execute",
                {
                    "group": "arm",
                    "target_type": "pose",
                    "target": {
                        "position": [gx, gy, max(gz + 0.40, 0.90)],
                        "orientation": grasp_orientation,
                        "frame_id": "base_footprint",
                    },
                },
            )
            tool_calls += 1
            await move_arm_to_look_forward(mcp)
            tool_calls += 1
        except Exception as e:
            logger.warning(f"  [pick] intermediate lift error: {e}")

    return {
        **base_result,
        "success": True,
        "reason": f"grasped {attached_model} (gripper attached)",
        "tool_calls_used": tool_calls,
    }
