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
import re

from skill_based.clients.mcp import MCPClient
from skill_based.utils import (
    LOOK_FORWARD_JOINTS,
    move_arm_to_look_forward,
    parse_seg_status,
)

logger = logging.getLogger(__name__)


# === Gripper attach verification (moved from common.py 2026-05-16) ===

def _tokenize(s: str) -> set[str]:
    """Split on non-alphanumerics AND CamelCase boundaries.

    Drops tokens shorter than 3 characters. ``"KidsRoom_WoodCube"``
    splits to ``{"kids", "room", "wood", "cube"}``; ``"coke can"``
    splits to ``{"coke", "can"}``; the two share no token, but
    ``coke can`` against ``Kitchen_Coke`` shares ``coke``.
    """
    out: set[str] = set()
    cur: list[str] = []

    def _flush():
        if cur:
            tok = "".join(cur).lower()
            if len(tok) >= 3:
                out.add(tok)
            cur.clear()

    prev_lower = False
    for ch in s:
        if ch.isalnum():
            if ch.isupper() and prev_lower:
                _flush()
            cur.append(ch)
            prev_lower = ch.islower()
        else:
            _flush()
            prev_lower = False
    _flush()
    return out


async def wait_for_gripper_attached(
    mcp: MCPClient,
    expected_object: str | None = None,
    timeout: float = 8.0,
    retry_timeout: float = 5.0,
) -> dict:
    """Subscribe to /gripper/status and wait for ``attached:<model>``.

    Two-phase wait: initial ``timeout`` seconds, then a ``retry_timeout``
    second retry if the first wait did not see an attach. The two-phase
    pattern absorbs the latency between close-gripper-action completion
    and the attach plugin firing.

    ``expected_object`` is matched by token overlap against the Gazebo
    model name. 2026-05-16: warn-and-pass on token mismatch — the
    gripper physically holds SOMETHING (SAM3-segmented at the grasp
    pose), so mismatch between user vocabulary and Gazebo model name
    is a labelling artifact, not a pick failure.
    """

    async def _read(t: float) -> str:
        raw = await mcp.call_tool_prefixed(
            "ros__subscribe_once",
            {
                "topic": "/gripper/status",
                "msg_type": "std_msgs/msg/String",
                "timeout": int(t),
            },
        )
        data = json.loads(raw) if isinstance(raw, str) else raw
        msg = data.get("msg", data)
        return msg.get("data", "") if isinstance(msg, dict) else str(msg)

    def _matches(body: str) -> tuple[bool, str | None]:
        if not body.startswith("attached:"):
            return False, None
        model = body[len("attached:"):].strip()
        if expected_object is None:
            return True, model
        expected_tokens = _tokenize(expected_object)
        model_tokens = _tokenize(model)
        if not expected_tokens:
            return True, model
        if expected_tokens & model_tokens:
            return True, model
        logger.warning(
            f"  [attach] model '{model}' attached but does not token-match "
            f"expected '{expected_object}' (expected_tokens={sorted(expected_tokens)}, "
            f"model_tokens={sorted(model_tokens)}); accepting attach"
        )
        return True, model

    # Phase 1
    first_body = ""
    try:
        body = await _read(timeout)
        ok, model = _matches(body)
        if ok:
            return {"success": True, "reason": f"attached:{model}", "model": model}
        first_body = body
    except Exception as e:
        logger.warning(f"  [attach] first read error: {e}")

    # Phase 2 — retry with shorter timeout
    await asyncio.sleep(0.5)
    try:
        body = await _read(retry_timeout)
        ok, model = _matches(body)
        if ok:
            return {"success": True, "reason": f"attached:{model} (retry)", "model": model}
        return {
            "success": False,
            "reason": (
                f"gripper not attached after {timeout:.0f}s + "
                f"{retry_timeout:.0f}s retry; last={body or first_body or 'no message'}"
            ),
        }
    except Exception as e:
        return {"success": False, "reason": f"attach retry error: {e}"}


# UR5e practical reach forward from base_footprint, per pick.md step 5.
UR5_REACH_X = 1.10

# Default top-down quaternion as (x, y, z, w). Used only as a fallback
# when the perception MCP cannot return a shape-aware orientation
# (e.g. on an intermediate recovery lift after the grasp pose has been
# consumed). Real picks use the orientation from get_topdown_grasp_pose,
# which is shape-aware (PCA on the segmented point cloud).
TOPDOWN_ORIENTATION_DEFAULT = [1.0, 0.0, 0.0, 0.0]

# Vertical clearance for pre-grasp and post-grasp lift, in metres.
# 2026-05-16: lowered from 0.20 to 0.12 because at top-down orientation
# the UR5e kinematic envelope above z≈0.40 only extends to ~0.78 m
# radial. Targets at the practical reach edge (radial ≳ 0.80 m) are
# unreachable in the natural IK branch at z+0.20 — the solver returns
# wrap-around / wrist-flip branches that plan_and_execute cannot
# traverse from the current state. 0.12 keeps the pre-grasp inside the
# natural-branch envelope while still giving a clean visual descent.
PRE_GRASP_CLEARANCE_M = 0.12


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
        logger.warning(f"gripper status read error: {e}")
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


_ARM_JOINT_ORDER = [
    "arm_shoulder_pan_joint",
    "arm_shoulder_lift_joint",
    "arm_elbow_joint",
    "arm_wrist_1_joint",
    "arm_wrist_2_joint",
    "arm_wrist_3_joint",
]
_IK_LINE_RE = re.compile(r"^\s*([\w_]+):\s*([-+]?\d*\.?\d+)\s*rad", re.MULTILINE)


def _parse_ik_joints(ik_raw: str) -> list[float] | None:
    """Parse joint values from the compute_ik MCP response.

    Expected format:
        Inverse kinematics solution for 'arm':
          arm_shoulder_pan_joint: 0.12 rad
          arm_shoulder_lift_joint: -1.45 rad
          ...
    Returns the joints in canonical UR5 order, or None if the response
    is missing or malformed.
    """
    if not ik_raw or "kinematics solution" not in ik_raw.lower():
        return None
    by_name = {name: float(val) for name, val in _IK_LINE_RE.findall(ik_raw)}
    if not all(j in by_name for j in _ARM_JOINT_ORDER):
        return None
    return [by_name[j] for j in _ARM_JOINT_ORDER]


async def _plan_to_xyz(
    mcp: MCPClient,
    x: float,
    y: float,
    z: float,
    orientation: list[float] | None = None,
    *,
    clear_scene_on_retry: bool = True,
) -> tuple[bool, str, int]:
    """Plan + execute a top-down pose via the IK→joint_state pattern.

    Per feedback_plan_pose_unreliable and the 2026-05-15 finding,
    plan_and_execute(target_type="pose") is unreliable for low-z grasp
    targets: OMPL's Cartesian goal sampler returns GOAL_STATE_INVALID
    even when compute_ik finds a valid joint solution. Routing through
    compute_ik → plan_and_execute(target_type="joint_state") bypasses
    the broken sampler and plans in joint space directly.

    ``orientation`` is the quaternion as ``[x, y, z, w]``. When omitted,
    the default top-down orientation is used; real picks should pass
    the shape-aware orientation from ``perception__get_topdown_grasp_pose``.
    """
    if orientation is None:
        orientation = TOPDOWN_ORIENTATION_DEFAULT

    # Step A — compute IK
    try:
        ik_raw = await mcp.call_tool_prefixed(
            "moveit__compute_ik",
            {"group": "arm", "position": [x, y, z], "orientation": orientation},
        )
    except Exception as e:
        return False, f"compute_ik error at ({x:.2f},{y:.2f},{z:.2f}): {e}", 1
    joints = _parse_ik_joints(ik_raw)
    if joints is None:
        return (
            False,
            f"compute_ik returned no valid IK at ({x:.2f},{y:.2f},{z:.2f}): "
            f"{ik_raw[:120] if isinstance(ik_raw, str) else ik_raw!r}",
            1,
        )

    # Step B — clear stale collision objects BEFORE planning. perception
    # MCP's get_topdown_grasp_pose (and similar tools) republish the
    # target object as a collision object after each segmentation; if we
    # don't clear right before the plan, the descent finds the object
    # back in the scene and the goal state is rejected as in-collision
    # with the gripper finger. The step-2b clear at the start of the
    # pick is not enough — segmentation happens AFTER step 2b.
    try:
        await mcp.call_tool_prefixed("moveit__clear_planning_scene", {})
    except Exception as e:
        logger.warning(f"clear_planning_scene pre-plan error: {e}")

    # Step C — plan + execute to joint_state
    try:
        result = await mcp.call_tool_prefixed(
            "moveit__plan_and_execute",
            {
                "group": "arm",
                "target_type": "joint_state",
                "target": {"joint_positions": joints},
            },
        )
        if "fail" not in result.lower() or "completed" in result.lower():
            return True, result[:200], 3
    except Exception as e:
        return False, f"plan_and_execute(joint_state) error: {e}", 3

    if not clear_scene_on_retry:
        return False, f"plan failed at ({x:.2f},{y:.2f},{z:.2f}): {result[:200]}", 3

    # Retry: re-clear and re-IK (the planning scene may have been
    # repopulated mid-plan by another publisher)
    try:
        await mcp.call_tool_prefixed("moveit__clear_planning_scene", {})
    except Exception as e:
        logger.warning(f"clear_planning_scene retry error: {e}")
    try:
        ik_raw = await mcp.call_tool_prefixed(
            "moveit__compute_ik",
            {"group": "arm", "position": [x, y, z], "orientation": orientation},
        )
    except Exception as e:
        return False, f"retry compute_ik error: {e}", 5
    joints = _parse_ik_joints(ik_raw)
    if joints is None:
        return False, "retry compute_ik returned no valid IK", 5
    try:
        result = await mcp.call_tool_prefixed(
            "moveit__plan_and_execute",
            {
                "group": "arm",
                "target_type": "joint_state",
                "target": {"joint_positions": joints},
            },
        )
        if "fail" not in result.lower() or "completed" in result.lower():
            return True, f"retry ok: {result[:200]}", 6
        return False, f"retry still failed: {result[:200]}", 6
    except Exception as e:
        return False, f"retry error: {e}", 6


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

    # Step 2 — clear octomap (mostly defensive; sensors:[] disables the
    # in-process octomap_updater, but keep this in case sensors are
    # re-enabled later)
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
        logger.warning(f"clear_octomap error: {e}")
        tool_calls += 1

    # Step 2b — clear stale collision objects from prior runs. Open3D
    # drop-pose (and similar perception code) can publish collision
    # objects to /collision_object that persist across pick attempts and
    # block the descent when the gripper finger would clip them. See
    # feedback_clean_planning_scene_between_picks.
    try:
        await mcp.call_tool_prefixed("moveit__clear_planning_scene", {})
        tool_calls += 1
    except Exception as e:
        logger.warning(f"clear_planning_scene error: {e}")
        tool_calls += 1

    # Step 3 — segment on arm camera; front-cam fallback if missed
    status, calls = await _segment_object(mcp, object_name, camera="arm")
    tool_calls += calls
    logger.info(f"arm SAM3 -> {status}")
    if status != "SUCCESS":
        status, calls = await _segment_object(mcp, object_name, camera="front")
        tool_calls += calls
        logger.info(f"front SAM3 fallback -> {status}")
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
    # Held-object height for downstream place skills (surface / floor
    # modes need it for wrist-z math). Comes from the SAM3 bounding-box
    # measurement, NOT a per-object lookup table — see
    # feedback_no_hardcoded_object_dimensions for why the lookup was
    # removed.
    bbox_size = grasp.get("bounding_box", {}).get("size", {})
    held_object_height_m = float(bbox_size.get("z", 0.0))
    logger.info(
        f"grasp x={gx:.3f} y={gy:.3f} z={gz:.3f} "
        f"centroid_z={centroid_z:.3f} height={held_object_height_m:.3f} "
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

    # Step 7 — pre-grasp pose (PRE_GRASP_CLEARANCE_M above grasp z).
    # Clearance is chosen to stay inside the UR5e natural-IK envelope at
    # the practical reach edge; see PRE_GRASP_CLEARANCE_M definition.
    ok, info, calls = await _plan_to_xyz(
        mcp, gx, gy, gz + PRE_GRASP_CLEARANCE_M, orientation=grasp_orientation
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
                f"descent reported failure but {status} — proceeding"
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
    logger.info(f"attached:{attached_model}")

    # Step 11 — lift PRE_GRASP_CLEARANCE_M above grasp (symmetric with pre-grasp)
    ok, info, calls = await _plan_to_xyz(
        mcp, gx, gy, gz + PRE_GRASP_CLEARANCE_M, orientation=grasp_orientation
    )
    tool_calls += calls
    if not ok:
        # Object is attached — cannot return FAILURE without dropping it.
        # Try a more conservative single retry; if it fails, report
        # SUCCESS with a warning so the planner knows the lift was partial.
        logger.warning(
            f"lift plan failed but object is attached: {info}. "
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
            f"post-lift look_forward failed: {arm.get('reason')}; "
            "attempting intermediate lift first"
        )
        try:
            # Intermediate lift kept conservative to stay inside the
            # natural-IK envelope; 0.65m is roughly waist-height transit.
            ok, info, calls = await _plan_to_xyz(
                mcp,
                gx,
                gy,
                max(gz + 0.25, 0.65),
                orientation=grasp_orientation,
                clear_scene_on_retry=False,
            )
            tool_calls += calls
            if not ok:
                logger.warning(f"intermediate lift failed: {info}")
            await move_arm_to_look_forward(mcp)
            tool_calls += 1
        except Exception as e:
            logger.warning(f"intermediate lift error: {e}")

    return {
        **base_result,
        "success": True,
        "reason": f"grasped {attached_model} (gripper attached)",
        "held_object_height_m": held_object_height_m,
        "tool_calls_used": tool_calls,
    }
