"""Shared low-level helpers used by multiple deterministic skills.

This module is intentionally invisible in the operator log — each
function returns a structured result (or raises) and does no internal
logging. Callers (the skills in ``skill_based/skills/``) log through
their own per-skill loggers so the trace reads as ``[PICK] / [PLACE] /
[APPROACH]`` only, never ``[UTILS]``.

Contents are limited to helpers needed by 2+ skills. Single-caller
utilities live in the calling skill's own file.
"""

import asyncio
import json
import re
import time
from typing import Any

from skill_based.clients.mcp import MCPClient


# Canonical look_forward joint positions. Used by every skill that
# needs to reset the arm to the transit / sensing posture.
#
# Use joint_state PRIMARY (not named_state). In Gazebo,
# plan_and_execute(named_state="look_forward") sometimes reports
# planning success while the physical arm has not moved (state
# divergence between MoveIt's perceived state and the actual robot).
# Explicit numeric joint targets eliminate this silent-success failure.
LOOK_FORWARD_JOINTS = [-0.0001, -0.2429, -2.8291, -0.7983, 1.5622, 0.0]

# Floor under the robot as a collision box, so no plan goes through it.
FLOOR_BOX = {
    "object_id": "floor",
    "position": [0.0, 0.0, -0.055],
    "dimensions": [4.0, 4.0, 0.1],
    "frame_id": "base_footprint",
}

# Robotiq 2F-140 GripperCommand positions (finger_joint). On the robot: open
# 0.003-0.009, closed empty 0.69-0.70, stalled on objects 0.35-0.44.
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.7
HOLD_MIN = 0.05
HOLD_MAX = 0.68


def motion_ok(result: str) -> bool:
    """True only when the moveit MCP reports a finished, successful motion.

    A refused call ("still running", the stop_arm latch) carries no operation
    and must not pass as success.
    """
    return "execution completed successfully" in (result or "").lower()


async def reset_scene(mcp: MCPClient) -> None:
    """Clear the planning scene, then put the floor back.

    Raises RuntimeError when either call fails: the moveit MCP reports
    errors as text, so without this check a lost floor would go unnoticed.
    """
    for tool, args in (("moveit__clear_planning_scene", {}), ("moveit__add_collision_box", FLOOR_BOX)):
        reply = await mcp.call_tool_prefixed(tool, args)
        if not str(reply).startswith("Successfully"):
            raise RuntimeError(f"{tool}: {str(reply)[:200]}")


async def gripper_command(mcp: MCPClient, position: float, max_effort: float = 50.0) -> dict:
    """Send a GripperCommand goal and return its result fields.

    Returns ``{"status", "position", "stalled", "reached_goal"}``; a stalled
    close ends with status 6 (aborted), which is the normal case on an object.
    """
    raw = await mcp.call_tool_prefixed(
        "ros__send_action_goal",
        {
            "action_name": "/robotiq_gripper_controller/gripper_cmd",
            "action_type": "control_msgs/action/GripperCommand",
            "goal": {"command": {"position": position, "max_effort": max_effort}},
            "timeout": 15,
        },
    )
    data = json.loads(raw) if isinstance(raw, str) else raw
    result = data.get("result") or {}
    return {
        "status": data.get("status"),
        "position": float(result.get("position", float("nan"))),
        "stalled": bool(result.get("stalled", False)),
        "reached_goal": bool(result.get("reached_goal", False)),
    }


async def gripper_position(mcp: MCPClient, attempts: int = 5) -> float | None:
    """finger_joint from /joint_states, or None.

    /joint_states also carries the wheel messages, so a few reads may be
    needed until one with the gripper comes by.
    """
    for _ in range(attempts):
        raw = await mcp.call_tool_prefixed(
            "ros__subscribe_once",
            {"topic": "/joint_states", "msg_type": "sensor_msgs/msg/JointState", "timeout": 3},
        )
        data = json.loads(raw) if isinstance(raw, str) else raw
        msg = data.get("msg", data) if isinstance(data, dict) else {}
        names = msg.get("name", [])
        positions = msg.get("position", [])
        if "finger_joint" in names and len(positions) == len(names):
            return float(positions[names.index("finger_joint")])
    return None


def holding(gripper_result: dict) -> bool:
    """True if a close stalled on something between the pads."""
    pos = gripper_result.get("position", float("nan"))
    return bool(gripper_result.get("stalled")) and HOLD_MIN <= pos <= HOLD_MAX


def parse_seg_status(seg_raw: Any) -> str:
    """Extract ``status`` from a segment_objects response (str or dict).

    Returns ``"UNKNOWN"`` on parse failure so callers can branch on
    SUCCESS / NO_OBJECTS_FOUND / UNKNOWN.
    """
    if not isinstance(seg_raw, str):
        seg_raw = str(seg_raw)
    try:
        return json.loads(seg_raw).get("status", "UNKNOWN")
    except (json.JSONDecodeError, AttributeError):
        return "UNKNOWN"


def geometric_fallback_prompts(target: str) -> list[str]:
    """Return a try-in-order list of SAM3 prompts for ``target``.
    
    First entry is the literal target; subsequent entries are
    geometric / colour descriptors known to anchor when category names
    fail. SAM3's open-vocabulary detector is reliable on shape / colour
    descriptors but flaky on object category nouns; the fallback chain
    trades semantic precision for recall.
    """
    prompts = [target]
    t = target.lower()
    if "bin" in t or "trash" in t:
        prompts.append("tall rectangular container")
        prompts.append("brown rectangular container on the floor")
        prompts.append("wastebasket")
        prompts.append("brown bucket")
        prompts.append("dark opening on the floor")
    elif "table" in t or "counter" in t or "shelf" in t or "desk" in t:
        # "wooden surface" is NOT used as a fallback: SAM3 has no
        # distance-plausibility gate and the generic prompt latches onto
        # far wooden objects (kitchen counters through doorways, distant
        # wood floors), producing multi-metre bbox.x_min values that the
        # approach drive then refuses. Use class-specific alternatives
        # that anchor on the close intended target.
        prompts.append("wooden coffee table")
        prompts.append("coffee table")
        prompts.append("wooden table")
        prompts.append("low wooden table")
    elif "shoe rack" in t:
        prompts.append("red shoe on the floor")
    elif "cube" in t:
        prompts.append("white cube on the floor")
    elif "can" in t or "coke" in t:
        prompts.append("red can on the floor")
    return prompts


async def move_arm_to_look_forward(mcp: MCPClient) -> dict:
    """Reset the arm to the canonical look_forward configuration.

    Tries joint_state first (more reliable than named_state in Gazebo,
    where named_state planning can report success without moving the
    arm); falls back to named_state if joint_state planning fails.
    Returns ``{"success": bool, "reason": str, "raw"?: str}``. The
    caller is responsible for logging the outcome.
    """
    # Primary: explicit joint positions
    try:
        result = await mcp.call_tool_prefixed(
            "moveit__plan_and_execute",
            {
                "group": "arm",
                "target_type": "joint_state",
                "target": {"joint_positions": LOOK_FORWARD_JOINTS},
            },
        )
        if motion_ok(result):
            return {"success": True, "reason": "joint_state look_forward", "raw": result[:200]}
    except Exception as e:
        # Swallow and try fallback; caller sees outcome via the return dict.
        last_error = str(e)
    else:
        last_error = None

    # Fallback: named state
    try:
        result = await mcp.call_tool_prefixed(
            "moveit__plan_and_execute",
            {
                "group": "arm",
                "target_type": "named_state",
                "target": {"state_name": "look_forward"},
            },
        )
        if motion_ok(result):
            return {
                "success": True,
                "reason": "named_state look_forward (joint_state fallback)",
                "raw": result[:200],
            }
        reason = f"named_state failed: {result[:200]}"
        if last_error:
            reason = f"joint_state error: {last_error}; " + reason
        return {"success": False, "reason": reason}
    except Exception as e:
        reason = f"named_state error: {e}"
        if last_error:
            reason = f"joint_state error: {last_error}; " + reason
        return {"success": False, "reason": reason}


async def wait_until_still(
    mcp: MCPClient,
    timeout: float = 3.0,
    vel_threshold: float = 0.02,
    poll_s: float = 0.15,
    post_settle: float = 0.25,
) -> None:
    """Block until /odom reports the base has decelerated to rest.

    nav2 reports complete as soon as it stops *commanding*, but the
    velocity_smoother keeps decelerating for ~500 ms. Segmenting during
    that window catches the camera mid-motion. Polls /odom twist until
    |linear.x| and |angular.z| are under ``vel_threshold``, then waits
    ``post_settle`` for the camera ring buffer to flush.

    On timeout this returns silently after the post-settle delay; it
    does not raise. Callers that need to know the base was still
    should observe the wall-clock cost.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            raw = await mcp.call_tool_prefixed(
                "ros__subscribe_once",
                {
                    "topic": "/odom",
                    "msg_type": "nav_msgs/msg/Odometry",
                    "timeout": 1,
                },
            )
            data = json.loads(raw) if isinstance(raw, str) else raw
            msg = data.get("msg", data)
            tw = msg.get("twist", {}).get("twist", {})
            vx = abs(tw.get("linear", {}).get("x", 0.0))
            wz = abs(tw.get("angular", {}).get("z", 0.0))
            if vx < vel_threshold and wz < vel_threshold:
                await asyncio.sleep(post_settle)
                return
        except Exception:
            pass  # silent — utilities don't log
        await asyncio.sleep(poll_s)
    # Timeout: still wait the post-settle so caller has a consistent guarantee.
    await asyncio.sleep(post_settle)

# Top-down gripper orientation (x, y, z, w) when no other is given.
TOPDOWN_ORIENTATION = [1.0, 0.0, 0.0, 0.0]

ARM_JOINT_ORDER = [
    "arm_shoulder_pan_joint",
    "arm_shoulder_lift_joint",
    "arm_elbow_joint",
    "arm_wrist_1_joint",
    "arm_wrist_2_joint",
    "arm_wrist_3_joint",
]
JOINT_LINE_RE = re.compile(r"^\s*([\w_]+):\s*([-+]?\d*\.?\d+)\s*rad", re.MULTILINE)


def parse_joints(raw: str) -> list[float] | None:
    """Joint values in UR5 order from a compute_ik / get_current_joint_state answer."""
    if not raw:
        return None
    by_name = {name: float(val) for name, val in JOINT_LINE_RE.findall(raw)}
    if not all(j in by_name for j in ARM_JOINT_ORDER):
        return None
    return [by_name[j] for j in ARM_JOINT_ORDER]


async def ik_joints(mcp: MCPClient, x: float, y: float, z: float, orientation: list[float]) -> list[float] | None:
    raw = await mcp.call_tool_prefixed(
        "moveit__compute_ik",
        {"group": "arm", "position": [x, y, z], "orientation": orientation},
    )
    if not isinstance(raw, str) or "kinematics solution" not in raw.lower():
        return None
    return parse_joints(raw)


async def plan_to_xyz(
    mcp: MCPClient,
    x: float,
    y: float,
    z: float,
    orientation: list[float] | None = None,
    reset_on_retry: bool = True,
) -> tuple[bool, str, int]:
    """Plan + execute a top-down pose via the IK→joint_state pattern.

    plan_and_execute(target_type="pose") is unreliable for low-z grasp
    targets (OMPL's goal sampler rejects poses compute_ik solves), so the
    skill solves IK and plans in joint space. One retry, after a scene reset
    unless ``reset_on_retry`` is False (it would drop the caller's boxes).
    """
    if orientation is None:
        orientation = TOPDOWN_ORIENTATION
    calls = 0
    result = ""
    for attempt in range(2):
        calls += 1
        try:
            joints = await ik_joints(mcp, x, y, z, orientation)
        except Exception as e:
            return False, f"compute_ik error at ({x:.2f},{y:.2f},{z:.2f}): {e}", calls
        if joints is None:
            return False, f"no IK at ({x:.2f},{y:.2f},{z:.2f})", calls
        calls += 1
        try:
            result = await mcp.call_tool_prefixed(
                "moveit__plan_and_execute",
                {
                    "group": "arm",
                    "target_type": "joint_state",
                    "target": {"joint_positions": joints},
                },
            )
        except Exception as e:
            return False, f"plan_and_execute(joint_state) error: {e}", calls
        if motion_ok(result):
            return True, result[-200:], calls
        if attempt == 0 and reset_on_retry:
            calls += 2
            try:
                await reset_scene(mcp)
            except Exception as e:
                return False, f"scene reset before the retry failed: {e}", calls
    return False, f"plan failed at ({x:.2f},{y:.2f},{z:.2f}): {result[-200:]}", calls
