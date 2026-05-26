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
        if "fail" not in result.lower() or "completed" in result.lower():
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
        if "fail" not in result.lower() or "completed" in result.lower():
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
