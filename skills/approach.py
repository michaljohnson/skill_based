"""Approach skill — deterministic Python find-and-approach primitive.

The skill takes a named destination, an optional target object, and a
mode that selects standoff distance. It tucks the arm, drives the base
to the entry pose for the destination, waits for the base to settle,
and (when ``target_object`` is given) refines the approach via
``approach_target`` plus a fall-back ``spin_search``.

The four-phase contract (coarse drive → area settle → target search →
fine approach to standoff) is the reason this skill is named ``approach``
rather than ``navigate``: it does substantially more than pose-to-pose
navigation, and ``approach`` reads naturally alongside ``pick`` and
``place`` in the planner's tool list.

The planner LLM never sees the underlying MCP tools; it only sees this
skill's signature and the structured success/failure result.
"""

import asyncio
import logging

from skill_based.clients.mcp import MCPClient
from skill_based.skills.common import (
    STANDOFF_BY_MODE,
    approach_target,
    geometric_fallback_prompts,
    move_arm_to_look_forward,
    parse_seg_status,
    spin_search,
    wait_until_still,
)

logger = logging.getLogger(__name__)


# === Named-area entry poses ===
#
# Map-frame (x, y, yaw) measured in Gazebo on the small_house_test world.
# Replace with your own coordinates for a different environment. Keys are
# lower-cased; the ``run`` lookup normalises whitespace and case so the
# planner can pass natural phrasing ("Kids Room", "kids_room", "kids room").

NAMED_AREA_POSES: dict[str, dict[str, float]] = {
    "living room": {"x": 1.21, "y": -0.63, "yaw": 1.53},
    "living room couch": {"x": 1.21, "y": -0.63, "yaw": 1.53},
    "living room tv": {"x": 1.02, "y": -1.12, "yaw": -1.55},
    "living room shoe rack": {"x": 2.84, "y": -3.87, "yaw": 0.0},
    "bedroom": {"x": -4.14, "y": 0.38, "yaw": 3.01},
    "kids room": {"x": -3.52, "y": -3.28, "yaw": -3.13},
    "kitchen": {"x": 5.14, "y": -0.86, "yaw": -0.52},
    "dining": {"x": 5.15, "y": -0.86, "yaw": 0.80},
    "dining area": {"x": 5.15, "y": -0.86, "yaw": 0.80},
}


def _resolve_destination(name: str) -> dict[str, float] | None:
    key = " ".join(name.lower().replace("_", " ").split())
    return NAMED_AREA_POSES.get(key)


async def run(
    mcp: MCPClient,
    destination: str,
    mode: str,
    target_object: str | None = None,
) -> dict:
    """Drive the robot to ``destination``, optionally approaching ``target_object``.

    Args:
        mcp: shared MCP client.
        destination: named area key (must be in ``NAMED_AREA_POSES``).
        mode: one of ``pick``, ``surface_place``, ``container_place``,
            ``floor_place``; selects the standoff for the optional
            approach refinement.
        target_object: optional surface or object to approach within the
            destination. When given, the skill segments it on the front
            camera, drives to standoff, and falls back to spin-search if
            the first segmentation misses.

    Returns:
        ``{"success": bool, "reason": str, "tool_calls_used": int}``
    """
    tool_calls = 0

    if mode not in STANDOFF_BY_MODE:
        return {
            "success": False,
            "reason": f"unknown mode: {mode}",
            "tool_calls_used": tool_calls,
        }

    pose = _resolve_destination(destination)
    if pose is None:
        return {
            "success": False,
            "reason": (
                f"unknown destination '{destination}'; known: "
                f"{sorted(NAMED_AREA_POSES.keys())}"
            ),
            "tool_calls_used": tool_calls,
        }

    standoff_m = STANDOFF_BY_MODE[mode]
    logger.info(
        f"approach -> dest='{destination}' mode={mode} "
        f"standoff={standoff_m:.2f}m target='{target_object}'"
    )

    # Step 1 — Tuck arm. A low arm reads as an obstacle in the front
    # costmap and prevents the base from planning a path. Mandatory
    # before driving.
    arm_reset = await move_arm_to_look_forward(mcp)
    tool_calls += 1
    if not arm_reset.get("success"):
        logger.warning(
            f"  [nav] arm reset failed: {arm_reset.get('reason')}; continuing"
        )

    # Step 2 — Drive to entry pose.
    NAV_WALL_TIMEOUT = 90.0
    try:
        await asyncio.wait_for(
            mcp.call_tool_prefixed(
                "nav2__navigate_to_pose",
                {"x": pose["x"], "y": pose["y"], "yaw": pose["yaw"]},
            ),
            timeout=NAV_WALL_TIMEOUT,
        )
        tool_calls += 1
    except asyncio.TimeoutError:
        # nav2 may have arrived but the result message didn't propagate
        # in time (a known nav2 + bridged-action quirk); check caller-side
        # via the approach step instead of bailing. The wall-timeout is
        # not authoritative for arrival outcome.
        logger.warning(
            f"  [nav] navigate_to_pose wall-timeout after {NAV_WALL_TIMEOUT:.0f}s; "
            f"checking outcome via approach"
        )
        tool_calls += 1
    except Exception as e:
        # One retry after clear_costmaps for transient failures
        logger.warning(f"  [nav] first navigate_to_pose error: {e}; clear_costmaps + retry")
        try:
            await mcp.call_tool_prefixed("nav2__clear_costmaps", {})
            tool_calls += 1
        except Exception as e2:
            logger.warning(f"  [nav] clear_costmaps error: {e2}")
        try:
            await asyncio.wait_for(
                mcp.call_tool_prefixed(
                    "nav2__navigate_to_pose",
                    {"x": pose["x"], "y": pose["y"], "yaw": pose["yaw"]},
                ),
                timeout=NAV_WALL_TIMEOUT,
            )
            tool_calls += 1
        except Exception as e2:
            return {
                "success": False,
                "reason": f"navigate_to_pose failed twice: {e}; retry: {e2}",
                "tool_calls_used": tool_calls,
            }

    # Step 3 — Settle on /odom.
    await wait_until_still(mcp, timeout=4.0)

    # Step 4 — If no target_object, navigation succeeds at entry pose.
    if not target_object:
        return {
            "success": True,
            "reason": (
                f"arrived at '{destination}' entry pose "
                f"({pose['x']:.2f},{pose['y']:.2f},{pose['yaw']:.2f})"
            ),
            "tool_calls_used": tool_calls,
        }

    # Step 5 — Verify target visibility (front cam at standoff). Try the
    # literal target first, then geometric fallback prompts before falling
    # back to spin-search.
    prompts = geometric_fallback_prompts(target_object)
    status = "ERROR"
    for prompt in prompts:
        try:
            seg_raw = await mcp.call_tool_prefixed(
                "perception__segment_objects",
                {"prompt": prompt, "camera": "front", "timeout": 20},
            )
            tool_calls += 1
            status = parse_seg_status(seg_raw)
            logger.info(f"  [nav] front-cam SAM3 '{prompt}' -> {status}")
        except Exception as e:
            status = "ERROR"
            logger.error(f"  [nav] front-cam SAM3 error: {e}")
            tool_calls += 1
        if status == "SUCCESS":
            break

    if status != "SUCCESS":
        # Spin-search to bring target into view (also tries fallback prompts per spin)
        spin = await spin_search(mcp, target_object, max_spins=6, camera="front")
        tool_calls += spin.get("tool_calls_used", 0)
        if not spin.get("success"):
            return {
                "success": False,
                "reason": (
                    f"target '{target_object}' not visible after spin-search "
                    f"at '{destination}'"
                ),
                "tool_calls_used": tool_calls,
            }

    # Step 6 — Drive to standoff distance from segmented target.
    approach = await approach_target(mcp, target_object, standoff_m=standoff_m)
    tool_calls += approach.get("tool_calls_used", 0)
    if not approach.get("success"):
        # Approach failed but the target was visible; report partial
        # success with a clear reason so the planner can decide.
        return {
            "success": False,
            "reason": (
                f"target '{target_object}' visible but approach to "
                f"{standoff_m:.2f}m failed: {approach.get('reason')}"
            ),
            "tool_calls_used": tool_calls,
        }

    # Step 7 — Final stillness wait so downstream perception sees a
    # stationary scene.
    await wait_until_still(mcp, timeout=3.0)

    return {
        "success": True,
        "reason": (
            f"approached '{target_object}' in '{destination}' "
            f"to {standoff_m:.2f}m standoff: {approach.get('reason')}"
        ),
        "tool_calls_used": tool_calls,
    }
