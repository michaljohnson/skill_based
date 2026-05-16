"""Approach skill — deterministic Python find-and-approach primitive.

The skill takes a named target_area, an optional target object, and a
``next_action`` flag that selects standoff distance. It tucks the arm,
drives the base to the entry pose for the target_area, waits for the
base to settle, and (when ``object_name`` is given) refines the
approach via ``approach_target`` plus a fall-back ``spin_search``.

The four-phase contract (coarse drive → area settle → target search →
fine approach to standoff) is the reason this skill is named ``approach``
rather than ``navigate``: it does substantially more than pose-to-pose
navigation, and ``approach`` reads naturally alongside ``pick`` and
``place`` in the planner's tool list.

The planner LLM never sees the underlying MCP tools; it only sees this
skill's signature and the structured success/failure result.
"""

import asyncio
import json
import logging
import math

from skill_based.clients.mcp import MCPClient
from skill_based.utils import (
    geometric_fallback_prompts,
    move_arm_to_look_forward,
    parse_seg_status,
    wait_until_still,
)

logger = logging.getLogger(__name__)


# === Standoff distance by next_action ===
#
# The right standoff depends on what the manipulation step does once
# the approach skill hands off:
#   - pick: UR5 reaches forward at low z (~0.40m); 0.85m centroid
#     distance leaves comfortable headroom for grasp pose math.
#   - surface_place: wrist must be HIGH (surface_z + 0.31m for can on
#     coffee table = 0.66m). UR5 top-down reach at z=0.66m caps near
#     x=0.55m, so the approach skill must deliver closer (~0.45m).
#   - container_place: drop INTO the bin from above; wrist sits 35cm
#     above rim. Same UR5 high-z constraints apply but rim is usually
#     at moderate height; 0.65m gives margin.
#   - floor_place: similar to pick — soft set-down at low z.

STANDOFF_BY_NEXT_ACTION = {
    "pick": 0.85,
    "surface_place": 0.45,
    "container_place": 0.65,
    "floor_place": 0.85,
}


# === Approach helpers (moved from common.py 2026-05-16) ===

def _parse_robot_pose(raw) -> tuple[float | None, float | None, float | None]:
    """Extract (x, y, yaw) from nav2__get_robot_pose response."""
    if not isinstance(raw, str):
        raw = str(raw)
    try:
        d = json.loads(raw)
        body = d.get("result", d)
        if isinstance(body, str):
            body = json.loads(body)
        pos = body["position"]
        ori = body["orientation"]
        return float(pos["x"]), float(pos["y"]), float(ori["yaw"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None, None, None


async def approach_target(
    mcp: MCPClient,
    target_object: str,
    standoff_m: float = 0.85,
) -> dict:
    """Drive to ``standoff_m`` from the segmented target.

    Reads the cached segmentation centroid via perception MCP, then
    dispatches to the shared ``nav2__approach_target`` primitive. The
    primitive owns all geometry + motion (spin to face + drive_on_heading);
    this wrapper only bridges perception (centroid lookup) to nav2.

    All three architectures (LLM + tools, multi-agent, skill-based) share
    this primitive to ensure they implement approach identically.

    Returns ``{"success": bool, "reason": str, "tool_calls_used": int}``.
    """
    tool_calls = 0

    # 1. Read cached centroid from perception
    try:
        grasp_raw = await mcp.call_tool_prefixed(
            "perception__get_topdown_grasp_pose",
            {"object_name": target_object},
        )
        tool_calls += 1
        grasp = json.loads(grasp_raw) if isinstance(grasp_raw, str) else grasp_raw
        target_x_base = float(grasp["centroid_base_frame"]["x"])
        target_y_base = float(grasp["centroid_base_frame"]["y"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
        return {
            "success": False,
            "reason": f"failed to read cached centroid: {e}",
            "tool_calls_used": tool_calls,
        }

    target_dist = math.hypot(target_x_base, target_y_base)
    logger.info(
        f"target_base=({target_x_base:.2f},{target_y_base:.2f}) "
        f"dist={target_dist:.2f}m standoff={standoff_m:.2f}m -> nav2__approach_target"
    )

    # 2. Delegate to the MCP primitive
    try:
        result_raw = await asyncio.wait_for(
            mcp.call_tool_prefixed(
                "nav2__approach_target",
                {
                    "target_x_base": target_x_base,
                    "target_y_base": target_y_base,
                    "standoff_m": standoff_m,
                },
            ),
            timeout=60.0,
        )
        tool_calls += 1
    except asyncio.TimeoutError:
        return {
            "success": False,
            "reason": "nav2__approach_target wall-timeout after 60s",
            "tool_calls_used": tool_calls,
        }
    except Exception as e:
        return {
            "success": False,
            "reason": f"nav2__approach_target error: {e}",
            "tool_calls_used": tool_calls,
        }

    text = result_raw if isinstance(result_raw, str) else str(result_raw)
    if "error" in text.lower():
        return {
            "success": False,
            "reason": text,
            "tool_calls_used": tool_calls,
        }

    # Settle: nav2 reports complete before robot fully decelerates AND
    # before camera buffers flush from the new pose.
    await asyncio.sleep(1.5)

    return {
        "success": True,
        "reason": text,
        "tool_calls_used": tool_calls,
    }


async def spin_search(
    mcp: MCPClient,
    target_object: str,
    max_spins: int = 8,
    spin_angle: float = 1.047,  # ~60 deg
    camera: str = "front",
) -> dict:
    """Spin in place searching for ``target_object``.

    Spins ``spin_angle`` radians at a time and calls SAM3 segmentation
    on the requested camera after each spin, trying the literal target
    plus geometric fallback prompts. Returns SUCCESS as soon as any
    prompt anchors the target.
    """
    tool_calls = 0
    prompts = geometric_fallback_prompts(target_object)
    for i in range(max_spins):
        try:
            await mcp.call_tool_prefixed(
                "nav2__spin_robot", {"angle": spin_angle}
            )
            tool_calls += 1
            logger.info(
                f"  [spin-search {i+1}/{max_spins}] spun {spin_angle:.2f}rad"
            )
        except Exception as e:
            logger.error(f"  [spin-search] spin failed: {e}")
            tool_calls += 1
            continue

        await wait_until_still(mcp)

        anchored = False
        for prompt in prompts:
            try:
                seg_raw = await mcp.call_tool_prefixed(
                    "perception__segment_objects",
                    {"prompt": prompt, "camera": camera, "timeout": 20},
                )
                tool_calls += 1
                status = parse_seg_status(seg_raw)
                logger.info(
                    f"  [spin-search {i+1}/{max_spins}] SAM3 {camera} '{prompt}' -> {status}"
                )
            except Exception as e:
                logger.error(f"  [spin-search] segmentation failed: {e}")
                tool_calls += 1
                continue
            if status == "SUCCESS":
                anchored = True
                break
        if anchored:
            return {
                "success": True,
                "reason": (
                    f"found '{target_object}' after {i+1} spin(s) via prompt '{prompt}'"
                ),
                "tool_calls_used": tool_calls,
            }

    return {
        "success": False,
        "reason": f"'{target_object}' not found after {max_spins} spins",
        "tool_calls_used": tool_calls,
    }


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


def _resolve_target_area(name: str) -> dict[str, float] | None:
    key = " ".join(name.lower().replace("_", " ").split())
    return NAMED_AREA_POSES.get(key)


async def run(
    mcp: MCPClient,
    target_area: str,
    next_action: str,
    object_name: str,
) -> dict:
    """Drive the robot to ``target_area`` and approach ``object_name``.

    Args:
        mcp: shared MCP client.
        target_area: named area key (must be in ``NAMED_AREA_POSES``).
        next_action: one of ``pick``, ``surface_place``, ``container_place``,
            ``floor_place``; declares what the planner intends to do
            immediately after this skill returns. Selects the standoff
            for the approach refinement.
        object_name: surface or object to approach within the
            target_area. The skill segments it on the front camera, drives
            to standoff, and falls back to spin-search if the first
            segmentation misses. The skill is named ``approach`` because
            it always approaches a specific named target. For pure
            relocation (return-to-home, exploration) add a separate
            skill rather than overloading this one with optional args.

    Returns:
        ``{"success": bool, "reason": str, "tool_calls_used": int}``
    """
    if not object_name:
        raise ValueError("object_name must be a non-empty string")

    tool_calls = 0

    if next_action not in STANDOFF_BY_NEXT_ACTION:
        return {
            "success": False,
            "reason": f"unknown next_action: {next_action}",
            "tool_calls_used": tool_calls,
        }

    pose = _resolve_target_area(target_area)
    if pose is None:
        return {
            "success": False,
            "reason": (
                f"unknown target_area '{target_area}'; known: "
                f"{sorted(NAMED_AREA_POSES.keys())}"
            ),
            "tool_calls_used": tool_calls,
        }

    standoff_m = STANDOFF_BY_NEXT_ACTION[next_action]
    logger.info(
        f"approach -> dest='{target_area}' next_action={next_action} "
        f"standoff={standoff_m:.2f}m target='{object_name}'"
    )

    # Step 1 — Tuck arm. A low arm reads as an obstacle in the front
    # costmap and prevents the base from planning a path. Mandatory
    # before driving.
    arm_reset = await move_arm_to_look_forward(mcp)
    tool_calls += 1
    if not arm_reset.get("success"):
        logger.warning(
            f"arm reset failed: {arm_reset.get('reason')}; continuing"
        )

    # Step 2 — Drive to entry pose.
    # Wall-timeout sized for slow Gazebo RTF; matches multi_agent's
    # 180s budget so cross-architecture comparisons aren't biased by
    # timeout differences.
    NAV_WALL_TIMEOUT = 180.0
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
            f"navigate_to_pose wall-timeout after {NAV_WALL_TIMEOUT:.0f}s; "
            f"checking outcome via approach"
        )
        tool_calls += 1
    except Exception as e:
        # One retry after clear_costmaps for transient failures
        logger.warning(f"first navigate_to_pose error: {e}; clear_costmaps + retry")
        try:
            await mcp.call_tool_prefixed("nav2__clear_costmaps", {})
            tool_calls += 1
        except Exception as e2:
            logger.warning(f"clear_costmaps error: {e2}")
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

    # Step 4 — Verify target visibility (front cam at standoff). Try the
    # literal target first, then geometric fallback prompts before falling
    # back to spin-search.
    prompts = geometric_fallback_prompts(object_name)
    status = "ERROR"
    for prompt in prompts:
        try:
            seg_raw = await mcp.call_tool_prefixed(
                "perception__segment_objects",
                {"prompt": prompt, "camera": "front", "timeout": 20},
            )
            tool_calls += 1
            status = parse_seg_status(seg_raw)
            logger.info(f"front-cam SAM3 '{prompt}' -> {status}")
        except Exception as e:
            status = "ERROR"
            logger.error(f"front-cam SAM3 error: {e}")
            tool_calls += 1
        if status == "SUCCESS":
            break

    if status != "SUCCESS":
        # Spin-search to bring target into view (also tries fallback prompts per spin)
        spin = await spin_search(mcp, object_name, max_spins=6, camera="front")
        tool_calls += spin.get("tool_calls_used", 0)
        if not spin.get("success"):
            return {
                "success": False,
                "reason": (
                    f"target '{object_name}' not visible after spin-search "
                    f"at '{target_area}'"
                ),
                "tool_calls_used": tool_calls,
            }

    # Step 5 — Drive to standoff distance from segmented target.
    approach = await approach_target(mcp, object_name, standoff_m=standoff_m)
    tool_calls += approach.get("tool_calls_used", 0)
    if not approach.get("success"):
        # Approach failed but the target was visible; report partial
        # success with a clear reason so the planner can decide.
        return {
            "success": False,
            "reason": (
                f"target '{object_name}' visible but approach to "
                f"{standoff_m:.2f}m failed: {approach.get('reason')}"
            ),
            "tool_calls_used": tool_calls,
        }

    # Step 6 — Final stillness wait so downstream perception sees a
    # stationary scene.
    await wait_until_still(mcp, timeout=3.0)

    return {
        "success": True,
        "reason": (
            f"approached '{object_name}' in '{target_area}' "
            f"to {standoff_m:.2f}m standoff: {approach.get('reason')}"
        ),
        "tool_calls_used": tool_calls,
    }
