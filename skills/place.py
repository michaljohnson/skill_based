"""Place skill — deterministic Python release primitive.

Surface and container modes share the front-cam stage 1; only container
mode re-segments at stage 2 from the arm camera. Stage 2 is skipped for
surfaces because re-segmenting from the arm-cam look-down view biases
the centroid toward the near edge of large flat surfaces (the arm-cam
sees only the close strip), making the stage-1 front-cam centroid more
reliable for surface placement.

Floor placement is intentionally NOT supported in this skill; it
requires dynamic held-object-height measurement that is out of scope
for the current build.
"""

import asyncio
import json
import logging

from skill_based.clients.mcp import MCPClient
from skill_based.skills.common import (
    geometric_fallback_prompts,
    move_arm_to_look_forward,
    parse_seg_status,
    wait_until_still,
)

logger = logging.getLogger(__name__)


# Per-object held-object heights (metres). Used to compute the surface
# wrist_z. Container mode ignores object_height_m.
HELD_OBJECT_HEIGHT_M: dict[str, float] = {
    "coke can": 0.12,
    "coca cola can": 0.12,
    "red can": 0.12,
    "white cube": 0.05,
    "wood cube": 0.10,
    "shoe": 0.10,
    "red shoe": 0.10,
    "small ball": 0.06,
}
DEFAULT_OBJECT_HEIGHT_M = 0.10  # safe default for unknown small objects

# Reach budget for surface placements (UR5 top-down at high wrist z)
PLACE_REACH_XY = 0.70

# Top-down orientation in base_footprint (w, x, y, z)
TOPDOWN_ORIENTATION = [1.0, 0.0, 0.0, 0.0]

# Container vs surface mode keywords
_CONTAINER_KEYWORDS = (
    "bin", "trash", "basket", "bowl", "drainer", "wagon", "box",
)


def _looks_like_container(target: str) -> bool:
    t = target.lower()
    return any(kw in t for kw in _CONTAINER_KEYWORDS)


def _resolve_object_height(object_name: str | None) -> float:
    if not object_name:
        return DEFAULT_OBJECT_HEIGHT_M
    key = object_name.lower().strip()
    if key in HELD_OBJECT_HEIGHT_M:
        return HELD_OBJECT_HEIGHT_M[key]
    for k, v in HELD_OBJECT_HEIGHT_M.items():
        if k in key or key in k:
            return v
    return DEFAULT_OBJECT_HEIGHT_M


async def _segment(
    mcp: MCPClient, prompt: str, camera: str
) -> tuple[str, int]:
    try:
        raw = await mcp.call_tool_prefixed(
            "perception__segment_objects",
            {"prompt": prompt, "camera": camera, "timeout": 20},
        )
        return parse_seg_status(raw), 1
    except Exception as e:
        logger.warning(f"  [place] segment_objects({camera}) error: {e}")
        return "ERROR", 1


async def _placing_pose(
    mcp: MCPClient,
    target: str,
    *,
    surface: bool,
    object_height_m: float,
    pointcloud_topic: str,
) -> tuple[dict | None, int]:
    """Compute a placing pose. ``surface=True`` uses object-height math;
    ``surface=False`` (container) uses 35cm clearance and ignores object height."""
    args: dict = {
        "object_name": target,
        "pointcloud_topic": pointcloud_topic,
    }
    if surface:
        args.update({
            "top_clearance_m": 0.05,
            "object_height_m": object_height_m,
            "x_bias_m": 0.0,
        })
    else:
        args.update({
            "top_clearance_m": 0.35,
            "object_height_m": 0.0,
            "x_bias_m": 0.0,
        })
    try:
        raw = await mcp.call_tool_prefixed(
            "perception__get_topdown_placing_pose", args
        )
        data = json.loads(raw) if isinstance(raw, str) else raw
        if "place_pose" not in data and "position" not in data:
            logger.warning(f"  [place] placing pose missing position: {data}")
            return None, 1
        return data, 1
    except Exception as e:
        logger.warning(f"  [place] get_topdown_placing_pose error: {e}")
        return None, 1


async def _plan_to_xyz(
    mcp: MCPClient, x: float, y: float, z: float
) -> tuple[bool, str, int]:
    target = {
        "position": [x, y, z],
        "orientation": TOPDOWN_ORIENTATION,
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
        return False, f"plan error: {e}", 1
    # Retry once after clearing the planning scene
    try:
        await mcp.call_tool_prefixed("moveit__clear_planning_scene", {})
    except Exception as e:
        logger.warning(f"  [place] clear_planning_scene error: {e}")
    try:
        result = await mcp.call_tool_prefixed(
            "moveit__plan_and_execute",
            {"group": "arm", "target_type": "pose", "target": target},
        )
        if "fail" not in result.lower() or "completed" in result.lower():
            return True, f"retry ok: {result[:200]}", 3
        return False, f"retry failed: {result[:200]}", 3
    except Exception as e:
        return False, f"retry error: {e}", 3


def _extract_xyz(data: dict) -> tuple[float, float, float]:
    """Pull (x, y, z) out of a placing-pose response.

    The perception MCP returns positions as ``{"x": ..., "y": ..., "z": ...}``
    dicts wrapped under ``place_pose``. This helper accepts both the
    wrapped form and a bare position dict for resilience.
    """
    if "place_pose" in data:
        pose = data["place_pose"]
        pos = pose.get("position", pose)
    else:
        pos = data.get("position", data)
    if isinstance(pos, dict):
        return float(pos["x"]), float(pos["y"]), float(pos["z"])
    # Legacy list shape, kept for safety
    return float(pos[0]), float(pos[1]), float(pos[2])


async def _force_detach(mcp: MCPClient) -> int:
    try:
        await mcp.call_tool_prefixed(
            "ros__publish_once",
            {
                "topic": "/gripper/force_detach_str",
                "msg_type": "std_msgs/msg/String",
                "msg": {"data": "release"},
            },
        )
        return 1
    except Exception as e:
        logger.warning(f"  [place] force_detach error: {e}")
        return 1


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


async def _verify_detached(
    mcp: MCPClient, object_name: str | None, timeout: float = 4.0
) -> tuple[bool, str, int]:
    """Confirm /gripper/status reports detached after release."""
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
        body = msg.get("data", "") if isinstance(msg, dict) else str(msg)
        if body == "detached" or not body.startswith("attached:"):
            return True, body or "no message (assumed detached)", 1
        return False, body, 1
    except Exception as e:
        logger.warning(f"  [place] verify_detached error: {e}")
        return True, f"verify error treated as PASS: {e}", 1


async def _verify_object_no_longer_visible(
    mcp: MCPClient, object_name: str
) -> tuple[bool, str, int]:
    """For container placement: check the object is no longer visible on
    the front cam (i.e. it fell into the container)."""
    status, calls = await _segment(mcp, object_name, camera="front")
    if status == "SUCCESS":
        return (
            False,
            f"'{object_name}' still segmentable on front camera — likely missed container",
            calls,
        )
    return True, f"'{object_name}' no longer visible on front camera", calls


async def run(
    mcp: MCPClient,
    target_location: str,
    object_name: str,
) -> dict:
    """Release the held object onto/into ``target_location``.

    Preconditions: robot is holding an object (gripper attached) and is
    positioned within working distance of the target. The approach skill
    owns positioning; place is a pure manipulation primitive.

    Args:
        mcp: shared MCP client.
        target_location: name of the surface or container.
        object_name: name of the held object. Required because (a) the
            post-release visibility verify needs it to segment the right
            object on the front cam, and (b) the object-height lookup
            table in surface-place mode keys off it. Without ``object_name``
            both checks silently degrade, so the contract requires it.

    Returns:
        ``{"success": bool, "reason": str, "target_location": str, "tool_calls_used": int}``
    """
    if not object_name:
        raise ValueError("object_name must be a non-empty string")

    tool_calls = 0
    base_result = {"target_location": target_location}

    if "floor" in target_location.lower():
        return {
            **base_result,
            "success": False,
            "reason": (
                "floor placement not supported — requires dynamic object-height "
                "measurement (pick.bbox.size_z → place input). "
                "Use surface or container placement instead."
            ),
            "tool_calls_used": tool_calls,
        }

    is_container = _looks_like_container(target_location)
    object_height_m = _resolve_object_height(object_name) if not is_container else 0.0
    logger.info(
        f"place -> target='{target_location}' object='{object_name}' "
        f"mode={'container' if is_container else 'surface'} "
        f"obj_h={object_height_m:.2f}m"
    )

    # Step 1 — arm to look_forward (transit / sensing pose)
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
        logger.warning(f"  [place] clear_octomap error: {e}")
        tool_calls += 1

    # Pre-segment settle: navigator's approach may have just driven the
    # base; the front-cam ring buffer can hold up to ~100ms of stale
    # frames. Wait for the base to be still + an extra beat to flush.
    await wait_until_still(mcp, timeout=3.0, post_settle=0.5)

    # Step 3a — segmentation with shared fallback chain. Try the FRONT
    # camera first (its wider FOV usually catches surfaces and tall
    # containers cleanly). If every front-cam prompt misses, fall back
    # to the ARM camera at look_forward — the held object blocks the
    # centre of the frame but the target's upper portion is usually
    # visible above the gripper.
    fallbacks = geometric_fallback_prompts(target_location)
    seg_ok = False
    seg_camera = "front"
    last_status = "ERROR"
    for prompt in fallbacks:
        last_status, calls = await _segment(mcp, prompt, camera="front")
        tool_calls += calls
        logger.info(f"  [place] front SAM3 '{prompt}' -> {last_status}")
        if last_status == "SUCCESS":
            seg_ok = True
            break
    if not seg_ok:
        for prompt in fallbacks:
            last_status, calls = await _segment(mcp, prompt, camera="arm")
            tool_calls += calls
            logger.info(f"  [place] arm SAM3 '{prompt}' -> {last_status}")
            if last_status == "SUCCESS":
                seg_ok = True
                seg_camera = "arm"
                break
    if not seg_ok:
        return {
            **base_result,
            "success": False,
            "reason": (
                f"both-camera segmentation failed for '{target_location}' "
                f"and fallbacks {fallbacks[1:]}"
            ),
            "tool_calls_used": tool_calls,
        }

    # Step 3b — coarse placing pose from whichever camera anchored
    pc_topic = (
        "/front/segmented_pointcloud" if seg_camera == "front"
        else "/segmented_pointcloud"
    )
    coarse, calls = await _placing_pose(
        mcp,
        target_location,
        surface=not is_container,
        object_height_m=object_height_m,
        pointcloud_topic=pc_topic,
    )
    tool_calls += calls
    if coarse is None:
        return {
            **base_result,
            "success": False,
            "reason": "stage-1 placing pose failed",
            "tool_calls_used": tool_calls,
        }
    cx, cy, cz = _extract_xyz(coarse)
    logger.info(f"  [place] stage-1 pose=({cx:.2f},{cy:.2f},{cz:.2f})")

    # Step 4 — reach check
    dist = (cx ** 2 + cy ** 2) ** 0.5
    if dist > PLACE_REACH_XY:
        return {
            **base_result,
            "success": False,
            "reason": (
                f"stage-1 xy distance {dist:.2f}m > {PLACE_REACH_XY:.2f}m. "
                "Navigator must redeliver closer."
            ),
            "tool_calls_used": tool_calls,
        }

    place_x, place_y, place_z = cx, cy, cz

    # Step 5+6 — stage 2 refinement (CONTAINER only; surface uses stage 1)
    if is_container:
        ok2, info2, calls = await _plan_to_xyz(mcp, cx, cy, cz + 0.30)
        tool_calls += calls
        if ok2:
            status, calls = await _segment(mcp, target_location, camera="arm")
            tool_calls += calls
            logger.info(f"  [place] stage-2 arm SAM3 -> {status}")
            if status == "SUCCESS":
                refined, calls = await _placing_pose(
                    mcp,
                    target_location,
                    surface=False,
                    object_height_m=0.0,
                    pointcloud_topic="/segmented_pointcloud",
                )
                tool_calls += calls
                if refined is not None:
                    rx, ry, rz = _extract_xyz(refined)
                    refined_dist = (rx ** 2 + ry ** 2) ** 0.5
                    logger.info(f"  [place] stage-2 refined=({rx:.2f},{ry:.2f},{rz:.2f})")
                    # Stage-2 can latch onto the bin's far rim, biasing the
                    # centroid forward. If the refined pose is past UR5
                    # reach (or notably farther than stage-1), prefer stage-1.
                    if refined_dist > PLACE_REACH_XY:
                        logger.warning(
                            f"  [place] stage-2 dist {refined_dist:.2f}m > "
                            f"reach {PLACE_REACH_XY:.2f}m; reverting to stage-1"
                        )
                    elif refined_dist > dist + 0.10:
                        logger.warning(
                            f"  [place] stage-2 pushed centroid +{refined_dist - dist:.2f}m "
                            "(likely far-rim bias); reverting to stage-1"
                        )
                    else:
                        place_x, place_y, place_z = rx, ry, rz
                else:
                    logger.info("  [place] stage-2 placing pose missed; falling back to stage-1")
            else:
                logger.info("  [place] stage-2 segmentation missed; falling back to stage-1")
        else:
            logger.warning(
                f"  [place] stage-2 overview pose failed: {info2}; using stage-1"
            )

    # Step 7 — pre-place above target
    ok, info, calls = await _plan_to_xyz(
        mcp, place_x, place_y, place_z + 0.15
    )
    tool_calls += calls
    if not ok:
        return {
            **base_result,
            "success": False,
            "reason": f"pre-place plan failed: {info}",
            "tool_calls_used": tool_calls,
        }

    # Step 8 — descend straight down to drop pose
    ok, info, calls = await _plan_to_xyz(mcp, place_x, place_y, place_z)
    tool_calls += calls
    if not ok:
        return {
            **base_result,
            "success": False,
            "reason": f"descent plan failed: {info}",
            "tool_calls_used": tool_calls,
        }

    # Step 9 — force-detach + open gripper
    tool_calls += await _force_detach(mcp)
    await asyncio.sleep(0.5)
    try:
        tool_calls += await _open_gripper(mcp)
    except Exception as e:
        logger.warning(f"  [place] open_gripper error: {e}")
        tool_calls += 1

    # Step 10 — verify gripper detached
    detached, body, calls = await _verify_detached(mcp, object_name)
    tool_calls += calls
    if not detached:
        return {
            **base_result,
            "success": False,
            "reason": f"gripper still {body} after release",
            "tool_calls_used": tool_calls,
        }
    logger.info(f"  [place] gripper status -> {body}")

    # Step 11 — lift clear before transit
    ok, info, calls = await _plan_to_xyz(
        mcp, place_x, place_y, place_z + 0.15
    )
    tool_calls += calls
    if not ok:
        logger.warning(f"  [place] lift-clear plan failed: {info}; continuing")

    # Step 12 — return to look_forward (transit posture)
    arm_back = await move_arm_to_look_forward(mcp)
    tool_calls += 1
    if not arm_back.get("success"):
        logger.warning(
            f"  [place] post-release look_forward failed: {arm_back.get('reason')}"
        )

    # Step 13 — for containers, post-release visibility check
    if is_container:
        visible_ok, vinfo, calls = await _verify_object_no_longer_visible(
            mcp, object_name
        )
        tool_calls += calls
        if not visible_ok:
            return {
                **base_result,
                "success": False,
                "reason": f"placement verify failed: {vinfo}",
                "tool_calls_used": tool_calls,
            }
        return {
            **base_result,
            "success": True,
            "reason": f"placed in {target_location}: {vinfo}",
            "tool_calls_used": tool_calls,
        }

    return {
        **base_result,
        "success": True,
        "reason": (
            f"placed on {target_location} at ({place_x:.2f},{place_y:.2f},{place_z:.2f})"
        ),
        "tool_calls_used": tool_calls,
    }
