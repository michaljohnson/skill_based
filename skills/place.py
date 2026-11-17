"""Place skill: deterministic release primitive.

The planner picks the mode with the ``mode`` argument:

  container  drop INTO a bin or basket: flange 35 cm above the segmented rim,
             the held object falls in. Adapted to the real robot.
  surface    put ON a table or shelf.  Not adapted to the real robot yet.
  floor      put NEXT TO an object on the floor.  Not adapted yet.

Surface and floor need the held object's height, which pick returns as
``held_object_height_m``; container mode ignores it. Their simulation
version is in git history (branch ``summit``) as a template.
"""

import asyncio
import json
import logging
from typing import Literal

from skill_based.clients.mcp import MCPClient
from skill_based.utils import (
    GRIPPER_OPEN,
    HOLD_MAX,
    HOLD_MIN,
    TOPDOWN_ORIENTATION as _TOPDOWN,
    geometric_fallback_prompts,
    gripper_command,
    gripper_position,
    move_arm_to_look_forward,
    parse_seg_status,
    plan_to_xyz,
    reset_scene,
)

logger = logging.getLogger(__name__)


Mode = Literal["container", "surface", "floor"]
_VALID_MODES: tuple[str, ...] = ("container", "surface", "floor")


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
        logger.warning(f"segment_objects({camera}) error: {e}")
        return "ERROR", 1



async def _verify_object_visible_from_above(
    mcp: MCPClient, object_name: str
) -> tuple[bool, str, int]:
    """Verify the released object is at the drop pose by looking from
    above with the arm camera.

    Replaces the older front-cam "object no longer visible" gate, which
    per [[feedback_place_visibility_not_containment]] was too easy to
    spoof: a cube that fell BEHIND the robot was also "not visible from
    front cam" and silently passed the gate. The arm camera at the
    post-release lift pose is directly above the drop point with the
    EE in top-down orientation, so its view is narrow and targeted at
    the drop location.

    Logic (same for container AND surface modes):
      - arm cam segments the object   -> SUCCESS (cube is at drop pose,
        either inside the container's opening or on the surface)
      - arm cam returns NO_OBJECTS_FOUND -> FAILURE (cube fell off the
        surface, missed the container, or is occluded by some
        geometry the place pose didn't anticipate)

    PRECONDITION: caller must invoke this BEFORE moving the arm to
    look_forward — at look_forward the wrist camera points sideways,
    not down.
    """
    # Detach lag (Gazebo DynamicGripperAttach): the released object can ride
    # up with the gripper on the post-release lift and settle onto the surface
    # only a few seconds later. Wait, verify, retry once — so a delayed but
    # correct drop is not false-negatived. (Demo-prep robustness; out of the
    # frozen evaluation matrix.)
    calls = 0
    status = "ERROR"
    for attempt in range(2):
        await asyncio.sleep(3.0)
        status, c = await _segment(mcp, object_name, camera="arm")
        calls += c
        logger.info(
            f"  [place-verify {attempt+1}/2] arm SAM3 '{object_name}' -> {status}"
        )
        if status == "SUCCESS":
            return (
                True,
                f"'{object_name}' visible on arm camera at drop pose "
                f"(settled, attempt {attempt+1})",
                calls,
            )
    return (
        False,
        f"'{object_name}' NOT visible on arm camera at drop pose after "
        f"settle+retry (seg status: {status}) — likely missed target / fell elsewhere",
        calls,
    )


# Container mode on the real robot. Flange this far above the rim: the held
# object's bottom then clears the rim by ~10 cm (pads ~6 cm above its bottom).
CONTAINER_RIM_CLEARANCE_M = 0.35
# Reach to the bin centre from base_footprint; a drop at 0.95 m planned with the
# elbow bent, at 1.08 m it was nearly straight.
CONTAINER_REACH_REAL_M = 1.00
# Footprint of the bin's collision box; keeps the arm off the rim on the way.
CONTAINER_BOX_M = 0.40


async def _segment_any(
    mcp: MCPClient, prompts: list[str], cameras: tuple[str, ...]
) -> tuple[str | None, int]:
    """First (camera, prompt) that segments; returns (prompt, calls)."""
    calls = 0
    for camera in cameras:
        for prompt in prompts:
            status, c = await _segment(mcp, prompt, camera=camera)
            calls += c
            logger.info(f"{camera} SAM3 '{prompt}' -> {status}")
            if status == "SUCCESS":
                return prompt, calls
    return None, calls


async def _run_container(mcp: MCPClient, target_location: str, object_name: str) -> dict:
    """Drop the held object into a container in front of the robot.

    The bin's centre comes from the grasp tool's near-face rule on the bin
    cloud (the placing tool's point mean is pulled metres away by stray depth
    points on a dark bin); its rim from the highest bin points. The bin goes
    into the planning scene, the flange goes straight to rim + clearance, the
    gripper opens and the arm camera checks the object from above.
    """
    tool_calls = 0
    base = {"target_location": target_location}

    def fail(reason: str) -> dict:
        return {**base, "success": False, "reason": reason, "tool_calls_used": tool_calls}

    logger.info(f"place -> container '{target_location}' object='{object_name}'")

    finger = await gripper_position(mcp)
    tool_calls += 1
    if finger is None or not HOLD_MIN <= finger <= HOLD_MAX:
        return fail(f"nothing held (finger_joint {finger}): pick first")

    arm = await move_arm_to_look_forward(mcp)
    tool_calls += 1
    if not arm.get("success"):
        return fail(f"arm reset failed: {arm.get('reason')}")
    tool_calls += 2
    try:
        await reset_scene(mcp)
    except Exception as e:
        return fail(f"planning scene reset failed, no floor in the scene: {e}")

    prompt, calls = await _segment_any(
        mcp, geometric_fallback_prompts(target_location), ("arm", "front")
    )
    tool_calls += calls
    if prompt is None:
        return fail(f"'{target_location}' not segmented on either camera")

    try:
        raw = await mcp.call_tool_prefixed(
            "perception__get_topdown_grasp_pose", {"object_name": prompt}
        )
        data = json.loads(raw) if isinstance(raw, str) else raw
        cx = float(data["grasp_pose"]["position"]["x"])
        cy = float(data["grasp_pose"]["position"]["y"])
        rim = float(data["bbox_base_frame"]["z_max"])
    except Exception as e:
        return fail(f"no container pose from the segmented cloud: {e}")
    tool_calls += 1
    drop_z = rim + CONTAINER_RIM_CLEARANCE_M
    dist = (cx ** 2 + cy ** 2) ** 0.5
    logger.info(f"container centre=({cx:.3f},{cy:.3f}) rim={rim:.3f} drop_z={drop_z:.3f} dist={dist:.2f}")
    if dist > CONTAINER_REACH_REAL_M:
        return fail(
            f"container at {dist:.2f} m > {CONTAINER_REACH_REAL_M:.2f} m. "
            "Navigator must redeliver closer."
        )

    try:
        reply = await mcp.call_tool_prefixed(
            "moveit__add_collision_box",
            {
                "object_id": "container",
                "position": [cx, cy, rim / 2],
                "dimensions": [CONTAINER_BOX_M, CONTAINER_BOX_M, rim],
                "frame_id": "base_footprint",
            },
        )
        if not str(reply).startswith("Successfully"):
            raise RuntimeError(str(reply)[:200])
    except Exception as e:
        await move_arm_to_look_forward(mcp)
        return fail(f"container collision box not added: {e}")
    tool_calls += 1

    ok, info, calls = await plan_to_xyz(mcp, cx, cy, drop_z, _TOPDOWN, reset_on_retry=False)
    tool_calls += calls
    if not ok:
        await move_arm_to_look_forward(mcp)
        tool_calls += 1
        return fail(f"drop pose plan failed: {info}")

    try:
        opened = await gripper_command(mcp, GRIPPER_OPEN)
    except Exception as e:
        return fail(f"open_gripper error: {e} (arm above the container, object maybe still held)")
    tool_calls += 1
    if not opened["position"] <= HOLD_MIN:  # also fails on a missing position (NaN)
        return fail(f"gripper did not open (position {opened['position']:.3f}); arm above the container")

    visible, vinfo, calls = await _verify_object_visible_from_above(mcp, object_name)
    tool_calls += calls
    logger.info(f"arm-cam verify -> {vinfo}")

    arm = await move_arm_to_look_forward(mcp)
    tool_calls += 1
    if not arm.get("success"):
        logger.warning(f"post-release look_forward failed: {arm.get('reason')}")
    if not visible:
        return fail(f"placement verify failed: {vinfo}")
    return {
        **base,
        "success": True,
        "reason": f"dropped into {target_location} at ({cx:.2f},{cy:.2f},{drop_z:.2f}); {vinfo}",
        "tool_calls_used": tool_calls,
    }


async def run(
    mcp: MCPClient,
    target_location: str,
    object_name: str,
    mode: Mode,
    object_height_m: float,
) -> dict:
    """Release the held object at ``target_location`` per ``mode``.

    Preconditions: robot is holding an object (gripper attached) and is
    positioned within working distance of the target. The approach skill
    owns positioning; place is a pure manipulation primitive.

    Args:
        mcp: shared MCP client.
        target_location: name of the reference object to segment.
            - mode="container": the container itself (e.g. "trash bin").
            - mode="surface":   the surface (e.g. "wooden coffee table").
            - mode="floor":     a REFERENCE object already on the floor
                                  (e.g. "white shoe"); the held object is
                                  dropped NEXT TO it at floor height.
        object_name: name of the held object. Required for the
            post-release arm-cam verify segmentation prompt.
        mode: "container" | "surface" | "floor". Selected by the planner
            from task context.
        object_height_m: height of the held object in metres, measured
            during pick (bounding_box.size.z from the grasp pose). Required
            by surface and floor modes for wrist-z math; ignored by
            container mode (it falls THROUGH the opening).

    Returns:
        ``{"success": bool, "reason": str, "target_location": str, "tool_calls_used": int}``
    """
    if not object_name:
        raise ValueError("object_name must be a non-empty string")
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
    if mode == "container":
        return await _run_container(mcp, target_location, object_name)
    # surface and floor still use the simulation's release checks
    return {
        "target_location": target_location,
        "success": False,
        "reason": f"mode={mode!r} is not adapted to the real robot yet (only 'container')",
        "tool_calls_used": 0,
    }
