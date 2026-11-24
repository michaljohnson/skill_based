"""Place skill: deterministic release primitive.

The planner picks the mode with the ``mode`` argument:

  container  drop INTO a bin or basket: flange 35 cm above the segmented rim,
             the held object falls in.
  surface    put ON a table or box: the object's bottom ends a few cm above
             the segmented top, then the gripper opens.
  floor      put NEXT TO a reference object on the floor, between it and the
             robot; the floor height is known (z = 0), so the object is set down.

Surface and floor need the held object's height, which pick returns as
``held_object_height_m``; container mode ignores it.

Every result says whether the gripper still holds the object (``holding``:
true, false, or null when unknown); a failure adds its kind (``failure``) and
the log lines of the call (``steps``).
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
    SEG_ANSWERS,
    TOPDOWN_ORIENTATION as _TOPDOWN,
    collect_steps,
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
        status = parse_seg_status(raw)
        if status == "UNKNOWN":
            logger.warning(f"segment_objects({camera}) '{prompt}': {raw}")
            # SAM3 found it, only the point cloud is missing: enough to know it is there
            if "reported SUCCESS" in str(raw):
                status = "SUCCESS_NO_CLOUD"
        return status, 1
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
        if status in ("SUCCESS", "SUCCESS_NO_CLOUD"):
            return (
                True,
                f"'{object_name}' visible on arm camera above the drop point "
                f"(settled, attempt {attempt+1})",
                calls,
            )
    if status != "NO_OBJECTS_FOUND":
        return (
            False,
            f"could not verify '{object_name}': segmentation error ({status}); "
            "the drop itself may have worked",
            calls,
        )
    return (
        False,
        f"'{object_name}' NOT visible on arm camera above the drop point after "
        f"settle+retry (seg status: {status}) — likely missed target / fell elsewhere",
        calls,
    )


# Container mode on the real robot. Flange this far above the rim: the held
# object's bottom then clears the rim by ~10 cm (pads ~6 cm above its bottom).
CONTAINER_RIM_CLEARANCE_M = 0.35
# Reach to the drop point from base_footprint; a drop at 0.95 m planned with the
# elbow bent, at 1.08 m it was nearly straight.
CONTAINER_REACH_REAL_M = 1.00
# Footprint of the bin's collision box; keeps the arm off the rim on the way.
CONTAINER_BOX_M = 0.40

# Surface mode. Flange to the middle of the finger pads (perception's GRIPPER_FINGER_OFFSET_M).
PAD_OFFSET_M = 0.188
# Object bottom this far above the top when the gripper opens. The top is perception's top_z
# (99th percentile of the heights; without it the highest point, often cm high). Pick measures how far
# the pads sit above the object's bottom; without that the object is counted as hanging
# fully below the pads, with more clearance.
SURFACE_CLEARANCE_M = 0.005
SURFACE_CLEARANCE_UNMEASURED_M = 0.03
# Wrist flange to the 2F-140 fingertips (perception's _FINGERTIP_OFFSET_M); after a deep grasp they
# reach below the held object, so they, not the object, decide how low the flange may go.
FINGERTIP_OFFSET_M = 0.225
FINGERTIP_CLEARANCE_M = 0.01
# The flange goes this far above the drop pose first, then straight down.
SURFACE_APPROACH_M = 0.08
# A top lower than this is the floor (floor mode); higher is out of top-down reach.
SURFACE_TOP_MIN_M = 0.05
SURFACE_TOP_MAX_M = 0.80
# A surface whose middle is out of reach gets the object this far behind its near edge.
SURFACE_EDGE_INSET_M = 0.15

# Floor mode: the object goes this far in front of the reference's near face, towards the
# robot (never behind it, where the arm would reach over the reference): the open 2F-140
# fingers spread ~7 cm to each side, plus half a small object and a margin.
FLOOR_GAP_M = 0.13
# Closer than this the arm would have to reach in front of its own base.
FLOOR_REACH_MIN_M = 0.50
# Surface and floor: straight up this far after the release, so the open fingers clear the
# object and the arm camera is far enough above it to segment it (at the drop pose it is too close).
VERIFY_LIFT_M = 0.20
# A target segmented farther than this is another object of the same kind (the shelf object, 4.3 m).
IMPLAUSIBLE_DIST_M = 2.0


async def _segment_any(
    mcp: MCPClient, prompts: list[str], cameras: tuple[str, ...]
) -> tuple[str | None, int, list[str]]:
    """First (camera, prompt) that segments; returns (prompt, calls, statuses seen)."""
    calls = 0
    statuses = []
    for camera in cameras:
        for prompt in prompts:
            status, c = await _segment(mcp, prompt, camera=camera)
            calls += c
            statuses.append(f"{camera}: {status}")
            logger.info(f"{camera} SAM3 '{prompt}' -> {status}")
            if status == "SUCCESS":
                return prompt, calls, statuses
    return None, calls, statuses


async def _run_drop(
    mcp: MCPClient,
    target_location: str,
    object_name: str,
    mode: Mode,
    object_height_m: float,
    pads_above_bottom_m: float | None = None,
) -> dict:
    """Release the held object into a container, onto a surface or onto the floor.

    Every mode segments the target, measures it with the grasp tool (near-face
    centre, top from the highest points), adds it to the planning scene, moves
    the flange to the drop point, opens the gripper and checks the object from
    above with the arm camera. They differ only in the drop point and the box.
    """
    with collect_steps(logger) as steps:
        return await _drop(mcp, target_location, object_name, mode, object_height_m, pads_above_bottom_m, steps)


async def _drop(
    mcp: MCPClient,
    target_location: str,
    object_name: str,
    mode: Mode,
    object_height_m: float,
    pads_above_bottom_m: float | None,
    steps: list[str],
) -> dict:
    tool_calls = 0
    base = {"target_location": target_location}
    unmoved = f"Arm at look_forward, still holding '{object_name}'."

    def fail(failure: str, reason: str, holding: bool | None = True) -> dict:
        return {**base, "success": False, "failure": failure, "reason": reason, "holding": holding,
                "tool_calls_used": tool_calls, "steps": list(steps)}

    logger.info(f"place -> {mode} '{target_location}' object='{object_name}' obj_h={object_height_m:.3f}")
    if mode in ("surface", "floor") and not object_height_m > 0:
        return fail("invalid_args", f"{mode} mode needs the held object's height: pass pick's "
                                    "held_object_height_m. Nothing moved.")

    finger = await gripper_position(mcp)
    tool_calls += 1
    if finger is None or not HOLD_MIN <= finger <= HOLD_MAX:
        return fail("not_holding", f"Nothing held (finger_joint {finger}): pick first. Nothing moved.",
                    holding=False if finger is not None else None)

    arm = await move_arm_to_look_forward(mcp)
    tool_calls += 1
    if not arm.get("success"):
        return fail("arm_failed", f"Moving the arm to look_forward failed ({arm.get('reason')}), still "
                                  f"holding '{object_name}'. Usually the UR program is not running.")
    tool_calls += 2
    try:
        await reset_scene(mcp)
    except Exception as e:
        return fail("scene_failed", f"Planning scene reset failed ({e}); without the floor box the arm "
                                    f"is not planned safely. {unmoved}")

    prompt, calls, looks = await _segment_any(
        mcp, geometric_fallback_prompts(target_location), ("arm", "front")
    )
    tool_calls += calls
    if prompt is None:
        if all(look.split(": ", 1)[1] in SEG_ANSWERS for look in looks):
            return fail("target_not_found", (
                f"'{target_location}' not found by the arm or the front camera from here. {unmoved} "
                f"Approach it first (next_action {mode}_place), or it is not here."))
        return fail("perception_error", (
            f"Segmentation of '{target_location}' did not work ({', '.join(looks)}); SUCCESS_NO_CLOUD means "
            f"SAM3 found it but the camera sent no point cloud (camera hung). {unmoved}"))

    try:
        raw = await mcp.call_tool_prefixed(
            "perception__get_topdown_grasp_pose", {"object_name": prompt}
        )
        data = json.loads(raw) if isinstance(raw, str) else raw
        cx = float(data["grasp_pose"]["position"]["x"])
        cy = float(data["grasp_pose"]["position"]["y"])
        top = float(data["bbox_base_frame"]["z_max"])
        if mode == "surface" and data.get("top_z") is not None:
            # perception's robust top (99th percentile); z_max is one point, often cm high.
            # Container and floor keep z_max: a thin rim or a box that is only an obstacle.
            top = float(data["top_z"])
        if mode in ("surface", "floor"):
            bbox = {k: float(data["bbox_base_frame"][k]) for k in ("x_min", "x_max", "y_min", "y_max")}
    except Exception as e:
        return fail("perception_error", f"No {mode} pose from the segmented cloud ({e}). {unmoved}")
    tool_calls += 1

    def set_down_z(support: float) -> float:
        """Flange z that puts the held object's bottom just above ``support``."""
        if pads_above_bottom_m is not None and 0 < pads_above_bottom_m <= object_height_m:
            z = support + SURFACE_CLEARANCE_M + pads_above_bottom_m + PAD_OFFSET_M
        else:
            z = support + SURFACE_CLEARANCE_UNMEASURED_M + object_height_m + PAD_OFFSET_M
        return max(z, support + FINGERTIP_CLEARANCE_M + FINGERTIP_OFFSET_M)

    if mode == "container":
        drop_z = top + CONTAINER_RIM_CLEARANCE_M
        box_pos = [cx, cy, top / 2]
        box_dim = [CONTAINER_BOX_M, CONTAINER_BOX_M, top]
    elif mode == "floor":
        ref = (cx ** 2 + cy ** 2) ** 0.5
        # Near face: the reference's bbox point closest to the robot.
        near = (bbox["x_min"] ** 2 + min(max(0.0, bbox["y_min"]), bbox["y_max"]) ** 2) ** 0.5
        offset = max(0.0, ref - near) + FLOOR_GAP_M
        if ref <= offset:
            return fail("too_close", f"'{target_location}' at {ref:.2f} m is too close to put anything in "
                                     f"front of it. {unmoved} Back up.")
        cx, cy = cx * (1 - offset / ref), cy * (1 - offset / ref)
        drop_z = set_down_z(0.0)
        box_pos = [(bbox["x_min"] + bbox["x_max"]) / 2, (bbox["y_min"] + bbox["y_max"]) / 2, top / 2]
        box_dim = [bbox["x_max"] - bbox["x_min"], bbox["y_max"] - bbox["y_min"], top]
    else:
        if not SURFACE_TOP_MIN_M <= top <= SURFACE_TOP_MAX_M:
            far = (cx ** 2 + cy ** 2) ** 0.5
            if far > IMPLAUSIBLE_DIST_M:
                return fail("implausible_target", (
                    f"Segmented a '{target_location}' {far:.1f} m away with its top at {top:.2f} m: almost "
                    f"certainly another object, not the target. {unmoved} The target is probably not in view."))
            return fail("invalid_target", (
                f"'{target_location}' top at {top:.2f} m is outside {SURFACE_TOP_MIN_M:.2f}-"
                f"{SURFACE_TOP_MAX_M:.2f} m (lower: use floor mode; higher: out of top-down reach). {unmoved}"))
        if (cx ** 2 + cy ** 2) ** 0.5 > CONTAINER_REACH_REAL_M:
            cx = min(cx, bbox["x_min"] + SURFACE_EDGE_INSET_M)
        drop_z = set_down_z(top)
        box_pos = [(bbox["x_min"] + bbox["x_max"]) / 2, (bbox["y_min"] + bbox["y_max"]) / 2, top / 2]
        box_dim = [bbox["x_max"] - bbox["x_min"], bbox["y_max"] - bbox["y_min"], top]

    dist = (cx ** 2 + cy ** 2) ** 0.5
    logger.info(f"{mode} drop=({cx:.3f},{cy:.3f}) top={top:.3f} drop_z={drop_z:.3f} dist={dist:.2f}")
    if dist > CONTAINER_REACH_REAL_M:
        if dist > IMPLAUSIBLE_DIST_M:
            return fail("implausible_target", (
                f"Segmented a '{target_location}' {dist:.1f} m away: probably another object, not the target. "
                f"{unmoved} The target is probably not in view."))
        return fail("out_of_reach", (
            f"'{target_location}' drop point {dist:.2f} m away, beyond the arm's {CONTAINER_REACH_REAL_M:.2f} m. "
            f"{unmoved} Approach it again (next_action {mode}_place)."))
    if mode == "floor" and dist < FLOOR_REACH_MIN_M:
        return fail("too_close", (
            f"Floor drop point {dist:.2f} m away, closer than {FLOOR_REACH_MIN_M:.2f} m to the robot. {unmoved} "
            "Back up."))

    box_id = "reference" if mode == "floor" else mode  # "floor" is the planning scene's floor box
    try:
        reply = await mcp.call_tool_prefixed(
            "moveit__add_collision_box",
            {"object_id": box_id, "position": box_pos, "dimensions": box_dim, "frame_id": "base_footprint"},
        )
        if not str(reply).startswith("Successfully"):
            raise RuntimeError(str(reply)[:200])
    except Exception as e:
        await move_arm_to_look_forward(mcp)
        return fail("scene_failed", f"The {box_id} collision box was not added ({e}). {unmoved}")
    tool_calls += 1

    stops = [drop_z] if mode == "container" else [drop_z + SURFACE_APPROACH_M, drop_z]
    for z in stops:
        ok, info, calls = await plan_to_xyz(mcp, cx, cy, z, _TOPDOWN, reset_on_retry=False)
        tool_calls += calls
        if not ok:
            await move_arm_to_look_forward(mcp)
            tool_calls += 1
            return fail("motion_failed", (
                f"Moving to the drop pose ({cx:.2f}, {cy:.2f}, {z:.2f}) m failed: {info.rstrip('.')}. {unmoved}"))

    try:
        opened = await gripper_command(mcp, GRIPPER_OPEN)
    except Exception as e:
        return fail("gripper_failed", (
            f"Opening the gripper failed ({e}); '{object_name}' may still be held. Arm above the {mode}."),
            holding=None)
    tool_calls += 1
    if not opened["position"] <= HOLD_MIN:  # also fails on a missing position (NaN)
        return fail("gripper_failed", (
            f"The gripper did not open (position {opened['position']:.3f}), still holding '{object_name}'. "
            f"Arm above the {mode}."))

    if mode != "container":
        ok, info, calls = await plan_to_xyz(mcp, cx, cy, drop_z + VERIFY_LIFT_M, _TOPDOWN, reset_on_retry=False)
        tool_calls += calls
        if not ok:
            logger.warning(f"lift after release failed: {info}; verifying from the drop pose")

    visible, vinfo, calls = await _verify_object_visible_from_above(mcp, object_name)
    tool_calls += calls
    logger.info(f"arm-cam verify -> {vinfo}")

    arm = await move_arm_to_look_forward(mcp)
    tool_calls += 1
    if not arm.get("success"):
        logger.warning(f"post-release look_forward failed: {arm.get('reason')}")
    arm_at = "Arm at look_forward." if arm.get("success") else f"Arm above the {mode} (look_forward failed)."
    if not visible:
        failure = "verify_failed" if "NOT visible" in vinfo else "verify_unavailable"
        return fail(failure, f"Released '{object_name}' at ({cx:.2f}, {cy:.2f}, {drop_z:.2f}) m, then: {vinfo}. "
                             f"{arm_at}", holding=False)
    where = {"container": "into", "surface": "onto", "floor": "next to"}[mode]
    return {
        **base,
        "success": True,
        "reason": f"Placed '{object_name}' {where} '{target_location}' at ({cx:.2f}, {cy:.2f}, {drop_z:.2f}) m; "
                  f"{vinfo}. {arm_at}",
        "holding": False,
        "tool_calls_used": tool_calls,
    }


async def run(
    mcp: MCPClient,
    target_location: str,
    object_name: str,
    mode: Mode,
    object_height_m: float,
    pads_above_bottom_m: float | None = None,
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
            during pick (pick's ``held_object_height_m``). Required by
            surface mode for the drop height; ignored by container mode
            (the object falls through the opening).
        pads_above_bottom_m: how far the finger pads sit above the held
            object's bottom (pick's ``pads_above_bottom_m``). With it, surface
            mode sets the object down 0.5 cm above the measured top; without it, it
            assumes the worst case and drops from a few cm higher.

    Returns:
        ``{"success": bool, "reason": str, "target_location": str, "tool_calls_used": int}``
    """
    if not object_name:
        raise ValueError("object_name must be a non-empty string")
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
    return await _run_drop(mcp, target_location, object_name, mode, object_height_m, pads_above_bottom_m)
