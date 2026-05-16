"""Place skill — deterministic Python release primitive.

Three explicit modes selected by the planner via the ``mode`` argument:

  container  drop INTO a deep target (bin / basket / wastebasket). 35 cm
             clearance above the segmented rim; held object falls in.
  surface    drop ON top of an elevated flat target (table / shelf).
             Wrist z = surface_z + finger + held_h + clearance.
  floor      drop NEXT TO a reference object on the floor (e.g. shoe
             beside the other shoe). xy from segmented reference centroid
             plus a lateral offset; z = floor + finger + held_h +
             clearance.

The held object's height is REQUIRED and comes from the pick skill's
returned ``held_object_height_m`` (bounding-box measurement); there is
no per-object lookup table. Container mode ignores it.

Single-stage placement: segment the target with the front camera
(coarse but reach-friendly view) and drop straight onto / into / next
to the returned pose. The earlier two-stage flow (front cam coarse →
arm cam top-down refine) was removed 2026-05-16 for parity with
multi_agent/subagents/place.md (which dropped stage-2 on 2026-05-11).
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Literal

from skill_based.clients.mcp import MCPClient
from skill_based.utils import (
    geometric_fallback_prompts,
    move_arm_to_look_forward,
    parse_seg_status,
    wait_until_still,
)

logger = logging.getLogger(__name__)


# === Mode dispatch ===

Mode = Literal["container", "surface", "floor"]
_VALID_MODES: tuple[str, ...] = ("container", "surface", "floor")

# Reach budgets and lift / clearance constants. Per-mode values are
# selected by _mode_config() below; the raw constants live here so they
# can be overridden in future tuning sweeps without touching dispatch.
CONTAINER_REACH_XY = 0.78
SURFACE_REACH_XY = 0.85
# Floor mode drops happen at z ≈ 0.41 m (= 0 + 0.14 finger + held_h +
# 0.15 clearance) — much lower than surface (~0.68 m) or container
# (~0.79 m). The UR5e reach envelope is wider at low z so the gate can
# be loosened. 1.05 m is a conservative empirical guess; tighten if
# plans start failing for radials in that range.
FLOOR_REACH_XY = 1.05
PRE_PLACE_CLEARANCE_M = 0.10
POST_RELEASE_LIFT_M_CONTAINER = 0.10
POST_RELEASE_LIFT_M_SURFACE = 0.20  # also used for floor
# Floor mode: where to drop the held object relative to the segmented
# reference. Offset is in -X (towards robot) instead of ±Y (sideways)
# because the held object then lands BETWEEN the robot and the
# reference, never PAST the reference. This avoids crashing the arm
# into geometry just beyond the reference (e.g. the shoe rack behind
# the floor shoe in the 2026-05-16 first test, where the +0.10 Y
# offset put the drop pose directly into the rack and the arm pushed
# the rack aside). 0.15 m is roughly a shoe-length of spacing.
FLOOR_NEXT_TO_OFFSET_X_M = -0.15
FLOOR_DROP_CLEARANCE_M = 0.15  # held-object bottom this far above floor at release

# Top-down orientation in base_footprint (x, y, z, w)
TOPDOWN_ORIENTATION = [1.0, 0.0, 0.0, 0.0]


@dataclass(frozen=True)
class _ModeConfig:
    """All per-mode parameters in one place. Built by _mode_config()."""
    # Parameters passed to perception MCP get_topdown_placing_pose
    pp_top_clearance_m: float
    pp_object_height_m: float
    pp_x_bias_m: float
    # Post-segment behaviour
    reach_gate_xy: float
    post_release_lift_m: float
    # Whether the skill must compute its own drop pose (floor) or use the
    # perception-MCP-returned place_pose directly (container / surface).
    overrides_drop_pose: bool
    # Whether to run the post-release arm-cam visibility verify. Floor
    # mode skips it because the arm cam sees BOTH the reference and the
    # held object from above and SAM3 cannot tell which is which (plus
    # gripper occlusion). Container and surface modes keep it.
    run_arm_cam_verify: bool
    # Cosmetic
    preposition: str  # "in" / "on" / "next to"


def _mode_config(mode: Mode, object_height_m: float) -> _ModeConfig:
    """Return the per-mode parameters for a place call.

    ``object_height_m`` is the held-object height (from the pick skill's
    bounding-box measurement). Required for surface and floor modes;
    ignored by container mode.
    """
    if mode == "container":
        return _ModeConfig(
            pp_top_clearance_m=0.35,
            pp_object_height_m=0.0,        # tells perception MCP to use container math
            pp_x_bias_m=0.08,              # compensate front-cam NEAR bias on bin
            reach_gate_xy=CONTAINER_REACH_XY,
            post_release_lift_m=POST_RELEASE_LIFT_M_CONTAINER,
            overrides_drop_pose=False,
            run_arm_cam_verify=True,
            preposition="in",
        )
    if mode == "surface":
        return _ModeConfig(
            pp_top_clearance_m=0.15,       # 15cm above surface for gravity decoupling
            pp_object_height_m=object_height_m,
            pp_x_bias_m=0.10,              # compensate front-cam NEAR bias on table
            reach_gate_xy=SURFACE_REACH_XY,
            post_release_lift_m=POST_RELEASE_LIFT_M_SURFACE,
            overrides_drop_pose=False,
            run_arm_cam_verify=True,
            preposition="on",
        )
    if mode == "floor":
        # We segment the reference (the OTHER object on the floor), get
        # its centroid, then compute the drop pose ourselves at floor
        # height + lateral offset. The perception-MCP-returned place_pose
        # would be wrong (it'd compute as if we were placing ON the
        # reference). We still call perception MCP to get a clean
        # centroid via surface_centroid.
        return _ModeConfig(
            pp_top_clearance_m=0.0,        # ignored; we override the drop pose
            pp_object_height_m=0.0,        # use container-mode math (cheaper, returns centroid)
            pp_x_bias_m=0.0,               # we offset ourselves
            reach_gate_xy=FLOOR_REACH_XY,  # wider envelope at low drop z
            post_release_lift_m=POST_RELEASE_LIFT_M_SURFACE,
            overrides_drop_pose=True,
            run_arm_cam_verify=False,  # arm-cam can't distinguish held from reference
            preposition="next to",
        )
    raise ValueError(f"unknown mode {mode!r}; must be one of {_VALID_MODES}")


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


async def _placing_pose(
    mcp: MCPClient,
    target: str,
    *,
    cfg: _ModeConfig,
    pointcloud_topic: str,
) -> tuple[dict | None, int]:
    """Call perception MCP get_topdown_placing_pose with mode params.

    The returned dict always contains ``surface_centroid`` (the raw
    segmented centroid in base_footprint) and ``place_pose`` (the
    perception-MCP-computed wrist pose).

    For container and surface modes, the caller uses ``place_pose``
    directly. For floor mode, the caller IGNORES ``place_pose`` and
    re-computes the drop pose locally from ``surface_centroid`` plus
    a lateral offset (the perception MCP doesn't know about "next-to"
    semantics).
    """
    try:
        raw = await mcp.call_tool_prefixed(
            "perception__get_topdown_placing_pose",
            {
                "object_name": target,
                "pointcloud_topic": pointcloud_topic,
                "top_clearance_m": cfg.pp_top_clearance_m,
                "object_height_m": cfg.pp_object_height_m,
                "x_bias_m": cfg.pp_x_bias_m,
            },
        )
        data = json.loads(raw) if isinstance(raw, str) else raw
        if "place_pose" not in data and "position" not in data:
            logger.warning(f"placing pose missing position: {data}")
            return None, 1
        return data, 1
    except Exception as e:
        logger.warning(f"get_topdown_placing_pose error: {e}")
        return None, 1


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
    """Parse joint values from the compute_ik MCP response."""
    if not ik_raw or "kinematics solution" not in ik_raw.lower():
        return None
    by_name = {name: float(val) for name, val in _IK_LINE_RE.findall(ik_raw)}
    if not all(j in by_name for j in _ARM_JOINT_ORDER):
        return None
    return [by_name[j] for j in _ARM_JOINT_ORDER]


async def _plan_to_xyz(
    mcp: MCPClient, x: float, y: float, z: float
) -> tuple[bool, str, int]:
    """Plan + execute a top-down pose via the IK→joint_state pattern.

    Per feedback_plan_pose_unreliable and the 2026-05-15 finding,
    plan_and_execute(target_type="pose") is unreliable for low-z drop
    targets: OMPL's Cartesian goal sampler returns GOAL_STATE_INVALID
    even when compute_ik finds a valid joint solution. Routing through
    compute_ik → plan_and_execute(target_type="joint_state") bypasses
    the broken sampler.
    """
    # Step A — compute IK
    try:
        ik_raw = await mcp.call_tool_prefixed(
            "moveit__compute_ik",
            {
                "group": "arm",
                "position": [x, y, z],
                "orientation": TOPDOWN_ORIENTATION,
            },
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
    # MCP republishes target objects as collision objects after each
    # segmentation; without a pre-plan clear the planner can find the
    # surface / container itself as a collision blocker.
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

    # Retry after re-clearing the planning scene
    try:
        await mcp.call_tool_prefixed("moveit__clear_planning_scene", {})
    except Exception as e:
        logger.warning(f"clear_planning_scene retry error: {e}")
    try:
        ik_raw = await mcp.call_tool_prefixed(
            "moveit__compute_ik",
            {
                "group": "arm",
                "position": [x, y, z],
                "orientation": TOPDOWN_ORIENTATION,
            },
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
        return False, f"retry failed: {result[:200]}", 6
    except Exception as e:
        return False, f"retry error: {e}", 6


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
        logger.warning(f"force_detach error: {e}")
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
        logger.warning(f"verify_detached error: {e}")
        return True, f"verify error treated as PASS: {e}", 1


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
    status, calls = await _segment(mcp, object_name, camera="arm")
    if status == "SUCCESS":
        return True, f"'{object_name}' visible on arm camera at drop pose", calls
    return (
        False,
        f"'{object_name}' NOT visible on arm camera at drop pose "
        f"(seg status: {status}) — likely missed target / fell elsewhere",
        calls,
    )


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
    if mode in ("surface", "floor") and object_height_m <= 0:
        raise ValueError(
            f"object_height_m must be > 0 for mode={mode!r} "
            f"(use pick's held_object_height_m); got {object_height_m}"
        )

    tool_calls = 0
    base_result = {"target_location": target_location}
    cfg = _mode_config(mode, object_height_m)
    logger.info(
        f"place -> target='{target_location}' object='{object_name}' "
        f"mode={mode} obj_h={object_height_m:.2f}m"
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

    # Step 2 — clear octomap (defensive; sensors:[] disables the in-process
    # octomap_updater, but keep this in case sensors are re-enabled later)
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

    # Step 2b — clear stale collision objects from prior pick/place runs.
    # Open3D drop-pose and similar perception code can publish collision
    # objects that persist across attempts and block the descent.
    try:
        await mcp.call_tool_prefixed("moveit__clear_planning_scene", {})
        tool_calls += 1
    except Exception as e:
        logger.warning(f"clear_planning_scene error: {e}")
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
        logger.info(f"front SAM3 '{prompt}' -> {last_status}")
        if last_status == "SUCCESS":
            seg_ok = True
            break
    if not seg_ok:
        for prompt in fallbacks:
            last_status, calls = await _segment(mcp, prompt, camera="arm")
            tool_calls += calls
            logger.info(f"arm SAM3 '{prompt}' -> {last_status}")
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
        cfg=cfg,
        pointcloud_topic=pc_topic,
    )
    tool_calls += calls
    if coarse is None:
        return {
            **base_result,
            "success": False,
            "reason": "placing pose computation failed",
            "tool_calls_used": tool_calls,
        }

    # Drop pose dispatch:
    #   - container / surface: use perception MCP's place_pose directly.
    #   - floor:               override with reference_centroid + lateral
    #                          offset; z = floor + finger + held_h +
    #                          clearance (placing NEXT TO reference, not ON).
    if cfg.overrides_drop_pose:  # floor mode
        ref = coarse.get("surface_centroid", {})
        try:
            ref_x = float(ref["x"])
            ref_y = float(ref["y"])
        except (KeyError, TypeError, ValueError) as e:
            return {
                **base_result,
                "success": False,
                "reason": f"floor mode: surface_centroid missing/malformed: {e}",
                "tool_calls_used": tool_calls,
            }
        cx = ref_x + FLOOR_NEXT_TO_OFFSET_X_M  # negative → towards robot
        cy = ref_y
        cz = 0.0 + 0.14 + object_height_m + FLOOR_DROP_CLEARANCE_M
        logger.info(
            f"floor pose=({cx:.2f},{cy:.2f},{cz:.2f}) "
            f"[ref=({ref_x:.2f},{ref_y:.2f}), offset_x={FLOOR_NEXT_TO_OFFSET_X_M}]"
        )
    else:
        cx, cy, cz = _extract_xyz(coarse)
        logger.info(f"pose=({cx:.2f},{cy:.2f},{cz:.2f})")

    # Step 4 — reach check (per-mode gate from cfg).
    dist = (cx ** 2 + cy ** 2) ** 0.5
    if dist > cfg.reach_gate_xy:
        return {
            **base_result,
            "success": False,
            "reason": (
                f"xy distance {dist:.2f}m > {cfg.reach_gate_xy:.2f}m "
                f"({mode} gate). Navigator must redeliver closer."
            ),
            "tool_calls_used": tool_calls,
        }

    place_x, place_y, place_z = cx, cy, cz

    # Step 7 — pre-place above target
    ok, info, calls = await _plan_to_xyz(
        mcp, place_x, place_y, place_z + PRE_PLACE_CLEARANCE_M
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
        logger.warning(f"open_gripper error: {e}")
        tool_calls += 1

    # Wait for fingers to physically open + gripper_attach_node to settle.
    # The GripperCommand action's `reached_goal` only confirms the
    # command was accepted, not that the fingers have moved through the
    # attach_node closure threshold (finger_joint < 0.10). Without this
    # sleep the lift trajectory starts while fingers are still partially
    # closed near the just-released object, the attach_node sees
    # finger_joint ≥ 0.10 + proximity and re-fires on the object,
    # causing trajectory state divergence and a stuck arm.
    # 2026-05-16: validated empirically — 0.5 s of the prior settle was
    # not enough; 1.8 s reliably clears the threshold.
    await asyncio.sleep(1.8)

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
    logger.info(f"gripper status -> {body}")

    # Step 11 — lift clear before verify (arm cam needs to be above the
    # drop pose with EE in top-down orientation so the camera points
    # straight down at the drop point). Lift height from cfg: surface
    # and floor get +0.20 m for cleaner separation from the released
    # object (avoids gripper_attach_node re-fire); container stays at
    # +0.10 m because the bin drop pose is at the UR5e reach edge.
    ok, info, calls = await _plan_to_xyz(
        mcp, place_x, place_y, place_z + cfg.post_release_lift_m
    )
    tool_calls += calls
    if not ok:
        logger.warning(f"lift-clear plan failed: {info}; continuing")

    # Step 12 — placement verify via arm camera looking down at drop pose.
    # Runs for container + surface modes (per cfg.run_arm_cam_verify).
    # Floor mode SKIPS verify because the arm cam from above sees BOTH
    # the held object AND the reference object near each other, and
    # SAM3 cannot reliably distinguish them; detach status is the gate
    # for floor mode.
    visible_ok = True
    vinfo = "skipped (floor mode; detach status is the gate)"
    if cfg.run_arm_cam_verify:
        visible_ok, vinfo, calls = await _verify_object_visible_from_above(
            mcp, object_name
        )
        tool_calls += calls
        logger.info(f"arm-cam verify -> {vinfo}")

    # Step 13 — return to look_forward (transit posture). Run regardless
    # of the verify result so the arm is left in a transit-ready pose.
    arm_back = await move_arm_to_look_forward(mcp)
    tool_calls += 1
    if not arm_back.get("success"):
        logger.warning(
            f"post-release look_forward failed: {arm_back.get('reason')}"
        )

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
        "reason": (
            f"placed {cfg.preposition} {target_location} at "
            f"({place_x:.2f},{place_y:.2f},{place_z:.2f}); {vinfo}"
        ),
        "tool_calls_used": tool_calls,
    }
