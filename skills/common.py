"""Shared helpers used by multiple deterministic skills.

Each helper encapsulates a small, repeatable robot-control sequence
(arm reset, drive-to-target, gripper-status wait, etc.). They exist
as compiled Python so the planner LLM never has to reason about them.

All helpers are coroutines that take an :class:`MCPClient` as the first
positional argument and return either ``None``, a primitive, or a
structured result dict ``{"success": bool, "reason": str, ...}``.
"""

import asyncio
import json
import logging
import math
import time

from skill_based.clients.mcp import MCPClient

logger = logging.getLogger(__name__)


# === Standoff distance by next_action (parallels multi_agent's _STANDOFF_BY_NEXT_ACTION) ===
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


# === Canonical look_forward joint positions ===
#
# Use joint_state PRIMARY (not named_state). In Gazebo,
# plan_and_execute(named_state="look_forward") sometimes reports
# planning success while the physical arm has not moved (state
# divergence between MoveIt's perceived state and the actual robot).
# Explicit numeric joint targets eliminate this silent-success failure.

LOOK_FORWARD_JOINTS = [-0.0001, -0.2429, -2.8291, -0.7983, 1.5622, 0.0]


# === Internal parsing helpers ===

def _parse_robot_pose(raw) -> tuple[float | None, float | None, float | None]:
    """Extract (x, y, yaw) from nav2__get_robot_pose response.

    Returns ``(None, None, None)`` on parse failure so callers can fall
    back gracefully.
    """
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


def parse_seg_status(seg_raw) -> str:
    """Extract ``status`` from a segment_objects response (str or dict)."""
    if not isinstance(seg_raw, str):
        seg_raw = str(seg_raw)
    try:
        return json.loads(seg_raw).get("status", "UNKNOWN")
    except (json.JSONDecodeError, AttributeError):
        return "UNKNOWN"


def geometric_fallback_prompts(target: str) -> list[str]:
    """Return a try-in-order list of SAM3 prompts for ``target``.

    First entry is always the literal target; subsequent entries are
    geometric / colour descriptors known to anchor when category names
    fail. SAM3's open-vocabulary detector is reliable on shape/colour
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
        prompts.append("wooden surface")
    elif "shoe rack" in t:
        prompts.append("red shoe on the floor")
    elif "cube" in t:
        prompts.append("white cube on the floor")
    elif "can" in t or "coke" in t:
        prompts.append("red can on the floor")
    return prompts


# === Arm reset ===

async def move_arm_to_look_forward(mcp: MCPClient) -> dict:
    """Reset the arm to the canonical look_forward configuration.

    Tries joint_state first (more reliable than named_state in Gazebo,
    where named_state planning can report success without moving the
    arm); falls back to named_state if joint_state planning fails.
    Returns a structured result with the path taken so callers can log it.
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
        logger.warning(f"  [look_forward] joint_state error: {e}")

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
        return {"success": False, "reason": f"named_state failed: {result[:200]}"}
    except Exception as e:
        return {"success": False, "reason": f"named_state error: {e}"}


# === Wait for stillness ===

async def wait_until_still(
    mcp: MCPClient,
    timeout: float = 3.0,
    vel_threshold: float = 0.02,
    poll_s: float = 0.15,
    post_settle: float = 0.25,
) -> None:
    """Block until /odom reports the base has decelerated to rest.

    nav2 reports complete as soon as it stops *commanding*, but the
    velocity_smoother keeps decelerating for ~500ms. Segmenting during
    that window catches the camera mid-motion. Polls /odom twist until
    |linear.x| and |angular.z| are under ``vel_threshold``, then waits
    ``post_settle`` for the camera ring buffer to flush.
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
        except Exception as e:
            logger.debug(f"  [wait-still] odom poll error: {e}")
        await asyncio.sleep(poll_s)
    logger.warning(f"  [wait-still] timeout after {timeout}s; proceeding")
    await asyncio.sleep(post_settle)


# === Approach target (delegates to nav2-mcp `approach_target` primitive) ===

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
        f"  [approach] target_base=({target_x_base:.2f},{target_y_base:.2f}) "
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

    # nav2-mcp returns a string on success; treat presence of "error" as failure
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


# === Spin search ===

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


# === Gripper attach verification ===

def _tokenize(s: str) -> set[str]:
    """Split on non-alphanumerics AND CamelCase boundaries.

    Drops tokens shorter than 3 characters. ``"KidsRoom_WoodCube"``
    splits to ``{"kids", "room", "wood", "cube"}``; ``"coke can"``
    splits to ``{"coke", "can"}``; the two share no token, but ``coke can``
    against ``Kitchen_Coke`` shares ``coke``.
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
            # Camel-case boundary: previous char was lowercase letter,
            # this one is uppercase. Flush before continuing.
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
    and the attach plugin firing — a single short wait misses real attach
    events; a single long wait wastes time on real failures.

    ``expected_object`` is matched by token overlap against the Gazebo
    model name. The natural-language prompt the agent uses (e.g.
    "coke can") is split into alphanumeric tokens; the model name
    ("Kitchen_Coke") is normalised the same way; if any token of length
    >= 3 appears in both, the match passes. This handles the common
    Gazebo naming convention where models are ``<Room>_<ObjectName>``.
    Pass None to accept any attach.

    Returns a dict with the attached model name on success.
    """

    async def _read() -> tuple[str, str]:
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
        return body, raw if isinstance(raw, str) else json.dumps(raw)

    def _matches(body: str) -> tuple[bool, str | None]:
        if not body.startswith("attached:"):
            return False, None
        model = body[len("attached:"):].strip()
        if expected_object is None:
            return True, model
        expected_tokens = _tokenize(expected_object)
        model_tokens = _tokenize(model)
        if not expected_tokens:
            return True, model  # nothing to compare; trust the attach
        return bool(expected_tokens & model_tokens), model

    # Phase 1
    try:
        body, _ = await _read()
        ok, model = _matches(body)
        if ok:
            return {
                "success": True,
                "reason": f"attached:{model}",
                "model": model,
            }
        first_body = body
    except Exception as e:
        logger.warning(f"  [attach] first read error: {e}")
        first_body = ""

    # Phase 2 — retry with shorter timeout
    await asyncio.sleep(0.5)
    try:
        raw = await mcp.call_tool_prefixed(
            "ros__subscribe_once",
            {
                "topic": "/gripper/status",
                "msg_type": "std_msgs/msg/String",
                "timeout": int(retry_timeout),
            },
        )
        data = json.loads(raw) if isinstance(raw, str) else raw
        msg = data.get("msg", data)
        body = msg.get("data", "") if isinstance(msg, dict) else str(msg)
        ok, model = _matches(body)
        if ok:
            return {
                "success": True,
                "reason": f"attached:{model} (retry)",
                "model": model,
            }
        return {
            "success": False,
            "reason": (
                f"gripper not attached after {timeout:.0f}s + "
                f"{retry_timeout:.0f}s retry; last={body or first_body or 'no message'}"
            ),
        }
    except Exception as e:
        return {
            "success": False,
            "reason": f"attach retry error: {e}",
        }


# === AMCL re-seed ===

async def reseed_amcl(
    mcp: MCPClient,
    x: float,
    y: float,
    yaw: float,
) -> dict:
    """Publish /initialpose to re-seed AMCL.

    Necessary after a ROS restart (AMCL defaults to (0, 2.0, 0) with
    fake-confident covariance). The initial pose covariance is set high
    enough that AMCL accepts the seed without resampling on the first
    laser update.
    """
    cos_h = math.cos(yaw / 2.0)
    sin_h = math.sin(yaw / 2.0)
    msg = {
        "header": {"frame_id": "map"},
        "pose": {
            "pose": {
                "position": {"x": x, "y": y, "z": 0.0},
                "orientation": {
                    "x": 0.0, "y": 0.0, "z": sin_h, "w": cos_h,
                },
            },
            "covariance": [
                0.25, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.25, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.07,
            ],
        },
    }
    try:
        await mcp.call_tool_prefixed(
            "ros__publish_once",
            {
                "topic": "/initialpose",
                "msg_type": "geometry_msgs/msg/PoseWithCovarianceStamped",
                "msg": msg,
            },
        )
        await asyncio.sleep(1.0)
        return {"success": True, "reason": f"re-seeded AMCL at ({x:.2f},{y:.2f},{yaw:.2f})"}
    except Exception as e:
        return {"success": False, "reason": f"reseed_amcl failed: {e}"}
