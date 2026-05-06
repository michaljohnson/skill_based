"""Shared helpers used by multiple deterministic skills.

This module hosts cross-skill utilities such as the navigation creeper,
AMCL re-seed logic, gripper-status waits, and arm-state safety resets.
The functions are extracted from ``multi_agent/navigator.py`` and
``multi_agent/pick.py`` so the skill-based architecture does not import
the LLM-driven sub-agent code directly.

TODO (next session): port the following from ``multi_agent/navigator.py``
and the relevant skill prompt files:
  * ``approach_target(mcp, target_object, mode)`` from navigator's
    ``_approach_target`` (mode-aware standoff lookup, drive_on_heading,
    SAM3 fallback when arm-cam centroid fails).
  * ``spin_search(mcp, target_object)`` from navigator's
    ``_try_spin_search`` (deterministic 3-step spin + perception check).
  * ``reseed_amcl(mcp, world_pose)`` from
    ``feedback_amcl_initialpose_after_restart.md``.
  * ``wait_until_still(mcp, timeout=5.0)`` from
    ``feedback_wait_until_still.md``.
  * ``move_arm_to_look_forward(mcp)`` using joint_state primary
    per ``feedback_moveit_state_divergence.md`` (do NOT use
    plan_to_named_state).
  * ``wait_for_gripper_attached(mcp, timeout=8.0, retry_timeout=5.0)``
    per ``feedback_gripper_attach_verify_timing.md``.

All helpers must be coroutines that take an ``MCPClient`` as the first
positional argument and return either ``None`` or a structured result
dict ``{"success": bool, "reason": str, ...}``.
"""

from skill_based.mcp_client import MCPClient


# === Mode-aware standoff distances (from navigator.py _STANDOFF_BY_MODE) ===

STANDOFF_BY_MODE = {
    "pick": 0.85,             # arm-cam grasp pipeline
    "surface_place": 0.45,    # close enough to reach over the table edge
    "container_place": 0.55,  # leave room for descent above bin opening
    "floor_place": 0.55,      # currently descoped; placeholder
}


# === Stub: approach_target ===

async def approach_target(
    mcp: MCPClient,
    target_object: str,
    mode: str,
) -> dict:
    """Drive the robot from initial nav2 pose to standoff-distance from target.

    TODO: port from ``multi_agent/navigator.py::_approach_target``.
    Behaviour expected:
        1. segment_objects on arm camera; fall back to front camera on miss.
        2. compute target xy in base_footprint frame.
        3. nav2 drive_on_heading toward target until standoff distance reached.
        4. final_verify_gate: re-segment to confirm target still visible.
    """
    raise NotImplementedError("approach_target not yet ported")


async def spin_search(
    mcp: MCPClient,
    target_object: str,
) -> dict:
    """Spin in place looking for target_object (front camera, SAM3).

    TODO: port from ``multi_agent/navigator.py::_try_spin_search``.
    """
    raise NotImplementedError("spin_search not yet ported")


async def reseed_amcl(
    mcp: MCPClient,
    x: float,
    y: float,
    yaw: float,
) -> dict:
    """Publish /initialpose to re-seed AMCL after a ROS restart.

    TODO: port from ``feedback_amcl_initialpose_after_restart.md``.
    """
    raise NotImplementedError("reseed_amcl not yet ported")


async def wait_until_still(
    mcp: MCPClient,
    timeout: float = 5.0,
) -> bool:
    """Block until /odom reports the base has decelerated to rest.

    TODO: port from ``feedback_wait_until_still.md``.
    """
    raise NotImplementedError("wait_until_still not yet ported")


async def move_arm_to_look_forward(mcp: MCPClient) -> dict:
    """Reset arm to the look_forward configuration via joint_state.

    Uses joint_state PRIMARY (not named_state) per
    ``feedback_moveit_state_divergence.md``.

    TODO: port from ``multi_agent/skills/pick.md`` step 1.
    """
    raise NotImplementedError("move_arm_to_look_forward not yet ported")


async def wait_for_gripper_attached(
    mcp: MCPClient,
    timeout: float = 8.0,
    retry_timeout: float = 5.0,
) -> dict:
    """Subscribe to /gripper/status and wait for attached:<model>.

    Two-phase wait per ``feedback_gripper_attach_verify_timing.md``:
    initial 8s, then a 5s retry if no attach detected on the first wait.

    TODO: port from ``multi_agent/pick.py``.
    """
    raise NotImplementedError("wait_for_gripper_attached not yet ported")
