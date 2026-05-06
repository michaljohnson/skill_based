"""Pick skill — deterministic Python grasp primitive.

The skill takes an object name and runs the canonical grasp pipeline:
arm to look_forward, segment_objects on the arm camera, top-down grasp
pose, plan-and-execute pre-grasp, descend, close gripper, and verify
attachment via /gripper/status. On failure it returns a structured
reason string; the planner does not retry inside the skill, but may
choose to call pick again after re-positioning.

Translated from the canonical sequence documented in
``multi_agent/skills/pick.md`` and the LLM-driven flow in
``multi_agent/pick.py``. The deterministic version removes the LLM's
turn-by-turn reasoning and bakes the canonical sequence into Python.
"""

import logging

from skill_based.mcp_client import MCPClient
from skill_based.skills.common import (
    move_arm_to_look_forward,
    wait_for_gripper_attached,
)

logger = logging.getLogger(__name__)


async def run(
    mcp: MCPClient,
    object_name: str,
) -> dict:
    """Grasp ``object_name`` from the surface in front of the robot.

    Preconditions: robot already positioned at standoff distance from
    the surface holding the target object. Arm in any state.

    Returns:
        ``{"success": bool, "reason": str, "object_name": str}``
    """
    # Step 1 — arm to look_forward (use joint_state PRIMARY per
    # feedback_moveit_state_divergence.md).
    # await move_arm_to_look_forward(mcp)

    # Step 2 — segment_objects on arm camera with object_name as text prompt.
    # Per feedback_pick_uses_arm_camera.md, MUST be camera="arm".

    # Step 3 — get_topdown_grasp_pose from segmentation pointcloud.
    # Per feedback_approach_target_centroid_bug.md, check for centroid_base_frame
    # presence in result; abort if missing (stale pointcloud).

    # Step 4 — clear_octomap before first pre-grasp motion
    # (feedback_clear_octomap_before_grasp.md).

    # Step 5 — plan_and_execute pre-grasp pose (z + 0.10 above grasp_z).

    # Step 6 — open_gripper.

    # Step 7 — plan_and_execute grasp pose (descend).

    # Step 8 — close_gripper.

    # Step 9 — wait_for_gripper_attached (8s + 5s retry per
    # feedback_gripper_attach_verify_timing.md).

    # Step 10 — lift to z + 0.10 above grasp_z.

    # Step 11 — return arm to look_forward (lift first to z=1.0 if near
    # tall surface per feedback_lift_before_look_forward.md).

    # Step 12 — return success only if /gripper/status reported attached.

    raise NotImplementedError("pick.run not yet ported")
