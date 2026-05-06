"""Place skill — deterministic Python release primitive.

The skill takes a target container or surface name and runs the
canonical release pipeline: front-cam segmentation of the target,
top-down drop pose computation, optional arm-cam refinement (stage 2),
pre-place above, cartesian descent, open gripper, post-release
gripper-status verify, and arm reset.

Translated from ``multi_agent/skills/place.md`` and the LLM-driven flow
in ``multi_agent/place.py``. Surface and container modes share the
front-cam stage 1; only container mode re-segments at stage 2 from the
arm camera (per feedback_dont_resegment_after_lookdown_for_surfaces.md).
"""

import logging

from skill_based.mcp_client import MCPClient
from skill_based.skills.common import (
    move_arm_to_look_forward,
)

logger = logging.getLogger(__name__)


async def run(
    mcp: MCPClient,
    target_container: str,
    object_name: str | None = None,
) -> dict:
    """Release the held object onto/into ``target_container``.

    Preconditions: robot is holding an object (gripper attached) and is
    positioned within working distance of the target. Arm in
    look_forward (carry posture).

    Returns:
        ``{"success": bool, "reason": str, "target_container": str}``
    """
    # Step 1 — segment_objects on FRONT camera with target_container as
    # text prompt (per feedback_place_uses_front_camera.md, MUST be
    # camera="front" in place mode).

    # Step 2 — get_topdown_drop_pose stage 1 (front-cam coarse pose).

    # Step 3 — for container mode: arm-cam stage 2 refinement
    # (re-segment + re-compute drop pose at surface_z+0.40m).
    # For surface mode: skip stage 2 (SAM3 cannot segment bare planes).

    # Step 4 — sanity check: reject stage-2 refinement if it differs
    # from stage 1 by >0.20m in any axis.

    # Step 5 — plan_and_execute pre-place above (drop_z + 0.10).

    # Step 6 — cartesian descent to drop_z (guarantees top-down approach).

    # Step 7 — open_gripper.

    # Step 8 — wait briefly, then check /gripper/status reports
    # detached (post-release ground-truth verify per
    # feedback_place_no_ground_truth_verify.md).

    # Step 9 — re-segment from arm-cam to confirm object is on target
    # (post-release visibility verify, see project_session_2026_04_22_e2e.md).

    # Step 10 — return arm to look_forward (intermediate lift to z=1.0
    # first if near tall container per feedback_lift_before_look_forward.md).

    raise NotImplementedError("place.run not yet ported")
