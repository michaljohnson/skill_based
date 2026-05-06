"""Navigate skill — deterministic Python navigation primitive.

The skill takes a named destination, an optional target object to refine
the approach, and a mode that controls the standoff distance. It calls
Nav2 MCP tools to drive the base, then optionally refines via the
``approach_target`` helper from ``common.py``.

The planner LLM never sees the underlying MCP tools; it only sees this
skill's signature and the structured success/failure result.

Inspired by ``multi_agent/navigator.py`` but stripped of the LLM loop.
The deterministic logic is what remains.
"""

import logging

from skill_based.mcp_client import MCPClient
from skill_based.skills.common import (
    STANDOFF_BY_MODE,
    approach_target,
    move_arm_to_look_forward,
    spin_search,
    wait_until_still,
)

logger = logging.getLogger(__name__)


# === Named-area pose lookup table ===
# TODO: populate from `multi_agent/skills/navigator.md` after re-mapping
# the world (see project_tomorrow_plan_2026_05_06.md).
NAMED_AREA_POSES = {
    # "kitchen": {"x": 0.0, "y": 0.0, "yaw": 0.0},
    # "living room": {"x": 0.0, "y": 0.0, "yaw": 0.0},
    # "kids room": {"x": 0.0, "y": 0.0, "yaw": 0.0},
    # "bedroom": {"x": 0.0, "y": 0.0, "yaw": 0.0},
}


async def run(
    mcp: MCPClient,
    destination: str,
    mode: str,
    target_object: str | None = None,
) -> dict:
    """Drive the robot to ``destination``, optionally approaching ``target_object``.

    Args:
        mcp: shared MCP client.
        destination: named area key (must be in NAMED_AREA_POSES).
        mode: one of "pick", "surface_place", "container_place" (selects standoff).
        target_object: optional surface or object to approach within destination.

    Returns:
        ``{"success": bool, "reason": str}``
    """
    if mode not in STANDOFF_BY_MODE:
        return {"success": False, "reason": f"unknown mode: {mode}"}

    # Step 1 — arm to look_forward (clear front-cam FOV before nav).
    # See feedback_arm_before_nav.md.
    # TODO: wire when common.py is ported.
    # await move_arm_to_look_forward(mcp)

    # Step 2 — nav2 navigate_to_pose to the named area.
    # TODO: implement after NAMED_AREA_POSES is populated.
    raise NotImplementedError("navigate.run not yet ported")

    # Step 3 — wait_until_still on /odom.
    # await wait_until_still(mcp)

    # Step 4 — if target_object given, refine approach via creeper.
    # if target_object:
    #     refine = await approach_target(mcp, target_object, mode)
    #     if not refine.get("success"):
    #         spin = await spin_search(mcp, target_object)
    #         if not spin.get("success"):
    #             return {"success": False, "reason": "target_object not found after spin search"}
    #
    # return {"success": True, "reason": f"navigated to {destination}"}
