"""Planner agent for the skill-based architecture.

The planner is a (typically small, open-weights) LLM that receives a
natural-language task instruction and decides which deterministic skill
to invoke next: ``approach``, ``pick``, or ``place``. It does NOT see
the underlying MCP tool surface; its decision space is the three skills
and their parameters.

Each skill is a regular Python function in ``skill_based/skills/`` that
internally calls MCP tools through the shared ``MCPClient``. The planner
treats skills as black-box LLM tools (via the LiteLLM tool-use protocol),
receiving a structured success/failure result back from each call.

This separation is the design point of the architecture: the
deterministic skill layer absorbs the per-step reasoning load, allowing
the planner to use a smaller open-weights model than a tool-using
agent that sees the raw MCP surface would require.
"""

import json
import logging
import os
from pathlib import Path

from skill_based.clients.llm import (
    assistant_message,
    call_llm,
    get_text_content,
    get_tool_calls,
    is_done,
    tool_result_message,
    wants_tool_use,
)
from skill_based.clients.mcp import MCPClient
from skill_based.skills import approach as approach_skill
from skill_based.skills import pick as pick_skill
from skill_based.skills import place as place_skill

logger = logging.getLogger(__name__)

_PROMPT_FILE = Path(__file__).parent / "planner.md"


def _resolve_default_model() -> str:
    """Return PLANNER_MODEL or LLM_MODEL from the environment.

    Raises ``RuntimeError`` if neither is set. Resolved lazily (at
    ``run_planner`` call time, not module import time) so importing
    this module for unit tests or REPL exploration does not require
    a configured ``.env``.
    """
    model = os.environ.get("PLANNER_MODEL") or os.environ.get("LLM_MODEL")
    if not model:
        raise RuntimeError(
            "no LLM model configured: set LLM_MODEL (or PLANNER_MODEL) "
            "in skill_based/.env. See skill_based/.env.example for the "
            "supported options."
        )
    return model

# === Skill tool schemas (LiteLLM / OpenAI format) ===
# These are what the planner LLM sees. The names and arguments are the
# only public surface of the architecture; everything else (MCP calls,
# motion planning, perception) lives inside the skill implementations.

PLANNER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "approach",
            "description": (
                "Move the robot base to a named area, optionally approaching a target object. "
                "Returns success when the robot is positioned within working distance of the target."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_area": {
                        "type": "string",
                        "description": "Named area (e.g. 'kitchen', 'living room', 'kids room', 'bedroom').",
                    },
                    "object_name": {
                        "type": "string",
                        "description": (
                            "Surface or object to approach within the target_area "
                            "(e.g. 'wooden coffee table', 'trash bin'). Required: "
                            "the skill always approaches a specific named target. "
                            "For pure relocation use a dedicated skill instead of "
                            "this one."
                        ),
                    },
                    "next_action": {
                        "type": "string",
                        "enum": ["pick", "surface_place", "container_place", "floor_place"],
                        "description": "What the planner intends to do immediately after this call. Controls the approach standoff distance.",
                    },
                },
                "required": ["target_area", "object_name", "next_action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pick",
            "description": (
                "Grasp the named object from the surface or floor in front of the robot. "
                "Assumes the robot is already positioned within working distance "
                "(call approach first). Returns success only when /gripper/status confirms attachment."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "object_name": {
                        "type": "string",
                        "description": "Object to grasp (e.g. 'red coke can', 'wooden cube').",
                    },
                },
                "required": ["object_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "place",
            "description": (
                "Release the held object at the named target per the selected mode. "
                "Three modes:\n"
                "  container: drop INTO a deep target (bin / basket).\n"
                "  surface:   drop ON an elevated flat target (table / shelf / rack).\n"
                "  floor:     drop NEXT TO a reference object on the floor "
                "(target_location is the REFERENCE object, e.g. the other shoe).\n"
                "Assumes the robot is already holding an object and positioned near "
                "the target. Returns success when /gripper/status confirms release "
                "(and, for container mode, when the arm-cam verify sees the object "
                "at the drop pose)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "target_location": {
                        "type": "string",
                        "description": (
                            "Reference object to segment for the drop pose. "
                            "container mode: the container (e.g. 'trash bin'). "
                            "surface mode: the surface (e.g. 'wooden coffee table'). "
                            "floor mode: a REFERENCE object on the floor "
                            "(e.g. 'white shoe' — the held object is dropped next to it)."
                        ),
                    },
                    "object_name": {
                        "type": "string",
                        "description": (
                            "Name of the held object (the object IN the gripper). "
                            "Required for the post-release arm-cam verify segmentation."
                        ),
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["container", "surface", "floor"],
                        "description": (
                            "Placement mode. Choose from task context: 'container' "
                            "for drops INTO a bin/basket; 'surface' for ON a table/shelf; "
                            "'floor' for NEXT TO a reference object on the floor."
                        ),
                    },
                    "object_height_m": {
                        "type": "number",
                        "description": (
                            "Height of the held object in metres, from the PRIOR "
                            "pick call's `held_object_height_m` result field. "
                            "Required by surface and floor modes for wrist-z math. "
                            "Container mode ignores it (pass 0.0 if unknown)."
                        ),
                    },
                },
                "required": ["target_location", "object_name", "mode", "object_height_m"],
            },
        },
    },
]


# === Skill dispatch ===

async def _dispatch_skill(mcp: MCPClient, name: str, args: dict) -> dict:
    """Route a planner skill call to the corresponding Python implementation."""
    if name == "approach":
        return await approach_skill.run(
            mcp=mcp,
            target_area=args["target_area"],
            next_action=args["next_action"],
            object_name=args["object_name"],
        )
    if name == "pick":
        return await pick_skill.run(
            mcp=mcp,
            object_name=args["object_name"],
        )
    if name == "place":
        return await place_skill.run(
            mcp=mcp,
            target_location=args["target_location"],
            object_name=args["object_name"],
            mode=args["mode"],
            object_height_m=float(args["object_height_m"]),
        )
    return {"success": False, "reason": f"unknown skill: {name}"}


# === Planner loop ===

def _load_system_prompt() -> str:
    return _PROMPT_FILE.read_text(encoding="utf-8")


async def run_planner(
    mcp: MCPClient,
    task: str,
    model: str | None = None,
    max_turns: int = 30,
) -> dict:
    """Run the planner LLM on a natural-language task.

    Returns:
        ``{"summary": str, "turns_used": int, "skill_tool_calls_total": int,
        "success": bool}``

        ``skill_tool_calls_total`` sums the MCP tool calls each skill
        made internally. Parallels ``subagent_tool_calls_total`` in the
        multi-agent orchestrator so the thesis comparison matrix can
        compare like-with-like across architectures.
    """
    model = model or _resolve_default_model()
    messages = [
        {"role": "system", "content": _load_system_prompt()},
        {"role": "user", "content": task},
    ]
    skill_tool_calls_total = 0

    for turn in range(max_turns):
        logger.info(f"=== PLANNER decision {turn + 1}/{max_turns} ===")
        response = call_llm(messages=messages, tools=PLANNER_TOOLS, model=model)

        messages.append(assistant_message(response))

        if is_done(response):
            text = get_text_content(response)
            return {
                "summary": text,
                "turns_used": turn + 1,
                "skill_tool_calls_total": skill_tool_calls_total,
                "success": True,
            }

        if not wants_tool_use(response):
            continue

        for tool_call_id, name, args in get_tool_calls(response):
            logger.info(f"-> {name}({json.dumps(args)})")
            result = await _dispatch_skill(mcp, name, args)
            success = result.get("success")
            reason = result.get("reason", "")
            calls = result.get("tool_calls_used", "?")
            try:
                skill_tool_calls_total += int(calls)
            except (TypeError, ValueError):
                pass
            if success:
                logger.info(f"<- {name} : True ({calls} calls) {reason}")
            else:
                logger.warning(
                    f"<- {name} : False ({calls} calls) {reason}"
                )
            messages.append(tool_result_message(tool_call_id, json.dumps(result)))

    return {
        "summary": "max planner turns exceeded",
        "turns_used": max_turns,
        "skill_tool_calls_total": skill_tool_calls_total,
        "success": False,
    }
