#!/usr/bin/env python3
"""Skill-based architecture entry point.

Usage:
    python3 -m skill_based.main --task "pick up the coke can on the kitchen table and bring it to the trash bin"
    python3 -m skill_based.main --test-pick "red coke can"
    python3 -m skill_based.main --test-place "trash bin"
    python3 -m skill_based.main --test-approach "kitchen" --target-object "wooden coffee table"
"""

import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from skill_based.clients.mcp import MCPClient
from skill_based.planner import run_planner
from skill_based.skills import approach as approach_skill
from skill_based.skills import pick as pick_skill
from skill_based.skills import place as place_skill

# === CONFIGURATION ===

# Default LLM model for the planner. The skill-based architecture
# intentionally targets a smaller open-weights model: the deterministic
# Python skills absorb the per-step reasoning load, so the planner only
# needs to pick from three skills and supply structured arguments.
#
# Any LiteLLM-supported model works. For an OpenAI-compatible endpoint
# (vLLM, Ollama, LocalAI, etc.) set OPENAI_API_BASE + OPENAI_API_KEY
# in `.env`; for Anthropic set ANTHROPIC_API_KEY. See `.env.example`.
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/cyankiwi/Qwen3.6-27B-AWQ-INT4")
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", LLM_MODEL)


async def test_pick(object_name: str) -> None:
    print(f"\n=== Testing pick skill: '{object_name}' ===\n")
    async with MCPClient() as mcp:
        t0 = time.perf_counter()
        result = await pick_skill.run(mcp=mcp, object_name=object_name)
        result["wall_seconds"] = round(time.perf_counter() - t0, 2)
        print(f"\n=== Result ===")
        print(json.dumps(result, indent=2))


async def test_place(
    target_container: str,
    object_name: str | None = None,
) -> None:
    print(f"\n=== Testing place skill: target='{target_container}' object='{object_name}' ===\n")
    async with MCPClient() as mcp:
        t0 = time.perf_counter()
        result = await place_skill.run(
            mcp=mcp,
            target_container=target_container,
            object_name=object_name,
        )
        result["wall_seconds"] = round(time.perf_counter() - t0, 2)
        print(f"\n=== Result ===")
        print(json.dumps(result, indent=2))


async def test_approach(
    destination: str,
    next_action: str,
    target_object: str | None = None,
) -> None:
    print(f"\n=== Testing approach skill: '{destination}' (next_action={next_action}) ===")
    if target_object:
        print(f"    Target object: '{target_object}'")
    async with MCPClient() as mcp:
        t0 = time.perf_counter()
        result = await approach_skill.run(
            mcp=mcp,
            destination=destination,
            next_action=next_action,
            target_object=target_object,
        )
        result["wall_seconds"] = round(time.perf_counter() - t0, 2)
        print(f"\n=== Result ===")
        print(json.dumps(result, indent=2))


async def run_full(task: str) -> None:
    print(f"\n=== Task: {task} ===")
    print(f"Planner: {PLANNER_MODEL}\n")
    async with MCPClient() as mcp:
        t0 = time.perf_counter()
        result = await run_planner(mcp=mcp, task=task, model=PLANNER_MODEL)
        wall_seconds = round(time.perf_counter() - t0, 2)
        print(f"\n=== Final Report ===")
        print(result["summary"])
        print(f"\nPlanner turns used:        {result['turns_used']}")
        print(f"Skill tool calls total:    {result['skill_tool_calls_total']}")
        print(f"Wall-clock total:          {wall_seconds}s ({wall_seconds / 60:.1f} min)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Skill-based agentic architecture")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Natural language task (e.g. 'pick up the can and put it in the trash bin')",
    )
    parser.add_argument(
        "--test-pick",
        type=str,
        metavar="OBJECT",
        default=None,
        help="Test pick skill on a single object (no planner)",
    )
    parser.add_argument(
        "--test-place",
        type=str,
        metavar="CONTAINER",
        default=None,
        help="Test place skill on a single target (no planner). Robot must be holding an object.",
    )
    parser.add_argument(
        "--test-approach",
        type=str,
        nargs="+",
        metavar="DEST",
        default=None,
        help="Test approach skill to DEST (no planner). Combine with --next-action and optionally --target-object.",
    )
    parser.add_argument(
        "--next-action",
        type=str,
        choices=["pick", "surface_place", "container_place", "floor_place"],
        default="pick",
        help="What the planner intends to do after the approach (selects standoff distance). Used with --test-approach.",
    )
    parser.add_argument(
        "--target-object",
        type=str,
        default=None,
        help="Optional target object for approach skill (e.g. 'wooden coffee table')",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    else:
        class _SkillTagFormatter(logging.Formatter):
            _TAGS = {
                "skill_based.planner":   "[PLANNER] ",
                "skill_based.skills.approach": "[APPROACH]",
                "skill_based.skills.pick":     "[PICK]    ",
                "skill_based.skills.place":    "[PLACE]   ",
                "skill_based.skills.common":   "[COMMON]  ",
                "skill_based.clients.mcp":      "[MCP]     ",
            }

            def format(self, record: logging.LogRecord) -> str:
                tag = self._TAGS.get(record.name, f"[{record.name}]")
                msg = record.getMessage()
                if record.levelno >= logging.WARNING:
                    return f"{tag} {record.levelname}: {msg}"
                return f"{tag} {msg}"

        handler = logging.StreamHandler()
        handler.setFormatter(_SkillTagFormatter())
        root = logging.getLogger()
        root.handlers = [handler]
        root.setLevel(logging.INFO)

    framework_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.getLogger("httpx").setLevel(framework_level)
    logging.getLogger("mcp").setLevel(framework_level)
    logging.getLogger("LiteLLM").setLevel(framework_level)
    logging.getLogger("litellm").setLevel(framework_level)
    logging.getLogger("skill_based.clients.mcp").setLevel(framework_level)

    if args.test_pick:
        asyncio.run(test_pick(args.test_pick))
    elif args.test_place:
        asyncio.run(test_place(args.test_place, args.target_object))
    elif args.test_approach:
        dest = " ".join(args.test_approach)
        asyncio.run(test_approach(dest, args.next_action, args.target_object))
    elif args.task:
        asyncio.run(run_full(args.task))
    else:
        parser.error(
            "no action specified — pass --task, --test-pick, --test-place, "
            "or --test-approach"
        )


if __name__ == "__main__":
    main()
