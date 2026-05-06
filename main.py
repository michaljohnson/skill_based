#!/usr/bin/env python3
"""Skill-based architecture entry point.

Usage:
    python3 -m skill_based.main --task "pick up the coke can on the kitchen table and bring it to the trash bin"
    python3 -m skill_based.main --test-pick "red coke can"
    python3 -m skill_based.main --test-place "trash bin"
    python3 -m skill_based.main --test-navigate "kitchen" --target-object "wooden coffee table"
"""

import argparse
import asyncio
import json
import logging
import os

from skill_based.mcp_client import MCPClient
from skill_based.planner import run_planner
from skill_based.skills import navigate as navigate_skill
from skill_based.skills import pick as pick_skill
from skill_based.skills import place as place_skill

# === CONFIGURATION ===

# Default LLM model for the planner. Skill-based intentionally targets
# a smaller model (Claude Haiku) because the deterministic skills absorb
# the per-step reasoning load. See methodology Section 2.1.3.
LLM_MODEL = os.environ.get("LLM_MODEL", "anthropic/claude-haiku-4-5-20251001")
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", LLM_MODEL)


async def test_pick(object_name: str) -> None:
    print(f"\n=== Testing pick skill: '{object_name}' ===\n")
    async with MCPClient() as mcp:
        result = await pick_skill.run(mcp=mcp, object_name=object_name)
        print(f"\n=== Result ===")
        print(json.dumps(result, indent=2))


async def test_place(target_container: str) -> None:
    print(f"\n=== Testing place skill: target='{target_container}' ===\n")
    async with MCPClient() as mcp:
        result = await place_skill.run(mcp=mcp, target_container=target_container)
        print(f"\n=== Result ===")
        print(json.dumps(result, indent=2))


async def test_navigate(
    destination: str,
    mode: str,
    target_object: str | None = None,
) -> None:
    print(f"\n=== Testing navigate skill: '{destination}' (mode={mode}) ===")
    if target_object:
        print(f"    Target object: '{target_object}'")
    async with MCPClient() as mcp:
        result = await navigate_skill.run(
            mcp=mcp,
            destination=destination,
            mode=mode,
            target_object=target_object,
        )
        print(f"\n=== Result ===")
        print(json.dumps(result, indent=2))


async def run_full(task: str) -> None:
    print(f"\n=== Task: {task} ===")
    print(f"Planner: {PLANNER_MODEL}\n")
    async with MCPClient() as mcp:
        result = await run_planner(mcp=mcp, task=task, model=PLANNER_MODEL)
        print(f"\n=== Final Report ===")
        print(result["summary"])
        print(f"\nPlanner turns used: {result['turns_used']}")


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
        "--test-navigate",
        type=str,
        nargs="+",
        metavar="DEST",
        default=None,
        help="Test navigate skill to DEST (no planner). Combine with --mode and optionally --target-object.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["pick", "surface_place", "container_place"],
        default="pick",
        help="Navigation mode (selects standoff distance). Used with --test-navigate.",
    )
    parser.add_argument(
        "--target-object",
        type=str,
        default=None,
        help="Optional target object for navigator (e.g. 'wooden coffee table')",
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
                "skill_based.skills.navigate": "[NAVIGATE]",
                "skill_based.skills.pick":     "[PICK]    ",
                "skill_based.skills.place":    "[PLACE]   ",
                "skill_based.skills.common":   "[COMMON]  ",
                "skill_based.mcp_client":      "[MCP]     ",
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
    logging.getLogger("skill_based.mcp_client").setLevel(framework_level)

    if args.test_pick:
        asyncio.run(test_pick(args.test_pick))
    elif args.test_place:
        asyncio.run(test_place(args.test_place))
    elif args.test_navigate:
        dest = " ".join(args.test_navigate)
        asyncio.run(test_navigate(dest, args.mode, args.target_object))
    elif args.task:
        asyncio.run(run_full(args.task))
    else:
        parser.error(
            "no action specified — pass --task, --test-pick, --test-place, "
            "or --test-navigate"
        )


if __name__ == "__main__":
    main()
