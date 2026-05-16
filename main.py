#!/usr/bin/env python3
"""Skill-based architecture entry point.

Usage:
    python3 -m skill_based.main --task "pick up the coke can on the kitchen table and bring it to the trash bin"
    python3 -m skill_based.main --test-pick "red coke can"
    python3 -m skill_based.main --test-place "trash bin" --object-name "white cube" --mode container --object-height-m 0
    python3 -m skill_based.main --test-place "wooden coffee table" --object-name "coke can" --mode surface --object-height-m 0.12
    python3 -m skill_based.main --test-place "white shoe" --object-name "red shoe" --mode floor --object-height-m 0.10
    python3 -m skill_based.main --test-approach "kitchen" --object-name "wooden coffee table"
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

# LLM model for the planner. The skill-based architecture intentionally
# targets a smaller open-weights model: the deterministic Python skills
# absorb the per-step reasoning load, so the planner only needs to pick
# from three skills and supply structured arguments.
#
# Required env var. Any LiteLLM-supported model works. For an
# OpenAI-compatible endpoint (vLLM, Ollama, LocalAI, etc.) set
# OPENAI_API_BASE + OPENAI_API_KEY in `.env`; for Anthropic set
# ANTHROPIC_API_KEY. See `.env.example` for the full menu.
LLM_MODEL = os.environ.get("LLM_MODEL")
if not LLM_MODEL:
    raise SystemExit(
        "LLM_MODEL is not set. Copy skill_based/.env.example to "
        "skill_based/.env and uncomment exactly one LLM_MODEL line."
    )
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
    target_location: str,
    object_name: str,
    mode: str,
    object_height_m: float,
) -> None:
    print(
        f"\n=== Testing place skill: target='{target_location}' "
        f"object='{object_name}' mode={mode} obj_h={object_height_m:.2f} ===\n"
    )
    async with MCPClient() as mcp:
        t0 = time.perf_counter()
        result = await place_skill.run(
            mcp=mcp,
            target_location=target_location,
            object_name=object_name,
            mode=mode,
            object_height_m=object_height_m,
        )
        result["wall_seconds"] = round(time.perf_counter() - t0, 2)
        print(f"\n=== Result ===")
        print(json.dumps(result, indent=2))


async def test_approach(
    target_area: str,
    next_action: str,
    object_name: str,
) -> None:
    print(f"\n=== Testing approach skill: '{target_area}' (next_action={next_action}) ===")
    print(f"    Target object: '{object_name}'")
    async with MCPClient() as mcp:
        t0 = time.perf_counter()
        result = await approach_skill.run(
            mcp=mcp,
            target_area=target_area,
            next_action=next_action,
            object_name=object_name,
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
        print(f"\nPlanner decisions made:    {result['turns_used']}")
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
        help="Test approach skill to DEST (no planner). Combine with --next-action and optionally --object-name.",
    )
    parser.add_argument(
        "--next-action",
        type=str,
        choices=["pick", "surface_place", "container_place", "floor_place"],
        default="pick",
        help="What the planner intends to do after the approach (selects standoff distance). Used with --test-approach.",
    )
    parser.add_argument(
        "--object-name",
        type=str,
        default=None,
        help="Target object (e.g. 'wooden coffee table'). Required with --test-approach and --test-place.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["container", "surface", "floor"],
        default=None,
        help="Place mode. Required with --test-place.",
    )
    parser.add_argument(
        "--object-height-m",
        type=float,
        default=0.0,
        help=(
            "Held object height in metres (from pick.held_object_height_m). "
            "Required with --test-place when --mode=surface or --mode=floor. "
            "Ignored for --mode=container."
        ),
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
            # Uppercase [PLANNER] signals the LLM-reasoning agent;
            # lowercase skill tags signal deterministic Python execution
            # underneath. Section-break "=== PLANNER decision N/M ===" lines
            # bracket each planner decision (handled in format() below).
            _TAGS = {
                "skill_based.planner":   "[PLANNER]  ",
                "skill_based.skills.approach": "[approach] ",
                "skill_based.skills.pick":     "[pick]     ",
                "skill_based.skills.place":    "[place]    ",
                "skill_based.clients.mcp":     "[mcp]      ",
            }

            def format(self, record: logging.LogRecord) -> str:
                msg = record.getMessage()
                # Section-break lines (e.g. "=== PLANNER decision 1/30 ===")
                # are printed without a tag prefix so they stand out as
                # visual boundaries between planner reasoning and skill
                # execution.
                if msg.startswith("==="):
                    return msg
                tag = self._TAGS.get(record.name, f"[{record.name}]")
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
        if not args.object_name:
            parser.error("--test-place requires --object-name (the held object's name)")
        if not args.mode:
            parser.error("--test-place requires --mode (container | surface | floor)")
        if args.mode in ("surface", "floor") and args.object_height_m <= 0:
            parser.error(
                f"--test-place --mode={args.mode} requires --object-height-m > 0 "
                "(measured in metres; from pick.held_object_height_m)"
            )
        asyncio.run(test_place(
            args.test_place,
            args.object_name,
            args.mode,
            args.object_height_m,
        ))
    elif args.test_approach:
        if not args.object_name:
            parser.error("--test-approach requires --object-name")
        dest = " ".join(args.test_approach)
        asyncio.run(test_approach(dest, args.next_action, args.object_name))
    elif args.task:
        asyncio.run(run_full(args.task))
    else:
        parser.error(
            "no action specified — pass --task, --test-pick, --test-place, "
            "or --test-approach"
        )


if __name__ == "__main__":
    main()
