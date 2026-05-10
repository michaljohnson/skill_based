# Skill-based architecture

A deterministic-skills agentic architecture for long-horizon mobile-manipulation tasks. Inspired by the CaP-X programmatic skill-abstraction pattern (Fu et al., 2026), implemented as a deliberately less-effort variant: small Python skills wrap the canonical MCP tool sequences for `navigate`, `pick`, and `place`, and a planner LLM decides which skill to call next.

Originally built as one of three architectures compared in a BA thesis on "where the policy should live" in agentic robotics; released so others can reuse the pattern.

![Architecture diagram](docs/architecture.png)

The headline claim: ~60 MCP tools across 4 servers are never exposed to the planner LLM. The planner sees exactly 3 skill schemas. All low-level decisions (which MCP tool, in what order, with what arguments, with what error handling) are absorbed by deterministic Python.

## Comparison-axis position (vs. siblings)

| Architecture | Where the policy lives | Planner LLM |
|---|---|---|
| Single-agent | LLM context, raw MCP tool surface | Frontier (e.g. Claude Opus) |
| Multi-agent | Orchestrator + 3 LLM sub-agents, narrow MCP subsets per agent | Frontier (e.g. Claude Opus) |
| **Skill-based** | **Inside Python skills, hidden from the LLM** | **Small open-weights or frontier — both supported** |

The smaller-model + smarter-skills pairing is this architecture's design point, not a confound. Any LiteLLM-supported model works; see `.env.example`.

## Requirements

- Python 3.10+
- `litellm`, `python-dotenv`, an `mcp` Python client (any FastMCP-style streamable-http client works)
- One or more MCP servers exposing the tool families this code expects:
  - **nav2** — drive, navigate-to-pose, spin, lifecycle, `approach_target`
  - **moveit** — plan/execute, IK, planning scene
  - **perception** — segmentation, top-down grasp/place pose, look
  - **ros** — generic topics, services, actions, parameters
- An LLM provider (Anthropic API, OpenAI-compatible vLLM, OpenAI, Ollama — anything LiteLLM supports)

The MCP servers are not part of this package; you bring your own. The architecture's contract with them is "any tool-calling LLM should be able to use them," which is exactly what an MCP server provides.

## Repository layout

```
skill_based/
  main.py                  CLI entry (--task / --test-{pick,place,navigate})
  planner.py               planner LLM agent + skill dispatch
  planner.md               planner system prompt (loaded by planner.py)
  skills/                  deterministic Python skills (the architecture's middle layer)
    __init__.py
    navigate.py            nav2 + approach sequence
    pick.py                grasp pipeline
    place.py               release pipeline
    common.py              shared helpers (approach_target, AMCL re-seed, gripper-status wait, segmentation fallback chain)
  clients/                 external system adapters (the architecture's low-layer interface)
    __init__.py
    llm.py                 LiteLLM wrapper with Hermes-XML tool-call fallback
    mcp.py                 MCP connection manager
  docs/
    architecture.png       3-level diagram (high/middle/low)
  .env.example             environment-variable template
  README.md                this file
```

## Quick start

```bash
cp skill_based/.env.example skill_based/.env
# edit .env: pick a planner LLM (Anthropic, OpenAI-compatible local vLLM, OpenAI, ...)

# Single-skill smoke tests (assume robot is pre-positioned for pick/place):
python3 -m skill_based.main --test-pick "red coke can"
python3 -m skill_based.main --test-place "trash bin" --target-object "red coke can"
python3 -m skill_based.main --test-navigate "kitchen" --mode pick --target-object "wooden coffee table"

# Full planner loop:
python3 -m skill_based.main --task "pick up the red coke can in the kitchen and place it on the wooden coffee table in the living room"
```

`.env` is loaded automatically via `python-dotenv` from the package root. Run from the parent directory of `skill_based/` so the package import resolves.

## How a turn works

1. The planner LLM receives the system prompt (`planner.md`), the user task, and three tool schemas (`navigate`, `pick`, `place`).
2. The planner emits a tool call. LiteLLM routes it back through `clients/llm.py`. If the model is served by a vLLM without the Hermes tool-call parser flag, a client-side parser extracts the tool call from `message.content`.
3. `planner.py` dispatches the call to the matching deterministic skill in `skills/`.
4. The skill runs its canonical MCP-tool sequence (typically 5–20 calls), returns `{success: bool, reason: str, tool_calls_used: int, ...}`.
5. The planner sees the structured result and decides the next skill, or returns a final report.

The planner does NOT see the underlying MCP tool surface. The skills do not call the LLM.

## Adapting to your environment

The skills assume specific MCP tool names (e.g. `perception__segment_objects`, `nav2__approach_target`, `moveit__plan_and_execute`). If your MCP servers expose different names, edit the calls inside `skills/*.py`. The architectural pattern (planner → deterministic skill → MCP tool) is independent of the specific tool names; only the strings need updating.

The hardcoded entry-pose table in `skills/navigate.py` is keyed to a particular simulated home environment. Replace with your own room/area coordinates.

## Design decisions

- **No CaP-X faithful reimplementation.** A faithful CaP-X reproduction would require extracting every MCP server tool as a native Python function. This architecture uses MCP tools internally inside the skills while still hiding them from the planner LLM — a related-but-distinct design point that keeps the architecture portable across MCP server implementations.
- **Planner is intentionally weak by default.** Small open-weights models (e.g. Qwen 3.6 27B AWQ-INT4) are sufficient because the deterministic skills absorb the per-step reasoning load. The planner only picks from three skills and supplies structured arguments.
- **Failures escalate, do not loop.** If a skill fails persistently, the planner returns overall failure rather than retrying indefinitely. This is the inverse of single-agent and multi-agent designs, which can iterate freely on tool errors.
- **Verification is baked into the skills, not the prompt.** Pick verifies via `/gripper/status`. Place verifies via post-release object visibility. These checks are sufficient for many cases but let some false-success cases through; see code comments in `skills/pick.py` and `skills/place.py` for the trade-offs.

## Citing

If you build on this work, please cite the BA thesis it originated from (forthcoming). The CaP-X paper that inspired the design is:

> Fu et al. (2026). *Code as Policies — extended skill library variant.*
