# Skill-based architecture

Third architecture for the BA thesis comparing agentic configurations on long-horizon mobile-manipulation tasks. Inspired by the CaP-X programmatic skill-abstraction pattern (Fu et al., 2026), implemented as a deliberately less-effort variant: deterministic Python skills wrap the canonical MCP tool sequences for `navigate`, `pick`, and `place`, and a small planner LLM (Claude Haiku by default) decides which skill to call next.

## Comparison-axis position

| Architecture | Where the policy lives | Model |
|---|---|---|
| Single-agent | LLM context, raw MCP tool surface | Claude Opus |
| Multi-agent | Orchestrator + 3 sub-agents, narrow MCP subsets | Claude Opus |
| **Skill-based** | **Inside Python skills, hidden from the LLM** | **Claude Haiku** |

The smaller-model + smarter-skills pairing is the design point of this architecture, not a confound.

## Repository layout

```
skill_based/
  main.py                  entry point (--task / --test-{pick,place,navigate})
  planner.py               planner LLM agent + skill dispatch
  mcp_client.py            shared MCP connection manager (copied verbatim from multi_agent)
  llm_client.py            LiteLLM wrapper (Haiku default)
  skills/
    __init__.py
    navigate.py            deterministic nav2 + creeper sequence
    pick.py                deterministic grasp pipeline
    place.py               deterministic release pipeline
    common.py              shared helpers (creeper, AMCL re-seed, gripper-status wait)
  prompts/
    planner.md             planner system prompt
  docs/                    design notes
  .env.example             environment variable template
  README.md                this file
```

## Status

This package is **scaffolded but not yet implemented**. The planner loop, skill dispatch, and MCP client are in place; each skill's `run()` raises `NotImplementedError` and lists the per-step canonical sequence to port from `multi_agent/`. See the TODO comments in each skill file.

## Quick start (once implemented)

```bash
cd /home/ros/rap
cp skill_based/.env.example skill_based/.env
# edit .env with ANTHROPIC_API_KEY

# Single-skill smoke tests (robot must be pre-positioned for pick/place):
python3 -m skill_based.main --test-pick "red coke can"
python3 -m skill_based.main --test-place "trash bin"
python3 -m skill_based.main --test-navigate "kitchen" --mode pick --target-object "wooden coffee table"

# Full planner loop:
python3 -m skill_based.main --task "pick up the coke can on the kitchen table and bring it to the trash bin"
```

## Reuse from `multi_agent/`

This repository is intentionally a separate git repo from `multi_agent/`. Reuse from `multi_agent/` is by copy-paste only, no symlinks or cross-imports, so the two architectures can be evaluated side by side without coupling.

| Source | Destination | Status |
|---|---|---|
| `multi_agent/mcp_client.py` | `mcp_client.py` | copied verbatim |
| `multi_agent/llm_client.py` | `llm_client.py` | copied; Haiku default |
| `multi_agent/navigator.py` `_approach_target` | `skills/common.py::approach_target` | TODO port |
| `multi_agent/navigator.py` `_try_spin_search` | `skills/common.py::spin_search` | TODO port |
| `multi_agent/skills/pick.md` canonical sequence | `skills/pick.py` | TODO translate to Python |
| `multi_agent/skills/place.md` canonical sequence | `skills/place.py` | TODO translate to Python |

## Design decisions

- **No CaP-X reproduction.** A faithful CaP-X reimplementation would require extracting every MCP server tool as a native Python function. That is out of scope for a bachelor thesis. This architecture uses MCP tools internally inside the skills while still hiding them from the planner LLM, which is a related-but-distinct design point.
- **Planner is intentionally weak.** Claude Haiku is chosen because the deterministic skills carry the per-step reasoning load. The planner only needs to pick from three skills and supply structured arguments.
- **Failures escalate, do not loop.** If a skill fails twice in a row, the planner returns overall failure rather than retrying a third time. This is the inverse of single-agent and multi-agent, which can iterate freely.

See `/home/ros/rap/ba26_michal/parts/02_main_matter/methodology.tex` Section 2.1.3 for the full design rationale and the methodology comparison framework.
