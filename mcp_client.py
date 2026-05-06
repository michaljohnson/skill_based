import asyncio
import json
import logging
import re
from contextlib import AsyncExitStack
from pathlib import Path
from datetime import timedelta

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

logger = logging.getLogger(__name__)

MCP_CONFIG_PATH = Path(__file__).parent.parent / ".mcp.json"

# Short names for cleaner tool prefixes
SERVER_SHORT_NAMES = {
    "ros-mcp-server": "ros",
    "moveit-mcp-server": "moveit",
    "nav2-mcp-server": "nav2",
    "perception-mcp-server": "perception",
}

# Per-server tool-call timeout (seconds). nav2 was originally capped
# tight at 45s to short-circuit zombie-goal hangs, but in practice nav2
# BT recovery loops + slow sim push legitimate navigates past 45s,
# triggering false timeouts that the agent then retries → 3+ wasted
# 45s waits per failed nav. Bumped to 90s on 2026-05-04 so legit slow
# nav succeeds first try; true hangs still bail in <2min.
#
# Bumped again to 180s on 2026-05-05 because Gazebo RTF was measured at
# ~0.24 (gz-sim CPU-bound with GUI). Cross-house hops (~8m, e.g. bedroom
# → living room) need ~100-150s wall-clock at that RTF. 180s budget
# absorbs the slow path; true zombie goals still bail in 3min.
# When/if RTF improves (close GUI, free cores), this can be tightened.
DEFAULT_TOOL_TIMEOUT = 120
SERVER_TOOL_TIMEOUTS = {
    "nav2-mcp-server": 180,
}


class MCPClient:
    """Manages connections to all MCP servers and provides unified tool access."""

    def __init__(self, config_path: Path = MCP_CONFIG_PATH):
        self._config_path = config_path
        self._exit_stack = AsyncExitStack()
        self._sessions: dict[str, ClientSession] = {}
        self._tools: dict[str, list] = {}  # server_name -> list of MCP Tool objects

    async def __aenter__(self) -> "MCPClient":
        config = json.loads(self._config_path.read_text())
        servers = config["mcpServers"]

        for server_name, server_config in servers.items():
            url = server_config["url"]
            try:
                transport = await self._exit_stack.enter_async_context(
                    streamablehttp_client(url=url, timeout=30)
                )
                read_stream, write_stream, _ = transport
                session = await self._exit_stack.enter_async_context(
                    ClientSession(read_stream, write_stream)
                )
                await session.initialize()

                result = await session.list_tools()
                self._sessions[server_name] = session
                self._tools[server_name] = result.tools

                short = SERVER_SHORT_NAMES.get(server_name, server_name)
                tool_names = [t.name for t in result.tools]
                logger.info(f"Connected to {server_name} ({short}): {tool_names}")
            except Exception as e:
                logger.error(f"Failed to connect to {server_name} at {url}: {e}")
                raise

        return self

    async def __aexit__(self, *exc):
        await self._exit_stack.aclose()

    async def call_tool(self, server_name: str, tool_name: str, args: dict) -> str:
        """Call a tool on a specific MCP server. Returns result as string."""
        session = self._sessions[server_name]
        timeout = SERVER_TOOL_TIMEOUTS.get(server_name, DEFAULT_TOOL_TIMEOUT)
        result = await session.call_tool(
            tool_name, args, read_timeout_seconds=timedelta(seconds=timeout)
        )
        # Combine all content blocks into a single string
        parts = []
        for block in result.content:
            if hasattr(block, "text"):
                parts.append(block.text)
            else:
                parts.append(str(block))
        return "\n".join(parts)

    async def call_tool_raw(
        self, server_name: str, tool_name: str, args: dict
    ) -> list[dict]:
        """Call a tool and return content as a list of OpenAI-format blocks
        (text + image_url), so images survive into the LLM's tool result.

        Use this for camera / image tools. Other callers should stick to
        `call_tool()` which flattens to a string.
        """
        session = self._sessions[server_name]
        timeout = SERVER_TOOL_TIMEOUTS.get(server_name, DEFAULT_TOOL_TIMEOUT)
        result = await session.call_tool(
            tool_name, args, read_timeout_seconds=timedelta(seconds=timeout)
        )
        blocks: list[dict] = []
        for block in result.content:
            btype = getattr(block, "type", None)
            if btype == "image":
                mime = getattr(block, "mimeType", "image/jpeg")
                data = getattr(block, "data", "")
                blocks.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{data}"},
                })
            elif hasattr(block, "text"):
                blocks.append({"type": "text", "text": block.text})
            else:
                blocks.append({"type": "text", "text": str(block)})
        return blocks

    async def call_tool_prefixed_raw(
        self, prefixed_name: str, args: dict
    ) -> list[dict]:
        """Prefixed-name variant of `call_tool_raw()`."""
        server_name, tool_name = self.route_tool_call(prefixed_name)
        return await self.call_tool_raw(server_name, tool_name, args)

    def route_tool_call(self, prefixed_name: str) -> tuple[str, str]:
        """Route a prefixed tool name to (server_name, tool_name).

        Example: "perception__segment_objects" -> ("perception-mcp-server", "segment_objects")
        """
        short, _, tool_name = prefixed_name.partition("__")
        for server_name, short_name in SERVER_SHORT_NAMES.items():
            if short_name == short:
                return server_name, tool_name
        raise ValueError(f"Unknown server prefix: {short} (from {prefixed_name})")

    # Tools that return an operation ID and need polling
    _ASYNC_TOOLS = {"plan_and_execute", "plan_to_named_state", "plan_to_joint_state"}

    async def _poll_execution(self, server_name: str, operation_id: str,
                               poll_interval: float = 2.0, timeout: float = 120.0) -> str:
        """Poll get_execution_status until the operation completes or fails."""
        elapsed = 0.0
        while elapsed < timeout:
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval
            status_result = await self.call_tool(
                server_name, "get_execution_status",
                {"operation_id": operation_id}
            )
            logger.debug(f"  Poll {operation_id[:8]}...: {status_result[:100]}")
            if "completed" in status_result.lower():
                return status_result
            if "failed" in status_result.lower() and "progress: 0" not in status_result.lower():
                return status_result
            # Check for definitive failure (not just "planning failed" mid-attempt)
            if "error" in status_result.lower() and "status: failed" in status_result.lower():
                return status_result
        return f"Polling timed out after {timeout}s for operation {operation_id}"

    async def call_tool_prefixed(self, prefixed_name: str, args: dict) -> str:
        """Call a tool using its prefixed name (e.g. 'moveit__plan_and_execute').

        For MoveIt async tools (plan_and_execute, etc.), automatically polls
        get_execution_status until the motion completes.
        """
        server_name, tool_name = self.route_tool_call(prefixed_name)
        result = await self.call_tool(server_name, tool_name, args)

        # If this is an async MoveIt tool, poll until done
        if tool_name in self._ASYNC_TOOLS and "Operation ID:" in result:
            match = re.search(r"Operation ID:\s*(\S+)", result)
            if match:
                operation_id = match.group(1)
                logger.info(f"  Waiting for motion {operation_id[:8]}...")
                status = await self._poll_execution(server_name, operation_id)
                return f"{result}\n\n--- Execution Result ---\n{status}"

        return result

    def get_tools(self) -> list[dict]:
        """Convert MCP tool schemas to OpenAI-format tool definitions.

        Returns all tools across all connected servers, prefixed with the
        server short name (e.g. 'perception__look'). Caller-side filtering
        is done with per-agent allowlists (e.g. NAVIGATOR_TOOLS).
        """
        tools = []
        for server_name, server_tools in self._tools.items():
            short = SERVER_SHORT_NAMES.get(server_name, server_name)
            for tool in server_tools:
                tools.append({
                    "type": "function",
                    "function": {
                        "name": f"{short}__{tool.name}",
                        "description": tool.description or "",
                        "parameters": tool.inputSchema,
                    },
                })
        return tools
