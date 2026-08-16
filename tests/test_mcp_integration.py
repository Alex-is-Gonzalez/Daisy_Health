"""Live MCP protocol tests required by the project rubric."""

import asyncio
import sys
import unittest
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from agent import build_tool_arguments, get_schema_properties


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_TOOLS = {
    "lookup_employee_profile",
    "check_pto_balance",
    "lookup_benefits_status",
    "search_policy_documents",
    "check_policy_compliance",
    "create_mock_hr_ticket",
    "draft_hr_email",
}


class MCPIntegrationTests(unittest.TestCase):
    def test_server_discovers_tools_and_calls_structured_data_tool(self):
        async def exercise_server():
            params = StdioServerParameters(
                command=sys.executable,
                args=[str(ROOT / "mcp" / "mcp_server.py")],
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    names = {tool.name for tool in tools.tools}
                    result = await session.call_tool(
                        "lookup_employee_profile", {"employee_id": "EMP-001"}
                    )
            text = "\n".join(getattr(item, "text", "") for item in result.content)
            return names, text, tools.tools

        names, text, discovered_tools = asyncio.run(exercise_server())
        self.assertSetEqual(names, REQUIRED_TOOLS)
        self.assertIn("Jordan Rivera", text)

        self.assertSetEqual(
            get_schema_properties(discovered_tools, "lookup_employee_profile"),
            {"employee_id"},
        )
        self.assertEqual(
            build_tool_arguments(
                "lookup_employee_profile",
                "test question",
                "EMP-001",
                "general",
                discovered_tools,
            ),
            {"employee_id": "EMP-001"},
        )
        self.assertEqual(
            build_tool_arguments(
                "create_mock_hr_ticket",
                "I need help with a workplace concern",
                "EMP-001",
                "hr_case",
                discovered_tools,
            ),
            {
                "employee_id": "EMP-001",
                "ticket_type": "workplace_concern",
                "subject": "Workplace concern",
                "description": "I need help with a workplace concern",
            },
        )
