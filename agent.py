"""
Daisy Health — Agent Orchestrator
Uses OpenRouter (Alexis's existing key) to run the agent.

To switch to Anthropic later, change JUST these two things:
1. Set USE_ANTHROPIC = True
2. Add ANTHROPIC_API_KEY to your .env file

Architecture:
    Streamlit UI
         ↓
    agent.py  ← this file (the brain)
         ↓              ↓
    mcp/mcp_server.py   rag_backend.py
         ↓              ↓
    mock_data/       Chroma Cloud
    JSON files       + OpenRouter

Install:
    pip install 'mcp[cli]' openai python-dotenv
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# ─────────────────────────────────────────────
# CONFIGURATION
# Switch between OpenRouter and Anthropic here
# ─────────────────────────────────────────────

# Set to True to use Anthropic (requires ANTHROPIC_API_KEY in .env)
# Set to False to use OpenRouter (requires OPENROUTER_API_KEY in .env)
USE_ANTHROPIC = False

if USE_ANTHROPIC:
    import anthropic
    LLM_MODEL = "claude-sonnet-4-6"
else:
    # OpenRouter uses the OpenAI SDK format
    from openai import OpenAI
    LLM_MODEL = "openai/gpt-oss-20b:free"
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# ─────────────────────────────────────────────
# MCP CLIENT
# ─────────────────────────────────────────────
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

# ─────────────────────────────────────────────
# RAG BACKEND (Alexis's existing code)
# ─────────────────────────────────────────────
try:
    from rag_backend import chat as rag_chat
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
MCP_SERVER_PATH = BASE_DIR / "mcp" / "mcp_server.py"
# Use the same Python executable that is running this script
# This ensures the venv Python is used, not the system Python
PYTHON_EXECUTABLE = sys.executable

MCP_SERVER_PARAMS = StdioServerParameters(
    command=PYTHON_EXECUTABLE,
    args=[str(MCP_SERVER_PATH)],
    env=None,
)

# ─────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are Daisy, the HR assistant for Daisy Health — a virtual primary care company.

Your job is to help employees with HR policy questions and operational tasks.

You have access to tools that let you:
- Look up employee profiles, PTO balances, and benefits from the employee database
- Search HR policy documents for accurate, cited policy information
- Check whether an employee request is compliant with policy
- Create mock HR tickets and draft HR emails

RULES:
1. Always look up the employee profile first with lookup_employee_profile
2. Always search policy documents with search_policy_documents to ground your answer
3. For PTO questions: check balance with check_pto_balance AND search PTO policy
4. For remote work: look up profile AND search remote work policy AND check compliance
5. For expense questions: search expense policy AND check compliance
6. For benefits questions: look up benefits status AND search benefits policy
7. For HR cases/workplace concerns: search conduct policy, create an HR ticket with create_mock_hr_ticket, AND draft an escalation email to People Operations with draft_hr_email — both tools are required for every HR case, not just one
8. Always cite the source policy document in your answer
9. Never make up policy — only use what tools return
10. If you cannot find information, direct to people@daisyhealth.com

Your answers should be warm, professional, grounded in policy, and personalized.
"""



# ─────────────────────────────────────────────
# TOOL CALLING HELPERS
# ─────────────────────────────────────────────
def format_tools_for_openrouter(mcp_tools):
    """
    Convert MCP tool definitions to OpenAI/OpenRouter format.
    OpenRouter expects tools in the OpenAI function-calling format.
"""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            }
        }
        for tool in mcp_tools
    ]



def _create_with_retry(client, max_retries=5, **kwargs):
    """
    Call client.chat.completions.create with retry + exponential backoff.
    Free-tier OpenRouter models share a rate-limited upstream pool and
    frequently return 429s under any kind of burst traffic (e.g. the
    evaluation harness running 25 questions back-to-back).
    """
    import time
    from openai import RateLimitError

    delay = 2
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 30)



def format_tools_for_anthropic(mcp_tools):
    """Convert MCP tool definitions to Anthropic format."""
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
        }
        for tool in mcp_tools
    ]



# ─────────────────────────────────────────────
# MAIN AGENT FUNCTION
# ─────────────────────────────────────────────
async def run_agent(question: str, employee_id: str) -> dict:
    """
    Run the full agentic pipeline.

    Args:
        question: The employee's HR question
        employee_id: The logged-in employee's ID (e.g. EMP-002)

    Returns:
        dict with answer, tool_trace, citations, error
    """
    tool_trace = []
    citations = []
    answer = ""

    try:
        async with stdio_client(MCP_SERVER_PARAMS) as (read, write):
            async with ClientSession(read, write) as session:

                # Initialize MCP connection
                await session.initialize()

                # Get available tools from MCP server
                tools_response = await session.list_tools()
                available_tools = tools_response.tools

                # ── Build the initial user message ──
                user_message = (
                    f"Employee ID: {employee_id}\n\n"
                    f"Question: {question}"
                )

                # ── Choose LLM provider ──
                if USE_ANTHROPIC:
                    answer, tool_trace, citations = await _run_anthropic_agent(
                        session, available_tools,
                        user_message, question, tool_trace, citations
                    )
                else:
                    answer, tool_trace, citations = await _run_openrouter_agent(
                        session, available_tools,
                        user_message, question, tool_trace, citations
                    )

    except Exception as e:
        import traceback
        full_error = traceback.format_exc()
        sys.stderr.write(f"Agent error: {full_error}\n")
        sys.stderr.flush()
        answer = (
            f"I encountered an error while processing your request. "
            f"Please try again or contact people@daisyhealth.com\n\n"
            f"Error: {str(e)}"
        )
        tool_trace.append({
            "tool": "Agent",
            "args": {"question": question},
            "result": str(e)[:200],
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "status": "✗ Error",
        })

    return {
        "answer": answer,
        "tool_trace": tool_trace,
        "citations": citations,
        "error": None,
    }
# ─────────────────────────────────────────────
# OPENROUTER AGENT LOOP
# Uses OpenAI-compatible function calling
# ─────────────────────────────────────────────
async def _run_openrouter_agent(
    session, available_tools,
    user_message, original_question,
    tool_trace, citations
):
    """Run the agentic loop using OpenRouter."""

    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url=OPENROUTER_BASE_URL,
    )

    tools = format_tools_for_openrouter(available_tools)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    answer = ""
    max_iterations = 10  # Prevent infinite loops

    for _ in range(max_iterations):

        response = _create_with_retry(
            client,
            model=LLM_MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        choice = response.choices[0]
        message = choice.message

        # Add assistant response to conversation
        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                }
                for tc in (message.tool_calls or [])
            ] or None,
        })

        # Check if agent wants to call tools
        if choice.finish_reason == "tool_calls" and message.tool_calls:

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_input = json.loads(tool_call.function.arguments)
                timestamp = datetime.now().strftime("%H:%M:%S")

                try:
                    # Call the MCP tool
                    mcp_result = await session.call_tool(tool_name, tool_input)

                    result_text = ""
                    for content in mcp_result.content:
                        if hasattr(content, "text"):
                            result_text += content.text

                    # Record in trace
                    tool_trace.append({
                        "tool": tool_name,
                        "args": tool_input,
                        "result": result_text[:200],
                        "timestamp": timestamp,
                        "status": "✓ Success",
                    })

                    # If policy search — also call Alexis's RAG
                    if tool_name == "search_policy_documents" and RAG_AVAILABLE:
                        try:
                            query = tool_input.get("query", original_question)
                            rag_result = rag_chat(query)
                            rag_docs = rag_result.get("documents", [])

                            for doc in rag_docs:
                                metadata = doc.metadata or {}
                                source = (
                                    metadata.get("source_file")
                                    or metadata.get("source")
                                    or "HR Policy Document"
                                )
                                page = metadata.get("page")
                                section = (
                                    f"Page {page + 1}" if page is not None
                                    else metadata.get("section", "")
                                )
                                snippet = doc.page_content.strip()[:500]
                                citations.append({
                                    "title": source,
                                    "section": section,
                                    "snippet": snippet,
                                    "policy_id": metadata.get("policy_id", ""),
                                })

                            if rag_result.get("answer"):
                                result_text += (
                                    "\n\nRAG context:\n"
                                    + rag_result["answer"]
                                )

                        except Exception as rag_err:
                            tool_trace.append({
                                "tool": "RAG Pipeline",
                                "args": {},
                                "result": f"RAG unavailable: {str(rag_err)[:100]}",
                                "timestamp": timestamp,
                                "status": ":warning: Warning",
                            })

                    # Add tool result to conversation
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    })

                except Exception as tool_err:
                    error_text = f"Tool error: {str(tool_err)}"
                    tool_trace.append({
                        "tool": tool_name,
                        "args": tool_input,
                        "result": error_text[:200],
                        "timestamp": timestamp,
                        "status": "✗ Error",
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": error_text,
                    })

        else:
            # Agent is done — extract final answer
            answer = message.content or ""
            break

    return answer, tool_trace, citations



# ─────────────────────────────────────────────
# ANTHROPIC AGENT LOOP
# Used when USE_ANTHROPIC = True
# ─────────────────────────────────────────────
async def _run_anthropic_agent(
    session, available_tools,
    user_message, original_question,
    tool_trace, citations
):
    """Run the agentic loop using Anthropic Claude."""

    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY")
    )

    tools = format_tools_for_anthropic(available_tools)
    messages = [{"role": "user", "content": user_message}]
    answer = ""

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            messages.append({
                "role": "assistant",
                "content": response.content,
            })

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input
                    timestamp = datetime.now().strftime("%H:%M:%S")

                    try:
                        mcp_result = await session.call_tool(tool_name, tool_input)
                        result_text = ""
                        for content in mcp_result.content:
                            if hasattr(content, "text"):
                                result_text += content.text

                        tool_trace.append({
                            "tool": tool_name,
                            "args": tool_input,
                            "result": result_text[:200],
                            "timestamp": timestamp,
                            "status": "✓ Success",
                        })

                        if tool_name == "search_policy_documents" and RAG_AVAILABLE:
                            try:
                                query = tool_input.get("query", original_question)
                                rag_result = rag_chat(query)
                                for doc in rag_result.get("documents", []):
                                    metadata = doc.metadata or {}
                                    source = metadata.get("source_file") or "HR Policy Document"
                                    page = metadata.get("page")
                                    section = f"Page {page + 1}" if page is not None else metadata.get("section", "")
                                    citations.append({
                                        "title": source,
                                        "section": section,
                                        "snippet": doc.page_content.strip()[:500],
                                        "policy_id": metadata.get("policy_id", ""),
                                    })
                                if rag_result.get("answer"):
                                    result_text += "\n\nRAG context:\n" + rag_result["answer"]
                            except Exception:
                                pass

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                        })

                    except Exception as tool_err:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Tool error: {str(tool_err)}",
                            "is_error": True,
                        })

            messages.append({"role": "user", "content": tool_results})

        else:
            for block in response.content:
                if hasattr(block, "text"):
                    answer += block.text
            break

    return answer, tool_trace, citations



# ─────────────────────────────────────────────
# SYNCHRONOUS WRAPPER
# Streamlit can't run async — use this instead
# ─────────────────────────────────────────────
def run_agent_sync(question: str, employee_id: str) -> dict:
    """
    Call this from Streamlit.

    Example:
        result = run_agent_sync("How much PTO do I have?", "EMP-001")
        print(result["answer"])
    """
    return asyncio.run(run_agent(question, employee_id))



# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print(":blossom: Testing Daisy Health Agent...\n")
    print(f"Using: {'Anthropic Claude' if USE_ANTHROPIC else 'OpenRouter'}\n")

    result = run_agent_sync(
        question="Can I expense a home office chair?",
        employee_id="EMP-002",
    )
    print("Answer:", result["answer"][:400])
    print(f"\nTools called: {len(result['tool_trace'])}")
    for t in result["tool_trace"]:
        print(f"  - {t['tool']}: {t['status']}")
    print(f"Citations: {len(result['citations'])}")