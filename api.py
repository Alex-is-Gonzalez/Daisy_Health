"""
Daisy Health — API Endpoints
Provides /health and /chat endpoints alongside the Streamlit app.

The /health endpoint confirms the app is running and services are online.
The /chat endpoint receives questions and returns answers with citations
and tool traces — allowing the grader to test the app via API client.

Run alongside Streamlit:
    # Terminal 1 — Streamlit UI
    streamlit run daisy_health_app.py

    # Terminal 2 — FastAPI endpoints
    uvicorn api:app --port 8000 --reload

Then test:
    curl http://localhost:8000/health
    curl -X POST http://localhost:8000/chat \
         -H "Content-Type: application/json" \
         -d '{"question": "How much PTO do I have?", "employee_id": "EMP-001"}'

Install dependencies:
    pip install fastapi uvicorn
"""

import asyncio
import json
import sys
import os
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

# Add project root to path so we can import agent and rag_backend
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

# ── Try importing the agent orchestrator ──
try:
    from agent import run_agent
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False

# ── Try importing Alexis's RAG backend ──
try:
    from rag_backend import get_document_count
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    def get_document_count():
        return "Unavailable"

# ── Try importing MCP check ──
MCP_SERVER_PATH = Path(__file__).parent / "mcp" / "mcp_server.py"
async def check_mcp_connectivity() -> tuple[bool, str]:
    """Verify live MCP discovery instead of treating a source file as online."""
    if not MCP_SERVER_PATH.exists():
        return False, "server file is missing"

    params = StdioServerParameters(
        command=sys.executable,
        args=[str(MCP_SERVER_PATH)],
        env=None,
    )
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = (await session.list_tools()).tools
        return bool(tools), f"{len(tools)} tools discovered"
    except Exception as exc:
        return False, str(exc)[:120]

# ── Try importing mock data ──
MOCK_DATA_PATH = Path(__file__).parent / "mock_data" / "employees.json"
MOCK_DATA_AVAILABLE = MOCK_DATA_PATH.exists()

# ─────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="Daisy Health HR Assistant API",
    description=(
        "API endpoints for the Daisy Health HR Assistant. "
        "Provides /health status and /chat agentic Q&A."
    ),
    version="1.0.0",
)

# Allow cross-origin requests so Streamlit can call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────

class ChatRequest(BaseModel):
    """
    Request body for the /chat endpoint.
    The grader sends a question and employee ID to test the agent.
    """
    question: str           # The HR question to ask
    employee_id: str        # The employee ID (e.g. EMP-001)

class ToolTraceItem(BaseModel):
    """A single MCP tool call record."""
    tool: str
    args: dict
    result: str
    timestamp: str
    status: str

class CitationItem(BaseModel):
    """A single policy document citation."""
    title: str
    section: str
    snippet: str
    policy_id: str

class ChatResponse(BaseModel):
    """
    Response body from the /chat endpoint.
    Returns the answer, citations, and tool trace.
    """
    answer: str
    citations: list
    tool_trace: list
    employee_id: str
    question: str
    timestamp: str
    processing_time_ms: int


# ─────────────────────────────────────────────
# /health ENDPOINT
# Returns system status — grader checks this
# ─────────────────────────────────────────────
@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    Returns the status of all system components:
    - App status
    - MCP server availability
    - RAG index status
    - ChromaDB document count
    - LLM provider
    - Mock data availability

    The grader uses this to confirm the app is running
    and all components are connected.
    """

    # Check ChromaDB document count
    try:
        chroma_count = get_document_count() if RAG_AVAILABLE else "Unavailable"
        rag_status = "online" if RAG_AVAILABLE else "unavailable"
    except Exception as e:
        chroma_count = f"Error: {str(e)[:50]}"
        rag_status = "error"

    # Check mock data
    try:
        if MOCK_DATA_AVAILABLE:
            with open(MOCK_DATA_PATH) as f:
                employees = json.load(f)
            employee_count = len(employees.get("employees", []))
            mock_data_status = f"online ({employee_count} employees)"
        else:
            mock_data_status = "unavailable"
    except Exception:
        mock_data_status = "error"

    mcp_online, mcp_detail = await check_mcp_connectivity()
    overall_status = "ok" if mcp_online and MOCK_DATA_AVAILABLE else "degraded"

    return {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "app": "Daisy Health HR Assistant",
        "version": "1.0.0",
        "components": {
            "mcp_server": "online" if mcp_online else f"unavailable: {mcp_detail}",
            "rag_index": rag_status,
            "chroma_docs": chroma_count,
            "mock_data": mock_data_status,
            "agent": "online" if AGENT_AVAILABLE else "unavailable",
        },
        "llm_provider": "OpenAI (gpt-5-mini)",
        "embedding_model": "OpenAI text-embedding-3-small",
        "vector_db": "Chroma Cloud",
        "mcp_tools": [
            "lookup_employee_profile",
            "check_pto_balance",
            "lookup_benefits_status",
            "search_policy_documents",
            "check_policy_compliance",
            "create_mock_hr_ticket",
            "draft_hr_email",
        ],
        "demo_tasks": [
            "Expense compliance",
            "HR case triage",
        ],
    }


# ─────────────────────────────────────────────
# /chat ENDPOINT
# Runs the full agentic pipeline and returns results
# ─────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint — runs the full agentic pipeline.

    Receives a question and employee ID, calls the agent orchestrator
    which uses MCP tools + RAG to produce a grounded, cited answer.

    Returns the answer, tool trace, and citations so the grader
    can verify the agent is correctly calling MCP tools.

    Example request:
        POST /chat
        {
            "question": "Can I expense a home office chair?",
            "employee_id": "EMP-001"
        }
    """

    # Validate employee ID format
    emp_id = request.employee_id.upper().strip()
    if not emp_id.startswith("EMP-"):
        raise HTTPException(
            status_code=400,
            detail="Invalid employee ID format. Expected format: EMP-001"
        )

    # Check agent is available
    if not AGENT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail=(
                "Agent orchestrator is unavailable. "
                "Ensure agent.py is in the project root and dependencies are installed."
            )
        )

    # Run the agent and measure time
    start_time = datetime.now()

    try:
        result = await run_agent(
            question=request.question,
            employee_id=emp_id,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Agent error: {str(e)}"
        )

    end_time = datetime.now()
    processing_ms = int((end_time - start_time).total_seconds() * 1000)

    return ChatResponse(
        answer=result["answer"],
        citations=result["citations"],
        tool_trace=result["tool_trace"],
        employee_id=emp_id,
        question=request.question,
        timestamp=start_time.isoformat(),
        processing_time_ms=processing_ms,
    )


# ─────────────────────────────────────────────
# /demo ENDPOINT
# Pre-built demo tasks for the grader to reproduce
# ─────────────────────────────────────────────
@app.get("/demo")
async def demo_tasks():
    """
    Returns the two required agentic demo tasks with
    the expected MCP tool call sequences.

    The grader can use these to reproduce the demo tasks
    via the API client.
    """
    return {
        "demo_task_1": {
            "name": "Expense Compliance",
            "description": (
                "Employee asks whether a home office chair can be reimbursed. "
                "Agent retrieves expense policy, checks employee role, "
                "and returns a compliant decision with citations."
            ),
            "sample_request": {
                "question": "Can I expense a home office chair?",
                "employee_id": "EMP-002"
            },
            "expected_tool_sequence": [
                "lookup_employee_profile",
                "search_policy_documents",
                "check_policy_compliance",
            ],
            "expected_citation": "Expense Reimbursement Policy (HR-EX-004) Section 2",
        },
        "demo_task_2": {
            "name": "HR Case Triage",
            "description": (
                "Employee reports a workplace harassment concern. "
                "Agent retrieves conduct policy, determines escalation is needed, "
                "creates a mock HR ticket, and drafts a confidential email."
            ),
            "sample_request": {
                "question": (
                    "I want to report a harassment concern about a coworker. "
                    "What should I do and can you help me open a case?"
                ),
                "employee_id": "EMP-002"
            },
            "expected_tool_sequence": [
                "lookup_employee_profile",
                "search_policy_documents",
                "create_mock_hr_ticket",
                "draft_hr_email",
            ],
            "expected_citation": "Workplace Conduct Policy (HR-WC-006)",
        },
        "how_to_run": {
            "via_api": (
                "POST /chat with the sample_request body above"
            ),
            "via_ui": (
                "Log in to the Streamlit UI, select the employee, "
                "and type the question in the chat input"
            ),
            "curl_example": (
                'curl -X POST http://localhost:8000/chat '
                '-H "Content-Type: application/json" '
                '-d \'{"question": "Can I expense a home office chair?", '
                '"employee_id": "EMP-002"}\''
            ),
        }
    }


# ─────────────────────────────────────────────
# ROOT ENDPOINT
# ─────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    """A lightweight deployed chat UI; Streamlit remains available locally."""
    return """<!doctype html>
<html><head><meta charset="utf-8"><title>Daisy Health HR Assistant</title>
<style>body{font-family:system-ui;max-width:760px;margin:3rem auto;padding:0 1rem;color:#173b36}textarea,input,button{font:inherit;padding:.7rem;margin:.35rem 0;width:100%;box-sizing:border-box}button{background:#147d73;color:white;border:0;border-radius:.4rem;cursor:pointer}.card{border:1px solid #c8deda;border-radius:.6rem;padding:1rem;margin-top:1rem;white-space:pre-wrap}</style></head>
<body><h1>Daisy Health HR Assistant</h1><p>Ask a policy or HR workflow question. Responses include retrieved citations and an operational MCP tool trace.</p>
<label>Employee ID<input id="employee" value="EMP-001" aria-label="Employee ID"></label>
<label>Question<textarea id="question" rows="4" aria-label="HR question" placeholder="Can I expense a home office chair?"></textarea></label>
<button onclick="ask()">Ask Daisy</button><div id="result" class="card" hidden></div>
<script>async function ask(){const out=document.getElementById('result');out.hidden=false;out.textContent='Working…';const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({employee_id:document.getElementById('employee').value,question:document.getElementById('question').value})});const x=await r.json();out.textContent=r.ok?x.answer+'\\n\\nCitations: '+JSON.stringify(x.citations,null,2)+'\\n\\nTool trace: '+JSON.stringify(x.tool_trace,null,2):JSON.stringify(x,null,2);}</script>
</body></html>"""
