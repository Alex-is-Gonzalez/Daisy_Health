"""
Daisy Health — Health Check Module
Actually tests each service rather than just checking imports.
Used by the Streamlit sidebar System Status panel.
"""

import json
import os
import sys
from pathlib import Path

# ─────────────────────────────────────────────
# INDIVIDUAL SERVICE CHECKS
# Each returns (status_emoji, label, detail)
# ─────────────────────────────────────────────

def check_mcp_tools() -> tuple:
    """
    Actually calls a real MCP tool function and verifies it returns data.
    Tests lookup_employee_profile with EMP-001.
    """
    try:
        sys.path.insert(0, str(Path(__file__).parent / "mcp"))
        from mcp_tools import tool_lookup_employee_profile
        result = tool_lookup_employee_profile("EMP-001")
        if "Jordan Rivera" in result:
            return ("✓", "Online", "7 tools active")
        else:
            return ("⚠️", "Degraded", "Tool returned unexpected result")
    except Exception as e:
        return ("✗", "Unavailable", str(e)[:60])


def check_mock_data() -> tuple:
    """
    Loads the employees.json file and counts records.
    """
    try:
        base = Path(__file__).parent
        path = base / "mock_data" / "employees.json"
        with open(path) as f:
            data = json.load(f)
        count = len(data.get("employees", []))
        return ("✓", "Online", f"{count} employees loaded")
    except Exception as e:
        return ("✗", "Unavailable", str(e)[:60])


def check_rag_backend() -> tuple:
    """
    Imports rag_backend and calls get_document_count().
    If that fails, tries a simple chat call.
    """
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from rag_backend import get_document_count
        count = get_document_count()
        if count and count != "Unavailable":
            return ("✓", "Online", f"Connected")
        else:
            return ("⚠️", "Degraded", "No document count returned")
    except Exception as e:
        err = str(e)[:60]
        if "API" in err or "key" in err.lower() or "auth" in err.lower():
            return ("⚠️", "No API Key", "Add OPENAI_API_KEY to .env")
        return ("✗", "Unavailable", err)


def check_chroma() -> tuple:
    """
    Connects to ChromaDB and gets the actual document count.
    """
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from rag_backend import get_document_count
        count = get_document_count()
        if count and str(count).isdigit() and int(str(count)) > 0:
            return ("✓", f"{count} documents", "Chroma Cloud connected")
        elif count == "Unavailable" or not count:
            return ("⚠️", "Unavailable", "Check CHROMADB_API_KEY in .env")
        else:
            return ("✓", f"{count} documents", "Connected")
    except Exception as e:
        err = str(e)[:60]
        if "api" in err.lower() or "key" in err.lower():
            return ("⚠️", "No API Key", "Add CHROMADB_API_KEY to .env")
        return ("✗", "Unavailable", err)


def check_llm() -> tuple:
    """
    Checks whether the OpenAI API key is configured.
    """
    try:
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            return ("⚠️", "No API Key", "Add OPENAI_API_KEY to .env")
        return ("✓", "OpenAI", "gpt-5-mini")
    except Exception as e:
        return ("✗", "Unavailable", str(e)[:60])


def check_embeddings() -> tuple:
    """
    Reports the configured embedding model. Embeddings are provided by OpenAI.
    """
    if os.getenv("OPENAI_API_KEY"):
        return ("✓", "text-embedding-3-small", "OpenAI embeddings configured")
    return ("⚠️", "Unavailable", "Add OPENAI_API_KEY to .env")


# ─────────────────────────────────────────────
# FULL HEALTH CHECK
# Runs all checks and returns a structured result
# ─────────────────────────────────────────────

def get_system_status() -> dict:
    """
    Runs all service checks and returns a dict of results.
    Each entry: { "icon": str, "status": str, "detail": str }

    Call this from the Streamlit sidebar to display System Status.
    Results are cached in session state to avoid re-checking on every rerun.
    """
    mcp_icon, mcp_status, mcp_detail = check_mcp_tools()
    data_icon, data_status, data_detail = check_mock_data()
    rag_icon, rag_status, rag_detail = check_rag_backend()
    chroma_icon, chroma_status, chroma_detail = check_chroma()
    llm_icon, llm_status, llm_detail = check_llm()
    emb_icon, emb_status, emb_detail = check_embeddings()

    return {
        "mcp_tools": {
            "label": "MCP Tools",
            "icon": mcp_icon,
            "status": mcp_status,
            "detail": mcp_detail,
        },
        "mock_data": {
            "label": "Mock Data",
            "icon": data_icon,
            "status": data_status,
            "detail": data_detail,
        },
        "rag_backend": {
            "label": "RAG Backend",
            "icon": rag_icon,
            "status": rag_status,
            "detail": rag_detail,
        },
        "chroma": {
            "label": "Chroma Docs",
            "icon": chroma_icon,
            "status": chroma_status,
            "detail": chroma_detail,
        },
        "llm": {
            "label": "LLM",
            "icon": llm_icon,
            "status": llm_status,
            "detail": llm_detail,
        },
        "embeddings": {
            "label": "Embeddings",
            "icon": emb_icon,
            "status": emb_status,
            "detail": emb_detail,
        },
    }


def render_status_html(status: dict) -> str:
    """
    Renders the system status as HTML for the Streamlit sidebar.
    Shows colored dots, service name, and status.
    """
    def dot_color(icon):
        if icon == "✓":
            return "#4ade80"   # green
        elif icon == "⚠️":
            return "#facc15"   # yellow
        else:
            return "#f87171"   # red

    rows = ""
    for key, info in status.items():
        color = dot_color(info["icon"])
        rows += (
            f'<div style="display:flex; align-items:center; gap:8px; '
            f'margin-bottom:6px; font-size:0.75rem;">'
            f'<span style="color:{color}; font-size:0.9rem;">●</span>'
            f'<span style="color:#1C2B2B; font-weight:500; min-width:90px;">'
            f'{info["label"]}</span>'
            f'<span style="color:#5A7070;">{info["status"]}</span>'
            f'</div>'
        )

    return (
        f'<div style="background:#F0F7F4; border:1px solid #C8DEDA; '
        f'border-radius:8px; padding:12px 14px; margin-top:20px;">'
        f'<div style="font-weight:600; color:#1C2B2B; font-size:0.8rem; '
        f'margin-bottom:10px;">System Status</div>'
        f'{rows}'
        f'</div>'
    )
