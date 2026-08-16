"""
Daisy Health — HR Assistant Web Application
Streamlit dashboard with:
- Employee login (ID + password)
- Teal sidebar with MCP tool trace
- Chat interface with real MCP tool calls
- Policy citations panel
- Real RAG retrieval via rag_backend.py

Architecture:
    Streamlit UI
        ↓
    agent.py — LLM tool-calling loop, MCP client (stdio)
        ↓
    mcp/mcp_server.py — MCP server (MCPServer, @tool)
        ↓                    ↓
    mock_data/ JSON      rag_backend.py → Chroma Cloud + OpenRouter

Run:
    streamlit run daisy_health_app.py

Requires:
    pip install streamlit python-dotenv
    .env with OPENROUTER_API_KEY, CHROMADB_API_KEY,
              CHROMADB_TENANT, CHROMADB_DB
"""

import html
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))


# ── RAG backend ──
try:
    from rag_backend import get_document_count
    RAG_AVAILABLE = True
    RAG_IMPORT_ERROR = None
except Exception as e:
    RAG_AVAILABLE = False
    RAG_IMPORT_ERROR = str(e)

    def get_document_count():
        return "Unavailable"


# ── Agent orchestrator ──
try:
    from agent import run_agent_sync
    AGENT_AVAILABLE = True
    AGENT_IMPORT_ERROR = None
except Exception as e:
    AGENT_AVAILABLE = False
    AGENT_IMPORT_ERROR = str(e)
# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Daisy Health HR Assistant",
    page_icon="🌼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

:root {
    --teal: #147D73;
    --teal-dark: #0F5F58;
    --teal-light: #E8F5F2;
    --sage-light: #F0F7F4;
    --border: #C8DEDA;
    --text: #1C2B2B;
    --muted: #5A7070;
    --white: #FFFFFF;
    --gold: #E8A838;
    --success: #2E7D5E;
    --danger: #B94A48;
}

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', sans-serif;
    background: #FAFCFB;
    color: var(--text);
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="collapsedControl"] { display: none !important; }

.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #147D73 0%, #0F5F58 100%) !important;
    min-width: 280px !important;
}
section[data-testid="stSidebar"] * { color: white !important; }

/* ── Header ── */
.dh-header {
    background: linear-gradient(135deg, #147D73 0%, #0F5F58 100%);
    border-radius: 12px;
    padding: 18px 28px;
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 14px;
}
.dh-header h1 { font-size: 1.6rem; color: white; margin: 0; }
.dh-header .subtitle { color: rgba(255,255,255,0.8); font-size: 0.82rem; margin: 0; }
.dh-daisy { font-size: 2.4rem; line-height: 1; }

/* ── Employee badge ── */
.emp-badge {
    background: var(--sage-light);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 6px 14px;
    font-size: 0.8rem;
    color: var(--teal);
    font-weight: 500;
    margin-bottom: 14px;
    display: inline-block;
}

/* ── Chat messages ── */
.msg-user {
    background: var(--teal);
    color: white;
    border-radius: 16px 16px 4px 16px;
    padding: 12px 16px;
    margin: 8px 0 4px 60px;
    font-size: 0.9rem;
    line-height: 1.5;
}
.msg-assistant {
    background: white;
    color: var(--text);
    border-radius: 16px 16px 16px 4px;
    padding: 14px 18px;
    margin: 8px 60px 4px 0;
    font-size: 0.9rem;
    line-height: 1.6;
    border: 1px solid var(--border);
}
.msg-meta { font-size: 0.72rem; color: var(--muted); margin: 0 4px 10px; }
.msg-meta-right { text-align: right; }

/* ── Tool trace ── */
.trace-card {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 8px;
    padding: 10px 12px;
    margin: 6px 0;
    font-size: 0.76rem;
}
.tool-name {
    font-weight: 600;
    color: #E8A838 !important;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.tool-status { font-size: 0.70rem; color: rgba(255,255,255,0.6) !important; margin-top: 2px; }
.tool-result { color: rgba(255,255,255,0.75) !important; margin-top: 4px; font-size: 0.73rem; line-height: 1.4; }

/* ── Citations ── */
.cite-card {
    background: white;
    border-left: 3px solid var(--teal);
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin: 8px 0;
    font-size: 0.8rem;
}
.cite-title { font-weight: 600; color: var(--teal); }
.cite-section { color: var(--muted); font-size: 0.72rem; margin-top: 2px; }
.cite-snippet { color: var(--text); font-size: 0.78rem; margin-top: 6px; font-style: italic; line-height: 1.4; }

/* ── Section labels ── */
.section-label {
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.6) !important;
    margin: 16px 0 6px;
    padding-bottom: 4px;
    border-bottom: 1px solid rgba(255,255,255,0.1);
}

/* ── Login ── */
.login-error {
    background: #FDF0EF; border: 1px solid #F5C6C2;
    border-radius: 8px; padding: 10px 14px;
    font-size: 0.82rem; color: var(--danger); margin-bottom: 14px;
}
.login-success {
    background: #EAF7F1; border: 1px solid #A8D5C2;
    border-radius: 8px; padding: 10px 14px;
    font-size: 0.82rem; color: var(--success); margin-bottom: 14px;
}
.demo-card {
    background: var(--sage-light); border: 1px solid var(--border);
    border-radius: 10px; padding: 12px 16px;
    font-size: 0.75rem; color: var(--teal); margin-top: 20px; line-height: 1.7;
}
.system-status {
    background: var(--sage-light); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 14px; font-size: 0.75rem; color: var(--text);
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# MOCK EMPLOYEES
# ============================================================
MOCK_EMPLOYEES = {
    "EMP-001": {"name": "Jordan Rivera", "role": "Care Coordinator", "type": "Clinical",
                "department": "Care Operations", "state": "California",
                "manager": "Dr. Priya Anand", "pto_balance": 14.5, "pto_used": 5.5,
                "benefits_plan": "Daisy Silver", "hsa_eligible": False},
    "EMP-002": {"name": "Morgan Chen", "role": "Senior Software Engineer", "type": "Non-Clinical",
                "department": "Engineering", "state": "Washington",
                "manager": "Aisha Thompson", "pto_balance": 18.0, "pto_used": 2.0,
                "benefits_plan": "Daisy Gold", "hsa_eligible": False},
    "EMP-003": {"name": "Dr. Simone Okafor", "role": "Primary Care Physician", "type": "Clinical",
                "department": "Clinical Operations", "state": "New York",
                "manager": "Dr. Priya Anand", "pto_balance": 20.0, "pto_used": 0.0,
                "benefits_plan": "Daisy Bronze", "hsa_eligible": True},
    "EMP-004": {"name": "Alex Nguyen", "role": "Clinical Pharmacist", "type": "Clinical",
                "department": "Pharmacy", "state": "Texas",
                "manager": "Dr. Priya Anand", "pto_balance": 9.0, "pto_used": 11.0,
                "benefits_plan": "Daisy Silver", "hsa_eligible": False},
    "EMP-005": {"name": "Taylor Brooks", "role": "HR Business Partner", "type": "Non-Clinical",
                "department": "People Operations", "state": "Colorado",
                "manager": "Aisha Thompson", "pto_balance": 22.0, "pto_used": 0.0,
                "benefits_plan": "Daisy Gold", "hsa_eligible": False},
}

SUGGESTED_QUESTIONS = [
    "How much PTO do I have left?",
    "Can I work remotely from another state?",
    "What health insurance plans are available?",
    "Can I expense a home office chair?",
    "I want to report a workplace concern.",
    "What is the parental leave policy?",
]

# ============================================================
# SESSION STATE
# ============================================================
def init_session():
    defaults = {
        "logged_in": False,
        "selected_emp_id": None,
        "messages": [],
        "tool_trace": [],
        "citations": [],
        "pending_input": None,
        "password_store": {},
        "login_mode": "login",
        "login_error": "",
        "login_success": "",
        "_setup_emp_id": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()

# ============================================================
# AUTH HELPERS
# ============================================================
def employee_exists(emp_id): return emp_id.upper() in MOCK_EMPLOYEES
def has_password(emp_id): return emp_id.upper() in st.session_state.password_store
def password_matches(emp_id, pw): return st.session_state.password_store.get(emp_id.upper(), "") == pw
def create_password(emp_id, pw): st.session_state.password_store[emp_id.upper()] = pw

def do_login(emp_id):
    st.session_state.logged_in = True
    st.session_state.selected_emp_id = emp_id.upper()
    st.session_state.messages = []
    st.session_state.tool_trace = []
    st.session_state.citations = []
    st.session_state.login_error = ""
    st.session_state.login_success = ""

def do_logout():
    for k in ["logged_in","selected_emp_id","messages","tool_trace","citations",
              "pending_input","login_mode","login_error","login_success"]:
        st.session_state[k] = False if k == "logged_in" else ([] if k in ["messages","tool_trace","citations"] else ("login" if k == "login_mode" else (None if k in ["selected_emp_id","pending_input"] else "")))

# ============================================================
# AGENT — real MCP-based orchestrator (agent.py)
# The LLM decides which MCP tools to call; agent.py spawns
# mcp/mcp_server.py over stdio and invokes tools via MCP's
# ClientSession.call_tool(). This function is a thin adapter that
# runs it synchronously for Streamlit and falls back gracefully.
# ============================================================
def run_agent(question: str, employee_id: str) -> dict:
    if not AGENT_AVAILABLE:
        return {
            "answer": (
                "The agent orchestrator is unavailable right now "
                f"({AGENT_IMPORT_ERROR}). Please contact people@daisyhealth.com."
            ),
            "tool_trace": [{
                "tool": "Agent", "args": {}, "result": AGENT_IMPORT_ERROR,
                "timestamp": datetime.now().strftime("%H:%M:%S"), "status": "✗ Error",
            }],
            "citations": [],
        }

    try:
        result = run_agent_sync(question, employee_id)
    except Exception as e:
        return {
            "answer": (
                "I encountered an error processing your request. "
                f"Please try again or contact people@daisyhealth.com\n\nError: {e}"
            ),
            "tool_trace": [{
                "tool": "Agent", "args": {}, "result": str(e)[:200],
                "timestamp": datetime.now().strftime("%H:%M:%S"), "status": "✗ Error",
            }],
            "citations": [],
        }

    return {
        "answer": result.get("answer", ""),
        "tool_trace": result.get("tool_trace", []),
        "citations": result.get("citations", []),
    }

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style="padding:8px 0 20px;">
        <div style="font-size:1.8rem;">🌼</div>
        <div style="font-size:1.2rem; font-weight:700; color:white; margin-top:4px;">Daisy Health</div>
        <div style="font-size:0.72rem; color:rgba(255,255,255,0.7); margin-top:1px;">HR Assistant</div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.logged_in:
        emp = MOCK_EMPLOYEES[st.session_state.selected_emp_id]

        st.markdown('<div class="section-label">Logged In As</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="trace-card">
            <div class="tool-name" style="font-size:0.70rem;">{html.escape(st.session_state.selected_emp_id)}</div>
            <div style="color:white !important; font-size:0.85rem; font-weight:500; margin:3px 0 6px;">{html.escape(emp['name'])}</div>
            <div class="tool-result">{html.escape(emp['role'])}<br>{html.escape(emp['department'])}<br>📍 {html.escape(emp['state'])}<br>🏥 {html.escape(emp['type'])}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:20px;">MCP Tool Trace</div>', unsafe_allow_html=True)

        if not st.session_state.tool_trace:
            st.markdown('<div style="font-size:0.73rem; color:rgba(255,255,255,0.4); padding:6px 0; font-style:italic;">MCP tool calls appear here as the agent works.</div>', unsafe_allow_html=True)
        else:
            for t in reversed(st.session_state.tool_trace):
                st.markdown(f"""
                <div class="trace-card">
                    <div class="tool-name">{html.escape(str(t['tool']))}</div>
                    <div class="tool-status">{html.escape(str(t['status']))} · {html.escape(str(t['timestamp']))}</div>
                    <div class="tool-result">{html.escape(str(t['result']))}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
        if st.button("🗑 Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.tool_trace = []
            st.session_state.citations = []
            st.rerun()
        if st.button("🚪 Sign Out", use_container_width=True):
            do_logout()
            st.rerun()
    else:
        st.markdown("""
        <div style="margin-top:16px; font-size:0.82rem; color:rgba(255,255,255,0.75); line-height:1.6;">
            Welcome to the Daisy Health HR Self-Service Portal.<br><br>
            Sign in with your <strong style="color:white;">Employee ID</strong> and password.
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# LOGIN PAGE
# ============================================================
if not st.session_state.logged_in:

    st.markdown("""
    <div class="dh-header">
        <div class="dh-daisy">🌼</div>
        <div>
            <h1>Daisy Health HR Assistant</h1>
            <p class="subtitle">Sign in to get personalized answers about PTO, benefits, remote work, and more.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, card_col, _ = st.columns([1, 1.4, 1])

    with card_col:
        if st.session_state.login_mode == "create_password":
            if st.session_state.login_success:
                st.markdown(f'<div class="login-success">✓ {html.escape(st.session_state.login_success)}</div>', unsafe_allow_html=True)
            if st.session_state.login_error:
                st.markdown(f'<div class="login-error">⚠️ {html.escape(st.session_state.login_error)}</div>', unsafe_allow_html=True)

            setup_id = st.session_state._setup_emp_id
            setup_emp = MOCK_EMPLOYEES[setup_id]
            st.markdown(f"""
            <div style="font-size:1.1rem; font-weight:600; color:#147D73; margin-bottom:4px;">Create your password</div>
            <div style="font-size:0.8rem; color:#5A7070; margin-bottom:16px;">
                First time logging in, {html.escape(setup_emp['name'].split()[0])}! Create a secure password.
            </div>
            <div style="background:#F0F7F4; border-radius:8px; padding:10px 14px; font-size:0.82rem; color:#147D73; margin-bottom:16px;">
                Setting up: <strong>{html.escape(setup_id)}</strong> — {html.escape(setup_emp['name'])}
            </div>
            """, unsafe_allow_html=True)

            new_pw = st.text_input("New password", type="password", placeholder="Min 6 characters", key="new_pw")
            confirm_pw = st.text_input("Confirm password", type="password", placeholder="Re-enter password", key="confirm_pw")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("← Back", use_container_width=True):
                    st.session_state.login_mode = "login"
                    st.session_state.login_error = ""
                    st.rerun()
            with col_b:
                if st.button("Create password", use_container_width=True, type="primary"):
                    if len(new_pw) < 6:
                        st.session_state.login_error = "Password must be at least 6 characters."
                        st.rerun()
                    elif new_pw != confirm_pw:
                        st.session_state.login_error = "Passwords do not match."
                        st.rerun()
                    else:
                        create_password(setup_id, new_pw)
                        do_login(setup_id)
                        st.rerun()
        else:
            if st.session_state.login_error:
                st.markdown(f'<div class="login-error">⚠️ {html.escape(st.session_state.login_error)}</div>', unsafe_allow_html=True)

            st.markdown("""
            <div style="font-size:1.1rem; font-weight:600; color:#147D73; margin-bottom:4px;">Sign in to your account</div>
            <div style="font-size:0.8rem; color:#5A7070; margin-bottom:20px;">Use your Employee ID and personal password.</div>
            """, unsafe_allow_html=True)

            emp_id_input = st.text_input("Employee ID", placeholder="e.g. EMP-001", key="login_emp_id").strip().upper()
            password_input = st.text_input("Password", type="password", placeholder="Enter your password", key="login_password")

            if st.button("Sign in →", use_container_width=True, type="primary"):
                st.session_state.login_error = ""
                if not emp_id_input:
                    st.session_state.login_error = "Please enter your Employee ID."
                    st.rerun()
                elif not employee_exists(emp_id_input):
                    st.session_state.login_error = f"Employee ID '{emp_id_input}' not found."
                    st.rerun()
                elif not has_password(emp_id_input):
                    st.session_state._setup_emp_id = emp_id_input
                    st.session_state.login_mode = "create_password"
                    st.session_state.login_success = f"Welcome, {MOCK_EMPLOYEES[emp_id_input]['name'].split()[0]}! Please create your password."
                    st.rerun()
                elif not password_matches(emp_id_input, password_input):
                    st.session_state.login_error = "Incorrect password. Please try again."
                    st.rerun()
                else:
                    do_login(emp_id_input)
                    st.rerun()

            st.markdown("""
            <div style="font-size:0.72rem; color:#8A9BAA; text-align:center; margin-top:14px; line-height:1.5;">
                First time? Enter your Employee ID and click Sign in to create your password.<br>
                Need help? Contact people@daisyhealth.com
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="demo-card">
            <strong>Demo Employee IDs:</strong><br>
            EMP-001 — Jordan Rivera (Care Coordinator)<br>
            EMP-002 — Morgan Chen (Software Engineer)<br>
            EMP-003 — Dr. Simone Okafor (Physician)<br>
            EMP-004 — Alex Nguyen (Clinical Pharmacist)<br>
            EMP-005 — Taylor Brooks (HR Business Partner)
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# MAIN DASHBOARD
# ============================================================
else:
    emp = MOCK_EMPLOYEES[st.session_state.selected_emp_id]

    st.markdown("""
    <div class="dh-header">
        <div class="dh-daisy">🌼</div>
        <div>
            <h1>Daisy Health HR Assistant</h1>
            <p class="subtitle">Ask questions about PTO, remote work, benefits, expenses, and more. Answers are grounded in Daisy Health policy documents.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_chat, col_cite = st.columns([2, 1], gap="large")

    with col_chat:
        st.markdown(f"""
        <div class="emp-badge">
            👤 {html.escape(emp['name'])} &nbsp;·&nbsp; {html.escape(emp['role'])} &nbsp;·&nbsp; {emp['pto_balance']} days PTO available
        </div>
        """, unsafe_allow_html=True)

        # Chat history
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f'<div class="msg-user">{html.escape(msg["content"])}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="msg-meta msg-meta-right">{msg["time"]}</div>', unsafe_allow_html=True)
            else:
                with st.chat_message("assistant", avatar="🌼"):
                    st.markdown(msg["content"])
                    st.caption(msg["time"])

        # Suggestion chips
        if not st.session_state.messages:
            st.markdown('<div style="margin:16px 0 8px; font-size:0.8rem; color:#5A7070;"><strong>Try asking:</strong></div>', unsafe_allow_html=True)
            chip_cols = st.columns(2)
            for i, s in enumerate(SUGGESTED_QUESTIONS):
                with chip_cols[i % 2]:
                    if st.button(s, key=f"chip_{i}", use_container_width=True):
                        st.session_state.pending_input = s
                        st.rerun()

        # Chat input
        user_input = st.chat_input(f"Ask an HR question, {emp['name'].split()[0]}…")
        if st.session_state.pending_input:
            user_input = st.session_state.pending_input
            st.session_state.pending_input = None

        if user_input:
            now = datetime.now().strftime("%I:%M %p")
            st.session_state.messages.append({"role": "user", "content": user_input, "time": now})

            with st.spinner("🌼 Agent working — calling MCP tools and searching policy documents..."):
                result = run_agent(user_input, st.session_state.selected_emp_id)

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "time": datetime.now().strftime("%I:%M %p"),
            })
            st.session_state.tool_trace.extend(result["tool_trace"])
            st.session_state.citations = result["citations"]
            st.rerun()

    with col_cite:
        st.markdown("""
        <div style="font-size:0.78rem; font-weight:600; color:#1C2B2B; letter-spacing:0.06em;
                    text-transform:uppercase; margin-bottom:10px; padding-bottom:6px;
                    border-bottom:2px solid #C8DEDA;">
            📄 Policy Citations
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state.citations:
            st.markdown("""
            <div style="text-align:center; padding:30px 10px; color:#5A7070;">
                <div style="font-size:2rem; margin-bottom:8px;">📋</div>
                <p style="font-size:0.78rem;">Policy documents cited by the agent will appear here after you ask a question.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            for cite in st.session_state.citations:
                title = html.escape(str(cite.get("title", "HR Policy")))
                section = html.escape(str(cite.get("section", "")))
                pid = html.escape(str(cite.get("policy_id", "")))
                snippet = html.escape(str(cite.get("snippet", "")))
                meta = f"{pid} · {section}" if pid and section else section or pid
                st.markdown(f"""
                <div class="cite-card">
                    <div class="cite-title">📄 {title}</div>
                    <div class="cite-section">{meta}</div>
                    <div class="cite-snippet">"{snippet}"</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Real system status check ──
        # Cache in session state so it doesn't re-check on every rerun
        if "system_status" not in st.session_state:
            try:
                from health_check import get_system_status, render_status_html
                st.session_state.system_status = get_system_status()
                st.session_state.status_html = render_status_html(
                    st.session_state.system_status
                )
            except Exception as e:
                st.session_state.status_html = (
                    f'<div class="system-status">'
                    f'<strong>System Status</strong><br>'
                    f'<span style="color:#f87171;">● Health check unavailable</span>'
                    f'</div>'
                )

        st.markdown(
            st.session_state.get("status_html", ""),
            unsafe_allow_html=True
        )

        # Refresh button
        if st.button("↻ Refresh Status", use_container_width=True):
            # Clear cached status to force re-check
            for k in ["system_status", "status_html"]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()
