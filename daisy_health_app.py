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
    MCP Tools (called directly as Python functions)
        ↓
    RAG backend (Alexis's rag_backend.py)
        ↓
    Chroma Cloud + OpenRouter

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

# ── Add project root to path ──
sys.path.insert(0, str(Path(__file__).parent))

# ── Import Alexis's RAG backend ──
try:
    from rag_backend import chat as rag_chat, get_document_count
    RAG_AVAILABLE = True
except Exception as e:
    RAG_AVAILABLE = False
    def get_document_count(): return "Unavailable"

# ── Import MCP tool functions from mcp_tools.py ──
# mcp_tools.py contains the tool functions without the MCP server startup
# This avoids the stdio conflict when importing mcp_server.py directly
try:
    sys.path.insert(0, str(Path(__file__).parent / "mcp"))
    from mcp_tools import (
        tool_lookup_employee_profile,
        tool_check_pto_balance,
        tool_lookup_benefits_status,
        tool_search_policy_documents,
        tool_check_policy_compliance,
        tool_create_mock_hr_ticket,
        tool_draft_hr_email,
    )
    MCP_AVAILABLE = True
except Exception as e:
    MCP_AVAILABLE = False

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
# AGENT — calls MCP tools directly + RAG
# ============================================================
def run_agent(question: str, employee_id: str) -> dict:
    """
    Agentic pipeline that calls MCP tools directly as Python functions.
    No subprocess needed — avoids stdio communication issues.
    """
    tool_trace = []
    citations = []
    answer = ""
    q = question.lower()
    now = datetime.now().strftime("%H:%M:%S")

    def trace(tool_name, args, result, status="✓ Success"):
        """Record a tool call in the trace."""
        tool_trace.append({
            "tool": tool_name,
            "args": args,
            "result": str(result)[:200],
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "status": status,
        })
        return result

    emp = MOCK_EMPLOYEES.get(employee_id.upper(), {})
    name = emp.get("name", "").split()[0] if emp else "there"

    try:
        # ── Step 1: Always look up employee profile first ──
        if MCP_AVAILABLE:
            profile = trace(
                "lookup_employee_profile",
                {"employee_id": employee_id},
                tool_lookup_employee_profile(employee_id)
            )
        else:
            profile = f"Employee: {emp.get('name', employee_id)}"

        # ── Step 2: Route to the right workflow ──

        # PTO / leave questions
        if any(w in q for w in ["pto", "time off", "vacation", "leave", "days off", "how much pto"]):
            if MCP_AVAILABLE:
                pto = trace("check_pto_balance", {"employee_id": employee_id},
                           tool_check_pto_balance(employee_id))
            else:
                pto = f"Available: {emp.get('pto_balance', 'N/A')} days"

            if MCP_AVAILABLE:
                policy = trace("search_policy_documents",
                              {"query": "PTO request approval process leave policy", "top_k": 3},
                              tool_search_policy_documents("PTO request approval process leave policy", 3))
            else:
                policy = "PTO policy retrieved"

            # Also call RAG if available
            if RAG_AVAILABLE:
                try:
                    rag_result = rag_chat("PTO request approval process")
                    for doc in rag_result.get("documents", []):
                        meta = doc.metadata or {}
                        citations.append({
                            "title": meta.get("source_file", "PTO and Leave Policy"),
                            "section": f"Page {meta.get('page', 0) + 1}",
                            "snippet": doc.page_content[:300],
                            "policy_id": "HR-PT-001",
                        })
                except Exception:
                    citations.append({
                        "title": "PTO and Leave Policy",
                        "section": "Section 2 — PTO Usage",
                        "snippet": "PTO may be used for any reason. Planned PTO requires at least 5 business days notice.",
                        "policy_id": "HR-PT-001",
                    })
            else:
                citations.append({
                    "title": "PTO and Leave Policy",
                    "section": "Section 2 — PTO Usage",
                    "snippet": "PTO may be used for any reason. Planned PTO requires at least 5 business days notice submitted through the HR Portal.",
                    "policy_id": "HR-PT-001",
                })

            balance = emp.get("pto_balance", "N/A")
            manager = emp.get("manager", "your manager")
            answer = (
                f"Hi {name}! Here's your PTO summary:\n\n"
                f"**Current PTO balance: {balance} days**\n"
                f"Used this year: {emp.get('pto_used', 0)} days\n\n"
                f"**To request time off:**\n"
                f"1. Log in to the HR Portal at hr.daisyhealth.com\n"
                f"2. Navigate to **Time Off → Request Time Off**\n"
                f"3. Select your dates and submit\n\n"
                f"Your manager **{manager}** will respond within 2 business days. "
                f"Please give at least 5 business days notice for planned PTO.\n\n"
                f"*Source: PTO and Leave Policy (HR-PT-001)*"
            )

        # Remote work questions
        elif any(w in q for w in ["remote", "work from", "relocat", "another state", "different state"]):
            if MCP_AVAILABLE:
                compliance = trace("check_policy_compliance",
                                  {"employee_id": employee_id, "policy_area": "remote_work",
                                   "scenario": question},
                                  tool_check_policy_compliance(employee_id, "remote_work", question))
                policy = trace("search_policy_documents",
                              {"query": "remote work eligibility approved states clinical licensure", "top_k": 3},
                              tool_search_policy_documents("remote work eligibility approved states clinical licensure", 3))

            if RAG_AVAILABLE:
                try:
                    rag_result = rag_chat("remote work policy eligibility")
                    for doc in rag_result.get("documents", []):
                        meta = doc.metadata or {}
                        citations.append({
                            "title": meta.get("source_file", "Remote Work Policy"),
                            "section": f"Page {meta.get('page', 0) + 1}",
                            "snippet": doc.page_content[:300],
                            "policy_id": "HR-RW-001",
                        })
                except Exception:
                    pass

            if not citations:
                citations.append({
                    "title": "Remote Work Policy",
                    "section": "Section 2 — Approved Work Locations",
                    "snippet": "Approved states include CA, NY, TX, FL, WA, IL, MA, CO, GA, AZ, VA, OR, PA, OH, NC.",
                    "policy_id": "HR-RW-001",
                })

            is_clinical = emp.get("type") == "Clinical"
            if is_clinical:
                answer = (
                    f"Hi {name}! As a **{emp.get('role', 'clinical employee')}**, remote work from another state involves extra steps.\n\n"
                    f"**Standard rules:**\n"
                    f"- Temporary (up to 4 weeks): notify your manager 5 days in advance\n"
                    f"- Extended (4+ weeks): submit a Location Change Request (up to 30 days to process)\n\n"
                    f"**As clinical staff, you also need:**\n"
                    f"- Active license in the state where your patients are located\n"
                    f"- Notify the **Credentialing team within 5 business days** of relocating\n"
                    f"- Confirm malpractice coverage extends to the new location\n\n"
                    f"**Next step:** Email credentialing@daisyhealth.com before making plans.\n\n"
                    f"*Sources: Remote Work Policy (HR-RW-001) §3, Licensure Policy (HR-LC-009)*"
                )
            else:
                answer = (
                    f"Hi {name}! Here's what you need to know about remote work:\n\n"
                    f"**Temporary (up to 4 weeks):**\n"
                    f"- Must be an approved state\n"
                    f"- Notify **{emp.get('manager', 'your manager')}** at least 5 days in advance\n\n"
                    f"**Extended (4+ weeks):**\n"
                    f"- Submit a Remote Work Location Change Request via HR Portal\n"
                    f"- Allow up to 30 business days for review\n\n"
                    f"**Approved states:** CA, NY, TX, FL, WA, IL, MA, CO, GA, AZ, VA, OR, PA, OH, NC\n\n"
                    f"*Source: Remote Work Policy (HR-RW-001), Sections 2 & 6*"
                )

        # Expense questions
        elif any(w in q for w in ["expense", "reimburse", "stipend", "chair", "laptop", "home office", "internet"]):
            if MCP_AVAILABLE:
                compliance = trace("check_policy_compliance",
                                  {"employee_id": employee_id, "policy_area": "expense",
                                   "scenario": question},
                                  tool_check_policy_compliance(employee_id, "expense", question))
                policy = trace("search_policy_documents",
                              {"query": "expense reimbursement home office stipend", "top_k": 3},
                              tool_search_policy_documents("expense reimbursement home office stipend", 3))

            if RAG_AVAILABLE:
                try:
                    rag_result = rag_chat("expense reimbursement home office stipend")
                    for doc in rag_result.get("documents", []):
                        meta = doc.metadata or {}
                        citations.append({
                            "title": meta.get("source_file", "Expense Reimbursement Policy"),
                            "section": f"Page {meta.get('page', 0) + 1}",
                            "snippet": doc.page_content[:300],
                            "policy_id": "HR-EX-004",
                        })
                except Exception:
                    pass

            if not citations:
                citations.append({
                    "title": "Expense Reimbursement Policy",
                    "section": "Section 2 — Home Office Stipend",
                    "snippet": "Full-time employees receive a one-time $500 home office stipend covering monitor, keyboard, desk, chair, webcam, or headset.",
                    "policy_id": "HR-EX-004",
                })

            answer = (
                f"Hi {name}! Here's what Daisy Health covers for home office expenses:\n\n"
                f"**One-time home office stipend: $500**\n"
                f"Covers: monitor, keyboard, desk, chair, webcam, headset\n"
                f"Submit receipts within 60 days of your hire date via the Expense Portal\n\n"
                f"**Monthly internet reimbursement: up to $50/month**\n"
                f"Submit your internet bill each month through the Expense Portal\n\n"
                f"**Not covered:** personal meals, alcohol, gym memberships, unapproved equipment\n\n"
                f"Submit at: expenses.daisyhealth.com\n\n"
                f"*Source: Expense Reimbursement Policy (HR-EX-004), Sections 2 & 3*"
            )

        # Benefits questions
        elif any(w in q for w in ["benefit", "health plan", "insurance", "hsa", "fsa", "dental", "vision", "401k"]):
            if MCP_AVAILABLE:
                benefits = trace("lookup_benefits_status",
                                {"employee_id": employee_id},
                                tool_lookup_benefits_status(employee_id))
                policy = trace("search_policy_documents",
                              {"query": "health insurance plans HSA FSA enrollment benefits", "top_k": 3},
                              tool_search_policy_documents("health insurance plans HSA FSA enrollment", 3))

            if RAG_AVAILABLE:
                try:
                    rag_result = rag_chat("health insurance benefits enrollment HSA FSA")
                    for doc in rag_result.get("documents", []):
                        meta = doc.metadata or {}
                        citations.append({
                            "title": meta.get("source_file", "Benefits and Insurance Policy"),
                            "section": f"Page {meta.get('page', 0) + 1}",
                            "snippet": doc.page_content[:300],
                            "policy_id": "HR-BI-002",
                        })
                except Exception:
                    pass

            if not citations:
                citations.append({
                    "title": "Benefits and Insurance Policy",
                    "section": "Section 2 — Health Insurance",
                    "snippet": "Three plans available: Bronze ($0/month), Silver ($85/month), Gold ($210/month). Open enrollment every November.",
                    "policy_id": "HR-BI-002",
                })

            plan = emp.get("benefits_plan", "Unknown")
            hsa = emp.get("hsa_eligible", False)
            answer = (
                f"Hi {name}! You're currently enrolled in the **{plan}** plan.\n\n"
                f"**Daisy Health's three health plans:**\n"
                f"| Plan | Premium | Deductible |\n"
                f"|---|---|---|\n"
                f"| Bronze | $0/month | $2,500 |\n"
                f"| Silver | $85/month | $1,000 |\n"
                f"| Gold | $210/month | $250 |\n\n"
                + (f"**HSA:** You're eligible — Daisy Health contributes $500/year. 2025 limit: $4,150.\n\n" if hsa
                   else f"**FSA:** You qualify for a Healthcare FSA (up to $3,050 in 2025).\n\n") +
                f"**Open enrollment:** Every November for January 1 coverage.\n\n"
                f"*Source: Benefits and Insurance Policy (HR-BI-002)*"
            )

        # HR case / harassment / conduct
        elif any(w in q for w in ["harass", "concern", "report", "conduct", "case", "triage", "complaint", "workplace issue"]):
            if MCP_AVAILABLE:
                policy = trace("search_policy_documents",
                              {"query": "harassment reporting workplace conduct HR case escalation", "top_k": 3},
                              tool_search_policy_documents("harassment reporting workplace conduct HR case", 3))
                ticket = trace("create_mock_hr_ticket",
                              {"employee_id": employee_id, "ticket_type": "HR Case",
                               "subject": "Workplace Concern Report",
                               "description": f"Employee reported: {question[:200]}",
                               "priority": "High"},
                              tool_create_mock_hr_ticket(employee_id, "HR Case",
                                  "Workplace Concern Report",
                                  f"Employee reported: {question[:200]}", "High"))
                email_draft = trace("draft_hr_email",
                                   {"employee_id": employee_id, "email_type": "hr_escalation",
                                    "context": question[:200]},
                                   tool_draft_hr_email(employee_id, "hr_escalation", question[:200]))

            if RAG_AVAILABLE:
                try:
                    rag_result = rag_chat("workplace harassment reporting conduct policy")
                    for doc in rag_result.get("documents", []):
                        meta = doc.metadata or {}
                        citations.append({
                            "title": meta.get("source_file", "Workplace Conduct Policy"),
                            "section": f"Page {meta.get('page', 0) + 1}",
                            "snippet": doc.page_content[:300],
                            "policy_id": "HR-WC-006",
                        })
                except Exception:
                    pass

            if not citations:
                citations.append({
                    "title": "Workplace Conduct Policy",
                    "section": "Section 8 — How to Report a Concern",
                    "snippet": "Employees may report concerns to People Operations, their manager, or anonymously via the Ethics Hotline at 1-800-DAISY-ETH.",
                    "policy_id": "HR-WC-006",
                })
                citations.append({
                    "title": "Workplace Conduct Policy",
                    "section": "Section 10 — Non-Retaliation",
                    "snippet": "Daisy Health strictly prohibits retaliation against any employee who reports a concern in good faith.",
                    "policy_id": "HR-WC-006",
                })

            # Extract ticket ID from trace
            ticket_id = "TKT-XXXX"
            for t in tool_trace:
                if t["tool"] == "create_mock_hr_ticket" and "TKT-" in t["result"]:
                    import re as re_module
                    match = re_module.search(r'TKT-\d+', t["result"])
                    if match:
                        ticket_id = match.group()

            answer = (
                f"Hi {name}, I'm sorry you're dealing with this. Your concern is being taken seriously.\n\n"
                f"**I've created an HR case for you:**\n"
                f"- Ticket ID: **{ticket_id}**\n"
                f"- Priority: High\n"
                f"- Assigned to: People Operations\n\n"
                f"**Your reporting options:**\n"
                f"- **People Operations:** people@daisyhealth.com\n"
                f"- **Anonymous Ethics Hotline:** 1-800-DAISY-ETH\n"
                f"- **Online:** ethics.daisyhealth.com\n\n"
                f"**Important:** Daisy Health strictly prohibits retaliation against anyone who reports a concern in good faith.\n\n"
                f"A draft escalation email has been prepared for you — ask me to show it and I'll display it.\n\n"
                f"*Source: Workplace Conduct Policy (HR-WC-006)*"
            )

        # Fallback
        else:
            if MCP_AVAILABLE:
                policy = trace("search_policy_documents",
                              {"query": question, "top_k": 3},
                              tool_search_policy_documents(question, 3))

            if RAG_AVAILABLE:
                try:
                    rag_result = rag_chat(question)
                    rag_answer = rag_result.get("answer", "")
                    for doc in rag_result.get("documents", []):
                        meta = doc.metadata or {}
                        citations.append({
                            "title": meta.get("source_file", "HR Policy Document"),
                            "section": f"Page {meta.get('page', 0) + 1}",
                            "snippet": doc.page_content[:300],
                            "policy_id": "",
                        })
                    if rag_answer:
                        answer = rag_answer
                    else:
                        raise Exception("No RAG answer")
                except Exception:
                    answer = (
                        f"Hi {name}! I searched Daisy Health's policy documents but didn't find "
                        f"a strong match for your question.\n\n"
                        f"Please reach out to:\n"
                        f"- **People Operations:** people@daisyhealth.com\n"
                        f"- **IT Support:** it@daisyhealth.com\n\n"
                        f"I can help with: PTO, remote work, benefits, expenses, onboarding, "
                        f"equipment, holidays, licensure, conduct, or performance."
                    )
            else:
                answer = (
                    f"Hi {name}! I searched Daisy Health's policy documents but didn't find "
                    f"a strong match for your question.\n\n"
                    f"Please reach out to:\n"
                    f"- **People Operations:** people@daisyhealth.com\n"
                    f"- **IT Support:** it@daisyhealth.com"
                )

    except Exception as e:
        answer = (
            f"I encountered an error processing your request. "
            f"Please try again or contact people@daisyhealth.com\n\nError: {str(e)}"
        )
        tool_trace.append({
            "tool": "Agent", "args": {}, "result": str(e)[:200],
            "timestamp": datetime.now().strftime("%H:%M:%S"), "status": "✗ Error",
        })

    return {"answer": answer, "tool_trace": tool_trace, "citations": citations}

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

        # System status
        try:
            chroma_count = get_document_count() if RAG_AVAILABLE else "Unavailable"
            rag_status = "Online" if RAG_AVAILABLE else "Unavailable"
        except Exception:
            chroma_count = "Unavailable"
            rag_status = "Error"

        mcp_status = "Online ✓" if MCP_AVAILABLE else "⚠️ Unavailable"

        st.markdown(f"""
        <div class="system-status">
            <strong>System Status</strong><br>
            <span style="color:#2E7D5E;">● MCP Server</span> &nbsp; {mcp_status}<br>
            <span style="color:#2E7D5E;">● RAG Index</span> &nbsp; {rag_status}<br>
            <span style="color:#2E7D5E;">● Chroma Docs</span> &nbsp; {chroma_count}<br>
            <span style="color:#2E7D5E;">● LLM</span> &nbsp; OpenRouter / Gemma<br>
            <span style="color:#2E7D5E;">● Embeddings</span> &nbsp; all-MiniLM-L6-v2
        </div>
        """, unsafe_allow_html=True)
