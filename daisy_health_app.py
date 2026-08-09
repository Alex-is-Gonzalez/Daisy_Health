"""
Daisy Health — HR Assistant Web Application

Streamlit dashboard with:

- Employee login
- Teal sidebar
- HR chat interface
- Real Chroma RAG retrieval
- OpenRouter LLM responses
- Real document citations
- RAG tool trace
- Employee demo profile

Run:

    streamlit run daisy_health_app.py

Requires:

    streamlit
    rag_backend.py
    .env
    Chroma Cloud credentials
    OpenRouter credentials
"""

import html
from datetime import datetime

import streamlit as st

from rag_backend import chat, get_document_count


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
#
# IMPORTANT:
# This MUST be wrapped in <style>...</style>.
# Do NOT put Markdown code fences around the CSS.
# ============================================================

st.markdown(
    """
    <style>

    /* ======================================================
       COLOR SYSTEM
    ====================================================== */

    :root {
        --teal: #147D73;
        --teal-dark: #0F5F58;
        --teal-light: #E8F5F2;
        --sage-light: #F0F7F4;
        --border: #C8DEDA;
        --text: #1C2B2B;
        --muted: #5A7070;
        --white: #FFFFFF;
        --success: #2E7D5E;
        --danger: #B94A48;
    }


    /* ======================================================
       GENERAL APP
    ====================================================== */

    .stApp {
        background: #FAFCFB;
    }

    [data-testid="stAppViewContainer"] {
        background: #FAFCFB;
    }

    [data-testid="stMain"] {
        background: #FAFCFB;
    }

    .main {
        padding-top: 1rem;
    }

    /* Remove excess top spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }


    /* ======================================================
       SIDEBAR
    ====================================================== */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #147D73 0%,
            #0F5F58 100%
        );
    }

    section[data-testid="stSidebar"] > div {
        background: transparent;
    }

    section[data-testid="stSidebar"] * {
        color: white;
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }

    .sidebar-brand {
        padding: 8px 0 20px 0;
    }

    .sidebar-brand-icon {
        font-size: 2rem;
        margin-bottom: 4px;
    }

    .sidebar-brand-name {
        font-size: 1.2rem;
        font-weight: 700;
        color: white !important;
    }

    .sidebar-brand-subtitle {
        font-size: 0.78rem;
        color: rgba(255,255,255,0.72) !important;
        margin-top: 2px;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 999px;
        padding: 5px 10px;
        font-size: 0.72rem;
        color: white !important;
    }

    .status-dot {
        width: 7px;
        height: 7px;
        background: #7EE2A8;
        border-radius: 50%;
        display: inline-block;
    }

    .section-label {
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: rgba(255,255,255,0.55) !important;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .trace-card {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 8px;
    }

    .tool-name {
        color: white !important;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .tool-status {
        color: rgba(255,255,255,0.55) !important;
        font-size: 0.67rem;
        margin-top: 2px;
    }

    .tool-result {
        color: rgba(255,255,255,0.72) !important;
        font-size: 0.72rem;
        line-height: 1.45;
        margin-top: 5px;
    }


    /* ======================================================
       SIDEBAR BUTTONS
    ====================================================== */

    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255,255,255,0.10);
        color: white !important;
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: 8px;
        font-weight: 500;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255,255,255,0.18);
        border-color: rgba(255,255,255,0.30);
        color: white !important;
    }


    /* ======================================================
       MAIN HEADER
    ====================================================== */

    .dh-header {
        display: flex;
        align-items: center;
        gap: 16px;
        background: white;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(20,125,115,0.04);
    }

    .dh-daisy {
        width: 52px;
        height: 52px;
        min-width: 52px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: var(--teal-light);
        border-radius: 12px;
        font-size: 1.8rem;
    }

    .dh-header h1 {
        margin: 0;
        color: var(--text) !important;
        font-size: 1.5rem;
        font-weight: 700;
    }

    .subtitle {
        margin: 4px 0 0 0;
        color: var(--muted) !important;
        font-size: 0.85rem;
        line-height: 1.5;
    }


    /* ======================================================
       LOGIN
    ====================================================== */

    .login-card-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text) !important;
        margin-bottom: 5px;
    }

    .login-card-sub {
        font-size: 0.82rem;
        color: var(--muted) !important;
        line-height: 1.5;
        margin-bottom: 18px;
    }

    .login-error {
        background: #FFF0EF;
        border: 1px solid #E8B7B5;
        color: var(--danger) !important;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 15px;
        font-size: 0.8rem;
    }

    .login-success {
        background: #EDF8F2;
        border: 1px solid #B8DCC6;
        color: var(--success) !important;
        border-radius: 8px;
        padding: 10px 12px;
        margin-bottom: 15px;
        font-size: 0.8rem;
    }

    .login-hint {
        text-align: center;
        color: var(--muted) !important;
        font-size: 0.72rem;
        line-height: 1.5;
        margin-top: 18px;
    }

    .demo-card {
        background: var(--sage-light);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 12px 14px;
        margin-top: 22px;
        color: var(--muted) !important;
        font-size: 0.72rem;
        line-height: 1.65;
    }


    /* ======================================================
       INPUTS
    ====================================================== */

    .stTextInput label,
    .stTextInput p {
        color: var(--text) !important;
    }

    .stTextInput input {
        background: white !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }

    .stTextInput input:focus {
        border-color: var(--teal) !important;
        box-shadow: 0 0 0 1px var(--teal) !important;
    }


    /* ======================================================
       EMPLOYEE BADGE
    ====================================================== */

    .emp-badge {
        background: var(--sage-light);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 9px 13px;
        color: var(--text) !important;
        font-size: 0.78rem;
        margin-bottom: 16px;
    }


    /* ======================================================
       CHAT
    ====================================================== */

    .msg-user {
        background: var(--teal);
        color: white !important;
        border-radius: 12px 12px 3px 12px;
        padding: 11px 14px;
        margin: 12px 0 3px auto;
        max-width: 82%;
        width: fit-content;
        font-size: 0.87rem;
        line-height: 1.5;
    }

    .msg-assistant {
        background: white;
        border: 1px solid var(--border);
        color: var(--text) !important;
        border-radius: 12px 12px 12px 3px;
        padding: 13px 15px;
        margin: 12px auto 3px 0;
        max-width: 92%;
        font-size: 0.87rem;
        line-height: 1.6;
        white-space: normal;
    }

    .msg-meta {
        color: #819292 !important;
        font-size: 0.64rem;
        margin-bottom: 8px;
    }

    .msg-meta-right {
        text-align: right;
    }


    /* ======================================================
       STREAMLIT CHAT MESSAGE
    ====================================================== */

    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        padding-left: 0 !important;
        padding-right: 0 !important;
    }

    [data-testid="stChatMessageContent"] {
        background: white;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 13px 15px;
        color: var(--text);
    }

    [data-testid="stChatMessageContent"] p {
        color: var(--text);
    }

    [data-testid="stChatMessageContent"] li {
        color: var(--text);
    }


    /* ======================================================
       CHAT INPUT
    ====================================================== */

    [data-testid="stChatInput"] {
        background: white;
    }

    [data-testid="stChatInput"] textarea {
        color: var(--text) !important;
    }


    /* ======================================================
       CITATIONS
    ====================================================== */

    .cite-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 9px;
        padding: 12px;
        margin-bottom: 10px;
    }

    .cite-title {
        color: var(--text) !important;
        font-size: 0.78rem;
        font-weight: 700;
        line-height: 1.4;
    }

    .cite-section {
        color: var(--teal) !important;
        font-size: 0.68rem;
        margin-top: 3px;
    }

    .cite-snippet {
        color: var(--muted) !important;
        font-size: 0.72rem;
        line-height: 1.5;
        margin-top: 8px;
        font-style: italic;
    }

    .empty-state {
        background: var(--sage-light);
        border: 1px dashed var(--border);
        border-radius: 9px;
        text-align: center;
        color: var(--muted) !important;
    }

    .empty-state .icon {
        font-size: 1.6rem;
        margin-bottom: 4px;
    }


    /* ======================================================
       BUTTONS
    ====================================================== */

    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
    }

    .stButton > button[kind="primary"] {
        background: var(--teal);
        border-color: var(--teal);
        color: white !important;
    }

    .stButton > button[kind="primary"]:hover {
        background: var(--teal-dark);
        border-color: var(--teal-dark);
        color: white !important;
    }


    /* ======================================================
       STATUS CARD
    ====================================================== */

    .system-status {
        background: #F0F7F4;
        border: 1px solid #C8DEDA;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.75rem;
        color: #1C2B2B;
        line-height: 1.7;
    }


    /* ======================================================
       RESPONSIVE
    ====================================================== */

    @media (max-width: 900px) {
        .dh-header {
            padding: 14px 16px;
        }

        .dh-header h1 {
            font-size: 1.25rem;
        }

        .dh-daisy {
            width: 44px;
            height: 44px;
            min-width: 44px;
            font-size: 1.5rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DEMO EMPLOYEE DATA
# ============================================================

MOCK_EMPLOYEES = {
    "EMP-001": {
        "name": "Jordan Rivera",
        "role": "Care Coordinator",
        "type": "Clinical",
        "department": "Care Operations",
        "state": "California",
        "manager": "Dr. Priya Anand",
        "pto_balance": 14.5,
        "pto_used": 5.5,
        "benefits_plan": "Daisy Silver",
        "hsa_eligible": False,
        "license_state": "California",
    },
    "EMP-002": {
        "name": "Morgan Chen",
        "role": "Senior Software Engineer",
        "type": "Non-Clinical",
        "department": "Engineering",
        "state": "Washington",
        "manager": "Aisha Thompson",
        "pto_balance": 18.0,
        "pto_used": 2.0,
        "benefits_plan": "Daisy Gold",
        "hsa_eligible": False,
        "license_state": None,
    },
    "EMP-003": {
        "name": "Dr. Simone Okafor",
        "role": "Primary Care Physician",
        "type": "Clinical",
        "department": "Clinical Operations",
        "state": "New York",
        "manager": "Dr. Priya Anand",
        "pto_balance": 20.0,
        "pto_used": 0.0,
        "benefits_plan": "Daisy Bronze",
        "hsa_eligible": True,
        "license_state": "New York",
    },
    "EMP-004": {
        "name": "Alex Nguyen",
        "role": "Clinical Pharmacist",
        "type": "Clinical",
        "department": "Pharmacy",
        "state": "Texas",
        "manager": "Dr. Priya Anand",
        "pto_balance": 9.0,
        "pto_used": 11.0,
        "benefits_plan": "Daisy Silver",
        "hsa_eligible": False,
        "license_state": "Texas",
    },
    "EMP-005": {
        "name": "Taylor Brooks",
        "role": "HR Business Partner",
        "type": "Non-Clinical",
        "department": "People Operations",
        "state": "Colorado",
        "manager": "Aisha Thompson",
        "pto_balance": 22.0,
        "pto_used": 0.0,
        "benefits_plan": "Daisy Gold",
        "hsa_eligible": False,
        "license_state": None,
    },
}


# ============================================================
# SUGGESTED QUESTIONS
# ============================================================

SUGGESTED_QUESTIONS = [
    "What does the PTO policy say?",
    "Can I work remotely from another state?",
    "What health insurance plans are available?",
    "How do I submit an expense report?",
    "When is open enrollment?",
    "What is the parental leave policy?",
]


# ============================================================
# SESSION STATE
# ============================================================

def init_session():
    """Initialize all Streamlit session variables."""

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

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session()


# ============================================================
# AUTH HELPERS
# ============================================================

def employee_exists(emp_id):
    """Return True if the employee ID exists."""
    return emp_id.upper() in MOCK_EMPLOYEES


def has_password(emp_id):
    """Return True if this employee has created a password."""
    return emp_id.upper() in st.session_state.password_store


def password_matches(emp_id, password):
    """Check demo password."""
    return (
        st.session_state.password_store.get(
            emp_id.upper(),
            "",
        )
        == password
    )


def create_password(emp_id, password):
    """Save demo password."""
    st.session_state.password_store[
        emp_id.upper()
    ] = password


def do_login(emp_id):
    """Log the employee in and reset the chat session."""

    st.session_state.logged_in = True
    st.session_state.selected_emp_id = emp_id.upper()

    st.session_state.messages = []
    st.session_state.tool_trace = []
    st.session_state.citations = []

    st.session_state.login_error = ""
    st.session_state.login_success = ""


def do_logout():
    """Log out and reset the current session."""

    st.session_state.logged_in = False
    st.session_state.selected_emp_id = None

    st.session_state.messages = []
    st.session_state.tool_trace = []
    st.session_state.citations = []

    st.session_state.pending_input = None

    st.session_state.login_mode = "login"
    st.session_state.login_error = ""
    st.session_state.login_success = ""


# ============================================================
# RAG HELPERS
# ============================================================

def build_citations(documents):
    """
    Convert LangChain Documents returned by Chroma
    into records used by the Streamlit citation panel.
    """

    citations = []

    for document in documents:

        metadata = document.metadata or {}

        source_file = (
            metadata.get("source_file")
            or metadata.get("source")
            or "HR Policy Document"
        )

        page = metadata.get("page")

        if page is not None:
            section = f"Page {page + 1}"
        else:
            section = (
                metadata.get("section")
                or metadata.get("heading")
                or ""
            )

        policy_id = (
            metadata.get("policy_id")
            or metadata.get("document_id")
            or ""
        )

        snippet = document.page_content.strip()

        if len(snippet) > 500:
            snippet = snippet[:500] + "..."

        citations.append(
            {
                "title": source_file,
                "section": section,
                "snippet": snippet,
                "policy_id": policy_id,
            }
        )

    return citations


def build_rag_trace(question, document_count):
    """
    Build a simple trace representing the actual RAG pipeline.
    """

    now = datetime.now().strftime("%H:%M:%S")

    return [
        {
            "tool": "Chroma Retriever",
            "args": {
                "query": question,
                "top_k": 4,
            },
            "result": (
                f"Retrieved {document_count} "
                "relevant document chunks."
            ),
            "timestamp": now,
            "status": "✓ Success",
        },
        {
            "tool": "OpenRouter LLM",
            "args": {
                "model": "google/gemma-4-26b-a4b-it:free",
            },
            "result": (
                "Generated response using "
                "retrieved HR context."
            ),
            "timestamp": now,
            "status": "✓ Success",
        },
    ]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    # --------------------------------------------------------
    # BRAND
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-icon">🌼</div>
            <div class="sidebar-brand-name">
                Daisy Health
            </div>
            <div class="sidebar-brand-subtitle">
                HR Assistant
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # SYSTEM STATUS
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="status-pill">
            <span class="status-dot"></span>
            System online
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # NOT LOGGED IN
    # --------------------------------------------------------

    if not st.session_state.logged_in:

        st.markdown(
            """
            <div style="
                margin-top:16px;
                font-size:0.82rem;
                color:rgba(255,255,255,0.75);
                line-height:1.6;
            ">
                Welcome to the Daisy Health HR
                Self-Service Portal.

                <br><br>

                Sign in with your
                <strong style="color:white;">
                    Employee ID
                </strong>
                and password to get answers about
                PTO, benefits, remote work,
                expenses, and more.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="
                margin-top:24px;
                font-size:0.72rem;
                color:rgba(255,255,255,0.45);
                line-height:1.6;
            ">
                Powered by Daisy Health RAG<br>
                LLM: OpenRouter + Gemma<br>
                Embeddings: Hugging Face<br>
                Vector DB: ChromaDB
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # LOGGED IN
    # --------------------------------------------------------

    else:

        emp = MOCK_EMPLOYEES[
            st.session_state.selected_emp_id
        ]

        st.markdown(
            '<div class="section-label">Logged In As</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="trace-card">

                <div class="tool-name"
                     style="font-size:0.72rem;">
                    {html.escape(
                        st.session_state.selected_emp_id
                    )}
                </div>

                <div style="
                    color:white !important;
                    font-size:0.85rem;
                    font-weight:500;
                    margin:3px 0 6px;
                ">
                    {html.escape(emp["name"])}
                </div>

                <div class="tool-result">
                    {html.escape(emp["role"])}<br>
                    {html.escape(emp["department"])}<br>
                    📍 {html.escape(emp["state"])}<br>
                    🏥 {html.escape(emp["type"])}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # TOOL TRACE
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="section-label"
                 style="margin-top:20px;">
                RAG Trace
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not st.session_state.tool_trace:

            st.markdown(
                """
                <div style="
                    font-size:0.75rem;
                    color:rgba(255,255,255,0.4);
                    padding:8px 0;
                    font-style:italic;
                ">
                    Retrieval and LLM activity
                    will appear here.
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            for trace in reversed(
                st.session_state.tool_trace
            ):

                st.markdown(
                    f"""
                    <div class="trace-card">

                        <div class="tool-name">
                            {html.escape(
                                str(trace["tool"])
                            )}
                        </div>

                        <div class="tool-status">
                            {html.escape(
                                str(trace["status"])
                            )}
                            ·
                            {html.escape(
                                str(trace["timestamp"])
                            )}
                        </div>

                        <div class="tool-result">
                            {html.escape(
                                str(trace["result"])
                            )}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ----------------------------------------------------
        # CLEAR CHAT
        # ----------------------------------------------------

        st.markdown(
            "<div style='margin-top:20px;'></div>",
            unsafe_allow_html=True,
        )

        if st.button(
            "🗑 Clear Conversation",
            use_container_width=True,
        ):

            st.session_state.messages = []
            st.session_state.tool_trace = []
            st.session_state.citations = []

            st.rerun()

        # ----------------------------------------------------
        # SIGN OUT
        # ----------------------------------------------------

        if st.button(
            "🚪 Sign Out",
            use_container_width=True,
        ):

            do_logout()
            st.rerun()


# ============================================================
# LOGIN PAGE
# ============================================================

if not st.session_state.logged_in:

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="dh-header">

            <div class="dh-daisy">
                🌼
            </div>

            <div>

                <h1>
                    Daisy Health HR Assistant
                </h1>

                <p class="subtitle">
                    Sign in to get answers about your PTO,
                    benefits, remote work, expenses,
                    and more.
                </p>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # LOGIN CARD
    # --------------------------------------------------------

    _, card_col, _ = st.columns(
        [1, 1.4, 1]
    )

    with card_col:

        # ====================================================
        # CREATE PASSWORD
        # ====================================================

        if (
            st.session_state.login_mode
            == "create_password"
        ):

            if st.session_state.login_success:

                st.markdown(
                    f"""
                    <div class="login-success">
                        ✓
                        {html.escape(
                            st.session_state.login_success
                        )}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            if st.session_state.login_error:

                st.markdown(
                    f"""
                    <div class="login-error">
                        ⚠️
                        {html.escape(
                            st.session_state.login_error
                        )}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            setup_emp_id = (
                st.session_state._setup_emp_id
            )

            setup_employee = MOCK_EMPLOYEES[
                setup_emp_id
            ]

            st.markdown(
                f"""
                <div class="login-card-title">
                    Create your password
                </div>

                <div class="login-card-sub">
                    First time logging in,
                    {html.escape(
                        setup_employee["name"].split()[0]
                    )}!

                    Please create a secure password
                    to protect your account.
                </div>

                <div style="
                    background:var(--sage-light);
                    border-radius:8px;
                    padding:10px 14px;
                    font-size:0.82rem;
                    color:var(--teal);
                    margin-bottom:16px;
                ">
                    Setting up:
                    <strong>
                        {html.escape(setup_emp_id)}
                    </strong>
                    —
                    {html.escape(setup_employee["name"])}
                </div>
                """,
                unsafe_allow_html=True,
            )

            new_pw = st.text_input(
                "New password",
                type="password",
                placeholder=(
                    "Choose a password "
                    "(min 6 characters)"
                ),
                key="new_pw",
            )

            confirm_pw = st.text_input(
                "Confirm password",
                type="password",
                placeholder="Re-enter your password",
                key="confirm_pw",
            )

            col_a, col_b = st.columns(2)

            with col_a:

                if st.button(
                    "← Back",
                    use_container_width=True,
                ):

                    st.session_state.login_mode = "login"
                    st.session_state.login_error = ""
                    st.session_state.login_success = ""

                    st.rerun()

            with col_b:

                if st.button(
                    "Create password",
                    use_container_width=True,
                    type="primary",
                ):

                    if len(new_pw) < 6:

                        st.session_state.login_error = (
                            "Password must be at least "
                            "6 characters."
                        )

                        st.rerun()

                    elif new_pw != confirm_pw:

                        st.session_state.login_error = (
                            "Passwords do not match. "
                            "Please try again."
                        )

                        st.rerun()

                    else:

                        create_password(
                            setup_emp_id,
                            new_pw,
                        )

                        do_login(setup_emp_id)

                        st.rerun()

        # ====================================================
        # NORMAL LOGIN
        # ====================================================

        else:

            if st.session_state.login_error:

                st.markdown(
                    f"""
                    <div class="login-error">
                        ⚠️
                        {html.escape(
                            st.session_state.login_error
                        )}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                """
                <div class="login-card-title">
                    Sign in to your account
                </div>

                <div class="login-card-sub">
                    Use your Daisy Health Employee ID
                    and your personal password.
                </div>
                """,
                unsafe_allow_html=True,
            )

            emp_id_input = st.text_input(
                "Employee ID",
                placeholder="e.g. EMP-001",
                key="login_emp_id",
            ).strip().upper()

            password_input = st.text_input(
                "Password",
                type="password",
                placeholder="Enter your password",
                key="login_password",
            )

            if st.button(
                "Sign in →",
                use_container_width=True,
                type="primary",
            ):

                st.session_state.login_error = ""
                st.session_state.login_success = ""

                if not emp_id_input:

                    st.session_state.login_error = (
                        "Please enter your Employee ID."
                    )

                    st.rerun()

                elif not employee_exists(
                    emp_id_input
                ):

                    st.session_state.login_error = (
                        f"Employee ID '{emp_id_input}' "
                        "not found. "
                        "Please check and try again."
                    )

                    st.rerun()

                elif not has_password(
                    emp_id_input
                ):

                    st.session_state._setup_emp_id = (
                        emp_id_input
                    )

                    st.session_state.login_mode = (
                        "create_password"
                    )

                    st.session_state.login_success = (
                        "Welcome, "
                        f"{MOCK_EMPLOYEES[emp_id_input]['name'].split()[0]}! "
                        "Please create your password "
                        "to continue."
                    )

                    st.rerun()

                elif not password_matches(
                    emp_id_input,
                    password_input,
                ):

                    st.session_state.login_error = (
                        "Incorrect password. "
                        "Please try again."
                    )

                    st.rerun()

                else:

                    do_login(emp_id_input)

                    st.rerun()

            st.markdown(
                """
                <div class="login-hint">
                    First time? Enter your Employee ID
                    and click Sign in to create your
                    password.
                    <br>
                    Need help?
                    Contact people@daisyhealth.com
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # DEMO EMPLOYEE REFERENCE
        # ----------------------------------------------------

        st.markdown(
            """
            <div class="demo-card">

                <strong>Demo Employee IDs:</strong><br>

                EMP-001 — Jordan Rivera
                (Care Coordinator)<br>

                EMP-002 — Morgan Chen
                (Software Engineer)<br>

                EMP-003 — Dr. Simone Okafor
                (Physician)<br>

                EMP-004 — Alex Nguyen
                (Clinical Pharmacist)<br>

                EMP-005 — Taylor Brooks
                (HR Business Partner)

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# MAIN DASHBOARD
# ============================================================

else:

    emp = MOCK_EMPLOYEES[
        st.session_state.selected_emp_id
    ]

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="dh-header">

            <div class="dh-daisy">
                🌼
            </div>

            <div>

                <h1>
                    Daisy Health HR Assistant
                </h1>

                <p class="subtitle">
                    Ask questions about PTO, remote work,
                    benefits, expenses, and more.
                    Answers are grounded in Daisy Health
                    policy documents.
                </p>

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # LAYOUT
    # --------------------------------------------------------

    col_chat, col_cite = st.columns(
        [2, 1],
        gap="large",
    )

    # ========================================================
    # CHAT COLUMN
    # ========================================================

    with col_chat:

        # ----------------------------------------------------
        # EMPLOYEE BADGE
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="emp-badge">

                👤
                {html.escape(emp["name"])}

                &nbsp;·&nbsp;

                {html.escape(emp["role"])}

                &nbsp;·&nbsp;

                {emp["pto_balance"]}
                days PTO available

            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # CHAT HISTORY
        # ----------------------------------------------------

        for msg in st.session_state.messages:

            if msg["role"] == "user":

                st.markdown(
                    f"""
                    <div class="msg-user">
                        {html.escape(
                            msg["content"]
                        )}
                    </div>

                    <div class="
                        msg-meta
                        msg-meta-right
                    ">
                        {html.escape(
                            msg["time"]
                        )}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            else:

                with st.chat_message(
                    "assistant",
                    avatar="🌼",
                ):

                    st.markdown(
                        msg["content"]
                    )

                    st.caption(
                        msg["time"]
                    )

        # ----------------------------------------------------
        # SUGGESTED QUESTIONS
        # ----------------------------------------------------

        if not st.session_state.messages:

            st.markdown(
                """
                <div style="
                    margin:16px 0 8px;
                    font-size:0.8rem;
                    color:#5A7070;
                ">
                    <strong>Try asking:</strong>
                </div>
                """,
                unsafe_allow_html=True,
            )

            chip_cols = st.columns(2)

            for i, suggestion in enumerate(
                SUGGESTED_QUESTIONS
            ):

                with chip_cols[i % 2]:

                    if st.button(
                        suggestion,
                        key=f"chip_{i}",
                        use_container_width=True,
                    ):

                        st.session_state.pending_input = (
                            suggestion
                        )

                        st.rerun()

        # ----------------------------------------------------
        # CHAT INPUT
        # ----------------------------------------------------

        user_input = st.chat_input(
            f"Ask an HR question, "
            f"{emp['name'].split()[0]}…"
        )

        # ----------------------------------------------------
        # HANDLE SUGGESTION CHIP
        # ----------------------------------------------------

        if st.session_state.pending_input:

            user_input = (
                st.session_state.pending_input
            )

            st.session_state.pending_input = None

        # ----------------------------------------------------
        # PROCESS USER QUESTION
        # ----------------------------------------------------

        if user_input:

            now = datetime.now().strftime(
                "%I:%M %p"
            )

            # -----------------------------------------------
            # Add user message
            # -----------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": user_input,
                    "time": now,
                }
            )

            # -----------------------------------------------
            # Run REAL RAG
            # -----------------------------------------------

            try:

                with st.spinner(
                    "🌼 Searching HR policy documents..."
                ):

                    rag_result = chat(
                        user_input
                    )

                answer = rag_result["answer"]

                documents = rag_result.get(
                    "documents",
                    [],
                )

                # -------------------------------------------
                # Build REAL citations
                # -------------------------------------------

                new_citations = build_citations(
                    documents
                )

                # -------------------------------------------
                # Build REAL RAG trace
                # -------------------------------------------

                new_tools = build_rag_trace(
                    user_input,
                    len(documents),
                )

                # -------------------------------------------
                # Add assistant message
                # -------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "time": datetime.now().strftime(
                            "%I:%M %p"
                        ),
                    }
                )

                # -------------------------------------------
                # Update trace and citations
                # -------------------------------------------

                st.session_state.tool_trace.extend(
                    new_tools
                )

                st.session_state.citations = (
                    new_citations
                )

            except Exception as error:

                # -------------------------------------------
                # Handle RAG errors
                # -------------------------------------------

                error_message = (
                    "I’m sorry, but I encountered an "
                    "error while searching the HR "
                    "documentation.\n\n"
                    "Please try again. If the problem "
                    "continues, contact People Operations."
                )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "time": datetime.now().strftime(
                            "%I:%M %p"
                        ),
                    }
                )

                st.session_state.tool_trace.append(
                    {
                        "tool": "RAG Pipeline",
                        "args": {
                            "query": user_input,
                        },
                        "result": str(error),
                        "timestamp": datetime.now().strftime(
                            "%H:%M:%S"
                        ),
                        "status": "✗ Error",
                    }
                )

                st.session_state.citations = []

            # -------------------------------------------
            # Rerun so the new answer appears
            # -------------------------------------------

            st.rerun()


    # ========================================================
    # CITATIONS COLUMN
    # ========================================================

    with col_cite:

        st.markdown(
            """
            <div style="
                font-size:0.78rem;
                font-weight:600;
                color:#1C2B2B;
                letter-spacing:0.06em;
                text-transform:uppercase;
                margin-bottom:10px;
                padding-bottom:6px;
                border-bottom:2px solid #C8DEDA;
            ">
                📄 Policy Citations
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # NO CITATIONS
        # ----------------------------------------------------

        if not st.session_state.citations:

            st.markdown(
                """
                <div class="empty-state"
                     style="padding:30px 10px;">

                    <div class="icon">
                        📋
                    </div>

                    <p style="
                        font-size:0.78rem;
                        color:#5A7070;
                    ">
                        Retrieved policy documents will
                        appear here after you ask a
                        question.
                    </p>

                </div>
                """,
                unsafe_allow_html=True,
            )

        # ----------------------------------------------------
        # REAL CITATIONS
        # ----------------------------------------------------

        else:

            for cite in (
                st.session_state.citations
            ):

                title = html.escape(
                    str(
                        cite.get(
                            "title",
                            "HR Policy Document",
                        )
                    )
                )

                section = html.escape(
                    str(
                        cite.get(
                            "section",
                            "",
                        )
                    )
                )

                policy_id = html.escape(
                    str(
                        cite.get(
                            "policy_id",
                            "",
                        )
                    )
                )

                snippet = html.escape(
                    str(
                        cite.get(
                            "snippet",
                            "",
                        )
                    )
                )

                if policy_id and section:

                    citation_metadata = (
                        f"{policy_id} · {section}"
                    )

                elif section:

                    citation_metadata = section

                elif policy_id:

                    citation_metadata = policy_id

                else:

                    citation_metadata = ""

                st.markdown(
                    f"""
                    <div class="cite-card">

                        <div class="cite-title">
                            📄 {title}
                        </div>

                        <div class="cite-section">
                            {citation_metadata}
                        </div>

                        <div class="cite-snippet">
                            "{snippet}"
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ----------------------------------------------------
        # SYSTEM STATUS
        # ----------------------------------------------------

        st.markdown(
            "<div style='margin-top:30px;'></div>",
            unsafe_allow_html=True,
        )

        try:

            chroma_count = get_document_count()
            rag_status = "Online"

        except Exception:

            chroma_count = "Unavailable"
            rag_status = "Error"

        st.markdown(
            f"""
            <div class="system-status">

                <strong>
                    System Status
                </strong>

                <br>

                <span style="color:#2E7D5E;">
                    ● RAG Index
                </span>
                &nbsp;
                {rag_status}

                <br>

                <span style="color:#2E7D5E;">
                    ● Chroma Documents
                </span>
                &nbsp;
                {chroma_count}

                <br>

                <span style="color:#2E7D5E;">
                    ● LLM Provider
                </span>
                &nbsp;
                OpenRouter / Gemma

                <br>

                <span style="color:#2E7D5E;">
                    ● Embeddings
                </span>
                &nbsp;
                all-MiniLM-L6-v2

                <br>

                <span style="color:#2E7D5E;">
                    ● Vector DB
                </span>
                &nbsp;
                Chroma Cloud

            </div>
            """,
            unsafe_allow_html=True,
        )