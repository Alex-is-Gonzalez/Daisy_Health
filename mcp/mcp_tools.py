"""
Daisy Health — MCP Tool Functions
These are the actual tool implementations, importable as plain Python functions.

The mcp_server.py wraps these with the MCP protocol for agent use.
The daisy_health_app.py imports these directly for Streamlit use.

This separation avoids the stdio conflict when importing mcp_server.py directly.
"""

import json
import re
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent  # Goes up from mcp/ to project root
MOCK_DATA_DIR = BASE_DIR / "mock_data"

import sys
sys.path.insert(0, str(BASE_DIR))

# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────
def load_employees():
    with open(MOCK_DATA_DIR / "employees.json") as f:
        data = json.load(f)
    return {e["employee_id"]: e for e in data["employees"]}

def load_pto_balances():
    with open(MOCK_DATA_DIR / "pto_balances.json") as f:
        data = json.load(f)
    return {r["employee_id"]: r for r in data["pto_balances"]}

def load_benefits():
    with open(MOCK_DATA_DIR / "benefits.json") as f:
        data = json.load(f)
    return {r["employee_id"]: r for r in data["benefits"]}

def load_hr_tickets():
    with open(MOCK_DATA_DIR / "hr_tickets.json") as f:
        return json.load(f)["hr_tickets"]

def save_hr_tickets(tickets):
    with open(MOCK_DATA_DIR / "hr_tickets.json", "w") as f:
        json.dump({"hr_tickets": tickets}, f, indent=2)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
APPROVED_STATES = [
    "California", "New York", "Texas", "Florida", "Washington",
    "Illinois", "Massachusetts", "Colorado", "Georgia", "Arizona",
    "Virginia", "Oregon", "Pennsylvania", "Ohio", "North Carolina",
    "Minnesota"
]

# ─────────────────────────────────────────────
# TOOL FUNCTIONS
# ─────────────────────────────────────────────

def tool_lookup_employee_profile(employee_id: str) -> str:
    emp_id = employee_id.upper().strip()
    employees = load_employees()
    if emp_id not in employees:
        return f"Employee ID '{emp_id}' not found."
    e = employees[emp_id]
    return (
        f"Employee Profile — {e['name']}\n"
        f"ID: {e['employee_id']}\n"
        f"Role: {e['role']}\n"
        f"Department: {e['department']}\n"
        f"Type: {e['employee_type']} ({e['employment_type']})\n"
        f"State: {e['state']}\n"
        f"Hire Date: {e['hire_date']}\n"
        f"Manager: {e['manager_name']} ({e['manager_id']})\n"
        f"Email: {e['email']}\n"
        f"Location: {e['office_location']}"
    )

def tool_check_pto_balance(employee_id: str) -> str:
    emp_id = employee_id.upper().strip()
    employees = load_employees()
    pto = load_pto_balances()
    if emp_id not in employees:
        return f"Employee ID '{emp_id}' not found."
    if emp_id not in pto:
        return f"No PTO record found for {emp_id}."
    e = employees[emp_id]
    b = pto[emp_id]
    return (
        f"PTO Balance — {e['name']} ({emp_id})\n"
        f"Available: {b['available_days']} days\n"
        f"Used this year: {b['used_days']} days\n"
        f"Total accrued: {b['total_accrued']} days\n"
        f"Carryover: {b['carryover_days']} days\n"
        f"Last updated: {b['last_updated']}"
    )

def tool_lookup_benefits_status(employee_id: str) -> str:
    emp_id = employee_id.upper().strip()
    employees = load_employees()
    benefits = load_benefits()
    if emp_id not in employees:
        return f"Employee ID '{emp_id}' not found."
    if emp_id not in benefits:
        return f"No benefits record for {emp_id}."
    e = employees[emp_id]
    b = benefits[emp_id]
    hsa = "Eligible ✓" if b["hsa_eligible"] else "Not eligible (not on Bronze plan)"
    fsa = f"Enrolled — ${b['fsa_balance']:.2f}" if b["fsa_enrolled"] else "Not enrolled"
    return (
        f"Benefits — {e['name']} ({emp_id})\n"
        f"Health Plan: {b['health_plan']}\n"
        f"HSA: {hsa}\nFSA: {fsa}\n"
        f"Dental: {'✓' if b['dental'] else '✗'} | "
        f"Vision: {'✓' if b['vision'] else '✗'} | "
        f"Life: {'✓' if b['life_insurance'] else '✗'}\n"
        f"STD: {'Eligible' if b['std_eligible'] else 'Not eligible'} | "
        f"LTD: {'Eligible' if b['ltd_eligible'] else 'Not eligible'}\n"
        f"401(k): {'✓' if b['retirement_401k'] else '✗'} ({b['employer_match_pct']}% match)\n"
        f"EAP: {'✓' if b['eap_enrolled'] else '✗'}"
    )

def tool_search_policy_documents(query: str, top_k: int = 3) -> str:
    top_k = max(1, min(top_k, 5))
    try:
        from rag_backend import retriever
    except Exception as e:
        return f"Policy search unavailable: RAG backend failed to load ({e})."

    docs = retriever.invoke(query)[:top_k]
    if not docs:
        return f"No results for '{query}'. Contact people@daisyhealth.com"

    out = f"Policy search: '{query}'\n" + "=" * 40 + "\n\n"
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata or {}
        source = meta.get("source_file") or meta.get("source") or "HR Policy Document"
        page = meta.get("page")
        section = f"Page {page + 1}" if page is not None else meta.get("section", "")
        snippet = doc.page_content.strip().replace("\n", " ")[:300]
        out += (
            f"[{i}] {Path(source).stem.replace('_', ' ').title()}\n"
            f"    Section: {section}\n"
            f"    Source: {source}\n"
            f"    Excerpt: {snippet}...\n\n"
        )
    return out

def tool_check_policy_compliance(employee_id: str, policy_area: str, scenario: str = "") -> str:
    emp_id = employee_id.upper().strip()
    employees = load_employees()
    if emp_id not in employees:
        return f"Employee ID '{emp_id}' not found."
    e = employees[emp_id]
    area = policy_area.lower()
    if "remote" in area:
        approved = e["state"] in APPROVED_STATES
        clinical = e["employee_type"] == "Clinical"
        result = (
            f"Remote Work Compliance — {e['name']}\n"
            f"State: {e['state']} — {'✓ Approved' if approved else '✗ Not approved'}\n"
        )
        if scenario:
            result += f"Scenario: {scenario}\n"
        result += "\n✓ Approved.\n" if approved else "\n⚠️ Location Change Request required.\n"
        if clinical:
            result += (
                "\n🏥 CLINICAL: Active license in patient state required. "
                "Notify Credentialing within 5 days. Source: HR-RW-001 §3, HR-LC-009"
            )
        else:
            result += "\nSource: HR-RW-001 — Temporary (<4 weeks): notify manager 5 days ahead."
        return result
    elif "pto" in area:
        pto = load_pto_balances()
        available = pto.get(emp_id, {}).get("available_days", 0)
        result = f"PTO Compliance — {e['name']}\nAvailable: {available} days\n\n"
        if scenario:
            nums = re.findall(r'\d+\.?\d*', scenario)
            if nums:
                requested = float(nums[0])
                result += (
                    f"✓ COMPLIANT: {requested} days requested, {available} available."
                    if requested <= available else
                    f"⚠️ INSUFFICIENT: {requested} requested but only {available} available."
                )
        return result
    elif "expense" in area:
        result = f"Expense Compliance — {e['name']} ({e['employment_type']})\n\n"
        if scenario:
            s = scenario.lower()
            if any(x in s for x in ["chair", "desk", "monitor", "keyboard", "webcam"]):
                result += "✓ COMPLIANT: $500 home office stipend covers this. Source: HR-EX-004 §2"
            elif "laptop" in s:
                result += "⚠️ REQUIRES APPROVAL: Manager + Finance must approve first. Source: HR-EX-004 §4"
            elif any(x in s for x in ["travel", "flight", "hotel"]):
                result += "✓ IF PRE-APPROVED: Economy class, $75/day meals. Source: HR-EX-004 §7"
            else:
                result += "Must be work-related with receipt. Over $500 needs pre-approval. Source: HR-EX-004"
        return result
    elif "benefit" in area:
        benefits = load_benefits()
        b = benefits.get(emp_id, {})
        result = f"Benefits Compliance — {e['name']}\nType: {e['employment_type']} | Plan: {b.get('health_plan','Unknown')}\n\n"
        result += (
            "ℹ️ Part-time: limited benefits. 30+ hrs/week needed. Source: HR-BI-002"
            if e["employment_type"] == "Part-Time" else
            "✓ Full-time: all benefits eligible. Open enrollment: November. Source: HR-BI-002"
        )
        return result
    return f"Unknown policy area '{policy_area}'. Options: remote_work, pto_request, expense, benefits_eligibility."

def tool_create_mock_hr_ticket(employee_id, ticket_type, subject, description, priority="Normal"):
    emp_id = employee_id.upper().strip()
    employees = load_employees()
    if emp_id not in employees:
        return f"Cannot create ticket — Employee ID '{emp_id}' not found."
    e = employees[emp_id]
    tickets = load_hr_tickets()
    ticket_id = f"TKT-{len(tickets) + 1:04d}"
    new_ticket = {
        "ticket_id": ticket_id,
        "employee_id": emp_id,
        "employee_name": e["name"],
        "ticket_type": ticket_type,
        "status": "Open",
        "priority": priority,
        "subject": subject,
        "description": description,
        "created_date": datetime.now().strftime("%Y-%m-%d"),
        "resolved_date": None,
        "assigned_to": e["manager_name"],
    }
    tickets.append(new_ticket)
    save_hr_tickets(tickets)
    return (
        f"✓ HR Ticket Created (MOCK)\n"
        f"ID: {ticket_id} | Type: {ticket_type} | Priority: {priority}\n"
        f"Employee: {e['name']} | Assigned: {e['manager_name']}\n"
        f"Subject: {subject}\nStatus: Open\n"
        f"Note: Mock ticket for demonstration purposes."
    )

def tool_draft_hr_email(employee_id, email_type, context=""):
    emp_id = employee_id.upper().strip()
    employees = load_employees()
    if emp_id not in employees:
        return f"Employee ID '{emp_id}' not found."
    e = employees[emp_id]
    et = email_type.lower()
    today = datetime.now().strftime("%B %d, %Y")
    mgr = e["manager_name"].split()[0] if e["manager_name"] else "there"
    if "pto" in et:
        subject = f"PTO Request — {e['name']}"
        to = f"{e['manager_name']} <{e['manager_id'].lower()}@daisyhealth.com>"
        body = f"Hi {mgr},\n\nI would like to request paid time off.\n\n{context or 'Please see my request in the HR portal.'}\n\nThank you,\n{e['name']}\n{e['role']}\n{e['email']}"
    elif "remote" in et:
        subject = f"Remote Work Request — {e['name']}"
        to = f"{e['manager_name']} <{e['manager_id'].lower()}@daisyhealth.com>"
        body = f"Hi {mgr},\n\nI am requesting a temporary remote work location change.\n\n{context or 'I would like to discuss the details.'}\n\nThank you,\n{e['name']}\n{e['role']}\n{e['email']}"
    elif "escalat" in et or "case" in et:
        subject = f"HR Concern — Confidential — {e['name']}"
        to = "people@daisyhealth.com"
        body = f"Hi People Operations,\n\nI am reaching out regarding a concern.\n\n{context or 'I would like to discuss this confidentially.'}\n\nSincerely,\n{e['name']}\n{e['role']}\n{e['email']}"
    elif "benefit" in et:
        subject = f"Benefits Question — {e['name']}"
        to = "people@daisyhealth.com"
        body = f"Hi People Operations,\n\nI have a question about my benefits.\n\n{context or 'I would appreciate your guidance.'}\n\nThank you,\n{e['name']}\n{e['role']}\n{e['email']}"
    else:
        subject = f"HR Inquiry — {e['name']}"
        to = "people@daisyhealth.com"
        body = f"Hi People Operations,\n\n{context or 'I have a question and would appreciate guidance.'}\n\nThank you,\n{e['name']}\n{e['role']}\n{e['email']}"
    return (
        f"📧 DRAFT EMAIL (not sent)\n{'='*50}\n"
        f"To: {to}\nFrom: {e['name']} <{e['email']}>\n"
        f"Date: {today}\nSubject: {subject}\n{'='*50}\n\n"
        f"{body}\n\n{'='*50}\nReview before sending."
    )
