"""
Daisy Health — MCP Server
Compatible with mcp 2.0.0

Install: pip install 'mcp[cli]'
Run:     python mcp/mcp_server.py
"""

import json
import re
import asyncio
from datetime import datetime
from pathlib import Path

from mcp.server.mcpserver.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool, TextContent,
    ListToolsRequest, ListToolsResult,
    CallToolRequest, CallToolRequestParams, CallToolResult,
)

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent
MOCK_DATA_DIR = BASE_DIR / "mock_data"

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

def load_policy_docs():
    for d in [BASE_DIR/"data"/"handbooks", BASE_DIR/"docs", BASE_DIR/"data"]:
        if d.exists():
            docs = []
            for f in d.glob("*.md"):
                docs.append({
                    "title": f.stem.replace("_", " ").title(),
                    "filename": f.name,
                    "content": f.read_text(encoding="utf-8"),
                })
            if docs:
                return docs
    return []

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
    docs = load_policy_docs()
    if not docs:
        return "Policy documents not found. Ensure data/handbooks contains .md files."
    words = set(query.lower().split())
    scored = []
    for doc in docs:
        for section in re.split(r'\n## ', doc["content"]):
            if not section.strip():
                continue
            score = sum(1 for w in words if w in section.lower())
            if score > 0:
                lines = section.strip().split("\n")
                scored.append({
                    "score": score,
                    "doc_title": doc["title"],
                    "filename": doc["filename"],
                    "section": lines[0].replace("#", "").strip(),
                    "snippet": " ".join(lines[1:])[:300].strip(),
                })
    scored.sort(key=lambda x: x["score"], reverse=True)
    results = scored[:top_k]
    if not results:
        return f"No results for '{query}'. Contact people@daisyhealth.com"
    out = f"Policy search: '{query}'\n" + "=" * 40 + "\n\n"
    for i, r in enumerate(results, 1):
        out += (
            f"[{i}] {r['doc_title']}\n"
            f"    Section: {r['section']}\n"
            f"    Source: {r['filename']}\n"
            f"    Excerpt: {r['snippet']}...\n\n"
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
        result += "\n✓ Approved for remote work.\n" if approved else "\n⚠️ Location Change Request required.\n"
        if clinical:
            result += (
                "\n🏥 CLINICAL: Additional requirements:\n"
                "- Active license in patient's state\n"
                "- Notify Credentialing within 5 business days\n"
                "- Confirm malpractice coverage\n"
                "Source: HR-RW-001 §3, HR-LC-009"
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
                    f"✓ COMPLIANT: {requested} days requested, {available} available.\nSubmit via HR Portal."
                    if requested <= available else
                    f"⚠️ INSUFFICIENT: {requested} requested but only {available} available."
                )
        return result
    elif "expense" in area:
        result = f"Expense Compliance — {e['name']} ({e['employment_type']})\n\n"
        if scenario:
            s = scenario.lower()
            if any(x in s for x in ["chair", "desk", "monitor", "keyboard", "webcam"]):
                result += "✓ COMPLIANT: $500 home office stipend covers this.\nSource: HR-EX-004 §2"
            elif "laptop" in s:
                result += "⚠️ REQUIRES APPROVAL: Manager + Finance must approve first.\nSource: HR-EX-004 §4"
            elif any(x in s for x in ["travel", "flight", "hotel"]):
                result += "✓ IF PRE-APPROVED: Economy class, $75/day meals.\nSource: HR-EX-004 §7"
            else:
                result += "Must be work-related with receipt. Over $500 needs pre-approval.\nSource: HR-EX-004"
        return result
    elif "benefit" in area:
        benefits = load_benefits()
        b = benefits.get(emp_id, {})
        result = f"Benefits Compliance — {e['name']}\nType: {e['employment_type']} | Plan: {b.get('health_plan','Unknown')}\n\n"
        result += (
            "ℹ️ Part-time: limited benefits. 30+ hrs/week needed.\nSource: HR-BI-002"
            if e["employment_type"] == "Part-Time" else
            "✓ Full-time: all benefits eligible. Open enrollment: November.\nSource: HR-BI-002"
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

# ─────────────────────────────────────────────
# TOOL DEFINITIONS
# ─────────────────────────────────────────────
TOOLS = [
    Tool(name="lookup_employee_profile",
         description="Look up a Daisy Health employee profile by ID (e.g. EMP-001).",
         inputSchema={"type":"object","properties":{"employee_id":{"type":"string"}},"required":["employee_id"]}),
    Tool(name="check_pto_balance",
         description="Check PTO balance for a Daisy Health employee.",
         inputSchema={"type":"object","properties":{"employee_id":{"type":"string"}},"required":["employee_id"]}),
    Tool(name="lookup_benefits_status",
         description="Look up benefits enrollment for a Daisy Health employee.",
         inputSchema={"type":"object","properties":{"employee_id":{"type":"string"}},"required":["employee_id"]}),
    Tool(name="search_policy_documents",
         description="Search HR policy documents for relevant information.",
         inputSchema={"type":"object","properties":{"query":{"type":"string"},"top_k":{"type":"integer","default":3}},"required":["query"]}),
    Tool(name="check_policy_compliance",
         description="Check if an employee action complies with Daisy Health policy.",
         inputSchema={"type":"object","properties":{"employee_id":{"type":"string"},"policy_area":{"type":"string"},"scenario":{"type":"string","default":""}},"required":["employee_id","policy_area"]}),
    Tool(name="create_mock_hr_ticket",
         description="Create a mock HR support ticket (saved to hr_tickets.json).",
         inputSchema={"type":"object","properties":{"employee_id":{"type":"string"},"ticket_type":{"type":"string"},"subject":{"type":"string"},"description":{"type":"string"},"priority":{"type":"string","default":"Normal"}},"required":["employee_id","ticket_type","subject","description"]}),
    Tool(name="draft_hr_email",
         description="Draft an HR email for an employee. Does NOT send it.",
         inputSchema={"type":"object","properties":{"employee_id":{"type":"string"},"email_type":{"type":"string"},"context":{"type":"string","default":""}},"required":["employee_id","email_type"]}),
]

# ─────────────────────────────────────────────
# SERVER + HANDLERS
# ─────────────────────────────────────────────
server = Server("daisy-health-hr")

async def handle_list_tools(request: ListToolsRequest) -> ListToolsResult:
    """Return the list of available tools to the agent."""
    return ListToolsResult(tools=TOOLS)

async def handle_call_tool(request: CallToolRequest) -> CallToolResult:
    """Route tool calls to the right function."""
    name = request.params.name
    args = request.params.arguments or {}

    if name == "lookup_employee_profile":
        result = tool_lookup_employee_profile(args["employee_id"])
    elif name == "check_pto_balance":
        result = tool_check_pto_balance(args["employee_id"])
    elif name == "lookup_benefits_status":
        result = tool_lookup_benefits_status(args["employee_id"])
    elif name == "search_policy_documents":
        result = tool_search_policy_documents(args["query"], args.get("top_k", 3))
    elif name == "check_policy_compliance":
        result = tool_check_policy_compliance(args["employee_id"], args["policy_area"], args.get("scenario", ""))
    elif name == "create_mock_hr_ticket":
        result = tool_create_mock_hr_ticket(args["employee_id"], args["ticket_type"], args["subject"], args["description"], args.get("priority", "Normal"))
    elif name == "draft_hr_email":
        result = tool_draft_hr_email(args["employee_id"], args["email_type"], args.get("context", ""))
    else:
        result = f"Unknown tool: {name}"

    return CallToolResult(content=[TextContent(type="text", text=result)])

# Register handlers with correct signature: (method, params_type, handler)
server.add_request_handler("tools/list", ListToolsRequest, handle_list_tools)
server.add_request_handler("tools/call", CallToolRequestParams, handle_call_tool)

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
async def main():
    print("🌼 Daisy Health MCP Server starting...")
    print("   7 tools ready: lookup_employee_profile, check_pto_balance,")
    print("                  lookup_benefits_status, search_policy_documents,")
    print("                  check_policy_compliance, create_mock_hr_ticket,")
    print("                  draft_hr_email")
    print()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
