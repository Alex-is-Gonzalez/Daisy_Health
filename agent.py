"""
Daisy Health — HR Agent

Architecture:

    Streamlit
        ↓
    Deterministic workflow detection
        ↓
    MCP tools
        ↓
    One final OpenAI call when useful
        ↓
    Structured employee-facing response

Key guarantees:

- No LLM classification by default.
- MCP schemas are inspected before tool calls.
- Required MCP arguments are populated dynamically.
- Policy search runs at most once.
- Duplicate tool calls are prevented.
- Raw MCP output is never returned to the employee.
- Citations are always normalized dictionaries.
- HR-case responses explicitly describe completed actions.
- "Mock ticket" terminology never appears in employee-facing text.
- Policy citations are returned separately for Streamlit.
- GPT-5-compatible OpenAI parameters are used.
"""

import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).parent

sys.path.insert(0, str(BASE_DIR))

LLM_MODEL = "gpt-5-mini"

# Deterministic classification is faster and more predictable.
USE_LLM_CLASSIFICATION = False

MAX_POLICY_SEARCHES = 1
MAX_TOOL_RESULT_CHARS = 6000
MAX_FINAL_CONTEXT_CHARS = 18000
FINAL_MAX_COMPLETION_TOKENS = 1000

MCP_SERVER_PATH = BASE_DIR / "mcp" / "mcp_server.py"

MCP_SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=[str(MCP_SERVER_PATH)],
    env=None,
)


# ============================================================
# OPENAI
# ============================================================

def get_openai_client():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")

    return AsyncOpenAI(api_key=api_key)


# ============================================================
# EMPLOYEE-FACING SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Daisy, the HR assistant for Daisy Health.

Your response is written directly to the employee.

IMPORTANT:

- Never mention MCP, APIs, tools, retrieval, RAG, embeddings,
  Chroma, agents, LLMs, internal workflows, or implementation.
- Never expose raw JSON or raw system output.
- Never expose validation errors or technical errors.
- Never invent employee information.
- Never invent policy rules.
- Never invent ticket numbers, policy IDs, or email status.
- Only say an action was completed when the supplied HR information
  indicates that it succeeded.
- Never say that an email was sent unless the supplied information
  explicitly says it was sent.
- If an email was drafted but not sent, clearly say it was drafted
  for the employee to review.
- Keep the answer concise, polished, and employee-friendly.
- Use Markdown headings and bullets when useful.
- Do not repeat the employee's entire question.

For policy questions:

- Answer the question first.
- Explain the relevant rule in plain English.
- Mention the policy name and policy ID when available.

For PTO questions:

- State the employee's balance when available.
- Explain the applicable policy when available.

For workplace concerns:

- Be empathetic and professional.
- Clearly explain what was completed.
- Do not use the words "mock ticket", "mock HR ticket",
  "demo ticket", or "demonstration ticket".
- Refer to the result simply as an HR case or HR case ticket.

If information is insufficient:

"I don't have enough information from the available HR documentation.
Please contact people@daisyhealth.com."

Return only the employee-facing response.
"""


# ============================================================
# GENERAL HELPERS
# ============================================================

def truncate_text(text, limit=MAX_TOOL_RESULT_CHARS):
    text = str(text or "")

    if len(text) <= limit:
        return text

    return text[:limit] + "\n...[truncated]"


def extract_mcp_text(result):
    parts = []

    for content in getattr(result, "content", []):
        text = getattr(content, "text", None)

        if text:
            parts.append(str(text))

    return "\n".join(parts).strip()


def is_tool_error(text):
    if not text:
        return True

    return str(text).lower().startswith(
        (
            "error executing",
            "cannot execute",
            "tool error",
        )
    )


def first_name_from_profile(profile):
    if not profile:
        return "there"

    match = re.search(
        r"(?:Employee Profile\s*[—-]\s*|Name:\s*)([A-Za-z]+)",
        profile,
        re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return "there"


# ============================================================
# WORKFLOW DETECTION
# ============================================================

def detect_workflow(question):
    q = question.lower().strip()

    if any(
        term in q
        for term in (
            "harassment",
            "hostile",
            "discrimination",
            "retaliation",
            "bullying",
            "workplace concern",
            "workplace issue",
            "manager concern",
            "manager problem",
            "coworker",
            "coworker problem",
            "complaint",
            "report",
            "unsafe",
            "misconduct",
            "workplace behavior",
            "workplace complaint",
            "file a complaint",
            "report my manager",
            "report a coworker",
            "problem with my manager",
            "problem with my coworker",
        )
    ):
        return "hr_case"

    if any(
        term in q
        for term in (
            "pto",
            "paid time off",
            "vacation",
            "time off",
            "days off",
            "leave balance",
            "pto balance",
            "vacation balance",
        )
    ):
        return "pto"

    if any(
        term in q
        for term in (
            "benefits",
            "health insurance",
            "insurance",
            "dental",
            "vision",
            "401k",
            "401(k)",
            "retirement",
            "benefit plan",
            "coverage",
            "medical plan",
        )
    ):
        return "benefits"

    if any(
        term in q
        for term in (
            "remote",
            "work from home",
            "work remotely",
            "hybrid",
            "home office",
            "remote work",
        )
    ):
        return "remote_work"

    if any(
        term in q
        for term in (
            "expense",
            "expenses",
            "reimburse",
            "reimbursement",
            "receipt",
            "chair",
            "laptop",
            "office equipment",
            "home office equipment",
            "purchase",
            "business expense",
            "equipment",
            "desk",
            "webcam",
            "headset",
            "office supplies",
        )
    ):
        return "expense"

    return "general"


# ============================================================
# REQUIRED TOOLS
# ============================================================

def get_workflow_tools(workflow, available_names):
    available = set(available_names)
    required = []

    # Employee profile is useful for every workflow.
    if "lookup_employee_profile" in available:
        required.append("lookup_employee_profile")

    if workflow == "pto":
        for name in (
            "check_pto_balance",
            "search_policy_documents",
        ):
            if name in available:
                required.append(name)

    elif workflow == "benefits":
        for name in (
            "lookup_benefits",
            "search_policy_documents",
        ):
            if name in available:
                required.append(name)

    elif workflow == "remote_work":
        for name in (
            "search_policy_documents",
            "check_policy_compliance",
        ):
            if name in available:
                required.append(name)

    elif workflow == "expense":
        for name in (
            "search_policy_documents",
            "check_policy_compliance",
        ):
            if name in available:
                required.append(name)

    elif workflow == "hr_case":
        for name in (
            "search_policy_documents",
            "create_mock_hr_ticket",
            "draft_hr_email",
        ):
            if name in available:
                required.append(name)

    elif workflow == "general":
        if "search_policy_documents" in available:
            required.append("search_policy_documents")

    return required


# ============================================================
# MCP SCHEMA HELPERS
# ============================================================

def get_tool_schema(available_tools, tool_name):
    for tool in available_tools:
        if tool.name == tool_name:
            return getattr(tool, "input_schema", None) or {}

    return {}


def get_schema_properties(available_tools, tool_name):
    schema = get_tool_schema(
        available_tools,
        tool_name,
    )

    return set(
        (schema.get("properties") or {}).keys()
    )


def get_required_fields(available_tools, tool_name):
    schema = get_tool_schema(
        available_tools,
        tool_name,
    )

    return schema.get("required") or []


# ============================================================
# MCP ARGUMENT BUILDER
# ============================================================

def build_tool_arguments(
    tool_name,
    question,
    employee_id,
    workflow,
    available_tools,
):
    """
    Build arguments using the actual MCP schema.

    This prevents errors such as:

        policy_area Field required
        ticket_type Field required
        subject Field required
        email_type Field required
    """

    properties = get_schema_properties(
        available_tools,
        tool_name,
    )

    args = {}

    # --------------------------------------------------------
    # EMPLOYEE PROFILE
    # --------------------------------------------------------

    if tool_name == "lookup_employee_profile":

        if "employee_id" in properties:
            args["employee_id"] = employee_id

        return args

    # --------------------------------------------------------
    # PTO
    # --------------------------------------------------------

    if tool_name == "check_pto_balance":

        if "employee_id" in properties:
            args["employee_id"] = employee_id

        return args

    # --------------------------------------------------------
    # BENEFITS
    # --------------------------------------------------------

    if tool_name == "lookup_benefits":

        if "employee_id" in properties:
            args["employee_id"] = employee_id

        return args

    # --------------------------------------------------------
    # POLICY SEARCH
    # --------------------------------------------------------

    if tool_name == "search_policy_documents":

        if "query" in properties:
            args["query"] = question

        elif "question" in properties:
            args["question"] = question

        elif "search_query" in properties:
            args["search_query"] = question

        return args

    # --------------------------------------------------------
    # POLICY COMPLIANCE
    # --------------------------------------------------------

    if tool_name == "check_policy_compliance":

        if "employee_id" in properties:
            args["employee_id"] = employee_id

        if "question" in properties:
            args["question"] = question

        elif "request" in properties:
            args["request"] = question

        elif "scenario" in properties:
            args["scenario"] = question

        if "policy_area" in properties:
            args["policy_area"] = {
                "pto": "pto",
                "benefits": "benefits",
                "remote_work": "remote_work",
                "expense": "expense",
                "hr_case": "conduct",
                "general": "general",
            }.get(workflow, workflow)

        return args

    # --------------------------------------------------------
    # HR CASE
    # --------------------------------------------------------

    if tool_name == "create_mock_hr_ticket":

        if "employee_id" in properties:
            args["employee_id"] = employee_id

        if "description" in properties:
            args["description"] = question

        elif "issue" in properties:
            args["issue"] = question

        elif "details" in properties:
            args["details"] = question

        if "ticket_type" in properties:
            args["ticket_type"] = "workplace_concern"

        if "subject" in properties:
            args["subject"] = "Workplace concern"

        return args

    # --------------------------------------------------------
    # HR EMAIL
    # --------------------------------------------------------

    if tool_name == "draft_hr_email":

        if "employee_id" in properties:
            args["employee_id"] = employee_id

        if "issue" in properties:
            args["issue"] = question

        elif "description" in properties:
            args["description"] = question

        elif "details" in properties:
            args["details"] = question

        if "email_type" in properties:
            args["email_type"] = "hr_escalation"

        if "subject" in properties:
            args["subject"] = "Workplace concern"

        return args

    # --------------------------------------------------------
    # GENERIC FALLBACK
    # --------------------------------------------------------

    if "employee_id" in properties:
        args["employee_id"] = employee_id

    if "question" in properties:
        args["question"] = question

    return args


# ============================================================
# TOOL EXECUTION
# ============================================================

async def execute_tool(
    session,
    tool_name,
    tool_input,
    tool_trace,
):
    timestamp = datetime.now().strftime("%H:%M:%S")
    start = time.perf_counter()

    try:
        result = await session.call_tool(
            tool_name,
            tool_input,
        )

        duration = time.perf_counter() - start
        result_text = extract_mcp_text(result)

        print(
            f"[TIMING] MCP tool '{tool_name}': "
            f"{duration:.2f}s",
            flush=True,
        )

        print(
            f"[TIMING] MCP tool '{tool_name}' returned "
            f"{len(result_text):,} characters",
            flush=True,
        )

        tool_trace.append(
            {
                "tool": tool_name,
                "args": tool_input,
                "result": truncate_text(result_text),
                "timestamp": timestamp,
                "status": "✓ Success",
                "duration_seconds": round(duration, 2),
            }
        )

        return result_text

    except Exception as exc:
        duration = time.perf_counter() - start

        error_text = (
            f"Error executing tool {tool_name}: {exc}"
        )

        print(
            f"[TIMING] MCP tool '{tool_name}' ERROR: "
            f"{duration:.2f}s",
            flush=True,
        )

        print(
            f"[AGENT] {error_text}",
            flush=True,
        )

        tool_trace.append(
            {
                "tool": tool_name,
                "args": tool_input,
                "result": error_text,
                "timestamp": timestamp,
                "status": "✗ Error",
                "duration_seconds": round(duration, 2),
            }
        )

        return error_text


# ============================================================
# CITATIONS
# ============================================================

def extract_citations(policy_result):
    """
    Normalize policy search output into dictionaries.

    Streamlit can safely use:

        citation.get("title")
        citation.get("section")
        citation.get("policy_id")
        citation.get("snippet")
    """

    if not policy_result or is_tool_error(policy_result):
        return []

    text = str(policy_result)

    citations = []

    pattern = re.compile(
        r"""
        \[(?P<number>\d+)\]
        \s*
        (?P<title>.*?)
        \s+
        Section:\s*
        (?P<section>.*?)
        \s+
        Source:\s*
        (?P<source>.*?)
        \s+
        Excerpt:\s*
        (?P<snippet>.*?)
        (?=
            \n\s*\[\d+\]
            |
            \Z
        )
        """,
        re.IGNORECASE
        | re.DOTALL
        | re.VERBOSE,
    )

    for match in pattern.finditer(text):
        title = match.group("title").strip()
        section = match.group("section").strip()
        source = match.group("source").strip()
        snippet = match.group("snippet").strip()

        # Remove stray control characters from PDF extraction.
        title = clean_text(title)
        section = clean_text(section)
        source = clean_text(source)
        snippet = clean_text(snippet)

        citations.append(
            {
                "title": title or "HR Policy",
                "section": section,
                "policy_id": source,
                "source": source,
                "snippet": snippet,
            }
        )

    if citations:
        return citations

    # --------------------------------------------------------
    # Fallback parser
    # --------------------------------------------------------

    source_match = re.search(
        r"Source:\s*([^\n]+)",
        text,
        re.IGNORECASE,
    )

    section_match = re.search(
        r"Section:\s*([^\n]+)",
        text,
        re.IGNORECASE,
    )

    excerpt_match = re.search(
        r"Excerpt:\s*(.*?)(?=\n\s*\[\d+\]|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    source = (
        clean_text(source_match.group(1))
        if source_match
        else "HR Policy"
    )

    section = (
        clean_text(section_match.group(1))
        if section_match
        else ""
    )

    snippet = (
        clean_text(excerpt_match.group(1))
        if excerpt_match
        else clean_text(text)
    )

    return [
        {
            "title": "HR Policy",
            "section": section,
            "policy_id": source,
            "source": source,
            "snippet": truncate_text(snippet, 800),
        }
    ]


def clean_text(value):
    """
    Remove PDF/control-character artifacts.
    """

    value = str(value or "")

    value = "".join(
        char
        for char in value
        if char in "\n\t"
        or ord(char) >= 32
    )

    value = re.sub(
        r"[ \t]+",
        " ",
        value,
    )

    value = re.sub(
        r"\n{3,}",
        "\n\n",
        value,
    )

    return value.strip()


def format_policy_citations(citations):
    """
    Create a short employee-facing policy section.
    """

    if not citations:
        return ""

    lines = [
        "### Relevant policy",
        "",
    ]

    seen = set()

    for citation in citations:
        title = citation.get(
            "title",
            "HR Policy",
        )

        policy_id = citation.get(
            "policy_id",
            "",
        )

        section = citation.get(
            "section",
            "",
        )

        snippet = citation.get(
            "snippet",
            "",
        )

        key = (
            title,
            policy_id,
            section,
        )

        if key in seen:
            continue

        seen.add(key)

        label = title

        if policy_id and policy_id != title:
            label += f" ({policy_id})"

        if section:
            lines.append(
                f"- **{label} — {section}:** "
                f"{clean_policy_summary(snippet)}"
            )
        else:
            lines.append(
                f"- **{label}:** "
                f"{clean_policy_summary(snippet)}"
            )

    if len(lines) == 2:
        return ""

    return "\n".join(lines)


def clean_policy_summary(snippet):
    """
    Make policy excerpts readable without dumping PDF text.
    """

    text = clean_text(snippet)

    # Remove common PDF bullet artifacts.
    text = text.replace("", "")
    text = text.replace("•", " ")

    # Collapse repeated whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return truncate_text(text, 500)


# ============================================================
# FINAL LLM CONTEXT
# ============================================================

def build_final_context(tool_results):
    parts = []

    for tool_name, result in tool_results.items():
        if not result:
            continue

        parts.append(
            f"===== {tool_name} =====\n"
            f"{truncate_text(result)}"
        )

    return truncate_text(
        "\n\n".join(parts),
        MAX_FINAL_CONTEXT_CHARS,
    )


# ============================================================
# EXTRACT HR CASE DETAILS
# ============================================================

def extract_ticket_details(ticket_result):
    """
    Pull useful employee-safe details from an HR case result.
    """

    if not ticket_result or is_tool_error(ticket_result):
        return {}

    text = clean_text(ticket_result)

    details = {}

    patterns = {
        "ticket_id": [
            r"(?:ticket|case)\s*(?:id|number)?\s*[:#-]?\s*(TKT-\d+)",
            r"\b(TKT-\d+)\b",
        ],
        "priority": [
            r"priority\s*[:\-]\s*([A-Za-z]+)",
        ],
        "status": [
            r"status\s*[:\-]\s*([A-Za-z]+)",
        ],
        "ticket_type": [
            r"ticket[_ ]type\s*[:\-]\s*([A-Za-z_ -]+)",
            r"type\s*[:\-]\s*([A-Za-z_ -]+)",
        ],
    }

    for key, regexes in patterns.items():
        for regex in regexes:
            match = re.search(
                regex,
                text,
                re.IGNORECASE,
            )

            if match:
                details[key] = match.group(1).strip()
                break

    return details


# ============================================================
# HR CASE RESPONSE
# ============================================================

def build_hr_case_response(
    profile,
    ticket_result,
    email_result,
    citations,
):
    """
    HR case responses are deliberately deterministic.

    This guarantees the UX requested by the employee:

        Thanks, Jordan — I can help you report this.

        ### What I did for you right now

        - ...
        - ...
        - ...

        ### Relevant policy

        - ...
    """

    first_name = first_name_from_profile(
        profile
    )

    lines = [
        f"Thanks, {first_name} — I can help you report this.",
        "",
        "### What I did for you right now",
        "",
    ]

    # --------------------------------------------------------
    # Employee profile
    # --------------------------------------------------------

    if profile and not is_tool_error(profile):
        employee_match = re.search(
            r"Employee Profile\s*[—-]\s*(.*?)(?:\s+ID:|\Z)",
            profile,
            re.IGNORECASE,
        )

        employee_name = (
            employee_match.group(1).strip()
            if employee_match
            else first_name
        )

        id_match = re.search(
            r"\bID:\s*([A-Z0-9-]+)",
            profile,
            re.IGNORECASE,
        )

        employee_id = (
            id_match.group(1)
            if id_match
            else ""
        )

        if employee_id:
            lines.append(
                f"- Looked up your employee profile "
                f"({employee_name}, {employee_id})."
            )
        else:
            lines.append(
                f"- Looked up your employee profile "
                f"({employee_name})."
            )

    # --------------------------------------------------------
    # HR case
    # --------------------------------------------------------

    ticket_details = extract_ticket_details(
        ticket_result
    )

    if ticket_result and not is_tool_error(ticket_result):
        ticket_text = "Created an HR case for People Operations"

        if ticket_details.get("ticket_id"):
            ticket_text += (
                f": {ticket_details['ticket_id']}"
            )

        metadata = []

        if ticket_details.get("ticket_type"):
            ticket_type = ticket_details["ticket_type"]
            ticket_type = ticket_type.replace(
                "_",
                " ",
            ).title()
            metadata.append(ticket_type)

        if ticket_details.get("priority"):
            metadata.append(
                f"{ticket_details['priority'].title()} priority"
            )

        if ticket_details.get("status"):
            metadata.append(
                f"Status: {ticket_details['status'].title()}"
            )

        if metadata:
            ticket_text += (
                " (" + ", ".join(metadata) + ")"
            )

        lines.append(f"- {ticket_text}.")

    # --------------------------------------------------------
    # Email
    # --------------------------------------------------------

    if email_result and not is_tool_error(email_result):
        lines.append(
            "- Drafted an escalation email to People "
            "Operations for you to review (not sent)."
        )

    # --------------------------------------------------------
    # No successful actions
    # --------------------------------------------------------

    successful_ticket = (
        ticket_result
        and not is_tool_error(ticket_result)
    )

    successful_email = (
        email_result
        and not is_tool_error(email_result)
    )

    if not successful_ticket and not successful_email:
        lines.append(
            "- I wasn't able to complete the HR follow-up "
            "right now."
        )

    # --------------------------------------------------------
    # Policy
    # --------------------------------------------------------

    policy_section = format_policy_citations(
        citations
    )

    if policy_section:
        lines.extend(
            [
                "",
                policy_section,
            ]
        )

    # --------------------------------------------------------
    # Contact
    # --------------------------------------------------------

    if not successful_ticket and not successful_email:
        lines.extend(
            [
                "",
                "Please contact people@daisyhealth.com "
                "for assistance.",
            ]
        )

    return "\n".join(lines)


# ============================================================
# FINAL LLM ANSWER
# ============================================================

async def generate_final_answer(
    client,
    question,
    employee_id,
    workflow,
    tool_results,
    citations,
):
    """
    One final OpenAI call for normal informational requests.

    HR-case responses are handled deterministically because
    those responses contain explicit operational actions.
    """

    if workflow == "hr_case":
        return None

    context = build_final_context(
        tool_results
    )

    policy_context = ""

    if citations:
        policy_context = (
            "\n\nRelevant policy citations:\n"
            + json.dumps(
                citations,
                ensure_ascii=False,
            )
        )

    prompt = f"""
Employee ID:
{employee_id}

Employee question:
{question}

Request type:
{workflow}

HR information:
{context}

{policy_context}

Write the final employee-facing answer.

Requirements:

- Answer the employee's question directly.
- Be concise but useful.
- Do not mention internal tools or systems.
- Do not mention MCP, APIs, retrieval, RAG, Chroma, or LLMs.
- Do not expose technical errors.
- Do not invent facts.
- Use only the supplied HR information.
- If policy information is available, explain it in plain English.
- Mention the policy name or policy ID when useful.
- Do not dump the policy excerpt verbatim.
- Use Markdown headings/bullets when helpful.
- If there is insufficient information, direct the employee to
  people@daisyhealth.com.

Return ONLY the final employee-facing answer.
"""

    start = time.perf_counter()

    try:
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            max_completion_tokens=FINAL_MAX_COMPLETION_TOKENS,
        )

        duration = time.perf_counter() - start

        print(
            f"[TIMING] Final OpenAI LLM call: "
            f"{duration:.2f}s",
            flush=True,
        )

        answer = (
            response.choices[0]
            .message.content
            or ""
        ).strip()

        if not answer:
            print(
                "[AGENT] Final OpenAI returned no answer; "
                "using deterministic fallback.",
                flush=True,
            )
            return None

        return answer

    except Exception as exc:
        duration = time.perf_counter() - start

        print(
            f"[TIMING] Final OpenAI call ERROR: "
            f"{duration:.2f}s",
            flush=True,
        )

        print(
            f"[AGENT] Final OpenAI error: {exc}",
            flush=True,
        )

        return None


# ============================================================
# DETERMINISTIC FALLBACK
# ============================================================

def deterministic_fallback(
    workflow,
    tool_results,
    citations,
):
    """
    Clean fallback when OpenAI does not return an answer.
    """

    if workflow == "hr_case":
        return build_hr_case_response(
            profile=tool_results.get(
                "lookup_employee_profile",
                "",
            ),
            ticket_result=tool_results.get(
                "create_mock_hr_ticket",
                "",
            ),
            email_result=tool_results.get(
                "draft_hr_email",
                "",
            ),
            citations=citations,
        )

    pto = tool_results.get(
        "check_pto_balance",
        "",
    )

    benefits = tool_results.get(
        "lookup_benefits",
        "",
    )

    policy = tool_results.get(
        "search_policy_documents",
        "",
    )

    compliance = tool_results.get(
        "check_policy_compliance",
        "",
    )

    # --------------------------------------------------------
    # PTO
    # --------------------------------------------------------

    if workflow == "pto":
        if pto and not is_tool_error(pto):
            answer = clean_text(pto)

            policy_section = format_policy_citations(
                citations
            )

            if policy_section:
                answer += (
                    "\n\n" + policy_section
                )

            return answer

    # --------------------------------------------------------
    # Benefits
    # --------------------------------------------------------

    if workflow == "benefits":
        if benefits and not is_tool_error(benefits):
            answer = clean_text(benefits)

            policy_section = format_policy_citations(
                citations
            )

            if policy_section:
                answer += (
                    "\n\n" + policy_section
                )

            return answer

    # --------------------------------------------------------
    # Policy questions
    # --------------------------------------------------------

    if workflow in {
        "remote_work",
        "expense",
        "general",
    }:

        if policy and not is_tool_error(policy):
            answer = clean_policy_summary(policy)

            if compliance and not is_tool_error(compliance):
                answer += (
                    "\n\n"
                    + clean_text(compliance)
                )

            policy_section = format_policy_citations(
                citations
            )

            if policy_section:
                answer = (
                    answer
                    + "\n\n"
                    + policy_section
                )

            return answer

    return (
        "I don't have enough information from the available "
        "HR documentation. Please contact "
        "people@daisyhealth.com."
    )


# ============================================================
# MAIN AGENT
# ============================================================

async def run_agent(
    question,
    employee_id,
):
    total_start = time.perf_counter()

    tool_trace = []
    citations = []

    if not question or not question.strip():
        return {
            "answer": "Please enter a question.",
            "tool_trace": [],
            "citations": [],
            "error": None,
            "runtime_seconds": 0,
        }

    question = question.strip()

    try:
        client = get_openai_client()

        # ----------------------------------------------------
        # WORKFLOW
        # ----------------------------------------------------

        workflow_start = time.perf_counter()

        if USE_LLM_CLASSIFICATION:
            response = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": """
                    Classify the HR request into exactly one category:
                            pto
                            benefits
                            remote_work
                            expense
                            hr_case
                            general

                            Return only the category.
                            """,
                    },
                    {
                        "role": "user",
                        "content": question,
                    },
                ],
                max_completion_tokens=20,
            )

            workflow = (
                response.choices[0]
                .message.content
                .strip()
                .lower()
            )

            if workflow not in {
                "pto",
                "benefits",
                "remote_work",
                "expense",
                "hr_case",
                "general",
            }:
                workflow = "general"

        else:
            workflow = detect_workflow(question)

        print(
            f"[AGENT] Workflow: {workflow}",
            flush=True,
        )

        print(
            f"[TIMING] Workflow detection: "
            f"{time.perf_counter() - workflow_start:.4f}s",
            flush=True,
        )

        # ----------------------------------------------------
        # MCP
        # ----------------------------------------------------

        async with stdio_client(
            MCP_SERVER_PARAMS
        ) as (read, write):

            print(
                f"[TIMING] MCP startup: "
                f"{time.perf_counter() - total_start:.2f}s",
                flush=True,
            )

            async with ClientSession(
                read,
                write,
            ) as session:

                start = time.perf_counter()

                await session.initialize()

                print(
                    f"[TIMING] MCP initialize: "
                    f"{time.perf_counter() - start:.2f}s",
                    flush=True,
                )

                start = time.perf_counter()

                tools_response = await session.list_tools()
                available_tools = tools_response.tools

                print(
                    f"[TIMING] MCP list_tools: "
                    f"{time.perf_counter() - start:.2f}s",
                    flush=True,
                )

                available_names = [
                    tool.name
                    for tool in available_tools
                ]

                required_tools = get_workflow_tools(
                    workflow,
                    available_names,
                )

                print(
                    "[AGENT] Required tools: "
                    + ", ".join(required_tools),
                    flush=True,
                )

                tool_results = {}
                executed_signatures = set()
                policy_search_count = 0

                # ------------------------------------------------
                # TOOL EXECUTION
                # ------------------------------------------------

                for tool_name in required_tools:

                    if (
                        tool_name
                        == "search_policy_documents"
                    ):
                        if (
                            policy_search_count
                            >= MAX_POLICY_SEARCHES
                        ):
                            continue

                    tool_input = build_tool_arguments(
                        tool_name=tool_name,
                        question=question,
                        employee_id=employee_id,
                        workflow=workflow,
                        available_tools=available_tools,
                    )

                    # --------------------------------------------
                    # Required schema validation
                    # --------------------------------------------

                    required_fields = get_required_fields(
                        available_tools,
                        tool_name,
                    )

                    missing_fields = [
                        field
                        for field in required_fields
                        if field not in tool_input
                    ]

                    if missing_fields:
                        error_text = (
                            f"Cannot execute {tool_name}: "
                            f"missing required arguments: "
                            f"{', '.join(missing_fields)}"
                        )

                        print(
                            f"[AGENT] {error_text}",
                            flush=True,
                        )

                        tool_trace.append(
                            {
                                "tool": tool_name,
                                "args": tool_input,
                                "result": error_text,
                                "timestamp": datetime.now().strftime(
                                    "%H:%M:%S"
                                ),
                                "status": "✗ Missing arguments",
                                "duration_seconds": 0,
                            }
                        )

                        tool_results[
                            tool_name
                        ] = error_text

                        continue

                    # --------------------------------------------
                    # Duplicate protection
                    # --------------------------------------------

                    signature = (
                        tool_name
                        + ":"
                        + json.dumps(
                            tool_input,
                            sort_keys=True,
                            default=str,
                        )
                    )

                    if signature in executed_signatures:
                        continue

                    executed_signatures.add(signature)

                    # --------------------------------------------
                    # Execute
                    # --------------------------------------------

                    result_text = await execute_tool(
                        session=session,
                        tool_name=tool_name,
                        tool_input=tool_input,
                        tool_trace=tool_trace,
                    )

                    tool_results[
                        tool_name
                    ] = result_text

                    if (
                        tool_name
                        == "search_policy_documents"
                    ):
                        policy_search_count += 1

                        citations = extract_citations(
                            result_text
                        )

                # ------------------------------------------------
                # HR CASE RESPONSE
                # ------------------------------------------------

                if workflow == "hr_case":
                    answer = build_hr_case_response(
                        profile=tool_results.get(
                            "lookup_employee_profile",
                            "",
                        ),
                        ticket_result=tool_results.get(
                            "create_mock_hr_ticket",
                            "",
                        ),
                        email_result=tool_results.get(
                            "draft_hr_email",
                            "",
                        ),
                        citations=citations,
                    )

                # ------------------------------------------------
                # NORMAL RESPONSE
                # ------------------------------------------------

                else:
                    answer = await generate_final_answer(
                        client=client,
                        question=question,
                        employee_id=employee_id,
                        workflow=workflow,
                        tool_results=tool_results,
                        citations=citations,
                    )

                    if not answer:
                        answer = deterministic_fallback(
                            workflow=workflow,
                            tool_results=tool_results,
                            citations=citations,
                        )

                    # --------------------------------------------
                    # Always make citations visible to employee
                    # --------------------------------------------

                    policy_section = format_policy_citations(
                        citations
                    )

                    if (
                        policy_section
                        and "### Relevant policy"
                        not in answer
                    ):
                        answer = (
                            answer.rstrip()
                            + "\n\n"
                            + policy_section
                        )

        # ----------------------------------------------------
        # COMPLETE
        # ----------------------------------------------------

        total_time = (
            time.perf_counter()
            - total_start
        )

        print(
            f"[TIMING] TOTAL agent runtime: "
            f"{total_time:.2f}s",
            flush=True,
        )

        print(
            f"[TIMING] Tool calls: "
            f"{len(tool_trace)}",
            flush=True,
        )

        print(
            f"[AGENT] Returning answer to Streamlit "
            f"({len(answer)} characters).",
            flush=True,
        )

        return {
            "answer": answer.strip(),
            "tool_trace": tool_trace,
            "citations": citations,
            "error": None,
            "runtime_seconds": round(
                total_time,
                2,
            ),
        }

    except Exception as exc:
        import traceback

        traceback.print_exc()

        total_time = (
            time.perf_counter()
            - total_start
        )

        print(
            f"[AGENT] Top-level error: {exc}",
            flush=True,
        )

        print(
            f"[TIMING] TOTAL agent runtime: "
            f"{total_time:.2f}s",
            flush=True,
        )

        return {
            "answer": (
                "I encountered an error while processing "
                "your request. Please try again or contact "
                "people@daisyhealth.com."
            ),
            "tool_trace": tool_trace,
            "citations": citations,
            "error": str(exc),
            "runtime_seconds": round(
                total_time,
                2,
            ),
        }


# ============================================================
# STREAMLIT WRAPPER
# ============================================================

def run_agent_sync(
    question,
    employee_id,
):
    return asyncio.run(
        run_agent(
            question=question,
            employee_id=employee_id,
        )
    )


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print("\n🌸 Testing Daisy Health Agent...\n")

    result = run_agent_sync(
        question="How much PTO do I have?",
        employee_id="EMP-001",
    )

    print("\n==============================")
    print("ANSWER")
    print("==============================\n")
    print(result["answer"])

    print("\n==============================")
    print("PERFORMANCE")
    print("==============================")
    print(
        f"Total runtime: "
        f"{result.get('runtime_seconds', '?')}s"
    )
    print(
        f"Tool calls: "
        f"{len(result['tool_trace'])}"
    )

    print("\n==============================")
    print("TOOL TRACE")
    print("==============================")

    for tool in result["tool_trace"]:
        print(
            f"- {tool['tool']}: "
            f"{tool['status']} "
            f"({tool.get('duration_seconds', '?')}s)"
        )

    print(
        f"\nCitations: "
        f"{len(result['citations'])}"
    )

    for citation in result["citations"]:
        print(
            f"- {citation.get('title', 'HR Policy')} "
            f"— "
            f"{citation.get('policy_id', '')}"
        )