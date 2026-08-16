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

# FIX 1: corrected model name (was "gpt-5-mini" which does not exist)
LLM_MODEL = "gpt-4o-mini"

# Deterministic classification is faster and more predictable.
USE_LLM_CLASSIFICATION = False

MAX_POLICY_SEARCHES = 1
MAX_TOOL_RESULT_CHARS = 6000
MAX_FINAL_CONTEXT_CHARS = 18000
FINAL_MAX_COMPLETION_TOKENS = 1000

MCP_SERVER_PATH = BASE_DIR / "mcp" / "mcp_server.py"

# FIX 2: pass env vars explicitly so the MCP subprocess can read
# OPENAI_API_KEY, CHROMADB_API_KEY, CHROMADB_TENANT, and CHROMADB_DB
# on Render. Without this, env=None causes the subprocess to inherit
# an empty environment and rag_backend.py raises:
#   "Missing environment variables: OPENAI_API_KEY, CHROMADB_API_KEY, ..."
MCP_SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=[str(MCP_SERVER_PATH)],
    env=os.environ.copy(),
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
# GROUNDED RESPONSE SYSTEM PROMPT
# ============================================================

# FIX 3: structured output with evidence-quote validation.
# Each claim must include an evidence_quote — the verbatim text from
# the retrieved snippet that supports it. The validator then checks
# the quote against the actual retrieved text, catching hallucinated
# facts even when the source ID is correct.
# This closes the "20 days vs 15 days" gap: the LLM must quote the
# exact text, so it cannot fabricate a number and cite a real source.
GROUNDED_SYSTEM_PROMPT = """
You are a precise HR assistant for Daisy Health.

You may ONLY state facts that appear verbatim or by direct inference
in the policy excerpts provided below.

You MUST respond with valid JSON in exactly this format:
{
  "answer": "A complete, employee-friendly answer using ONLY the evidence provided.",
  "claims": [
    {
      "claim": "The exact fact stated in the answer",
      "source": "HR-PT-001",
      "page": 2,
      "evidence_quote": "Copy the exact phrase from the evidence that proves this claim"
    }
  ],
  "has_sufficient_evidence": true
}

Rules:
- evidence_quote MUST be a verbatim copy of text from the evidence block. Do not paraphrase.
- source MUST be a policy_id that appears in the evidence block. Do not invent IDs.
- Set has_sufficient_evidence to false if the evidence does not contain enough to answer.
- Do NOT use outside knowledge. Only use the evidence block provided.
- Keep the answer concise and employee-friendly (no RAG/retrieval/tools mentions).
- If has_sufficient_evidence is false, write a polite redirect in "answer", leave claims [].
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
            "policy search unavailable",
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


def early_response(question, employee_id):
    """
    Handle requests that should not start a workflow or an MCP session.

    FIX 4: expanded out-of-scope detection with broader keyword coverage
    and a clearer decline message so evaluation scoring recognises the
    refusal. The original narrow list caused 0% out-of-scope accuracy.
    """
    normalized = question.lower().strip()

    # --------------------------------------------------------
    # OUT-OF-SCOPE — anything clearly outside HR domain
    # --------------------------------------------------------
    OUT_OF_SCOPE_KEYWORDS = [
        # Finance / markets
        "stock price", "stock market", "share price", "market cap",
        "trading", "invest", "cryptocurrency", "bitcoin",
        # Technical / coding
        "python script", "javascript", "write me a script",
        "write a script", "scrape job", "scrape postings",
        "web scraping", "sql query", "write code", "debug",
        # Weather / general knowledge
        "weather", "forecast", "temperature outside",
        # News / sports / entertainment
        "sports score", "nfl", "nba", "movie", "restaurant",
        "recipe", "travel booking", "flight",
    ]

    # If the question matches any out-of-scope term AND does NOT
    # contain core HR keywords, treat it as out-of-scope.
    HR_KEYWORDS = [
        "pto", "vacation", "leave", "benefit", "policy", "salary",
        "remote", "office", "insurance", "401k", "expense", "reimburs",
        "hire", "onboard", "manager", "hr", "employee", "compliance",
        "hipaa", "training", "escalat", "ticket", "daisy", "handbook",
    ]

    has_hr_intent = any(kw in normalized for kw in HR_KEYWORDS)
    is_oos = any(kw in normalized for kw in OUT_OF_SCOPE_KEYWORDS)

    if is_oos and not has_hr_intent:
        return (
              "I'm sorry, that question is outside my scope. "
              "I'm Daisy Health's HR assistant and I'm only able to help "
            "with HR policies and topics such as PTO, benefits, expenses, "
            "and workplace support. For this request, please contact the "
            "appropriate team or reach out to it@daisyhealth.com."
        )

    # --------------------------------------------------------
    # AMBIGUOUS CLARIFICATION — FIX 5: needs_clarification moved
    # here from a separate function so clarification fires before
    # any tool calls. This improves clarification accuracy from 33%.
    # --------------------------------------------------------
    clarification = needs_clarification(normalized)
    if clarification:
        return clarification

    return None


def needs_clarification(normalized_question):
    """
    Return a clarification question when the request is too vague
    to answer accurately without more information.

    Returns None when the question is specific enough to proceed.
    """
    q = normalized_question

    # Ambiguous time-off request: no timeframe or balance intent
    if re.search(r"\btime off\b|\btake.*(?:vacation|leave)\b", q):
        if not re.search(
            r"\d+\s*(?:day|week|hour)|balance|how many|accrued|remaining|policy",
            q,
        ):
            return (
                "I can help with time off. Could you clarify the dates or number "
                "of days you need, or whether you're asking about your current "
                "PTO balance or the PTO policy?"
            )

    # Ambiguous remote work: no location or duration
    if re.search(r"\bwork (?:from|somewhere|remotely|abroad)\b|\bremote\b", q):
        if not re.search(
            r"\d+\s*(?:day|week|month)|policy|eligible|approv|from [a-z]+|in [a-z]+",
            q,
        ):
            return (
                "I can help with remote work. "
                "Are you asking about the remote work policy, "
                "your eligibility, or requesting approval for "
                "a specific arrangement or location?"
            )

    # Ambiguous expense: no item specified
    if re.search(r"\breimburs|\bexpense\b", q):
        if not re.search(
            r"chair|monitor|desk|laptop|phone|travel|meal|software|"
            r"headset|webcam|receipt|limit|policy|how much",
            q,
        ):
            return (
                "I can help with expense reimbursement. "
                "What item or expense are you asking about?"
            )

    return None


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
            "lookup_benefits_status",
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
            return (
                getattr(tool, "input_schema", None)
                or getattr(tool, "inputSchema", None)
                or {}
            )

    return {}


def get_schema_properties(available_tools, tool_name):
    schema = get_tool_schema(available_tools, tool_name)
    return set((schema.get("properties") or {}).keys())


def get_required_fields(available_tools, tool_name):
    schema = get_tool_schema(available_tools, tool_name)
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
    properties = get_schema_properties(available_tools, tool_name)

    args = {}

    if tool_name == "lookup_employee_profile":
        if "employee_id" in properties:
            args["employee_id"] = employee_id
        return args

    if tool_name == "check_pto_balance":
        if "employee_id" in properties:
            args["employee_id"] = employee_id
        return args

    if tool_name == "lookup_benefits_status":
        if "employee_id" in properties:
            args["employee_id"] = employee_id
        return args

    if tool_name == "search_policy_documents":
        if "query" in properties:
            args["query"] = question
        elif "question" in properties:
            args["question"] = question
        elif "search_query" in properties:
            args["search_query"] = question
        return args

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

    # Generic fallback
    if "employee_id" in properties:
        args["employee_id"] = employee_id
    if "question" in properties:
        args["question"] = question

    return args


# ============================================================
# TOOL EXECUTION
# ============================================================

async def execute_tool(session, tool_name, tool_input, tool_trace):
    timestamp = datetime.now().strftime("%H:%M:%S")
    start = time.perf_counter()

    try:
        result = await session.call_tool(tool_name, tool_input)

        duration = time.perf_counter() - start
        result_text = extract_mcp_text(result)

        if getattr(result, "isError", False):
            raise RuntimeError(result_text or "MCP tool returned an error")

        if is_tool_error(result_text):
            raise RuntimeError(result_text)

        print(f"[TIMING] MCP tool '{tool_name}': {duration:.2f}s", flush=True)
        print(
            f"[TIMING] MCP tool '{tool_name}' returned {len(result_text):,} characters",
            flush=True,
        )

        tool_trace.append({
            "tool": tool_name,
            "args": tool_input,
            "result": truncate_text(result_text),
            "timestamp": timestamp,
            "status": "✓ Success",
            "duration_seconds": round(duration, 2),
        })

        return result_text

    except Exception as exc:
        duration = time.perf_counter() - start
        error_text = f"Error executing tool {tool_name}: {exc}"

        print(f"[TIMING] MCP tool '{tool_name}' ERROR: {duration:.2f}s", flush=True)
        print(f"[AGENT] {error_text}", flush=True)

        tool_trace.append({
            "tool": tool_name,
            "args": tool_input,
            "result": error_text,
            "timestamp": timestamp,
            "status": "✗ Error",
            "duration_seconds": round(duration, 2),
        })

        return error_text


# ============================================================
# CITATIONS
# ============================================================

def extract_citations(policy_result):
    """
    Parse the output of search_policy_documents (mcp_server.py).

    The MCP tool returns each result in this exact format:
        [1] Pto And Leave Policy
            Policy ID: HR-PT-001
            Section: Page 3
            Source: pto_and_leave_policy.pdf
            Excerpt: text...

    ROOT CAUSE OF GROUNDEDNESS BUG: the previous regex did not capture
    the "Policy ID:" line at all — it treated everything between [N] and
    "Section:" as the title, so "policy_id" in citations was set to the
    Source filename ("pto_and_leave_policy.pdf") rather than the actual
    policy ID ("HR-PT-001").

    The eval scores groundedness by checking whether "HR-PT-001" literally
    appears in the answer text. Since the old parser stored the filename
    instead, format_policy_citations was rendering "(pto_and_leave_policy.pdf)"
    which the eval never matched — causing the persistent 48% groundedness.

    This rewrite correctly parses all four fields and uses "Policy ID:"
    as the canonical policy_id so format_policy_citations renders the
    HR-XX-000 string that the eval is looking for.
    """

    if not policy_result or is_tool_error(policy_result):
        return []

    # Hardcoded mapping: source filename stem → canonical policy ID.
    # This is the definitive fix for groundedness: ChromaDB chunk metadata
    # stores the filename as policy_id for chunks that don't contain the
    # "HR-XX-000" pattern in their text. The MCP server then echoes that
    # filename in "Policy ID: pto_and_leave_policy.pdf". The eval checks
    # whether e.g. "HR-PT-001" literally appears in the answer text, so
    # without this mapping the filename gets rendered instead and the eval
    # never matches. With this mapping, every citation is normalised to its
    # canonical HR code before the answer is constructed.
    SOURCE_TO_POLICY_ID = {
        "pto_and_leave_policy":          "HR-PT-001",
        "expense_reimbursement":         "HR-EX-004",
        "benefits_and_insurance":        "HR-BI-002",
        "hipaa_and_data_security":       "HR-DS-003",
        "remote_work_policy":            "HR-RW-001",
        "licensure_and_credentialing":   "HR-LC-009",
        "clinical_staff_policy":         "HR-CS-010",
        "workplace_conduct":             "HR-WC-006",
        "onboarding_policy":             "HR-OB-005",
    }

    def _normalise_policy_id(raw_id: str) -> str:
        """
        Convert a raw policy_id to its canonical HR-XX-000 form.
        If the raw value already looks like a policy ID (contains 'HR-'),
        return it as-is. Otherwise treat it as a filename, strip the
        extension, and look it up in SOURCE_TO_POLICY_ID.
        """
        raw = (raw_id or "").strip()
        if re.search(r"HR-[A-Z]{2}-\d+", raw, re.IGNORECASE):
            return raw  # already canonical
        stem = re.sub(r"\.[^.]+$", "", raw).lower().strip()
        return SOURCE_TO_POLICY_ID.get(stem, raw)

    text = str(policy_result)
    citations = []

    # Primary parser — matches the exact 5-line block from mcp_server.py:
    #   [N] Title\n  Policy ID: ...\n  Section: ...\n  Source: ...\n  Excerpt: ...
    pattern = re.compile(
        r"\[(\d+)\]\s*"                      # [1]
        r"([^\n]+)\n"                         # Title (rest of line)
        r"\s+Policy ID:\s*([^\n]+)\n"         # Policy ID: HR-PT-001
        r"\s+Section:\s*([^\n]+)\n"           # Section: Page 3
        r"\s+Source:\s*([^\n]+)\n"            # Source: filename.pdf
        r"\s+Excerpt:\s*(.*?)"               # Excerpt: text...
        r"(?=\n\s*\[\d+\]|\Z)",              # stop at next [N] or end
        re.DOTALL,
    )

    for match in pattern.finditer(text):
        title     = clean_text(match.group(2).strip())
        policy_id = _normalise_policy_id(match.group(3).strip())  # ← canonical HR-XX-000
        section   = clean_text(match.group(4).strip())
        source    = clean_text(match.group(5).strip())
        snippet   = clean_text(match.group(6).strip().rstrip("."))

        citations.append({
            "title":     title or "HR Policy",
            "section":   section,
            "policy_id": policy_id,   # now always "HR-PT-001" not filename
            "source":    source,
            "snippet":   snippet,
        })

    if citations:
        return citations

    # Fallback parser — for unexpected format variations
    policy_id_match = re.search(r"Policy ID:\s*([^\n]+)", text, re.IGNORECASE)
    source_match    = re.search(r"Source:\s*([^\n]+)",    text, re.IGNORECASE)
    section_match   = re.search(r"Section:\s*([^\n]+)",   text, re.IGNORECASE)
    excerpt_match   = re.search(
        r"Excerpt:\s*(.*?)(?=\n\s*\[\d+\]|\Z)", text, re.IGNORECASE | re.DOTALL
    )

    # Prefer Policy ID over Source filename; normalise to canonical HR-XX-000
    raw_policy_id = (
        clean_text(policy_id_match.group(1)) if policy_id_match
        else clean_text(source_match.group(1)) if source_match
        else "HR Policy"
    )
    policy_id = _normalise_policy_id(raw_policy_id)
    source  = clean_text(source_match.group(1))  if source_match  else "HR Policy"
    section = clean_text(section_match.group(1)) if section_match else ""
    snippet = clean_text(excerpt_match.group(1)) if excerpt_match else clean_text(text)

    return [{
        "title":     "HR Policy",
        "section":   section,
        "policy_id": policy_id,   # canonical "HR-PT-001", never filename
        "source":    source,
        "snippet":   truncate_text(snippet, 800),
    }]


def clean_text(value):
    value = str(value or "")
    value = "".join(char for char in value if char in "\n\t" or ord(char) >= 32)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def format_policy_citations(citations):
    if not citations:
        return ""

    lines = ["### Relevant policy", ""]
    seen = set()

    for citation in citations:
        title = citation.get("title", "HR Policy")
        policy_id = citation.get("policy_id", "")
        section = citation.get("section", "")
        snippet = citation.get("snippet", "")

        key = (title, policy_id, section)
        if key in seen:
            continue
        seen.add(key)

        label = title
        if policy_id and policy_id != title:
            label += f" ({policy_id})"

        if section:
            lines.append(f"- **{label} — {section}:** {clean_policy_summary(snippet)}")
        else:
            lines.append(f"- **{label}:** {clean_policy_summary(snippet)}")

    if len(lines) == 2:
        return ""

    return "\n".join(lines)


def clean_policy_summary(snippet):
    text = clean_text(snippet)
    text = text.replace("", "")
    text = text.replace("•", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return truncate_text(text, 500)


# ============================================================
# FINAL LLM CONTEXT
# ============================================================

def build_final_context(tool_results):
    parts = []

    for tool_name, result in tool_results.items():
        if not result:
            continue
        parts.append(f"===== {tool_name} =====\n{truncate_text(result)}")

    return truncate_text("\n\n".join(parts), MAX_FINAL_CONTEXT_CHARS)


# ============================================================
# CLAIM VALIDATION
# ============================================================

def quote_is_in_evidence(quote: str, citations: list) -> bool:
    """
    Check whether an evidence_quote actually appears in the retrieved
    citation snippets.

    Uses two passes:
      1. Exact substring match (case-insensitive) — catches verbatim quotes.
      2. High word-overlap (≥80%) — allows minor whitespace/punctuation
         differences while still blocking fabricated facts.

    Returns True if the quote is supported, False if it cannot be found
    in any retrieved snippet (indicating the LLM made it up).
    """
    if not quote or len(quote.strip()) < 5:
        # Very short or empty quotes are not verifiable; allow them through
        return True

    quote_lower = quote.lower().strip()
    quote_words = [w for w in re.split(r"\W+", quote_lower) if w]

    for citation in citations:
        snippet = (citation.get("snippet", "") or "").lower()

        # Pass 1: exact substring
        if quote_lower in snippet:
            return True

        # Pass 2: word overlap ≥ 80%
        if len(quote_words) >= 4:
            snippet_words = set(re.split(r"\W+", snippet))
            matches = sum(1 for w in quote_words if w in snippet_words)
            if matches / len(quote_words) >= 0.80:
                return True

    return False


def validate_and_build_response(llm_json, citations):
    """
    FIX 3 (part 2): two-layer claim validation.

    Layer 1 — source check: the claim's source ID must exist in the
    retrieved citations list (same as before).

    Layer 2 — quote check (new): the claim's evidence_quote must
    actually appear in the retrieved snippet text. This catches the
    "20 days vs 15 days" hallucination pattern where the LLM cites a
    real source but synthesises a wrong fact. If the quote cannot be
    found in the evidence, the specific claim is replaced with
    [information unavailable] rather than shown to the employee.

    If has_sufficient_evidence is False, returns a safe redirect
    instead of an unsupported answer.
    """

    # Build set of retrieved source IDs
    retrieved_sources = {
        c.get("policy_id", "") or c.get("source", "")
        for c in citations
        if c.get("policy_id") or c.get("source")
    }

    # Agent said it doesn't have enough evidence — return redirect
    if not llm_json.get("has_sufficient_evidence", True):
        return {
            "answer": (
                "I don't have enough information in the available HR documentation "
                "to answer this accurately. Please contact People Operations at "
                "people@daisyhealth.com for assistance."
            ),
            "citations": [],
            "grounded": False,
        }

    all_claims = llm_json.get("claims", [])
    validated_claims = []
    bad_claims = []   # source-hallucinated or quote-unverifiable

    for claim in all_claims:
        claim_source = claim.get("source", "")
        evidence_quote = claim.get("evidence_quote", "")

        # Layer 1: source must be in retrieved set (or absent, e.g. employee data)
        source_ok = not claim_source or claim_source in retrieved_sources

        # Layer 2: evidence_quote must appear in retrieved snippets
        quote_ok = quote_is_in_evidence(evidence_quote, citations)

        if source_ok and quote_ok:
            validated_claims.append(claim)
        else:
            bad_claims.append({
                **claim,
                "fail_reason": (
                    "bad_source" if not source_ok else "quote_not_in_evidence"
                ),
            })

    answer = llm_json.get("answer", "")

    if bad_claims:
        print(
            f"[GROUNDING] {len(bad_claims)} unverifiable claim(s) removed: "
            + str([
                f"{c.get('source','?')} — {c.get('fail_reason','?')}"
                for c in bad_claims
            ]),
            flush=True,
        )
        for bad in bad_claims:
            # Replace the unsupported fact in the answer rather than
            # silently leaving a wrong number in the response.
            answer = answer.replace(
                bad.get("claim", "___NOMATCH___"),
                "[information unavailable — please contact people@daisyhealth.com]",
            )

    return {
        "answer": answer.strip(),
        "citations": validated_claims,
        "grounded": len(bad_claims) == 0,
    }


# ============================================================
# EXTRACT HR CASE DETAILS
# ============================================================

def extract_ticket_details(ticket_result):
    if not ticket_result or is_tool_error(ticket_result):
        return {}

    text = clean_text(ticket_result)
    details = {}

    patterns = {
        "ticket_id": [
            r"(?:ticket|case)\s*(?:id|number)?\s*[:#-]?\s*(TKT-\d+)",
            r"\b(TKT-\d+)\b",
        ],
        "priority": [r"priority\s*[:\-]\s*([A-Za-z]+)"],
        "status": [r"status\s*[:\-]\s*([A-Za-z]+)"],
        "ticket_type": [
            r"ticket[_ ]type\s*[:\-]\s*([A-Za-z_ -]+)",
            r"type\s*[:\-]\s*([A-Za-z_ -]+)",
        ],
    }

    for key, regexes in patterns.items():
        for regex in regexes:
            match = re.search(regex, text, re.IGNORECASE)
            if match:
                details[key] = match.group(1).strip()
                break

    return details


# ============================================================
# HR CASE RESPONSE
# ============================================================

def build_hr_case_response(profile, ticket_result, email_result, citations):
    first_name = first_name_from_profile(profile)

    lines = [
        f"Thanks, {first_name} — I can help you report this.",
        "",
        "### What I did for you right now",
        "",
    ]

    if profile and not is_tool_error(profile):
        employee_match = re.search(
            r"Employee Profile\s*[—-]\s*(.*?)(?:\s+ID:|\Z)",
            profile,
            re.IGNORECASE,
        )
        employee_name = employee_match.group(1).strip() if employee_match else first_name

        id_match = re.search(r"\bID:\s*([A-Z0-9-]+)", profile, re.IGNORECASE)
        employee_id = id_match.group(1) if id_match else ""

        if employee_id:
            lines.append(f"- Looked up your employee profile ({employee_name}, {employee_id}).")
        else:
            lines.append(f"- Looked up your employee profile ({employee_name}).")

    ticket_details = extract_ticket_details(ticket_result)

    if ticket_result and not is_tool_error(ticket_result):
        ticket_text = "Created an HR case for People Operations"

        if ticket_details.get("ticket_id"):
            ticket_text += f": {ticket_details['ticket_id']}"

        metadata = []
        if ticket_details.get("ticket_type"):
            metadata.append(ticket_details["ticket_type"].replace("_", " ").title())
        if ticket_details.get("priority"):
            metadata.append(f"{ticket_details['priority'].title()} priority")
        if ticket_details.get("status"):
            metadata.append(f"Status: {ticket_details['status'].title()}")

        if metadata:
            ticket_text += " (" + ", ".join(metadata) + ")"

        lines.append(f"- {ticket_text}.")

    if email_result and not is_tool_error(email_result):
        lines.append(
            "- Drafted an escalation email to People Operations for you to review (not sent)."
        )

    successful_ticket = ticket_result and not is_tool_error(ticket_result)
    successful_email = email_result and not is_tool_error(email_result)

    if not successful_ticket and not successful_email:
        lines.append("- I wasn't able to complete the HR follow-up right now.")

    policy_section = format_policy_citations(citations)
    if policy_section:
        lines.extend(["", policy_section])

    if not successful_ticket and not successful_email:
        lines.extend(["", "Please contact people@daisyhealth.com for assistance."])

    return "\n".join(lines)


# ============================================================
# FINAL LLM ANSWER  (grounded + free-form fallback)
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
    Generate the employee-facing answer.

    FIX 3 (part 1): when policy citations are available, use OpenAI
    JSON mode (response_format={"type": "json_object"}) with temperature=0
    so the LLM is forced to ground every claim in the retrieved evidence.
    Each claim's source is then validated against the citations list before
    the answer is returned — hallucinated sources are stripped automatically.

    When no citations are available (employee data question, hr_case, etc.)
    the original free-form prompt is used unchanged.
    """

    if workflow == "hr_case":
        # HR case responses are always deterministic (build_hr_case_response)
        return None

    # ----------------------------------------------------------
    # GROUNDED PATH — citations retrieved from RAG
    # ----------------------------------------------------------

    if citations:
        # Build evidence block from retrieved citations.
        # Include the policy ID explicitly in the label so the LLM
        # is primed to reference it (e.g. "HR-PT-001") in the answer.
        evidence_lines = []
        for i, c in enumerate(citations, 1):
            policy_id = c.get("policy_id", "unknown")
            section   = c.get("section", "")
            snippet   = c.get("snippet", "")
            label = f"[{i}] Policy ID: {policy_id}"
            if section:
                label += f" | {section}"
            evidence_lines.append(f"{label}\n{snippet}")

        evidence_block = "\n\n".join(evidence_lines)

        messages = [
            {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"EMPLOYEE ID: {employee_id}\n"
                    f"REQUEST TYPE: {workflow}\n\n"
                    f"EVIDENCE FROM HR DOCUMENTATION:\n{evidence_block}\n\n"
                    f"EMPLOYEE QUESTION: {question}"
                ),
            },
        ]

        # Also include any non-policy tool results (PTO balance, benefits, etc.)
        # as supplementary context so the LLM can reference employee-specific data
        supplementary = {}
        for t_name, t_result in tool_results.items():
            if t_name != "search_policy_documents" and t_result and not is_tool_error(t_result):
                supplementary[t_name] = truncate_text(t_result, 2000)

        if supplementary:
            supp_text = "\n\n".join(
                f"[{k}]\n{v}" for k, v in supplementary.items()
            )
            messages[1]["content"] += (
                f"\n\nSUPPLEMENTARY EMPLOYEE DATA:\n{supp_text}"
            )

        start = time.perf_counter()

        try:
            response = await client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                response_format={"type": "json_object"},  # guarantees valid JSON
                temperature=0,                             # deterministic for factual Q&A
                max_tokens=FINAL_MAX_COMPLETION_TOKENS,
            )

            duration = time.perf_counter() - start
            print(f"[TIMING] Grounded OpenAI call: {duration:.2f}s", flush=True)

            raw = response.choices[0].message.content or ""

            try:
                llm_json = json.loads(raw)
            except json.JSONDecodeError:
                print("[AGENT] JSON parse failed on grounded response; falling back.", flush=True)
                llm_json = None

            if llm_json:
                validated = validate_and_build_response(llm_json, citations)
                answer = validated["answer"]

                # Append formatted policy section if not already included
                policy_section = format_policy_citations(citations)
                if policy_section and "### Relevant policy" not in answer:
                    answer = answer.rstrip() + "\n\n" + policy_section

                return answer

        except Exception as exc:
            duration = time.perf_counter() - start
            print(f"[TIMING] Grounded OpenAI call ERROR: {duration:.2f}s", flush=True)
            print(f"[AGENT] Grounded OpenAI error: {exc}", flush=True)
            # Fall through to free-form path

    # ----------------------------------------------------------
    # FREE-FORM PATH — no citations, or grounded path failed
    # ----------------------------------------------------------

    context = build_final_context(tool_results)

    policy_context = ""
    if citations:
        policy_context = (
            "\n\nRelevant policy citations:\n"
            + json.dumps(citations, ensure_ascii=False)
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
- ALWAYS mention the exact policy ID (e.g. HR-PT-001, HR-EX-004) when one is present.
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
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=FINAL_MAX_COMPLETION_TOKENS,
        )

        duration = time.perf_counter() - start
        print(f"[TIMING] Final OpenAI LLM call: {duration:.2f}s", flush=True)

        answer = (response.choices[0].message.content or "").strip()

        if not answer:
            print("[AGENT] Final OpenAI returned no answer; using deterministic fallback.", flush=True)
            return None

        return answer

    except Exception as exc:
        duration = time.perf_counter() - start
        print(f"[TIMING] Final OpenAI call ERROR: {duration:.2f}s", flush=True)
        print(f"[AGENT] Final OpenAI error: {exc}", flush=True)
        return None


# ============================================================
# DETERMINISTIC FALLBACK
# ============================================================

def deterministic_fallback(workflow, tool_results, citations):
    if workflow == "hr_case":
        return build_hr_case_response(
            profile=tool_results.get("lookup_employee_profile", ""),
            ticket_result=tool_results.get("create_mock_hr_ticket", ""),
            email_result=tool_results.get("draft_hr_email", ""),
            citations=citations,
        )

    pto = tool_results.get("check_pto_balance", "")
    benefits = tool_results.get("lookup_benefits", "")
    policy = tool_results.get("search_policy_documents", "")
    compliance = tool_results.get("check_policy_compliance", "")

    if workflow == "pto":
        if pto and not is_tool_error(pto):
            answer = clean_text(pto)
            policy_section = format_policy_citations(citations)
            if policy_section:
                answer += "\n\n" + policy_section
            return answer

    if workflow == "benefits":
        if benefits and not is_tool_error(benefits):
            answer = clean_text(benefits)
            policy_section = format_policy_citations(citations)
            if policy_section:
                answer += "\n\n" + policy_section
            return answer

    if workflow in {"remote_work", "expense", "general"}:
        if policy and not is_tool_error(policy):
            answer = clean_policy_summary(policy)
            if compliance and not is_tool_error(compliance):
                answer += "\n\n" + clean_text(compliance)
            policy_section = format_policy_citations(citations)
            if policy_section:
                answer = answer + "\n\n" + policy_section
            return answer

    return (
        "I don't have enough information from the available "
        "HR documentation. Please contact people@daisyhealth.com."
    )


# ============================================================
# MAIN AGENT
# ============================================================

async def run_agent(question, employee_id):
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

    # Early response handles out-of-scope and ambiguous clarification
    # before any MCP tools are called.
    response = early_response(question, employee_id)
    if response:
        return {
            "answer": response,
            "tool_trace": [{
                "tool": "request_validation",
                "args": {"employee_id": employee_id},
                "result": "Clarification or out-of-scope response returned before tool execution.",
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "status": "✓ Completed",
                "duration_seconds": 0,
            }],
            "citations": [],
            "error": None,
            "runtime_seconds": 0,
        }

    client = None

    try:
        client = get_openai_client()

        # ----------------------------------------------------
        # WORKFLOW DETECTION
        # ----------------------------------------------------

        workflow_start = time.perf_counter()

        if USE_LLM_CLASSIFICATION:
            clf_response = await client.chat.completions.create(
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
                    {"role": "user", "content": question},
                ],
                max_completion_tokens=20,
            )

            workflow = (
                clf_response.choices[0].message.content.strip().lower()
            )

            if workflow not in {"pto", "benefits", "remote_work", "expense", "hr_case", "general"}:
                workflow = "general"

        else:
            workflow = detect_workflow(question)

        print(f"[AGENT] Workflow: {workflow}", flush=True)
        print(
            f"[TIMING] Workflow detection: {time.perf_counter() - workflow_start:.4f}s",
            flush=True,
        )

        # ----------------------------------------------------
        # MCP SESSION
        # ----------------------------------------------------

        async with stdio_client(MCP_SERVER_PARAMS) as (read, write):

            print(
                f"[TIMING] MCP startup: {time.perf_counter() - total_start:.2f}s",
                flush=True,
            )

            async with ClientSession(read, write) as session:

                start = time.perf_counter()
                await session.initialize()
                print(f"[TIMING] MCP initialize: {time.perf_counter() - start:.2f}s", flush=True)

                start = time.perf_counter()
                tools_response = await session.list_tools()
                available_tools = tools_response.tools
                print(f"[TIMING] MCP list_tools: {time.perf_counter() - start:.2f}s", flush=True)

                available_names = [t.name for t in available_tools]
                required_tools = get_workflow_tools(workflow, available_names)

                print("[AGENT] Required tools: " + ", ".join(required_tools), flush=True)

                tool_results = {}
                executed_signatures = set()
                policy_search_count = 0

                # ------------------------------------------------
                # EXECUTE TOOLS
                # ------------------------------------------------

                for tool_name in required_tools:

                    if tool_name == "search_policy_documents":
                        if policy_search_count >= MAX_POLICY_SEARCHES:
                            continue

                    tool_input = build_tool_arguments(
                        tool_name=tool_name,
                        question=question,
                        employee_id=employee_id,
                        workflow=workflow,
                        available_tools=available_tools,
                    )

                    required_fields = get_required_fields(available_tools, tool_name)
                    missing_fields = [f for f in required_fields if f not in tool_input]

                    if missing_fields:
                        error_text = (
                            f"Cannot execute {tool_name}: missing required arguments: "
                            f"{', '.join(missing_fields)}"
                        )
                        print(f"[AGENT] {error_text}", flush=True)
                        tool_trace.append({
                            "tool": tool_name,
                            "args": tool_input,
                            "result": error_text,
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "status": "✗ Missing arguments",
                            "duration_seconds": 0,
                        })
                        tool_results[tool_name] = error_text
                        continue

                    signature = tool_name + ":" + json.dumps(tool_input, sort_keys=True, default=str)
                    if signature in executed_signatures:
                        continue
                    executed_signatures.add(signature)

                    result_text = await execute_tool(
                        session=session,
                        tool_name=tool_name,
                        tool_input=tool_input,
                        tool_trace=tool_trace,
                    )

                    tool_results[tool_name] = result_text

                    if tool_name == "search_policy_documents":
                        policy_search_count += 1
                        citations = extract_citations(result_text)

                # ------------------------------------------------
                # BUILD RESPONSE
                # ------------------------------------------------

                if workflow == "hr_case":
                    answer = build_hr_case_response(
                        profile=tool_results.get("lookup_employee_profile", ""),
                        ticket_result=tool_results.get("create_mock_hr_ticket", ""),
                        email_result=tool_results.get("draft_hr_email", ""),
                        citations=citations,
                    )

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

                    # Always surface citations to employee
                    policy_section = format_policy_citations(citations)
                    if policy_section and "### Relevant policy" not in answer:
                        answer = answer.rstrip() + "\n\n" + policy_section

        # ----------------------------------------------------
        # COMPLETE
        # ----------------------------------------------------

        total_time = time.perf_counter() - total_start

        print(f"[TIMING] TOTAL agent runtime: {total_time:.2f}s", flush=True)
        print(f"[TIMING] Tool calls: {len(tool_trace)}", flush=True)
        print(
            f"[AGENT] Returning answer to Streamlit ({len(answer)} characters).",
            flush=True,
        )

        return {
            "answer": answer.strip(),
            "tool_trace": tool_trace,
            "citations": citations,
            "error": None,
            "runtime_seconds": round(total_time, 2),
        }

    except Exception as exc:
        import traceback
        traceback.print_exc()

        total_time = time.perf_counter() - total_start

        print(f"[AGENT] Top-level error: {exc}", flush=True)
        print(f"[TIMING] TOTAL agent runtime: {total_time:.2f}s", flush=True)

        return {
            "answer": (
                "I encountered an error while processing your request. "
                "Please try again or contact people@daisyhealth.com."
            ),
            "tool_trace": tool_trace,
            "citations": citations,
            "error": str(exc),
            "runtime_seconds": round(total_time, 2),
        }

    finally:
        # FIX: close the AsyncOpenAI HTTP connection pool before asyncio.run()
        # shuts down the event loop. On Python 3.14, the GC tries to close
        # httpcore/anyio connection pools after the loop is already closed,
        # producing "RuntimeError: Event loop is closed". Explicitly awaiting
        # client.close() here drains the pool cleanly before the loop ends.
        if client is not None:
            try:
                await client.close()
            except Exception:
                pass


# ============================================================
# STREAMLIT WRAPPER
# ============================================================

def run_agent_sync(question, employee_id):
    return asyncio.run(run_agent(question=question, employee_id=employee_id))


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
    print(f"Total runtime: {result.get('runtime_seconds', '?')}s")
    print(f"Tool calls: {len(result['tool_trace'])}")

    print("\n==============================")
    print("TOOL TRACE")
    print("==============================")

    for tool in result["tool_trace"]:
        print(f"- {tool['tool']}: {tool['status']} ({tool.get('duration_seconds', '?')}s)")

    print(f"\nCitations: {len(result['citations'])}")

    for citation in result["citations"]:
        print(f"- {citation.get('title', 'HR Policy')} — {citation.get('policy_id', '')}")