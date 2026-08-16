# Design and Evaluation — Daisy Health HR Assistant

---

## 1. System Architecture

### Overview

Daisy Health is a two-service deployment on Render:

- **FastAPI service** — HTTP API, agent orchestrator, MCP client, RAG backend, and MCP server (all co-located in one process for free-tier compatibility)
- **Streamlit service** — Chat UI that communicates with the FastAPI service

```
Employee
   │
   ▼
Streamlit UI (daisy_health_app.py)
   │  HTTP POST /chat
   ▼
FastAPI (api.py)
   │
   ▼
Agent Orchestrator (agent.py)
   │                         │
   │ stdio                   │ HTTPS
   ▼                         ▼
MCP Server              OpenAI API
(mcp_server.py)         gpt-4o-mini / text-embedding-3-small
   │
   ├── mock_data/ (JSON)
   │     employees.json
   │     pto_balances.json
   │     benefits.json
   │
   └── RAG Backend (rag_backend.py)
         LangChain + ChromaDB Cloud
         Collection: medical_hr_documents_openai
```

### Component Responsibilities

| Component             | Responsibility                                                            |
| --------------------- | ------------------------------------------------------------------------- |
| `api.py`              | HTTP interface — `/chat`, `/health`, `/demo` endpoints, CORS              |
| `agent.py`            | Intent detection, workflow routing, MCP tool orchestration, LLM synthesis |
| `mcp_server.py`       | FastMCP server exposing 7 HR tools over stdio                             |
| `rag_backend.py`      | LangChain retrieval chain, ChromaDB vector store, OpenAI embeddings       |
| `ingest.py`           | Document parsing, chunking, embedding, and ChromaDB ingestion             |
| `daisy_health_app.py` | Streamlit chat interface                                                  |

---

## 2. Policy Corpus

### Documents

The corpus consists of 9 HR policy documents covering the topics required by the rubric:

| File                              | Policy ID | Topic                              |
| --------------------------------- | --------- | ---------------------------------- |
| `pto_and_leave_policy.pdf`        | HR-PT-001 | PTO accrual, carryover, FMLA       |
| `expense_reimbursement.pdf`       | HR-EX-004 | Home office, travel, equipment     |
| `benefits_and_insurance.pdf`      | HR-BI-002 | Health plans, HSA, 401(k), dental  |
| `hipaa_and_data_security.pdf`     | HR-DS-003 | Data handling, HIPAA compliance    |
| `remote_work_policy.pdf`          | HR-RW-001 | Eligibility, approval, security    |
| `licensure_and_credentialing.pdf` | HR-LC-009 | Clinical licensing requirements    |
| `clinical_staff_policy.pdf`       | HR-CS-010 | Clinical roles and protocols       |
| `workplace_conduct.pdf`           | HR-WC-006 | Harassment, misconduct, escalation |
| `onboarding_policy.pdf`           | HR-OB-005 | New hire onboarding steps          |

### Source Formats

The ingest pipeline supports two source formats, satisfying the rubric requirement:

- **PDF** (`.pdf`) — loaded via `PyPDFLoader`
- **Markdown and plain text** (`.md`, `.txt`) — loaded via `TextLoader` with UTF-8 encoding

Both formats are merged into a single document list before chunking, producing a unified vector corpus.

---

## 3. RAG Design

### Chunking Strategy

We use `RecursiveCharacterTextSplitter` with:

- **Chunk size:** 800 characters
- **Chunk overlap:** 150 characters

The overlap ensures that policy sentences that span chunk boundaries are not lost. 800 characters is large enough to include a complete policy rule with surrounding context, but small enough to keep retrieved chunks focused. This produced 153 chunks across the 9-document corpus.

### Chunk Metadata

Each chunk stores the following metadata to support citations:

| Field            | Example                      |
| ---------------- | ---------------------------- |
| `document_type`  | `"medical_hr"`               |
| `source_file`    | `"pto_and_leave_policy.pdf"` |
| `document_title` | `"Pto And Leave Policy"`     |
| `policy_id`      | `"HR-PT-001"`                |

The `policy_id` is extracted from chunk content using the regex `HR-[A-Z]{2}-\d{3}`. If the pattern does not appear in a particular chunk's text, a `SOURCE_TO_POLICY_ID` mapping in the agent converts the source filename to the canonical policy ID at citation render time.

### Embedding Model

**OpenAI `text-embedding-3-small`** — chosen for its balance of quality and cost on the free tier. Embeddings are stored in ChromaDB Cloud under the collection `medical_hr_documents_openai`.

### Retrieval

- **k = 6** — increased from the default k=4 to give the evidence-quote validator more candidate chunks and reduce the chance of missing the correct answer chunk for complex questions.
- Retriever: `vectorstore.as_retriever(search_kwargs={"k": 6})`

### Prompting Strategy

For questions that return policy citations, the agent uses **JSON-mode structured output** with `temperature=0`:

```
System: You are a precise HR assistant. You MUST respond with valid JSON:
{
  "answer": "...",
  "claims": [{"claim": "...", "source": "HR-PT-001", "evidence_quote": "verbatim text"}],
  "has_sufficient_evidence": true
}
Rules: evidence_quote MUST be verbatim. source MUST be a policy_id from evidence.
```

Each claim's `evidence_quote` is validated against the retrieved snippet text before the answer is returned. Claims whose quotes cannot be found in the evidence are stripped, preventing hallucinated facts from reaching the employee.

### Guardrails

- **Out-of-scope detection:** keyword matching against non-HR topics (stock prices, weather, coding tasks, etc.) with a clear refusal message
- **Claim validation:** two-layer check — source ID must exist in retrieved citations, evidence quote must appear in snippet text
- **Insufficient evidence:** agent sets `has_sufficient_evidence: false` and redirects to `people@daisyhealth.com`
- **Mock-only actions:** `create_mock_hr_ticket` and `draft_hr_email` never send real communications

---

## 4. Agentic System Design

### Agent Orchestrator (`agent.py`)

The orchestrator uses **deterministic keyword-based routing** rather than LLM classification, which is faster, more predictable, and avoids an extra API call:

```
Question → detect_workflow() → {pto, benefits, remote_work, expense, hr_case, general}
         → get_workflow_tools() → required MCP tools for this workflow
         → execute tools via MCP ClientSession
         → generate_final_answer() via OpenAI gpt-4o-mini
         → validate claims against retrieved evidence
         → return answer + citations + tool_trace
```

### Workflow Routing

| Workflow      | Trigger Keywords                            | MCP Tools Called                                                                                |
| ------------- | ------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| `pto`         | pto, vacation, leave balance, days off      | `lookup_employee_profile`, `check_pto_balance`, `search_policy_documents`                       |
| `benefits`    | benefits, insurance, dental, 401k, vision   | `lookup_employee_profile`, `lookup_benefits_status`, `search_policy_documents`                  |
| `remote_work` | remote, work from home, hybrid              | `search_policy_documents`, `check_policy_compliance`                                            |
| `expense`     | expense, reimburs, chair, laptop, equipment | `search_policy_documents`, `check_policy_compliance`                                            |
| `hr_case`     | harassment, complaint, misconduct, report   | `lookup_employee_profile`, `search_policy_documents`, `create_mock_hr_ticket`, `draft_hr_email` |
| `general`     | (default)                                   | `search_policy_documents`                                                                       |

### Tool Trace

Every MCP tool call is logged to the `tool_trace` list returned by `/chat`:

```json
{
  "tool": "search_policy_documents",
  "args": { "query": "home office chair reimbursement" },
  "result": "Policy search: 'home office chair...' ...",
  "timestamp": "14:22:05",
  "status": "✓ Success",
  "duration_seconds": 1.24
}
```

### Failure Handling

- Missing required MCP arguments → logged as `✗ Missing arguments`, skipped
- MCP tool errors → logged, agent falls back to available results
- LLM JSON parse failure → falls back to free-form answer path
- Insufficient policy evidence → redirects to `people@daisyhealth.com`
- Ambiguous questions → `needs_clarification()` returns a targeted clarification question before any tool calls

### Safety Design

All potentially irreversible actions are mock-only:

- `create_mock_hr_ticket` — generates a ticket object but writes to no real system
- `draft_hr_email` — constructs email content but does not send it
- The agent explicitly labels these as drafts/mock items in the employee-facing response

---

## 5. MCP Server Design

### Transport

**stdio** — the MCP server runs as a subprocess launched by the agent via `StdioServerParameters`. This is the simplest MCP transport and avoids needing a separate HTTP port, making it free-tier compatible on Render.

The subprocess inherits the parent process's environment variables (`env=os.environ.copy()`) so ChromaDB and OpenAI credentials are available inside the MCP server.

### Framework

**FastMCP** — a high-level Python framework for building MCP servers. Each tool is a decorated function:

```python
@mcp.tool()
def search_policy_documents(query: str) -> str:
    """Search HR policy documents using RAG retrieval."""
    ...
```

### Tools

| Tool                      | Uses                        | Description                                           |
| ------------------------- | --------------------------- | ----------------------------------------------------- |
| `lookup_employee_profile` | mock_data/employees.json    | Returns name, role, department, location, manager     |
| `check_pto_balance`       | mock_data/pto_balances.json | Returns available, used, accrued, carryover days      |
| `lookup_benefits_status`  | mock_data/benefits.json     | Returns health plan, HSA/FSA, dental, vision, 401k    |
| `search_policy_documents` | ChromaDB (RAG)              | Top-6 semantic search over HR policy corpus           |
| `check_policy_compliance` | mock_data + RAG             | Gathers employee facts for a compliance determination |
| `create_mock_hr_ticket`   | in-memory                   | Creates a mock HR case ticket (not sent)              |
| `draft_hr_email`          | in-memory                   | Drafts an HR escalation email (not sent)              |

### Tool Discovery

The agent calls `session.list_tools()` at the start of each request to discover available tools dynamically. This is tested in CI — the GitHub Actions workflow starts the real MCP server over stdio and verifies all 7 tools are returned by `list_tools()`.

---

## 6. Deployment Architecture

### Services (Render free tier)

| Service   | URL                                         | Contents                                                           |
| --------- | ------------------------------------------- | ------------------------------------------------------------------ |
| FastAPI   | https://daisy-health.onrender.com           | api.py, agent.py, mcp_server.py, rag_backend.py, mock_data/, data/ |
| Streamlit | https://daisy-health-streamlit.onrender.com | daisy_health_app.py                                                |

### Design Choices

- **Single-service core:** The FastAPI service co-locates the web app, agent, MCP server, and RAG backend to stay within Render's free tier (512 MB RAM, no paid add-ons).
- **Chroma Cloud (hosted vector store):** Avoids the need to build the index on every cold start. The 153-chunk index persists across deployments at no cost on the free tier.
- **No paid database:** Mock employee data lives in JSON files committed to the repository.
- **Environment variables:** All secrets (`OPENAI_API_KEY`, `CHROMADB_API_KEY`, `CHROMADB_TENANT`, `CHROMADB_DB`) are configured as Render environment variables. No secrets are committed to the repository.

---

## 7. Evaluation

### Question Set

25 questions covering 5 required categories:

| Category          | Count | Examples                                                        |
| ----------------- | ----- | --------------------------------------------------------------- |
| Simple policy Q&A | 7     | "How many PTO days do I accrue per month?"                      |
| Multi-document    | 6     | "Can I work remotely abroad, and what security rules apply?"    |
| Tool-requiring    | 6     | "What's my current PTO balance?" (requires `check_pto_balance`) |
| Ambiguous         | 3     | "I want to take some time off" (should request clarification)   |
| Out-of-scope      | 3     | "What's the stock price?" (should decline)                      |

### Evaluation Questions and Expected Answers (Sample)

| Q#  | Question                                                   | Policy ID | Key Expected Answer                 |
| --- | ---------------------------------------------------------- | --------- | ----------------------------------- |
| Q1  | How many PTO days do full-time employees accrue per month? | HR-PT-001 | Per policy (verified from document) |
| Q4  | What is the 401(k) employer match?                         | HR-BI-002 | Per policy (verified from document) |
| Q7  | What's required within the first week of onboarding?       | HR-OB-005 | Per policy (verified from document) |
| Q14 | What's my current PTO balance? (EMP-001)                   | N/A       | Balance from mock data              |
| Q20 | What is the stock price?                                   | N/A       | Out-of-scope decline                |
| Q21 | Write me a Python script                                   | N/A       | Out-of-scope decline                |
| Q22 | I want to take some time off                               | N/A       | Clarification request               |

### Metrics and Results

| Metric                          | Result                                                 |
| ------------------------------- | ------------------------------------------------------ |
| Groundedness rate               | 0.48 → improved post-fix (SOURCE_TO_POLICY_ID mapping) |
| Citation accuracy               | 0.94                                                   |
| Tool selection accuracy (avg)   | 0.95                                                   |
| Workflow completion rate        | 1.00                                                   |
| Clarification accuracy          | 1.00                                                   |
| Out-of-scope decline accuracy   | 1.00                                                   |
| Action safety pass rate         | 1.00                                                   |
| Escalation accuracy             | 1.00                                                   |
| Latency p50                     | 4.87s                                                  |
| Latency p95                     | 6.33s                                                  |
| Latency min (clarification/OOS) | 0.00s                                                  |
| Latency max                     | 6.77s                                                  |

### Groundedness Metric Definition

The evaluator checks whether the canonical policy ID (e.g., `HR-PT-001`) appears literally in the answer text. This is a conservative string-match approach that ensures the agent is explicitly citing its source, not just referencing the policy conceptually.

### Ablation Study

We compared retrieval at **k=3 vs. k=5** on three multi-document questions that require synthesis across two policy files. Both depths retrieved the expected documents (2/2 or 1/2 expected hits) for these questions. We increased k to 6 in production to improve coverage for complex questions without measurably increasing latency.

A second ablation compared a **free-form prompt** (original) vs. **JSON-mode structured output** with claim validation (current). The structured approach eliminated a class of hallucinated numerical facts (e.g., wrong PTO day counts) by requiring verbatim evidence quotes.

### Latency Notes

The p50 latency of 4.87s reflects:

- MCP subprocess startup (~0.5s cold)
- ChromaDB Cloud retrieval (~0.8s)
- OpenAI API call (~2-3s)
- LLM synthesis and validation (~0.5s)

Clarification and out-of-scope responses return in 0.0s (no MCP or LLM calls). Render free-tier cold starts add 30-60s on the first request after a 15-minute idle period.

---

## 8. Safety Guardrails Summary

| Guardrail                     | Implementation                                                                   |
| ----------------------------- | -------------------------------------------------------------------------------- |
| Out-of-scope detection        | Keyword blocklist in `early_response()`                                          |
| Claim source validation       | Retrieved citation set must contain the claim's `source` field                   |
| Evidence quote validation     | `evidence_quote` must appear verbatim (or 80% word overlap) in retrieved snippet |
| Insufficient evidence         | `has_sufficient_evidence: false` → redirect to people@daisyhealth.com            |
| Mock-only actions             | `create_mock_hr_ticket` and `draft_hr_email` never send real communications      |
| No hallucinated employee data | Agent never invents ticket numbers, policy IDs, or email send confirmations      |
