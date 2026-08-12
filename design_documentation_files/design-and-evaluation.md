# Design and Evaluation — Daisy Health HR Assistant
**AI Engineering Techniques and Architectures — Quantic MSAIE**
**Team:** Jessica Huang & Alexis Gonzalez
**Project:** Daisy Health HR Assistant — Agentic RAG System

---

## 1. Architecture Overview

Daisy Health is a virtual primary care company. The HR Assistant is a full-stack agentic AI system that helps employees get personalized, grounded answers to HR policy questions.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    Employee (User)                       │
└─────────────────────────┬───────────────────────────────┘
                          │ HTTPS
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Streamlit Web Application                   │
│         daisy_health_app.py (port 8501)                  │
│   - Employee login (ID + password)                       │
│   - Chat interface with suggestion chips                 │
│   - Live MCP tool trace sidebar                          │
│   - Policy citations panel                              │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI API Layer                           │
│              api.py (port 8000)                          │
│   - GET  /health — system status                         │
│   - POST /chat  — agentic Q&A                           │
│   - GET  /demo  — demo task descriptions                 │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│              Agent Orchestrator                          │
│              agent.py                                    │
│   - Interprets employee intent                           │
│   - Decides which MCP tools to call                      │
│   - Runs agentic loop via OpenRouter / Anthropic         │
│   - Synthesizes final grounded answer                    │
└──────────────┬──────────────────────┬───────────────────┘
               │                      │
               ▼                      ▼
┌──────────────────────┐  ┌───────────────────────────────┐
│    MCP Server         │  │       RAG Backend              │
│    mcp/mcp_server.py  │  │       rag_backend.py           │
│    (stdio transport)  │  │                               │
│                      │  │  - Embedding: all-MiniLM-L6-v2 │
│  7 Tools:            │  │  - Vector DB: Chroma Cloud     │
│  - lookup_employee_  │  │  - LLM: OpenRouter + Gemma     │
│    profile           │  │  - top_k retrieval: 4 chunks   │
│  - check_pto_balance │  │  - Citation metadata stored    │
│  - lookup_benefits_  │  └───────────────────────────────┘
│    status            │
│  - search_policy_    │
│    documents         │
│  - check_policy_     │
│    compliance        │
│  - create_mock_      │
│    hr_ticket         │
│  - draft_hr_email    │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│                  Data Layer                              │
├──────────────────────┬──────────────────────────────────┤
│   Mock Structured    │      Policy Corpus               │
│   Data (JSON)        │      (data/handbooks/)           │
│                      │                                  │
│  - employees.json    │  12 policy documents:            │
│    (30 employees)    │  - pto_and_leave_policy.md       │
│  - pto_balances.json │  - remote_work_policy.md         │
│  - benefits.json     │  - benefits_and_insurance.md     │
│  - hr_tickets.json   │  - hipaa_and_data_security.md    │
│                      │  - expense_reimbursement.md      │
│                      │  - onboarding_policy.md          │
│                      │  - workplace_conduct.md           │
│                      │  - equipment_and_technology.md   │
│                      │  - holidays_and_schedule.md      │
│                      │  - licensure_and_credentialing.md│
│                      │  - clinical_staff_policy.md      │
│                      │  - performance_and_compensation.md│
└──────────────────────┴──────────────────────────────────┘
```

---

## 2. Agent Framework and Orchestration

### Approach
We chose **manual orchestration** over a framework like LangGraph for the agent loop. This gives us full control over tool selection, error handling, and the trace logged to the UI.

The agent uses a **perceive → decide → act** loop:
1. **Perceive:** Receive the employee's question and ID
2. **Decide:** Claude/Gemma decides which MCP tools to call based on the system prompt
3. **Act:** Call MCP tools via stdio transport, collect results, synthesize answer

### Why Manual Orchestration
- Full visibility into each tool call for the trace sidebar
- No framework abstraction hiding the MCP calls from the grader
- Simpler to debug and explain during the demo

### LLM Provider
- **Primary:** OpenRouter (google/gemma-3-27b-it:free) — zero cost
- **Fallback:** Anthropic claude-sonnet-4-6 — switch by setting `USE_ANTHROPIC = True` in `agent.py`

---

## 3. MCP Server Design

### Transport Choice
We use **stdio transport** — the MCP server runs as a local subprocess launched by the agent. This is the simplest and most reliable approach for a single-service deployment and avoids the need for a separate HTTP port.

### Tool Schemas

| Tool | Input | Output |
|---|---|---|
| `lookup_employee_profile` | `employee_id: str` | Employee name, role, department, state, manager |
| `check_pto_balance` | `employee_id: str` | Available days, used days, carryover |
| `lookup_benefits_status` | `employee_id: str` | Health plan, HSA/FSA, dental, vision, 401k |
| `search_policy_documents` | `query: str, top_k: int` | Top matching policy sections with citations |
| `check_policy_compliance` | `employee_id: str, policy_area: str, scenario: str` | Compliance assessment with policy source |
| `create_mock_hr_ticket` | `employee_id, ticket_type, subject, description, priority` | Ticket ID and confirmation (saved to JSON) |
| `draft_hr_email` | `employee_id: str, email_type: str, context: str` | Draft email with subject, to, body |

### How the Agent Discovers and Calls Tools
1. Agent starts MCP server as subprocess via `StdioServerParameters`
2. Agent calls `session.list_tools()` to discover all 7 tools
3. Tools are converted to LLM function-calling format and passed to OpenRouter/Anthropic
4. LLM decides which tools to call and with what arguments
5. Agent calls `session.call_tool(name, args)` for each tool
6. Results are returned to the LLM for synthesis

---

## 4. RAG Design

### Policy Corpus
- **12 documents** covering all required HR topics
- **Two formats:** Markdown (`.md`) and PDF (`.pdf`) — satisfies two-format requirement
- **Total size:** ~60 pages

### Chunking Strategy
Heading-aware chunking using LangChain's `RecursiveCharacterTextSplitter`:
- Chunk size: 500 tokens
- Chunk overlap: 50 tokens
- Split boundaries: `##` headings, then paragraphs, then sentences

**Justification:** Heading-aware chunking preserves section context, which improves citation accuracy — the agent can cite the specific section (e.g. "Section 3 — Clinical Staff Requirements") rather than just the document.

### Embedding Model
`sentence-transformers/all-MiniLM-L6-v2` (HuggingFace, free, local)

**Justification:** Free, fast, runs locally, no API key required, strong performance on short to medium text. Well suited for policy document chunks.

### Vector Store
**Chroma Cloud** — hosted free tier

**Justification:** Free, no infrastructure management, persistent across deployments, supports metadata filtering for citations.

### Retrieval
- Top-k retrieval with k=4 (default)
- Citation metadata stored per chunk: source file, page number, section heading
- No reranking (kept simple for free-tier performance)

### Guardrails
- Out-of-corpus questions redirected to `people@daisyhealth.com`
- Agent system prompt instructs: "Never make up policy — only use what tools return"
- Policy facts distinguished from recommendations in all answers

---

## 5. Deployment Architecture

### Platform
Render.com (free tier) — single web service deployment

### Single Service Design
All components run within one deployed service:
- Streamlit web UI
- Agent orchestrator
- MCP server (subprocess via stdio)
- FastAPI endpoints
- Chroma vector store (Chroma Cloud — hosted separately, free)
- Mock data (JSON files committed to repo)

### Environment Variables
All secrets stored as Render environment variables, never committed to GitHub:
- `OPENROUTER_API_KEY`
- `CHROMADB_API_KEY`
- `CHROMADB_TENANT`
- `CHROMADB_DB`
- `ANTHROPIC_API_KEY` (optional fallback)

### Cold Start Behavior
Render free tier spins down after 15 minutes of inactivity. Cold start takes approximately 30-60 seconds. The `/health` endpoint can be used to wake the service before the demo.

---

## 6. Safety Guardrails

| Guardrail | Implementation |
|---|---|
| Out-of-corpus questions | Agent redirects to people@daisyhealth.com |
| Irreversible actions | `create_mock_hr_ticket` is mock only — no real HR system updated |
| Email sending | `draft_hr_email` produces draft only — employee must send manually |
| Missing employee ID | Agent asks for clarification before calling tools |
| Policy vs recommendation | System prompt instructs agent to distinguish facts from advice |
| PHI handling | App uses mock data only — no real patient or employee data |

---

## 7. Two Required Agentic Demo Tasks

### Demo Task 1 — Expense Compliance

**Prompt:** "Can I expense a home office chair?"
**Employee:** EMP-002 Morgan Chen (Senior Software Engineer)

**Expected MCP Tool Sequence:**
1. `lookup_employee_profile(employee_id="EMP-002")` → confirms full-time status
2. `search_policy_documents(query="home office stipend expense reimbursement chair")` → retrieves HR-EX-004
3. `check_policy_compliance(employee_id="EMP-002", policy_area="expense", scenario="home office chair")` → confirms compliant

**Expected Citations:** Expense Reimbursement Policy (HR-EX-004) Section 2

**Expected Answer:** Yes — Morgan is eligible for the $500 home office stipend. Submit receipts within 60 days via the Expense Portal.

---

### Demo Task 2 — HR Case Triage

**Prompt:** "I want to report a harassment concern about a coworker. Can you help me open a case?"
**Employee:** EMP-001 Jordan Rivera (Care Coordinator)

**Expected MCP Tool Sequence:**
1. `lookup_employee_profile(employee_id="EMP-001")` → identifies employee
2. `search_policy_documents(query="harassment reporting workplace conduct HR case")` → retrieves HR-WC-006
3. `create_mock_hr_ticket(employee_id="EMP-001", ticket_type="HR Case", subject="Workplace Harassment Concern", ...)` → creates ticket TKT-XXXX
4. `draft_hr_email(employee_id="EMP-001", email_type="hr_escalation", ...)` → drafts confidential email to People Ops

**Expected Citations:** Workplace Conduct Policy (HR-WC-006) Sections 8 and 10

**Expected Answer:** HR ticket created, draft escalation email provided, anonymous hotline information shared.

---

## 8. Evaluation Results

*To be completed after running the evaluation set.*

### Answer Quality Metrics

| Metric | Result |
|---|---|
| Groundedness (Q01–Q19) | X / 19 answers grounded in retrieved policy |
| Citation accuracy (Q01–Q19) | X / 19 correct source citations |
| Exact/partial match | X exact, X partial, X no match |

### Agent Behavior Metrics

| Metric | Result |
|---|---|
| Tool selection accuracy (Q14–Q19) | X / 6 correct tool sequences |
| Workflow completion rate | X / 6 multi-step workflows completed |
| Escalation accuracy (Q18) | Correct / Incorrect |
| Clarification accuracy (Q20–Q22) | X / 3 correctly asked for clarification |
| Out-of-scope accuracy (Q23–Q25) | X / 3 correctly declined |
| Action safety pass rate | X / 6 tool-requiring tasks safe |

### System Metrics (Latency)

| Query Type | p50 | p95 |
|---|---|---|
| Simple policy question | X ms | X ms |
| Tool-requiring task | X ms | X ms |
| Multi-document question | X ms | X ms |
| Cold start (Render) | X ms | X ms |
| Warm start (Render) | X ms | X ms |

### Ablation — Retrieval k=3 vs k=5

| Question | k=3 Citation Accuracy | k=5 Citation Accuracy |
|---|---|---|
| Q08 (multi-doc remote work) | X | X |
| Q12 (multi-doc security) | X | X |
| Q13 (multi-doc licensure) | X | X |
| **Overall** | X / 3 | X / 3 |

**Finding:** *(Fill in after running ablation)*

---

## 9. Design Choices Summary

| Decision | Choice | Justification |
|---|---|---|
| Agent framework | Manual orchestration | Full control, clear MCP traces |
| MCP transport | stdio | Simple, single-service deployment |
| LLM provider | OpenRouter / Gemma | Free tier, no cost |
| Embedding model | all-MiniLM-L6-v2 | Free, local, strong performance |
| Chunking | Heading-aware, 500 tokens | Preserves section context for citations |
| Vector store | Chroma Cloud | Free hosted, persistent, metadata support |
| Retrieval k | 4 | Balance between coverage and noise |
| Web framework | Streamlit + FastAPI | Fast to build, clean UI, API endpoints |
| Deployment | Render free tier | Free, GitHub-connected, environment variables |
| Mock data | JSON files in repo | Simple, no paid database needed |
