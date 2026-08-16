# Daisy Health HR Assistant

An agentic AI system for HR policy and operations tasks at **Daisy Health**, a hypothetical medical company. The system combines Retrieval-Augmented Generation (RAG) over internal HR policy documents with an agentic orchestrator that plans, selects tools, and calls a Model Context Protocol (MCP) server to complete multi-step HR workflows.

**Deployed application:**

- FastAPI backend: https://daisy-health.onrender.com
- Streamlit UI: https://daisy-health-streamlit.onrender.com
- Health endpoint: https://daisy-health.onrender.com/health

## Project Overview

Daisy Health's HR assistant (named "Daisy") answers employee questions about PTO, benefits, expense reimbursement, remote work, workplace conduct, onboarding, and data security. The system retrieves relevant HR policy documents from a ChromaDB vector store, calls MCP-exposed tools for structured mock employee data, and produces grounded, cited responses.

### Architecture

```
Employee → Streamlit UI → FastAPI /chat
                              │
                         agent.py (orchestrator)
                              │
                    ┌─────────┴──────────┐
                    │                    │
              MCP Client          OpenAI gpt-4o-mini
              (stdio)             (LLM synthesis)
                    │
             mcp_server.py (FastMCP)
                    │
          ┌─────────┼──────────┐
          │         │          │
     mock_data/  rag_backend  compliance
     JSON files  ChromaDB     logic
```

---

## Repository Structure

```
Daisy_Health/
├── api.py                        # FastAPI app (/chat, /health, /demo)
├── agent.py                      # Agent orchestrator (workflow detection, MCP calls, LLM synthesis)
├── rag_backend.py                # LangChain + ChromaDB RAG backend
├── ingest.py                     # Document ingestion script
├── daisy_health_app.py           # Streamlit chat UI
├── requirements.txt              # Python dependencies
├── mcp/
│   └── mcp_server.py             # FastMCP server (7 tools over stdio)
├── mock_data/
│   ├── employees.json            # 30 synthetic employee profiles
│   ├── pto_balances.json         # PTO balances per employee
│   └── benefits.json             # Benefits elections per employee
├── data/
│   └── handbooks/                # HR policy documents (PDF + Markdown)
├── evaluation/
│   ├── evaluation_questions.py   # 25-question eval set
│   ├── run_eval.py               # Evaluation runner
│   └── results.json              # Latest evaluation results
├── .github/
│   └── workflows/
│       └── ci.yml                # GitHub Actions CI/CD pipeline
├── README.md                     # This file
├── design-and-evaluation.md      # Architecture and evaluation documentation
├── ai-tooling.md                 # AI tooling usage documentation
└── deployed.md                   # Deployment URLs and cold-start notes
```

---

## Local Setup

### Prerequisites

- Python 3.11 or 3.12
- An OpenAI API key
- A ChromaDB Cloud account (free tier)

### 1. Clone the repository

```bash
git clone https://github.com/Alex-is-Gonzalez/Daisy_Health.git
cd Daisy_Health
git checkout alex-openAILMM
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set environment variables

Create a `.env` file in the project root (never commit this file):

```env
OPENAI_API_KEY=sk-...
CHROMADB_API_KEY=...
CHROMADB_TENANT=...
CHROMADB_DB=...
```

### 5. Ingest HR policy documents

Place HR policy documents (PDF, Markdown, or TXT) in `data/handbooks/`, then run:

```bash
python ingest.py
```

This parses, chunks (800-char windows, 150-char overlap), embeds (text-embedding-3-small), and stores 153 chunks in ChromaDB Cloud.

---

## Running Locally

### Start the FastAPI backend

```bash
uvicorn api:app --reload --port 8000
```

The API is now available at http://localhost:8000.

**Endpoints:**

- `GET /health` — system status, MCP connectivity, document count
- `POST /chat` — submit an HR question and receive an answer with citations and tool trace
- `GET /demo` — pre-built demo task descriptions with expected tool sequences

### Start the Streamlit UI

In a separate terminal:

```bash
streamlit run daisy_health_app.py
```

The UI is available at http://localhost:8501.

### Test the API with curl

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Can I expense a home office chair?", "employee_id": "EMP-002"}'
```

---

## Reproducing the Two Agentic Demo Tasks

Both tasks can be reproduced via the `/demo` endpoint, the Streamlit UI, or direct API calls.

### Demo Task 1 — Expense Compliance

**Question:** `Can I expense a home office chair?`  
**Employee ID:** `EMP-002`  
**Expected tool sequence:** `lookup_employee_profile` → `search_policy_documents` → `check_policy_compliance`  
**Expected citation:** Expense Reimbursement Policy (HR-EX-004)

```bash
curl -X POST https://daisy-health.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Can I expense a home office chair?", "employee_id": "EMP-002"}'
```

### Demo Task 2 — HR Case Triage

**Question:** `I want to report a harassment concern about a coworker. What should I do and can you help me open a case?`  
**Employee ID:** `EMP-002`  
**Expected tool sequence:** `lookup_employee_profile` → `search_policy_documents` → `create_mock_hr_ticket` → `draft_hr_email`  
**Expected citation:** Workplace Conduct Policy (HR-WC-006)

```bash
curl -X POST https://daisy-health.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "I want to report a harassment concern about a coworker. What should I do and can you help me open a case?", "employee_id": "EMP-002"}'
```

---

## Running the Evaluation

```bash
python evaluation/run_eval.py
```

This runs all 25 evaluation questions against the live agent and writes results to `evaluation/results.json`. Requires all environment variables to be set. See `design-and-evaluation.md` for full evaluation methodology and results.

---

## Deployment

The application is deployed as two separate services on Render (free tier):

- **FastAPI service** (`api.py` + `agent.py` + `mcp_server.py`): handles all agentic logic
- **Streamlit service** (`daisy_health_app.py`): serves the chat UI

See `deployed.md` for full deployment details and cold-start behavior.

---

## CI/CD

GitHub Actions runs on every push and pull request. The pipeline:

1. Validates that all critical files exist
2. Checks JSON mock data integrity (minimum record counts)
3. Validates Python syntax on all core modules
4. Starts the FastAPI server and tests `/health` and `/demo` return HTTP 200
5. **Starts the real MCP server over stdio and calls `list_tools()` via `ClientSession`** to verify all 7 tools are discoverable through the MCP protocol
6. Deploys to Render only if all tests pass

---

## AI Tooling

See `ai-tooling.md` for a description of how Claude Code and related AI tools were used throughout this project.
