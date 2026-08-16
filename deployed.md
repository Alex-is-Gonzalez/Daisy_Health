# Deployment Notes — Daisy Health HR Assistant

---

## Deployed URLs

| Service         | URL                                         |
| --------------- | ------------------------------------------- |
| FastAPI backend | https://daisy-health.onrender.com           |
| Streamlit UI    | https://daisy-health-streamlit.onrender.com |
| Health endpoint | https://daisy-health.onrender.com/health    |
| Demo endpoint   | https://daisy-health.onrender.com/demo      |

---

## Health Check

The `/health` endpoint returns the status of all system components and can be used to verify the deployment is live:

```bash
curl https://daisy-health.onrender.com/health
```

Expected response:

```json
{
  "status": "ok",
  "app": "Daisy Health HR Assistant",
  "version": "1.0.0",
  "components": {
    "mcp_server": "online",
    "rag_index": "online",
    "chroma_docs": 153,
    "mock_data": "online (30 employees)",
    "agent": "online"
  },
  "llm_provider": "OpenAI (gpt-4o-mini)",
  "embedding_model": "OpenAI text-embedding-3-small",
  "vector_db": "Chroma Cloud",
  "mcp_tools": [
    "lookup_employee_profile",
    "check_pto_balance",
    "lookup_benefits_status",
    "search_policy_documents",
    "check_policy_compliance",
    "create_mock_hr_ticket",
    "draft_hr_email"
  ]
}
```

---

## Platform

Both services are deployed on **Render** (free tier):

- **FastAPI service:** Web service running `uvicorn api:app --host 0.0.0.0 --port $PORT`
- **Streamlit service:** Web service running `streamlit run daisy_health_app.py --server.port $PORT --server.address 0.0.0.0`

### Single-Service Architecture

To remain within Render's free-tier resource limits (512 MB RAM per service), the FastAPI service co-locates the web API, agent orchestrator, MCP client, MCP server subprocess, and RAG backend in a single deployed process. The MCP server (`mcp_server.py`) is launched as a subprocess over stdio each time a `/chat` request is processed.

### Vector Store

The policy document index is hosted on **ChromaDB Cloud** (free tier), not built locally at startup. This means the 153 embedded chunks persist across deployments and cold starts without requiring the ingestion script to run on the server. Connection credentials are configured as Render environment variables.

---

## Environment Variables

The following environment variables must be configured in Render's service settings (Settings → Environment):

| Variable           | Description                                               |
| ------------------ | --------------------------------------------------------- |
| `OPENAI_API_KEY`   | OpenAI API key for gpt-4o-mini and text-embedding-3-small |
| `CHROMADB_API_KEY` | ChromaDB Cloud API key                                    |
| `CHROMADB_TENANT`  | ChromaDB Cloud tenant name                                |
| `CHROMADB_DB`      | ChromaDB Cloud database name                              |

No secrets are committed to the repository. The `.env` file is listed in `.gitignore`.

---

## Free-Tier Cold-Start Behavior

Render's free tier spins down services after **15 minutes of inactivity**. When a request arrives after a spin-down, the service must restart before it can respond. This is expected behavior and not a deployment failure.

**Expected cold-start time:** 30–60 seconds for the FastAPI service.

**What happens during a cold start:**

1. Render starts the container (~10s)
2. Python imports and dependency loading (~5s)
3. First `/chat` request spawns the MCP subprocess and connects to ChromaDB Cloud (~15-30s total for first request)
4. Subsequent warm requests respond in 4–7 seconds (p50 ~4.87s, p95 ~6.33s)

**Recommendation for graders:** If the health endpoint or a chat request times out or returns a 503, wait 60 seconds and retry. The service is warming up, not broken. A successful `/health` response with `"status": "ok"` confirms the service is warm and all components are connected.

**The Streamlit UI** may also experience a cold start if it has been idle. The first page load may take 30–60 seconds.

---

## Reproducing the Demo Tasks

Both demo tasks can be reproduced via the API without the Streamlit UI:

### Task 1 — Expense Compliance

```bash
curl -X POST https://daisy-health.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "Can I expense a home office chair?", "employee_id": "EMP-002"}'
```

### Task 2 — HR Case Triage

```bash
curl -X POST https://daisy-health.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "I want to report a harassment concern about a coworker. What should I do and can you help me open a case?", "employee_id": "EMP-002"}'
```

Full task descriptions including expected tool sequences are available at:

```bash
curl https://daisy-health.onrender.com/demo
```

---

## Deployment History

| Date                              | Change                                                                |
| --------------------------------- | --------------------------------------------------------------------- |
| Initial deployment                | FastAPI + Streamlit on Render free tier                               |
| Fix: MCP subprocess env isolation | `env=os.environ.copy()` so credentials reach the MCP server           |
| Fix: Model name                   | `gpt-4o-mini` (was `gpt-5-mini`, which does not exist)                |
| Fix: Citation normalization       | `SOURCE_TO_POLICY_ID` mapping for groundedness                        |
| Fix: Event loop cleanup           | `await client.close()` in finally block for Python 3.14 compatibility |
