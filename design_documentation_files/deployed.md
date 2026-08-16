# Deployed — Daisy Health HR Assistant

**AI Engineering Techniques and Architectures — Quantic MSAIE**
**Team:** Jessica Huang & Alexis Gonzalez

---

## Deployed Application URL

**Deployed API and chat UI:**

```
FastAPI:
https://daisy-health.onrender.com

Health:
https://daisy-health.onrender.com/health

Demo:
https://daisy-health.onrender.com/demo

Streamlit:
https://daisy-health-streamlit.onrender.com
```

**Health Endpoint:**

```
https://daisy-health.onrender.com/health
```

---

## Health Endpoint Response

Once deployed, the `/health` endpoint returns:

```json
{
  "status": "ok",
  "app": "Daisy Health HR Assistant",
  "version": "1.0.0",
  "components": {
    "mcp_server": "online",
    "rag_index": "online",
    "chroma_docs": "[number of indexed chunks]",
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

## Deployment Platform

**Platform:** Render.com (free tier)
**Type:** Single web service
**Region:** Oregon (US West)

---

## Deployment Instructions

### Prerequisites

- GitHub repository: `https://github.com/Alex-is-Gonzalez/Daisy_Health`
- Render.com account (free)
- All API keys ready as environment variables

### Steps

**1. Connect GitHub to Render**

- Go to render.com → New Web Service
- Connect GitHub account
- Select `Alex-is-Gonzalez/Daisy_Health` repository

**2. Configure the service**

- Name: `daisy-health-hr-assistant`
- Runtime: Python 3
- Build command:

```
pip install -r requirements.txt
```

- Start command (already defined in `render.yaml`):

```
uvicorn api:app --host 0.0.0.0 --port $PORT
```

**3. Add environment variables**

Add these in the Render dashboard under Environment:

| Key                | Value                 |
| ------------------ | --------------------- |
| `OPENAI_API_KEY`   | Your OpenAI key       |
| `CHROMADB_API_KEY` | Your Chroma Cloud key |
| `CHROMADB_TENANT`  | Your Chroma tenant ID |
| `CHROMADB_DB`      | `HR_IT`               |

**4. Deploy**

- Click **Create Web Service**
- Wait 3-5 minutes for the first deploy to complete
- Visit the provided Render URL

---

## Cold Start Behavior

Render free tier **spins down after 15 minutes of inactivity.** When the service is inactive:

- First request after inactivity triggers a cold start
- Cold start takes approximately **30-60 seconds**
- Subsequent requests are fast (warm start: 2-5 seconds)

**For the demo:** Visit the health endpoint URL about 2 minutes before the demo begins to warm up the service:

```
[deployed-url]/health
```

---

## Local Development

To run locally without deployment:

**Terminal 1 — Streamlit UI:**

```bash
source .venv/bin/activate
streamlit run daisy_health_app.py
# Opens at http://localhost:8501
```

**Terminal 2 — FastAPI endpoints:**

```bash
source .venv/bin/activate
uvicorn api:app --port 8000
# Opens at http://localhost:8000
```

**Test the agent directly:**

```bash
python agent.py
```

---

## CI/CD

GitHub Actions workflow runs automatically on every push to `main`:

- Installs dependencies
- Validates project structure
- Validates mock data JSON files
- Checks Python syntax
- Runs API smoke test (`/health` returns 200)
- Verifies all 7 MCP tools are defined
- Tests mock data tool logic directly

Deployment is defined by `render.yaml` and should be enabled through Render's GitHub integration after CI passes.

View workflow runs at:

```
https://github.com/Alex-is-Gonzalez/Daisy_Health/actions
```
