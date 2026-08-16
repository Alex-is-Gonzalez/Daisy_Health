# AI Tooling — Daisy Health HR Assistant

This document describes how AI code generation tools were used during the development of the Daisy Health HR Assistant, in accordance with the project's academic integrity policy.

---

## Tools Used

### Claude (Anthropic) — Primary Development Tool

Claude was used extensively throughout the project via **Claude Code** (CLI) and **Claude Cowork** (cloud session). It was the primary AI tool for architecture design, code generation, debugging, and iterative improvement.

#### What Claude helped build

**Agent orchestrator (`agent.py`)**
Claude designed and implemented the full agent orchestration logic, including deterministic workflow routing (`detect_workflow`), MCP tool selection (`get_workflow_tools`), argument building (`build_tool_arguments`), JSON-mode structured output with evidence-quote validation (`validate_and_build_response`), and the `SOURCE_TO_POLICY_ID` citation normalization mapping. Multiple iterations were required to reach the final design — early versions had bugs in the citation parser regex that caused filenames to be used as policy IDs instead of canonical HR codes like `HR-PT-001`.

**MCP server (`mcp/mcp_server.py`)**
Claude scaffolded the FastMCP server with all 7 tools, including the RAG-backed `search_policy_documents` and mock-data tools (`lookup_employee_profile`, `check_pto_balance`, `lookup_benefits_status`, `create_mock_hr_ticket`, `draft_hr_email`). The tool output format — `[N] Title\n  Policy ID: ...\n  Section: ...\n  Source: ...\n  Excerpt: ...` — was designed to make citation parsing deterministic.

**RAG backend (`rag_backend.py`)**
Claude set up the LangChain retrieval chain with ChromaDB Cloud, selected the `text-embedding-3-small` embedding model, and configured the retriever with k=6 after ablation testing showed that k=4 missed relevant chunks for complex multi-document questions.

**Ingestion pipeline (`ingest.py`)**
Claude implemented the document loading pipeline supporting PDF, Markdown, and TXT formats, the RecursiveCharacterTextSplitter chunking strategy, and the stable chunk ID scheme that prevents duplicate ingestion.

**FastAPI application (`api.py`)**
Claude designed the `/chat`, `/health`, and `/demo` endpoints, including the `/health` response that exposes all component statuses (MCP connectivity, ChromaDB document count, mock data status) for grader inspection.

**CI/CD pipeline (`.github/workflows/ci.yml`)**
Claude wrote the GitHub Actions workflow, including the step that actually starts the MCP server over stdio and calls `list_tools()` via `ClientSession` to verify real MCP protocol connectivity — going beyond simple file-existence checks.

**Evaluation framework (`evaluation/`)**
Claude helped design the 25-question evaluation set across 5 categories, wrote the heuristic scoring logic in `run_eval.py`, and designed the ablation study comparing k=3 vs. k=5 retrieval and free-form vs. structured-output prompting.

---

## What Worked Well

**Rapid iteration on complex bugs.** The most difficult bug in the project was a multi-layer groundedness problem: the ChromaDB chunk metadata stored filenames instead of policy IDs for most chunks, the MCP server echoed those filenames in its output, the citation parser regex didn't capture the `Policy ID:` line at all, and the evaluation metric was a literal string match. Claude traced the entire chain in one session and identified all three failure points, proposing the `SOURCE_TO_POLICY_ID` mapping as a definitive fix.

**MCP protocol knowledge.** Claude understood the MCP `ClientSession` / `stdio_client` pattern and `StdioServerParameters` correctly, including the `env=os.environ.copy()` fix required for subprocess environment inheritance on Render — a non-obvious deployment issue that would have been hard to debug without knowing the cause.

**Code quality and consistency.** Claude produced consistent, well-commented code across all modules with clear separation of concerns, helpful print statements for Render log debugging, and explicit handling of edge cases (missing employee IDs, MCP errors, LLM JSON parse failures).

**Grounding architecture.** The two-layer claim validation system (source ID check + evidence quote check) was Claude's suggestion after the structured output approach alone wasn't improving the groundedness metric. The evidence_quote verbatim matching approach is a meaningful quality improvement over the original free-form prompting.

---

## What Required Manual Intervention

**Iterative debugging across multiple sessions.** Because Claude doesn't retain memory across separate conversations, some context had to be re-established at the start of each session. This occasionally led to redundant changes being proposed before Claude fully understood the current state of the code.

**Groundedness metric calibration.** Claude initially misunderstood the groundedness metric as measuring factual accuracy rather than literal string presence of the policy ID in the answer. This led to several rounds of structured output improvements that didn't move the metric, before the root cause was correctly identified as a citation rendering problem.

**Python 3.14 compatibility.** The `RuntimeError: Event loop is closed` error caused by `AsyncOpenAI`'s connection pool cleanup on Python 3.14 required Claude to understand the interaction between `asyncio.run()`, GC timing, and the httpcore/anyio backend — a subtle issue that took multiple attempts to diagnose correctly.

**Streamlit architecture.** The correct pattern for a two-service Render deployment (Streamlit calls FastAPI via HTTP, not direct Python import) required explicit correction. Claude's default was to use direct imports, which works locally but fails across separate deployed services.

---

## How AI Tooling Affected the Development Process

Using Claude as the primary development tool substantially accelerated the project timeline. The agent orchestrator, MCP server, RAG backend, ingestion pipeline, FastAPI app, CI/CD workflow, and evaluation framework were all produced in days rather than weeks. The ability to ask Claude to trace a multi-component bug through six files simultaneously (agent.py → citation parser → mcp_server.py output format → ingest.py metadata → ChromaDB → evaluation metric) was particularly valuable.

The main trade-off was that AI-generated code requires careful human review. Several bugs made it into early commits (the `gpt-5-mini` model name, the `env=None` subprocess isolation, the citation parser regex) that passed superficial inspection but broke at runtime. Maintaining a clear understanding of what each module does — not just trusting the AI's output — was essential to catching and fixing these issues.

The evaluation-driven development approach (run eval → measure → identify root cause → fix → re-run) worked well in combination with AI assistance. Claude's code changes are more targeted and correct when given specific metric results and failure cases rather than vague descriptions.
