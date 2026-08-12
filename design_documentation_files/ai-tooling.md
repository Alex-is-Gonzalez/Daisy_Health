# AI Tooling — Daisy Health HR Assistant
**AI Engineering Techniques and Architectures — Quantic MSAIE**
**Team:** Jessica Huang & Alexis Gonzalez

---

## Overview

This document describes how we used AI code generation tools throughout the development of the Daisy Health HR Assistant, what worked well, and what did not.

---

## Tools Used

### 1. Claude (Anthropic) — claude.ai
**Used by:** Jessica Huang
**How we used it:**
- Generated all 12 HR policy documents for the Daisy Health corpus (markdown format)
- Converted markdown documents to styled PDFs
- Built the Streamlit web application with login, chat interface, tool trace sidebar, and citations panel
- Generated all 30 synthetic employee profiles and mock data JSON files (employees, PTO balances, benefits, HR tickets)
- Built the MCP server (`mcp/mcp_server.py`) with 7 tools
- Built the agent orchestrator (`agent.py`) connecting MCP tools and RAG backend
- Built the FastAPI API layer (`api.py`) with `/health`, `/chat`, and `/demo` endpoints
- Built the GitHub Actions CI/CD workflow
- Generated the 25-question evaluation set with gold answers
- Generated this design documentation

**What worked well:**
- Extremely fast at generating well-structured policy documents with realistic HR content
- Strong at debugging Python errors step by step — would explain each fix clearly
- Generated complete, working code files with detailed inline comments
- Helped troubleshoot mcp 2.0 compatibility issues by inspecting the installed package source
- Excellent at explaining technical concepts in plain language during the build process

**What did not work well:**
- Required multiple iterations to get the MCP server working with mcp 2.0 — the correct class (`MCPServer` vs `Server`) and handler signatures changed between versions
- The `@tool` decorator approach required checking the actual installed package to find the right import path
- Some generated code needed to be adapted for the specific Python 3.14 environment on the development machine

---

### 2. Claude / AI Assistant — Alexis's Setup
**Used by:** Alexis Gonzalez
**How we used it:**
- Generated the RAG ingestion pipeline (`ingest.py`) for loading PDF policy documents into Chroma Cloud
- Built the RAG backend (`rag_backend.py`) with similarity search, citation extraction, and OpenRouter integration
- Set up the ChromaDB Cloud connection and embedding pipeline using `all-MiniLM-L6-v2`
- Debugged Streamlit integration with the RAG backend
- Generated the initial `daisy_health_app.py` with teal color scheme and sidebar layout
- Updated the README with setup and local run instructions

**What worked well:**
- Fast at generating LangChain-based RAG pipelines
- Strong at ChromaDB configuration and metadata handling
- Helped identify the correct chunking strategy for policy documents

**What did not work well:**
- Initial Streamlit app did not include MCP integration — required Jessica's agent layer to connect
- Some dependency conflicts between langchain versions required manual resolution

---

## Impact on Development Process

### Speed
Using AI tools allowed us to build a complete agentic RAG system in approximately [X] hours of active development. Manually coding the same system would have taken significantly longer, particularly the policy document corpus and mock data generation.

### Quality
AI-generated code required review and debugging — it was not always correct on the first attempt. The MCP server required approximately 8 iterations to find the correct mcp 2.0 API. We treated AI output as a strong first draft rather than final code.

### Division of Labor
AI tools allowed us to work in parallel — Jessica built the MCP layer, mock data, and Streamlit UI while Alexis built the RAG pipeline. This parallel development was only possible because AI tools accelerated both tracks simultaneously.

### Academic Integrity
All code was reviewed, understood, and verified by both team members before being committed. We are responsible for the correctness, security, and academic integrity of all submitted work. AI tool usage is documented here per Quantic's plagiarism policy.

---

## Lessons Learned

1. **Check the installed package version before assuming an API.** The mcp library changed significantly between 1.x and 2.0. Always inspect `site-packages` directly when debugging import errors.

2. **AI tools are fastest for boilerplate and slowest for version-specific APIs.** Policy documents, mock data, and standard Python code were generated quickly. Version-specific library integration required more iteration.

3. **Explain context, not just the task.** The more context we gave Claude about the full system architecture, the better the generated code fit together. Sharing the project rubric, company theme, and existing code led to much better outputs.

4. **Use AI for documentation too.** Generating design documentation, evaluation sets, and this ai-tooling.md itself was faster with AI assistance — and the outputs were more comprehensive than we would have written manually.
