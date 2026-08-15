"""
Daisy Health — Evaluation Runner
Executes evaluation_questions.py against the real agent (agent.run_agent_sync,
which drives the LLM tool-calling loop over the actual MCP server) and scores
the results against the metrics required by rubric Section 9:

  - Answer quality: groundedness, citation accuracy
  - Agent behavior: tool selection accuracy, workflow completion,
    escalation/clarification accuracy, action-safety pass rate
  - System: latency p50/p95
  - Ablation: retrieval k=3 vs k=5 on three multi-document questions

Scoring here is heuristic (regex/keyword-based), not an LLM judge — the
per-question raw answers, tool traces, and citations are all persisted to
results.json so a human grader can spot-check any score by hand.

Usage:
    python evaluation/run_eval.py
Requires the same .env as agent.py / rag_backend.py (OPENROUTER_API_KEY,
CHROMADB_API_KEY, CHROMADB_TENANT, CHROMADB_DB).
"""

import json
import re
import statistics
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from evaluation_questions import evaluation_questions  # noqa: E402
from agent import run_agent_sync  # noqa: E402

# ─────────────────────────────────────────────
# Employee mapping for tool-requiring questions
# (matches mock_data/employees.json + daisy_health_app.py MOCK_EMPLOYEES)
# ─────────────────────────────────────────────
NAME_TO_EMP_ID = {
    "jordan rivera": "EMP-001",
    "morgan chen": "EMP-002",
    "simone okafor": "EMP-003",
    "alex nguyen": "EMP-004",
    "taylor brooks": "EMP-005",
}
DEFAULT_EMP_ID = "EMP-002"

CLARIFY_PHRASES = [
    "could you", "can you tell me", "please provide", "let me know",
    "which employee", "what is your employee id", "how many days",
    "what dates", "need more information", "clarify", "which state",
    "what item", "how much did", "?",
]
DECLINE_PHRASES = [
    "outside the scope", "out of scope", "hr assistant", "cannot help with",
    "can only answer", "not something i can help", "redirect", "please contact",
]


def resolve_employee_id(question: str) -> str:
    q = question.lower()
    for name, emp_id in NAME_TO_EMP_ID.items():
        if name in q:
            return emp_id
    return DEFAULT_EMP_ID


def score_question(q: dict, result: dict, elapsed: float) -> dict:
    answer = (result.get("answer") or "").lower()
    tool_trace = result.get("tool_trace") or []
    citations = result.get("citations") or []
    called_tools = [t["tool"] for t in tool_trace if t.get("status", "").startswith("✓")]
    had_error = any(t.get("status", "").startswith("✗") for t in tool_trace) or not answer.strip()

    scores = {"id": q["id"], "type": q["type"], "elapsed_s": round(elapsed, 2)}

    # ── Groundedness: does the answer reference a specific policy ID from
    #    the gold policy_id field, or (for tool-only/ambiguous/out-of-scope
    #    questions with no policy_id) is it non-empty and coherent? ──
    policy_ids = [p.strip() for p in q.get("policy_id", "").split("+") if p.strip() and p.strip() != "N/A"]
    if policy_ids:
        grounded = any(pid.lower() in answer for pid in policy_ids)
    else:
        grounded = bool(answer.strip()) and not had_error
    scores["grounded"] = grounded

    # ── Citation accuracy: for questions with a real policy_id, did the
    #    agent return at least one citation object (from search_policy_documents
    #    + RAG) or explicitly cite a policy ID inline? ──
    if policy_ids:
        scores["citation_present"] = bool(citations) or any(pid.lower() in answer for pid in policy_ids)
    else:
        scores["citation_present"] = None  # N/A for ambiguous/out-of-scope

    # ── Tool selection accuracy: fraction of expected tools actually called ──
    expected_tools = q.get("tools_expected")
    if expected_tools:
        hit = sum(1 for t in expected_tools if t in called_tools)
        scores["tool_selection_accuracy"] = round(hit / len(expected_tools), 2)
        scores["tools_expected"] = expected_tools
        scores["tools_called"] = called_tools
    else:
        scores["tool_selection_accuracy"] = None

    # ── Workflow completion: answered without error and (if tools required)
    #    actually invoked at least one tool ──
    if q.get("requires_tools"):
        scores["workflow_completed"] = (not had_error) and len(called_tools) > 0
    else:
        scores["workflow_completed"] = not had_error

    # ── Escalation accuracy (Q18 — HR case triage) ──
    if q["type"] == "tool_required" and "create_mock_hr_ticket" in (expected_tools or []):
        scores["escalation_correct"] = "create_mock_hr_ticket" in called_tools

    # ── Clarification accuracy (ambiguous requests) ──
    if q["type"] == "ambiguous":
        scores["clarified"] = any(p in answer for p in CLARIFY_PHRASES)

    # ── Out-of-scope refusal accuracy ──
    if q["type"] == "out_of_scope":
        scores["declined"] = any(p in answer for p in DECLINE_PHRASES)

    # ── Action-safety: mock tickets/emails are always labeled MOCK/DRAFT
    #    (never actually sent) — verify that label survived into the trace ──
    if "create_mock_hr_ticket" in called_tools or "draft_hr_email" in called_tools:
        trace_text = " ".join(str(t.get("result", "")) for t in tool_trace).lower()
        scores["action_safe"] = ("mock" in trace_text) or ("draft" in trace_text) or ("not sent" in trace_text)

    scores["had_error"] = had_error
    return scores


def run_ablation():
    """Compare retrieval k=3 vs k=5 on three multi-document questions."""
    from rag_backend import vectorstore

    ablation_qs = [q for q in evaluation_questions if q["id"] in ("Q08", "Q12", "Q13")]
    results = []
    for q in ablation_qs:
        expected_docs = [
            d.strip().replace(".md", "").lower()
            for d in q["source_doc"].split("+")
        ]
        row = {"id": q["id"], "question": q["question"]}
        for k in (3, 5):
            retriever = vectorstore.as_retriever(search_kwargs={"k": k})
            docs = retriever.invoke(q["question"])
            found_sources = {
                Path(d.metadata.get("source_file") or d.metadata.get("source") or "").stem.lower()
                for d in docs
            }
            hits = sum(
                1 for exp in expected_docs
                if any(exp in src or src in exp for src in found_sources if src)
            )
            row[f"k={k}_docs_retrieved"] = len(docs)
            row[f"k={k}_expected_docs_hit"] = f"{hits}/{len(expected_docs)}"
        results.append(row)
    return results


def main():
    print("Daisy Health — Evaluation Run")
    print("=" * 50)

    all_results = []
    all_scores = []
    latencies = []

    for i, q in enumerate(evaluation_questions, 1):
        emp_id = resolve_employee_id(q["question"])
        print(f"[{i}/{len(evaluation_questions)}] {q['id']} ({q['type']}) — {emp_id}")

        start = time.time()
        try:
            result = run_agent_sync(q["question"], emp_id)
        except Exception as e:
            result = {"answer": "", "tool_trace": [{"tool": "Agent", "status": "✗ Error", "result": str(e)}], "citations": []}
        elapsed = time.time() - start
        latencies.append(elapsed)

        scores = score_question(q, result, elapsed)
        all_scores.append(scores)
        time.sleep(3)  # ease pressure on the shared free-tier rate limit
        all_results.append({
            "id": q["id"],
            "question": q["question"],
            "gold_answer": q["gold_answer"],
            "employee_id": emp_id,
            "agent_answer": result.get("answer", ""),
            "tool_trace": result.get("tool_trace", []),
            "citations": result.get("citations", []),
            "scores": scores,
        })

    # ── Aggregate metrics ──
    n = len(all_scores)
    grounded_n = sum(1 for s in all_scores if s["grounded"])
    cite_applicable = [s for s in all_scores if s["citation_present"] is not None]
    cite_n = sum(1 for s in cite_applicable if s["citation_present"])
    tool_scores = [s["tool_selection_accuracy"] for s in all_scores if s["tool_selection_accuracy"] is not None]
    workflow_n = sum(1 for s in all_scores if s["workflow_completed"])
    clarify_scores = [s for s in all_scores if "clarified" in s]
    decline_scores = [s for s in all_scores if "declined" in s]
    safety_scores = [s for s in all_scores if "action_safe" in s]
    escalation_scores = [s for s in all_scores if "escalation_correct" in s]

    sorted_lat = sorted(latencies)
    p50 = statistics.median(sorted_lat)
    p95 = sorted_lat[min(int(round(0.95 * (len(sorted_lat) - 1))), len(sorted_lat) - 1)]

    summary = {
        "total_questions": n,
        "groundedness_rate": round(grounded_n / n, 2),
        "citation_accuracy": round(cite_n / len(cite_applicable), 2) if cite_applicable else None,
        "tool_selection_accuracy_avg": round(statistics.mean(tool_scores), 2) if tool_scores else None,
        "workflow_completion_rate": round(workflow_n / n, 2),
        "clarification_accuracy": round(sum(1 for s in clarify_scores if s["clarified"]) / len(clarify_scores), 2) if clarify_scores else None,
        "out_of_scope_decline_accuracy": round(sum(1 for s in decline_scores if s["declined"]) / len(decline_scores), 2) if decline_scores else None,
        "action_safety_pass_rate": round(sum(1 for s in safety_scores if s["action_safe"]) / len(safety_scores), 2) if safety_scores else None,
        "escalation_accuracy": round(sum(1 for s in escalation_scores if s["escalation_correct"]) / len(escalation_scores), 2) if escalation_scores else None,
        "latency_p50_s": round(p50, 2),
        "latency_p95_s": round(p95, 2),
        "latency_min_s": round(min(latencies), 2),
        "latency_max_s": round(max(latencies), 2),
    }

    print("\nRunning retrieval ablation (k=3 vs k=5)...")
    ablation = run_ablation()

    out = {"summary": summary, "ablation_k3_vs_k5": ablation, "per_question": all_results}
    out_path = Path(__file__).parent / "results.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nFull results written to {out_path}")


if __name__ == "__main__":
    main()
