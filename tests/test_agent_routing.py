"""Deterministic routing checks that do not require external API credentials."""

import unittest

from agent import early_response, get_workflow_tools, is_tool_error


class AgentRoutingTests(unittest.TestCase):
    def test_benefits_workflow_uses_exposed_tool_name(self):
        available = ["lookup_employee_profile", "lookup_benefits_status", "search_policy_documents"]
        self.assertEqual(
            get_workflow_tools("benefits", available),
            ["lookup_employee_profile", "lookup_benefits_status", "search_policy_documents"],
        )

    def test_ambiguous_pto_request_gets_clarification(self):
        response = early_response("Can I take some time off?", "EMP-001")
        self.assertIn("dates or number of days", response)

    def test_pto_balance_question_is_not_treated_as_ambiguous(self):
        self.assertIsNone(
            early_response("How many PTO days does Alex Nguyen have available?", "EMP-004")
        )

    def test_out_of_scope_request_does_not_start_tools(self):
        response = early_response("What is the weather like in San Francisco today?", "EMP-001")
        self.assertIn("HR policies", response)

    def test_rag_dependency_error_is_not_treated_as_a_citation(self):
        self.assertTrue(is_tool_error("Policy search unavailable: RAG backend failed to load."))
