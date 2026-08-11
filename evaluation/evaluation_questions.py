# Daisy Health — Evaluation Set
# AI Engineering Techniques and Architectures — Quantic MSAIE
# 25 questions covering all required types per Section 9 of the rubric

evaluation_questions = [

    # ─────────────────────────────────────────────
    # TYPE 1: SIMPLE POLICY QUESTIONS (7 questions)
    # One document, one clear answer
    # ─────────────────────────────────────────────
    {
        "id": "Q01",
        "type": "simple_policy",
        "question": "How many days of PTO do individual contributors receive in their first two years at Daisy Health?",
        "gold_answer": "15 days (120 hours) per year, accruing at 4.62 hours per pay period.",
        "source_doc": "pto_and_leave_policy.md",
        "policy_id": "HR-PT-001",
        "requires_tools": False,
        "notes": "Tests basic PTO policy retrieval."
    },
    {
        "id": "Q02",
        "type": "simple_policy",
        "question": "What is the home office stipend amount for full-time Daisy Health employees?",
        "gold_answer": "$500 one-time stipend for full-time remote employees, to be claimed within 60 days of hire date.",
        "source_doc": "expense_reimbursement.md",
        "policy_id": "HR-EX-004",
        "requires_tools": False,
        "notes": "Tests expense policy retrieval."
    },
    {
        "id": "Q03",
        "type": "simple_policy",
        "question": "When is Daisy Health's open enrollment period for benefits?",
        "gold_answer": "Open enrollment occurs every November for coverage beginning January 1 of the following year.",
        "source_doc": "benefits_and_insurance.md",
        "policy_id": "HR-BI-002",
        "requires_tools": False,
        "notes": "Tests benefits policy retrieval."
    },
    {
        "id": "Q04",
        "type": "simple_policy",
        "question": "How much does Daisy Health match for employee 401k contributions?",
        "gold_answer": "Daisy Health matches 100% of contributions up to 4% of the employee's salary, with a 3-year graded vesting schedule.",
        "source_doc": "benefits_and_insurance.md",
        "policy_id": "HR-BI-002",
        "requires_tools": False,
        "notes": "Tests retirement benefits retrieval."
    },
    {
        "id": "Q05",
        "type": "simple_policy",
        "question": "What is the minimum notice required for planned PTO at Daisy Health?",
        "gold_answer": "At least 5 business days notice for planned PTO, submitted through the HR Portal.",
        "source_doc": "pto_and_leave_policy.md",
        "policy_id": "HR-PT-001",
        "requires_tools": False,
        "notes": "Tests PTO request process."
    },
    {
        "id": "Q06",
        "type": "simple_policy",
        "question": "How long is Daisy Health's parental leave for primary caregivers?",
        "gold_answer": "16 weeks fully paid leave for primary caregivers. Secondary caregivers receive 6 weeks fully paid.",
        "source_doc": "pto_and_leave_policy.md",
        "policy_id": "HR-PT-001",
        "requires_tools": False,
        "notes": "Tests parental leave policy retrieval."
    },
    {
        "id": "Q07",
        "type": "simple_policy",
        "question": "What HIPAA training is required for new Daisy Health employees?",
        "gold_answer": "All new employees must complete HIPAA Fundamentals training within 7 days of their start date through the Daisy Health Learning Management System (LMS).",
        "source_doc": "hipaa_and_data_security.md",
        "policy_id": "HR-DS-003",
        "requires_tools": False,
        "notes": "Tests HIPAA policy retrieval."
    },

    # ─────────────────────────────────────────────
    # TYPE 2: MULTI-DOCUMENT QUESTIONS (6 questions)
    # Answer requires retrieving from 2+ documents
    # ─────────────────────────────────────────────
    {
        "id": "Q08",
        "type": "multi_document",
        "question": "Can a clinical pharmacist at Daisy Health work remotely from a state where they are not licensed?",
        "gold_answer": "It depends. Clinical staff must hold an active license in the state where their patients are located. If working from a different state, they may also need a license in their physical work state. They must notify the Credentialing team within 5 business days of any relocation. Compact licensure (IMLC/NLC) may apply. Sources: Remote Work Policy HR-RW-001 Section 3 and Licensure and Credentialing Policy HR-LC-009 Section 1.",
        "source_doc": "remote_work_policy.md + licensure_and_credentialing.md",
        "policy_id": "HR-RW-001 + HR-LC-009",
        "requires_tools": False,
        "notes": "Tests multi-document retrieval across remote work and licensure policies."
    },
    {
        "id": "Q09",
        "type": "multi_document",
        "question": "If a full-time employee at Daisy Health is on the Bronze health plan, what savings account options are available to them and how much does Daisy Health contribute?",
        "gold_answer": "Employees on the Bronze plan (a high-deductible health plan) are eligible for a Health Savings Account (HSA). Daisy Health contributes $500/year for individual coverage and $1,000/year for employee plus dependents. The 2025 IRS limit is $4,150 for individuals and $8,300 for families. Sources: Benefits and Insurance Policy HR-BI-002 Sections 2 and 6.",
        "source_doc": "benefits_and_insurance.md",
        "policy_id": "HR-BI-002",
        "requires_tools": False,
        "notes": "Tests multi-section retrieval within benefits policy."
    },
    {
        "id": "Q10",
        "type": "multi_document",
        "question": "What happens to a clinical employee's patient care privileges if their license expires?",
        "gold_answer": "An expired license results in immediate suspension of patient care privileges until the license is renewed and verified by the Credentialing team. The Credentialing team sends renewal reminders 90 days before expiration. Sources: Licensure and Credentialing Policy HR-LC-009 Section 3 and Clinical Staff Policy HR-CS-010.",
        "source_doc": "licensure_and_credentialing.md + clinical_staff_policy.md",
        "policy_id": "HR-LC-009 + HR-CS-010",
        "requires_tools": False,
        "notes": "Tests multi-document retrieval across licensure and clinical staff policies."
    },
    {
        "id": "Q11",
        "type": "multi_document",
        "question": "Can a new employee at Daisy Health take PTO during their first 30 days, and what other restrictions apply during the probationary period?",
        "gold_answer": "New employees may not take PTO during their first 30 days without manager approval. During the 90-day probationary period, employees are also not eligible for internal transfers. PTO accrues from day one but is restricted in use during the first 30 days. Sources: PTO and Leave Policy HR-PT-001 Section 2 and Onboarding Policy HR-OB-005 Section 9.",
        "source_doc": "pto_and_leave_policy.md + onboarding_policy.md",
        "policy_id": "HR-PT-001 + HR-OB-005",
        "requires_tools": False,
        "notes": "Tests multi-document retrieval across PTO and onboarding policies."
    },
    {
        "id": "Q12",
        "type": "multi_document",
        "question": "What security requirements must a Daisy Health employee follow when working remotely with patient data?",
        "gold_answer": "Remote employees must use a Daisy Health-issued or IT-approved device, connect through the Daisy Health VPN when accessing internal systems or patient records, never use public Wi-Fi without VPN, ensure their workspace is private with no patient conversations where others can overhear, and lock their screen when stepping away. Clinical staff must use headphones for telehealth encounters. Sources: Remote Work Policy HR-RW-001 Section 4 and HIPAA and Data Security Policy HR-DS-003 Sections 3, 4, and 7.",
        "source_doc": "remote_work_policy.md + hipaa_and_data_security.md",
        "policy_id": "HR-RW-001 + HR-DS-003",
        "requires_tools": False,
        "notes": "Tests multi-document retrieval across remote work and HIPAA policies."
    },
    {
        "id": "Q13",
        "type": "multi_document",
        "question": "Does Daisy Health reimburse clinical staff for license renewal fees and continuing education?",
        "gold_answer": "Yes. Daisy Health reimburses state license renewal fees up to $300 per renewal cycle, additional state licenses up to $500, DEA registration in full, and board certification exams up to $500. The annual $1,000 professional development stipend can also be used for CE courses. Sources: Licensure and Credentialing Policy HR-LC-009 Section 5 and Expense Reimbursement Policy HR-EX-004 Section 6.",
        "source_doc": "licensure_and_credentialing.md + expense_reimbursement.md",
        "policy_id": "HR-LC-009 + HR-EX-004",
        "requires_tools": False,
        "notes": "Tests multi-document retrieval across licensure and expense policies."
    },

    # ─────────────────────────────────────────────
    # TYPE 3: TOOL-REQUIRING TASKS (6 questions)
    # Agent must call MCP tools to answer
    # ─────────────────────────────────────────────
    {
        "id": "Q14",
        "type": "tool_required",
        "question": "How many PTO days does Alex Nguyen have available?",
        "gold_answer": "Alex Nguyen (EMP-004) has 9.0 PTO days available. He has used 11.0 days this year.",
        "source_doc": "mock_data/pto_balances.json",
        "policy_id": "N/A",
        "requires_tools": True,
        "tools_expected": ["lookup_employee_profile", "check_pto_balance"],
        "notes": "Tests employee data lookup via MCP tools."
    },
    {
        "id": "Q15",
        "type": "tool_required",
        "question": "I am Jordan Rivera and I want to take 3 days off next week. Do I have enough PTO and what do I need to do?",
        "gold_answer": "Jordan Rivera (EMP-001) has 14.5 PTO days available, so 3 days is within their balance. To request PTO, Jordan should log in to the HR Portal at hr.daisyhealth.com, navigate to Time Off, select the dates, and submit. Manager Dr. Priya Anand must approve within 2 business days. Jordan should submit at least 5 business days in advance for planned PTO. Source: PTO and Leave Policy HR-PT-001.",
        "source_doc": "mock_data/pto_balances.json + pto_and_leave_policy.md",
        "policy_id": "HR-PT-001",
        "requires_tools": True,
        "tools_expected": ["lookup_employee_profile", "check_pto_balance", "search_policy_documents"],
        "notes": "Tests PTO guidance agentic workflow — Demo Task candidate."
    },
    {
        "id": "Q16",
        "type": "tool_required",
        "question": "Can Morgan Chen expense a home office chair?",
        "gold_answer": "Yes. Morgan Chen (EMP-002) is a full-time employee and is eligible for the $500 one-time home office stipend, which covers items like chairs, monitors, keyboards, and desks. Morgan should submit the receipt through the Expense Portal within 60 days of hire date. Source: Expense Reimbursement Policy HR-EX-004 Section 2.",
        "source_doc": "mock_data/employees.json + expense_reimbursement.md",
        "policy_id": "HR-EX-004",
        "requires_tools": True,
        "tools_expected": ["lookup_employee_profile", "search_policy_documents", "check_policy_compliance"],
        "notes": "Tests expense compliance agentic workflow — Demo Task 1."
    },
    {
        "id": "Q17",
        "type": "tool_required",
        "question": "What health plan is Dr. Simone Okafor enrolled in and is she eligible for an HSA?",
        "gold_answer": "Dr. Simone Okafor (EMP-003) is enrolled in the Daisy Bronze plan and is HSA-eligible. Daisy Health contributes $500/year to her HSA. The 2025 individual contribution limit is $4,150.",
        "source_doc": "mock_data/benefits.json + benefits_and_insurance.md",
        "policy_id": "HR-BI-002",
        "requires_tools": True,
        "tools_expected": ["lookup_employee_profile", "lookup_benefits_status", "search_policy_documents"],
        "notes": "Tests benefits lookup via MCP tools."
    },
    {
        "id": "Q18",
        "type": "tool_required",
        "question": "I am a colleague reporting a workplace harassment concern. What should happen next?",
        "gold_answer": "The agent should retrieve the Workplace Conduct Policy, determine this is an HR case requiring escalation, create a mock HR ticket, and draft a confidential escalation email to People Operations. The employee can also report anonymously via the Ethics Hotline at 1-800-DAISY-ETH. Source: Workplace Conduct Policy HR-WC-006.",
        "source_doc": "workplace_conduct.md",
        "policy_id": "HR-WC-006",
        "requires_tools": True,
        "tools_expected": ["lookup_employee_profile", "search_policy_documents", "create_mock_hr_ticket", "draft_hr_email"],
        "notes": "Tests HR case triage agentic workflow — Demo Task 2."
    },
    {
        "id": "Q19",
        "type": "tool_required",
        "question": "Can Taylor Brooks work remotely from Minnesota for 6 weeks?",
        "gold_answer": "Taylor Brooks (EMP-005) is a full-time non-clinical HR Business Partner based in Colorado. Minnesota is on Daisy Health's approved state list. However, 6 weeks exceeds the 4-week temporary relocation limit, so Taylor must submit a Remote Work Location Change Request through the HR Portal. Allow up to 30 business days for processing and do not relocate before receiving written approval. Source: Remote Work Policy HR-RW-001.",
        "source_doc": "mock_data/employees.json + remote_work_policy.md",
        "policy_id": "HR-RW-001",
        "requires_tools": True,
        "tools_expected": ["lookup_employee_profile", "search_policy_documents", "check_policy_compliance"],
        "notes": "Tests remote work eligibility agentic workflow."
    },

    # ─────────────────────────────────────────────
    # TYPE 4: AMBIGUOUS REQUESTS (3 questions)
    # Agent should ask for clarification
    # ─────────────────────────────────────────────
    {
        "id": "Q20",
        "type": "ambiguous",
        "question": "Can I take some time off?",
        "gold_answer": "The agent should ask for clarification — specifically the employee's ID and the dates or duration they are requesting. Without knowing who is asking or how much time, the agent cannot check PTO balance or advise on approval requirements.",
        "source_doc": "N/A",
        "policy_id": "N/A",
        "requires_tools": False,
        "notes": "Tests clarification behavior for ambiguous PTO request."
    },
    {
        "id": "Q21",
        "type": "ambiguous",
        "question": "I want to work from somewhere else for a while.",
        "gold_answer": "The agent should ask for clarification — specifically the employee's ID, the destination state or country, and the duration of the proposed remote work. The answer differs significantly for temporary vs extended relocation and for clinical vs non-clinical staff.",
        "source_doc": "N/A",
        "policy_id": "N/A",
        "requires_tools": False,
        "notes": "Tests clarification behavior for vague remote work request."
    },
    {
        "id": "Q22",
        "type": "ambiguous",
        "question": "Can I get reimbursed for something I bought?",
        "gold_answer": "The agent should ask for clarification — specifically what the item is, how much it cost, and the employee's ID. Reimbursement eligibility depends on the type of purchase (home office equipment, travel, professional development, etc.) and employment status.",
        "source_doc": "N/A",
        "policy_id": "N/A",
        "requires_tools": False,
        "notes": "Tests clarification behavior for vague expense request."
    },

    # ─────────────────────────────────────────────
    # TYPE 5: OUT-OF-SCOPE REQUESTS (3 questions)
    # Agent should decline and redirect
    # ─────────────────────────────────────────────
    {
        "id": "Q23",
        "type": "out_of_scope",
        "question": "What is the stock price of Daisy Health today?",
        "gold_answer": "This question is outside the scope of the Daisy Health HR Assistant. The assistant can only answer questions grounded in Daisy Health's internal HR policy documents and employee data. For financial information, please contact Finance or visit appropriate financial resources.",
        "source_doc": "N/A",
        "policy_id": "N/A",
        "requires_tools": False,
        "notes": "Tests out-of-scope refusal — financial question."
    },
    {
        "id": "Q24",
        "type": "out_of_scope",
        "question": "Can you write me a Python script to scrape job postings?",
        "gold_answer": "This question is outside the scope of the Daisy Health HR Assistant. The assistant is designed to answer HR policy and operations questions only. For technical requests, please contact the Engineering team or IT support at it@daisyhealth.com.",
        "source_doc": "N/A",
        "policy_id": "N/A",
        "requires_tools": False,
        "notes": "Tests out-of-scope refusal — technical request."
    },
    {
        "id": "Q25",
        "type": "out_of_scope",
        "question": "What is the weather like in San Francisco today?",
        "gold_answer": "This question is outside the scope of the Daisy Health HR Assistant. The assistant can only answer questions about Daisy Health HR policies, employee benefits, PTO, remote work, expenses, and other HR-related topics. For weather information, please use a weather service.",
        "source_doc": "N/A",
        "policy_id": "N/A",
        "requires_tools": False,
        "notes": "Tests out-of-scope refusal — completely unrelated question."
    },
]

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from collections import Counter
    types = Counter(q["type"] for q in evaluation_questions)
    tools_required = sum(1 for q in evaluation_questions if q["requires_tools"])

    print("Daisy Health — Evaluation Set Summary")
    print("=" * 40)
    print(f"Total questions: {len(evaluation_questions)}")
    print()
    print("By type:")
    for t, count in types.items():
        print(f"  {t}: {count}")
    print()
    print(f"Requires MCP tools: {tools_required}")
    print(f"Policy only: {len(evaluation_questions) - tools_required}")
