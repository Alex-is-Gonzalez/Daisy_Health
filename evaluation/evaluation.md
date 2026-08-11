# Daisy Health — Evaluation Set
**AI Engineering Techniques and Architectures — Quantic MSAIE**
**Project: Daisy Health HR Assistant**
**Total Questions: 25**

---

## Overview

This evaluation set covers all five required question types from the project rubric Section 9:

| Type | Count | Description |
|---|---|---|
| Simple policy questions | 7 | Single document, clear answer |
| Multi-document questions | 6 | Answer spans 2+ policy documents |
| Tool-requiring tasks | 6 | Agent must call MCP tools |
| Ambiguous requests | 3 | Agent should ask for clarification |
| Out-of-scope requests | 3 | Agent should decline and redirect |

---

## Type 1 — Simple Policy Questions

### Q01
**Question:** How many days of PTO do individual contributors receive in their first two years at Daisy Health?

**Gold Answer:** 15 days (120 hours) per year, accruing at 4.62 hours per pay period.

**Source:** PTO and Leave Policy (HR-PT-001)

---

### Q02
**Question:** What is the home office stipend amount for full-time Daisy Health employees?

**Gold Answer:** $500 one-time stipend for full-time remote employees, to be claimed within 60 days of hire date via the Expense Portal.

**Source:** Expense Reimbursement Policy (HR-EX-004) Section 2

---

### Q03
**Question:** When is Daisy Health's open enrollment period for benefits?

**Gold Answer:** Open enrollment occurs every November for coverage beginning January 1 of the following year.

**Source:** Benefits and Insurance Policy (HR-BI-002) Section 1

---

### Q04
**Question:** How much does Daisy Health match for employee 401k contributions?

**Gold Answer:** Daisy Health matches 100% of contributions up to 4% of salary, with a 3-year graded vesting schedule (33% after year 1, 66% after year 2, 100% after year 3).

**Source:** Benefits and Insurance Policy (HR-BI-002) Section 9

---

### Q05
**Question:** What is the minimum notice required for planned PTO at Daisy Health?

**Gold Answer:** At least 5 business days notice for planned PTO, submitted through the HR Portal. For 5 or more consecutive days, 2 weeks notice is required.

**Source:** PTO and Leave Policy (HR-PT-001) Section 2

---

### Q06
**Question:** How long is Daisy Health's parental leave for primary caregivers?

**Gold Answer:** 16 weeks fully paid leave for primary caregivers. Secondary caregivers receive 6 weeks fully paid.

**Source:** PTO and Leave Policy (HR-PT-001) Section 8

---

### Q07
**Question:** What HIPAA training is required for new Daisy Health employees?

**Gold Answer:** All new employees must complete HIPAA Fundamentals training within 7 days of their start date through the Daisy Health LMS. Annual refresher training is required every 12 months.

**Source:** HIPAA and Data Security Policy (HR-DS-003) Section 2

---

## Type 2 — Multi-Document Questions

### Q08
**Question:** Can a clinical pharmacist at Daisy Health work remotely from a state where they are not licensed?

**Gold Answer:** It depends. Clinical staff must hold an active license in the state where their patients are located. If working from a different state physically, they may also need a license in that state. They must notify the Credentialing team within 5 business days of any relocation. Compact licensure programs (IMLC/NLC) may apply and are covered by Daisy Health.

**Sources:** Remote Work Policy (HR-RW-001) Section 3 + Licensure and Credentialing Policy (HR-LC-009) Section 1

---

### Q09
**Question:** If a full-time employee is on the Bronze health plan, what savings account options are available and how much does Daisy Health contribute?

**Gold Answer:** Bronze plan employees are eligible for a Health Savings Account (HSA). Daisy Health contributes $500/year for individual coverage and $1,000/year for employee plus dependents. The 2025 IRS limit is $4,150 (individual) and $8,300 (family). HSA funds roll over year to year.

**Sources:** Benefits and Insurance Policy (HR-BI-002) Sections 2 and 6

---

### Q10
**Question:** What happens to a clinical employee's patient care privileges if their license expires?

**Gold Answer:** An expired license results in immediate suspension of patient care privileges until the license is renewed and verified by the Credentialing team. The Credentialing team sends renewal reminders 90 days before expiration.

**Sources:** Licensure and Credentialing Policy (HR-LC-009) Section 3 + Clinical Staff Policy (HR-CS-010) Section 1

---

### Q11
**Question:** Can a new employee at Daisy Health take PTO during their first 30 days, and what other restrictions apply during the probationary period?

**Gold Answer:** New employees may not take PTO during their first 30 days without manager approval. During the full 90-day probationary period, employees are not eligible for internal transfers. PTO accrues from day one but is restricted in use during the first 30 days.

**Sources:** PTO and Leave Policy (HR-PT-001) Section 2 + Onboarding Policy (HR-OB-005) Section 9

---

### Q12
**Question:** What security requirements must a Daisy Health employee follow when working remotely with patient data?

**Gold Answer:** Remote employees must use a Daisy Health-issued or IT-approved device, connect through the Daisy Health VPN when accessing patient records, never use public Wi-Fi without VPN active, ensure their workspace is private, and lock their screen when stepping away. Clinical staff must use headphones for telehealth encounters and ensure no patient data is visible in shared spaces.

**Sources:** Remote Work Policy (HR-RW-001) Section 4 + HIPAA and Data Security Policy (HR-DS-003) Sections 3, 4, and 7

---

### Q13
**Question:** Does Daisy Health reimburse clinical staff for license renewal fees and continuing education?

**Gold Answer:** Yes. Daisy Health reimburses: state license renewal fees up to $300/cycle, additional state licenses up to $500, DEA registration in full, and board certification exams up to $500/year. The annual $1,000 professional development stipend can also be used for CE courses.

**Sources:** Licensure and Credentialing Policy (HR-LC-009) Section 5 + Expense Reimbursement Policy (HR-EX-004) Section 6

---

## Type 3 — Tool-Requiring Tasks

### Q14
**Question:** How many PTO days does Alex Nguyen have available?

**Gold Answer:** Alex Nguyen (EMP-004) has 9.0 PTO days available. He has used 11.0 days this year out of a total accrual of 20.0 days.

**Tools Expected:** `lookup_employee_profile` → `check_pto_balance`

**Source:** mock_data/pto_balances.json

---

### Q15
**Question:** I am Jordan Rivera and I want to take 3 days off next week. Do I have enough PTO and what do I need to do?

**Gold Answer:** Jordan Rivera (EMP-001) has 14.5 PTO days available, so 3 days is within their balance. Jordan should submit the request through the HR Portal at hr.daisyhealth.com with at least 5 business days notice. Manager Dr. Priya Anand must approve within 2 business days.

**Tools Expected:** `lookup_employee_profile` → `check_pto_balance` → `search_policy_documents`

**Source:** mock_data/pto_balances.json + PTO and Leave Policy (HR-PT-001)

---

### Q16
**Question:** Can Morgan Chen expense a home office chair?

**Gold Answer:** Yes. Morgan Chen (EMP-002) is a full-time employee eligible for the $500 home office stipend which covers chairs, monitors, keyboards, and desks. Morgan should submit the receipt through the Expense Portal within 60 days of hire date.

**Tools Expected:** `lookup_employee_profile` → `search_policy_documents` → `check_policy_compliance`

**Source:** mock_data/employees.json + Expense Reimbursement Policy (HR-EX-004) Section 2

---

### Q17
**Question:** What health plan is Dr. Simone Okafor enrolled in and is she eligible for an HSA?

**Gold Answer:** Dr. Simone Okafor (EMP-003) is enrolled in the Daisy Bronze plan and is HSA-eligible. Daisy Health contributes $500/year to her HSA for individual coverage.

**Tools Expected:** `lookup_employee_profile` → `lookup_benefits_status` → `search_policy_documents`

**Source:** mock_data/benefits.json + Benefits and Insurance Policy (HR-BI-002)

---

### Q18
**Question:** I am reporting a workplace harassment concern. What should happen next?

**Gold Answer:** The agent should retrieve the Workplace Conduct Policy, escalate the case, create a mock HR ticket, and draft a confidential email to People Operations. The employee can also report anonymously via the Ethics Hotline at 1-800-DAISY-ETH or ethics.daisyhealth.com. Daisy Health prohibits retaliation against reporters.

**Tools Expected:** `lookup_employee_profile` → `search_policy_documents` → `create_mock_hr_ticket` → `draft_hr_email`

**Source:** Workplace Conduct Policy (HR-WC-006) Sections 8 and 10

---

### Q19
**Question:** Can Taylor Brooks work remotely from Minnesota for 6 weeks?

**Gold Answer:** Taylor Brooks (EMP-005) is a full-time non-clinical employee based in Colorado. Minnesota is on Daisy Health's approved state list. However, 6 weeks exceeds the 4-week temporary limit, so Taylor must submit a Remote Work Location Change Request through the HR Portal. Allow up to 30 business days for processing. Do not relocate before written approval.

**Tools Expected:** `lookup_employee_profile` → `search_policy_documents` → `check_policy_compliance`

**Source:** mock_data/employees.json + Remote Work Policy (HR-RW-001) Sections 2, 6, and 7

---

## Type 4 — Ambiguous Requests

### Q20
**Question:** Can I take some time off?

**Expected Behavior:** The agent should ask for clarification — specifically the employee's ID and the dates or duration requested. Without this information, the agent cannot check PTO balance or advise on approval requirements.

**Gold Answer:** Clarification request. Agent should not attempt to answer without employee ID and requested dates.

---

### Q21
**Question:** I want to work from somewhere else for a while.

**Expected Behavior:** The agent should ask for clarification — specifically the employee's ID, the destination state or country, and the duration. The answer differs significantly for clinical vs non-clinical staff and for temporary vs extended relocation.

**Gold Answer:** Clarification request. Agent should ask for employee ID, destination, and duration.

---

### Q22
**Question:** Can I get reimbursed for something I bought?

**Expected Behavior:** The agent should ask for clarification — specifically what the item is, how much it cost, and the employee's ID. Reimbursement eligibility depends on the type of purchase and employment status.

**Gold Answer:** Clarification request. Agent should ask for employee ID and item details before answering.

---

## Type 5 — Out-of-Scope Requests

### Q23
**Question:** What is the stock price of Daisy Health today?

**Expected Behavior:** The agent should politely decline and explain it can only answer HR policy questions. Redirect to appropriate resources.

**Gold Answer:** Out of scope. Agent should decline and redirect to Finance or external resources.

---

### Q24
**Question:** Can you write me a Python script to scrape job postings?

**Expected Behavior:** The agent should politely decline and explain it is an HR assistant only. Redirect to IT support at it@daisyhealth.com.

**Gold Answer:** Out of scope. Agent should decline and redirect to Engineering or IT.

---

### Q25
**Question:** What is the weather like in San Francisco today?

**Expected Behavior:** The agent should politely decline and explain it can only answer Daisy Health HR policy questions.

**Gold Answer:** Out of scope. Agent should decline and redirect to a weather service.

---

## Metrics to Report

Per the rubric, report the following after running the evaluation:

### Answer Quality Metrics
- **Groundedness:** Is the answer supported by retrieved policy evidence? (Yes/No per question)
- **Citation accuracy:** Does the answer correctly cite the source document? (Yes/No per question)
- **Exact/partial match:** Does the answer match the gold answer? (Full/Partial/No match)

### Agent Behavior Metrics
- **Tool selection accuracy:** Did the agent call the right tools? (Q14–Q19)
- **Workflow completion rate:** Did the agent complete the full multi-step workflow?
- **Escalation accuracy:** Did the agent correctly escalate HR cases? (Q18)
- **Action safety:** Did the agent avoid irreversible actions without confirmation?

### System Metrics
- **Latency p50/p95:** Measure response time for 10–20 representative queries
- **Cold start vs warm start:** Document if Render/Railway free tier affects latency

### Ablation Comparison
Compare retrieval with **k=3 vs k=5** (top_k parameter in search_policy_documents):
- Run Q08, Q12, Q13 with both settings
- Report whether citation accuracy improves with k=5
