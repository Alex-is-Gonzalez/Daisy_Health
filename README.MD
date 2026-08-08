# Build a Customer Service RAG Chatbot

## Setup

Install the packages you'll need **(DON'T FORGET TO SETUP A VIRTUAL ENVIRONMENT)**:

1. Create a virtual environment
   `python3 -m venv .venv`
2. Activate the environment
   `source .venv/bin/activate`

```
pip install -r requirements.txt
```

Create a `.env` file in your project folder with the following keys:

```
OPENAI_API_KEY=your_openai_api_key
CHROMADB_API_KEY=your-key-here
CHROMADB_TENANT=your-tenant-id-here
CHROMADB_DB=your-chroma-db-name-here
```

Replace the values with your actual keys and other information

## What can HR_IT do?

You are an HR assistant for a medical organization.

You answer questions about:

- PTO
- Benefits
- FMLA
- HIPAA training
- Employee handbook
- Compliance
- Credentialing
- Scheduling
- Leave policies

Never answer outside the provided context.

Always cite the document name.

## Project Structure

medical_hr_rag/
│
├── app.py
├── rag_backend.py
├── .env
├── chroma_db/ # Created automatically
├── documents/
│ ├── EmployeeHandbook.pdf
│ ├── Benefits.pdf
│ ├── HIPAA.pdf
│ ├── PTO.pdf
│ ├── LeavePolicy.pdf
│ └── Compliance.pdf
└── requirements.txt

