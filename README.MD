# Build a Customer Service RAG Chatbot

## Setup

Install the packages you'll need **(DON'T FORGET TO SETUP A VIRTUAL ENVIRONMENT)**:

```
pip install chromadb scikit-learn openai python-dotenv streamlit
```

Create a `.env` file in your project folder with the following keys:

```
DEEPSEEK_API_KEY=your-key-here
CHROMADB_API_KEY=your-key-here
CHROMADB_TENANT=your-tenant-id-here
CHROMADB_DB=your-chroma-db-name-here
```

Replace the values with your actual keys and other information
