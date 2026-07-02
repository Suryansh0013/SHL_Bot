# SHL Assessment Recommender

A conversational AI agent that helps recruiters and hiring managers discover the most appropriate **SHL Individual Test Solutions** through natural language conversation.

The agent accepts vague hiring requirements, asks clarifying questions when necessary, recommends grounded SHL assessments, supports iterative refinement, and compares assessments—all while ensuring recommendations are restricted to the official SHL catalog.

---

## Features

- Conversational assessment recommendation
- Clarifies vague hiring requirements
- Supports refinement during the conversation
- Compares SHL assessments using catalog data
- Strictly grounded on the SHL Individual Test Solutions catalog
- Stateless REST API using FastAPI
- Provider-agnostic LLM support (Groq, Gemini, OpenRouter)

---

# System Architecture

```
                 User Conversation
                        │
                        ▼
             AssessmentRetriever
          (TF-IDF Catalog Retrieval)
                        │
                        ▼
        Top Relevant SHL Assessments
                        │
                        ▼
              Prompt Construction
    (Conversation + Candidate Pool)
                        │
                        ▼
                   LLM Decision
        (Clarify / Recommend /
         Compare / Refuse)
                        │
                        ▼
         Validate Selected Assessment IDs
                        │
                        ▼
         Lookup Catalog Metadata & URLs
                        │
                        ▼
          FastAPI JSON Response
```

---

# Design Decisions

## Grounded Recommendations

The language model never generates assessment names or URLs.

Instead, it only selects assessment IDs from a retrieved candidate pool. The application maps these IDs back to catalog entries before returning results.

This guarantees that every recommendation originates from the official SHL catalog.

---

## Retrieval Strategy

The project uses **TF-IDF retrieval** over the complete SHL catalog.

Reasons:

- lightweight
- deterministic
- no embedding model downloads
- fast cold start
- suitable for a relatively small catalog

The retrieved candidates are then reranked by the LLM.

---

## Stateless API

Every request contains the complete conversation history.

No conversation state is stored on the server, making deployment simple and horizontally scalable.

---

## Fail-safe Behaviour

If the LLM fails or returns invalid JSON, the application returns a valid schema-compliant response instead of crashing.

---

# Project Structure

```
app/
├── agent.py
├── llm_client.py
├── main.py
├── prompts.py
├── retrieval.py
└── schemas.py

data/
├── catalog.json
└── traces/

scraper/
└── scrape_shl.py

tests/
├── manual_chat.py
└── replay_harness.py
```

---

# Installation

Clone the repository.

```bash
git clone <repository-url>

cd shl-agent
```

Create a virtual environment.

```bash
python -m venv .venv
```

Activate it.

macOS/Linux

```bash
source .venv/bin/activate
```

Windows

```bash
.venv\Scripts\activate
```

Install dependencies.

```bash
pip install -r requirements.txt
```

---

# Environment Variables

Create a `.env` file.

Example:

```env
LLM_PROVIDER=groq

GROQ_API_KEY=YOUR_API_KEY

LLM_MODEL=llama-3.1-8b-instant
```

Supported providers:

- Groq
- Gemini
- OpenRouter

---

# Running the Application

Start the FastAPI server.

```bash
uvicorn app.main:app --reload
```

Open Swagger UI.

```
http://127.0.0.1:8000/docs
```

Health endpoint.

```
GET /health
```

Chat endpoint.

```
POST /chat
```

---

# Example Request

```json
{
  "messages": [
    {
      "role": "user",
      "content": "I am hiring a Java developer with around 5 years of experience."
    }
  ]
}
```

---

# Example Response

```json
{
  "reply": "Based on the role, here are suitable SHL assessments.",
  "recommendations": [
    {
      "name": "Java 8 (New)",
      "url": "...",
      "test_type": "Knowledge"
    }
  ],
  "end_of_conversation": false
}
```

---

# Evaluation

Run the replay harness.

```bash
cd tests

python replay_harness.py --url http://localhost:8000
```

Evaluation measures:

- Schema compliance
- Recall@10
- Recommendation grounding
- Conversation quality
- Behavioral robustness

---

# Deployment

The application can be deployed to:

- Render
- Railway
- Fly.io
- Modal
- Hugging Face Spaces

For Render:

1. Push the repository to GitHub.
2. Create a new Web Service.
3. Connect the repository.
4. Add environment variables.
5. Deploy.

Verify:

```bash
curl https://<your-url>/health
```

---

# Technologies Used

- FastAPI
- Python
- Scikit-learn
- TF-IDF Retrieval
- OpenAI SDK
- Groq
- Gemini
- OpenRouter
- python-dotenv

---

# Future Improvements

- Hybrid BM25 + embedding retrieval
- Cross-encoder reranking
- FAISS vector search
- Better retrieval query expansion
- Automatic evaluation dashboard
- Streaming responses

---

