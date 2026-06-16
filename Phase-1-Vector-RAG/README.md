
# ⚖️ AI-Powered Legal Advisory Agent (Phase 1: Vector RAG)

A sophisticated, modular legal document intelligence system that leverages **Graph-based RAG (Retrieval-Augmented Generation)** to analyze, summarize, and query legal documents. Built using a decoupled architecture with **FastAPI**, **Flask**, and **LangGraph**, it ensures high-performance document processing and context-aware legal consultation.

![Project Status](https://img.shields.io/badge/Status-Phase_1_Complete-green)
![Tech Stack](https://img.shields.io/badge/Stack-FastAPI%20%7C%20Flask%20%7C%20LangGraph-blue)
![LLM](https://img.shields.io/badge/LLM-GPT--OSS--120B-orange)

---

## 📖 Table of Contents
1. [Architecture Overview](#-architecture-overview)
2. [Key Features](#-key-features)
3. [Deep Dive: The Retrieval Pipeline](#-deep-dive-the-retrieval-pipeline)
4. [LangGraph Workflow Logic](#-langgraph-workflow-logic)
5. [Tech Stack & Tooling](#-tech-stack--tooling)
6. [Installation & Setup](#-installation--setup)
7. [Current Limitations](#-current-limitations)
8. [Phase 2 Roadmap](#-phase-2-roadmap)

---

## 🏗 Architecture Overview

The system is built on a **Decoupled Dual-Server Architecture**:

*   **Backend (The Brain):** A **FastAPI** service that manages the LangChain/LangGraph logic, ChromaDB vector storage, and OpenRouter LLM integration. It acts as an asynchronous API handling heavy computation.
*   **Frontend (The Face):** A **Flask** web application providing a user-friendly interface. It manages file uploads via AJAX and provides a real-time, stateful chat experience.

### Folder Structure
```text
legal_agent/
├── api/                    # FastAPI Backend
│   ├── core/
│   │   ├── database.py     # ChromaDB & Embedding Configuration
│   │   ├── engine.py       # LangGraph Workflow & Agent Logic
│   │   └── processor.py    # PDF/DOCX Chunking & Metadata Extraction
│   └── main.py             # REST API Endpoints
├── gui/                    # Flask Frontend
│   ├── templates/          # Jinja2 HTML Templates
│   └── app.py              # Flask Web Routes
├── vector_db/              # Persistent Vector Storage (ChromaDB)
├── uploads/                # Temporary file staging
├── pyproject.toml          # UV Package Manager Config
└── .env                    # Environment Variables
```

---

## ✨ Key Features

*   **Multi-Document Knowledge Base:** Unlike basic RAG, this system tags every text chunk with its source filename. The LLM identifies and compares data across multiple PDFs/DOCX files simultaneously.
*   **Context-Aware Intent Guardrails:** A custom middleware node in the graph validates both the user query and the document content. If the user asks for a pizza recipe or uploads a non-legal file, the agent politely refuses to process.
*   **Automated Source Citation:** The generation engine is prompted to use `[SOURCE: filename.pdf]` tags, ensuring all legal advice is traceable to the uploaded evidence.
*   **Persistent Vector Memory:** Uses ChromaDB with local persistence, allowing users to return to their indexed knowledge base even after server restarts.
*   **Modern UI/UX:** Features a "Knowledge Base" sidebar, real-time upload progress tracking, auto-scrolling chat, and Markdown rendering for professional legal formatting (headers, lists, bolding).

---

## 🔍 Deep Dive: The Retrieval Pipeline

### 1. Advanced Processing (`processor.py`)
Legal documents are hierarchical. We use a **RecursiveCharacterTextSplitter** with a chunk size of 1200 characters and a 200-character overlap. This ensures that legal clauses aren't cut in half, maintaining the semantic integrity of the "Legalese."

### 2. Semantic Search (`database.py`)
We utilize the `all-MiniLM-L6-v2` Sentence-Transformer model for local embeddings. It transforms text into 384-dimensional vectors, allowing the agent to find "Termination" clauses even if the user asks about "ending the contract."

---

## 🔄 LangGraph Workflow Logic

The agent is not a linear script; it is a **State Machine** managed by LangGraph:

1.  **START** ➡️ **Retrieve Node**: Immediately fetches the top 8 relevant chunks from the database.
2.  **Validate Intent Node**: A "Guardrail" LLM call analyzes the question + the retrieved context.
3.  **Conditional Router**:
    *   If `is_legal == True` ➡️ **Generate Node** (Summarizes/Answers).
    *   If `is_legal == False` ➡️ **Out of Scope Node** (Refusal).
4.  **END**: Returns the state (Answer + Updated Chat History) to the user.

---

## 🛠 Tech Stack & Tooling

*   **UV Package Manager:** Used for 10x faster dependency installation and deterministic environments.
*   **LangChain-OpenRouter:** Provides access to the massive `gpt-oss-120b` model, offering high-reasoning legal capabilities for free.
*   **ChromaDB:** A high-performance vector database that runs locally without the need for complex infrastructure.
*   **Bootstrap 5 & Markdown-it:** Ensures the legal reports generated by the AI look professional and are easy to read.

---

## 🚦 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-repo/legal-rag-agent.git
cd legal-rag-agent
```

### 2. Install UV & Sync
```bash
pip install uv
uv sync
```

### 3. Configure Environment
Create a `.env` file:
```env
OPENROUTER_API_KEY=your_api_key_here
FASTAPI_URL=http://127.0.0.1:8000
```

### 4. Execution
**Terminal 1 (Backend):**
```bash
uv run python uvicorn api.main:app --reload --port 8000
```

**Terminal 2 (Frontend):**
```bash
uv run python gui/app.py
```

---

## ⚠️ Current Limitations (Phase 1)
*   **Vector Bias:** Semantic search can sometimes overlook specific numeric values or rare keyword strings.
*   **In-Memory Context:** Chat history is stored per session in the API's memory; refreshing the backend clears the conversation thread.
*   **Hardware Dependency:** Local embeddings run on CPU/GPU; large document uploads (100+ pages) may take 30-60 seconds to index.

---

## 🗺 Phase 2 Roadmap

1.  **Hybrid RAG (Vector + BM25):** Combining semantic search with keyword search for 100% precision on legal terminology.
2.  **Multi-Agent Orchestration:**
    *   *Analyst Agent:* Chunks and finds data.
    *   *Legal Auditor:* Checks the AI's answer against the actual text for hallucinations.
    *   *Professional Writer:* Formats the final response into a formal legal memo.
3.  **OCR Support:** Ability to process scanned images and non-searchable PDFs.
4.  **Persistence Layer:** PostgreSQL with PGVector for enterprise-grade data management and user accounts.
5.  **Dockerization:** One-click deployment using Docker Compose.

---

![UI Screenshot](screenshot.jpg)
*Figure 1: The interface showing the Knowledge Base sidebar and the AI citing sources from an uploaded PDF.*

---
*Disclaimer: This tool is intended for research and document analysis assistance only. It does not replace professional legal advice.*
