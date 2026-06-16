---

# ⚖️ AI-Powered Legal Advisory Agent (Phase 1)

A modular, graph-based RAG (Retrieval-Augmented Generation) system designed for high-precision legal document analysis. This agent uses **FastAPI** for core AI logic, **Flask** for the user interface, and **LangGraph** to manage complex decision-making workflows.

![Project Status](https://img.shields.io/badge/Status-Phase_1_Complete-green)
![Tech Stack](https://img.shields.io/badge/Stack-FastAPI%20%7C%20Flask%20%7C%20LangGraph-blue)

---

## 📂 Project Structure

```text
legal_agent/
├── .env                    # API Keys & Configuration
├── pyproject.toml          # uv dependency management
├── api/                    # Backend (The Brain)
│   ├── main.py             # FastAPI entry point
│   └── core/
│       ├── database.py     # ChromaDB & Embeddings setup
│       ├── engine.py       # LangGraph Orchestration & Guardrails
│       └── processor.py    # Document Parsing Logic
├── gui/                    # Frontend (The Face)
│   ├── app.py              # Flask Web Server
│   └── templates/
│       └── index.html      # Professional UI
├── vector_db/              # Persistent ChromaDB storage
├── uploads/                # Staging for processed documents
```

---

## 🧩 Data Chunking Strategy

Chunking is the process of breaking down long legal documents into smaller, searchable pieces. The quality of legal advice depends entirely on how these chunks are created.

### **Current Implementation: Recursive Character Chunking**
In Phase 1, we use the `RecursiveCharacterTextSplitter`.
- **How it works:** It splits text based on a hierarchy of characters: double newlines (paragraphs), single newlines (sentences), and spaces (words).
- **Why we chose it:** Legal documents are structured by clauses. By prioritizing double newlines, we keep legal clauses together as single semantic units, preventing the AI from "losing the thread" in the middle of a sentence.
- **Parameters:** `chunk_size=1200`, `chunk_overlap=200`. The overlap ensures that context from the end of one chunk is carried into the start of the next.

### **Future Strategy: Semantic & Layout-Aware Chunking**
For Phase 2, we will move beyond character-based splitting:
1.  **Layout-Aware Parsing:** Using tools like **LlamaParse**, the agent will recognize the difference between a "Header," a "Table," and a "Footer." This prevents the agent from confusing a page number or a footer for actual legal text.
2.  **Semantic Chunking:** Instead of splitting by length, the agent will split text where the **meaning** changes. It uses embeddings to detect shifts in topic (e.g., moving from "Governing Law" to "Signatures"), ensuring every chunk is a perfectly isolated legal concept.

---

## ✨ Key Features

- **Multi-Document Knowledge Base:** Tag-based retrieval ensures the AI knows exactly which file it is reading from.
- **Intent Guardrails:** Custom LangGraph nodes validate that both the user query and the document content are legal in nature before processing.
- **Source Attribution:** Mandatory [SOURCE: filename] citations in all AI responses.
- **Persistent Vector Memory:** Uses **ChromaDB** to store document vectors locally on disk.
- **Markdown-Ready UI:** Professional display of bullet points, bolded text, and legal headers.

---

## 🛠 Tech Stack

- **LLM:** `openai/gpt-oss-120b:free` (via OpenRouter)
- **Orchestration:** LangChain & LangGraph
- **Vector DB:** ChromaDB
- **Embeddings:** HuggingFace `all-MiniLM-L6-v2` (Local)
- **Package Manager:** `uv` (Fastest Python installer)

---

## 🚦 How to Run

### 1. Prerequisites
- Install **uv**: `pip install uv`
- Obtain an API Key from [OpenRouter](https://openrouter.ai/)

### 2. Setup
Create a `.env` file:
```env
OPENROUTER_API_KEY=your_api_key_here
FASTAPI_URL=http://127.0.0.1:8000
```

### 3. Install & Sync
```bash
uv sync
```

### 4. Run Servers
**Terminal 1 (Backend):**
```bash
uv run uvicorn api.main:app --reload --port 8000
```

**Terminal 2 (Frontend):**
```bash
uv run python gui/app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## ⚠️ Current Limitations
- **Semantic-Only Search:** Phase 1 lacks keyword search (BM25), which can sometimes miss specific case numbers or rare terminology.
- **Session History:** Chat history is held in-memory; it clears when the FastAPI server restarts.
- **Complex Tables:** Highly complex legal tables in PDFs may be fragmented by character-based chunking.

---

## 🗺 Phase 2 Roadmap
- [ ] **Hybrid RAG:** Implementing a combination of Vector Search + Keyword Search.
- [ ] **Agentic Review:** Adding a "Critic Agent" to verify AI answers against the original text.
- [ ] **OCR Integration:** Support for scanned/image-based legal documents.
- [ ] **LlamaParse Integration:** For high-fidelity, layout-aware legal document parsing.

---
![Project Preview](screenshot.jpg)
*Figure 1: Knowledge Base Sidebar and Source Attribution in action.*

---
*Disclaimer: This project is an AI tool for research and is not a substitute for professional legal advice.*
