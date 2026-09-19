# 🚀 LangChain Learning — RAG Chatbot

> **A practical LangChain project built with Google Gemini, Chroma, and FastAPI.**

[![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-RAG-purple?logo=chainlink&logoColor=white)](https://www.langchain.com/)
[![Google Gemini](https://img.shields.io/badge/Google-Gemini-4285F4?logo=google&logoColor=white)](https://ai.google.dev/)
[![Chroma](https://img.shields.io/badge/Chroma-Vector%20Store-orange)](https://www.trychroma.com/)
[![License](https://img.shields.io/badge/License-Learning%20Project-green)](#)

---

## 🧠 About This Project

This project is a hands-on implementation of a **Retrieval-Augmented Generation (RAG)** chatbot.

Instead of sending a question directly to an LLM, the application first searches a custom knowledge base, retrieves the most relevant information, and then provides that context to Google Gemini.

### The basic idea

```text
User Question
      ↓
   Retriever
      ↓
Chroma Vector Store
      ↓
Relevant Knowledge
      ↓
Prompt + Context
      ↓
Google Gemini
      ↓
Final Answer

The knowledge base currently contains Markdown files about:

🐍 Python
⚡ FastAPI
🔗 LangChain
✨ Key Features
📚 Markdown-based knowledge base
📄 Document loading with LangChain
✂️ Recursive text splitting
🧮 Semantic embeddings
🗄️ Chroma vector store
🔎 Similarity search
🎯 LangChain Retriever
🤖 Google Gemini integration
🧠 Retrieval-Augmented Generation
🚀 FastAPI REST API
📝 Pydantic request validation
🔐 Environment-variable based API key
📖 Swagger/OpenAPI documentation
🏗️ Architecture
                    KNOWLEDGE INGESTION

   python.md
   fastapi.md
   langchain.md
       │
       ▼
┌───────────────────┐
│  Document Loader  │
└─────────┬─────────┘
          ↓
┌───────────────────┐
│   Text Splitter   │
└─────────┬─────────┘
          ↓
       Chunks
          ↓
┌───────────────────┐
│ Gemini Embeddings │
└─────────┬─────────┘
          ↓
      Vectors
          ↓
┌───────────────────┐
│ Chroma Vector DB  │
└───────────────────┘


                    QUERY / RAG

User Question
      │
      ▼
┌──────────────┐
│   FastAPI    │
└──────┬───────┘
       ↓
┌──────────────┐
│  Retriever   │
└──────┬───────┘
       ↓
┌──────────────┐
│    Chroma    │
└──────┬───────┘
       ↓
Relevant Chunks
       ↓
    Context
       ↓
┌──────────────┐
│    Gemini    │
└──────┬───────┘
       ↓
   Final Answer
🧩 Project Structure
lang-chain-learning/
│
├── app/
│   │
│   ├── knowledge/
│   │   ├── python.md
│   │   ├── fastapi.md
│   │   └── langchain.md
│   │
│   ├── services/
│   │   ├── document_service.py
│   │   ├── embedding_service.py
│   │   ├── llm_service.py
│   │   └── rag_service.py
│   │
│   ├── main.py
│   └── schemas.py
│
├── chroma_db/
├── .env
├── .gitignore
└── README.md
🛠️ Technologies
Technology	Purpose
🐍 Python	Application language
⚡ FastAPI	REST API
🔗 LangChain	LLM application framework
🤖 Google Gemini	LLM + embeddings
🗄️ Chroma	Vector store
📦 Pydantic	Data validation
📝 Markdown	Knowledge source
🚀 Uvicorn	ASGI server
📚 Core Concepts
🔗 LangChain

LangChain is a framework for building applications powered by Large Language Models.

It provides reusable components for:

Models
Prompts
Documents
Text splitters
Embeddings
Vector stores
Retrievers
Tools
Agents
📄 Document Loader

A document loader reads external data and converts it into LangChain Document objects.

loader = DirectoryLoader(
    "app/knowledge",
    glob="*.md",
    loader_cls=TextLoader,
)

documents = loader.load()

A Document contains:

document.page_content
document.metadata
✂️ Text Splitter

Large documents are divided into smaller chunks.

RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
)
chunk_size

Controls the approximate size of each chunk.

chunk_overlap

Keeps some content from the previous chunk in the next chunk.

This helps preserve context.

🧮 Embeddings

An embedding converts text into a numerical vector representing its semantic meaning.

"What is FastAPI?"
        ↓
Embedding Model
        ↓
[0.12, -0.03, 0.44, ...]

We use:

GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)
Important distinction
Embedding Model
      ↓
Converts text → vectors

LLM
      ↓
Generates text → answers
🗄️ Vector Store

A vector store stores embeddings and associated documents and allows semantic similarity search.

This project uses Chroma.

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)
🔎 Similarity Search

Similarity search finds the chunks that are semantically closest to the user's question.

results = vector_store.similarity_search(
    "What is FastAPI?",
    k=2,
)

k=2 means:

Return the 2 most relevant chunks.

🎯 Retriever

A Retriever is responsible for retrieving relevant documents for a query.

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

Then:

documents = retriever.invoke(question)

The relationship is:

Chroma Vector Store
        ↓
   as_retriever()
        ↓
     Retriever
🧠 RAG

RAG = Retrieval-Augmented Generation

RAG combines:

Retrieval
    +
Generation

Our process:

1. User asks a question
2. Retriever searches Chroma
3. Relevant chunks are returned
4. Chunks become context
5. Context is added to the prompt
6. Gemini receives the prompt
7. Gemini generates the answer
🔄 Complete RAG Flow
Question
   ↓
Query Embedding
   ↓
Retriever
   ↓
Chroma
   ↓
Relevant Chunks
   ↓
Context
   ↓
Prompt
   ↓
Gemini
   ↓
Answer
🚀 Quick Start
1. Clone the repository
git clone https://github.com/sreenandpk/langchain-learning.git
cd langchain-learning
2. Create virtual environment
Windows
python -m venv venv
.\venv\Scripts\Activate.ps1
Linux/macOS
python3 -m venv venv
source venv/bin/activate
3. Install dependencies
pip install langchain langchain-community langchain-google-genai langchain-chroma langchain-text-splitters fastapi uvicorn python-dotenv
4. Configure API Key

Create .env in the project root:

GOOGLE_API_KEY=your_google_api_key

Never commit .env to GitHub.

5. Build the Vector Store

Run:

python app/services/embedding_service.py

This performs:

Markdown Files
      ↓
Documents
      ↓
Chunks
      ↓
Embeddings
      ↓
Chroma

Run this again whenever the knowledge files are changed.

6. Start FastAPI
uvicorn app.main:app --reload
7. Open Swagger

Visit:

http://127.0.0.1:8000/docs
💬 Example Request
Endpoint
POST /chat
Request
{
    "message": "What is FastAPI?"
}
Response
{
    "answer": "FastAPI is a modern Python web framework for building APIs..."
}
💡 Example Questions

Try asking:

What is Python?
What is FastAPI?
What is LangChain?
Which Python frameworks are mentioned?
What is an embedding?
🧪 Learning Journey

This project covers:

✅ LangChain
✅ Gemini LLM
✅ Document Loaders
✅ Documents
✅ Text Splitting
✅ Chunking
✅ Embeddings
✅ Vectors
✅ Chroma
✅ Similarity Search
✅ Retrievers
✅ Context
✅ Prompt Engineering
✅ RAG
✅ FastAPI
✅ Pydantic
🎯 Important Interview Definitions
LangChain

A framework for building LLM-powered applications by connecting models, prompts, documents, retrieval systems, tools, and workflows.

Embedding

A numerical representation of text that captures semantic meaning.

Vector Store

A system that stores embeddings and supports similarity-based retrieval.

Retriever

A component that retrieves relevant documents for a given query.

RAG

A technique that retrieves relevant external information and provides it to an LLM before generating an answer.

LLM

A Large Language Model that understands and generates natural language.

Chroma

A vector store used to store and search embeddings.

🔐 Security

The API key is stored in:

.env

The following should be ignored by Git:

.env
venv/
__pycache__/
*.pyc
chroma_db/

Never commit your Gemini API key.

📈 Future Improvements

Possible next steps:

💬 Conversation memory
🧠 Better prompt templates
📊 Retrieval evaluation
🔍 Metadata filtering
📄 PDF document loading
🌐 Web document loading
🗃️ PostgreSQL + pgvector
⚡ Streaming responses
🔐 Authentication
🧪 RAG testing
📈 Observability
🤖 LangChain tools
🧩 Agents
🚀 Production deployment
⭐ Learning Goal

The goal of this repository is not just to build a chatbot.

It is to understand how modern LLM applications are built:

Documents
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Database
    ↓
Retrieval
    ↓
Context
    ↓
LLM
    ↓
Application
🚀 Learn. Build. Improve. Repeat.

If this project helped you understand RAG and LangChain, consider giving the repository a ⭐.