LangChain Learning --- RAG Chatbot

A practical learning project for understanding LangChain, embeddings,
vector stores, retrievers, RAG, Gemini, Chroma, and FastAPI.

What We Built

This project is a Retrieval-Augmented Generation (RAG) chatbot.

Instead of:

User Question
     ↓
Gemini
     ↓
Answer

we built:

User Question
     ↓
FastAPI
     ↓
Retriever
     ↓
Chroma Vector Store
     ↓
Relevant Knowledge Chunks
     ↓
Prompt + Context
     ↓
Gemini
     ↓
Answer

The application retrieves relevant information from local Markdown files
before asking Gemini to generate the final answer.

Project Structure

lang-chain-learning/
│
├── app/
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
├── README.md
└── venv/

Technologies

Technology          Purpose

Python              Main programming language
FastAPI             API framework
LangChain           LLM application framework
Google Gemini       LLM used to generate answers
Gemini Embeddings   Converts text into vectors
Chroma              Local vector store
Pydantic            Request validation
Markdown            Knowledge-base format
Uvicorn             ASGI server
python-dotenv       Environment variables

Core Concepts

1. LangChain

LangChain is a framework for building applications powered by Large
Language Models.

It provides components for connecting:

LLMs

prompts

documents

text splitters

embeddings

vector stores

retrievers

tools

agents

workflows

In this project, LangChain connects Gemini, document processing,
embeddings, Chroma, and retrieval.

2. LLM

LLM means Large Language Model.

An LLM is an AI model trained on large amounts of text and capable of
understanding and generating natural language.

We use Google Gemini as our LLM:

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)

The LLM generates the final natural-language answer.

3. Document Loader

Our knowledge is stored in Markdown files:

app/knowledge/
├── python.md
├── fastapi.md
└── langchain.md

We load them with LangChain:

loader = DirectoryLoader(
    "app/knowledge",
    glob="*.md",
    loader_cls=TextLoader,
)

documents = loader.load()

The result is a list of LangChain Document objects.

A Document contains mainly:

document.page_content
document.metadata

Example:

page_content:
"FastAPI is a modern Python web framework..."

metadata:
{"source": "app/knowledge/fastapi.md"}

4. Text Splitting

Large documents are divided into smaller pieces called chunks.

We use:

RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
)

chunk_size

Controls the approximate target size of each chunk.

chunk_overlap

Keeps some text from the previous chunk in the next chunk.

This helps preserve context between chunks.

Original Document
       ↓
Chunk 1
Chunk 2
Chunk 3
...

5. Embeddings

An embedding is a numerical representation of text that captures
semantic meaning.

"What is FastAPI?"
        ↓
Embedding Model
        ↓
[0.12, -0.03, 0.44, ...]

The list of numbers is a vector.

We use:

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2"
)

The embedding model is different from the LLM:

Embedding Model
→ converts text into vectors

LLM
→ generates text

6. embed_query() vs embed_documents()

embed_query()

Used for a single query:

vector = embeddings.embed_query(
    "What is FastAPI?"
)

Question
   ↓
Query Vector

embed_documents()

Used for multiple documents:

vectors = embeddings.embed_documents(
    [
        "FastAPI is a Python framework.",
        "Python is a programming language.",
    ]
)

Document 1 → Vector 1
Document 2 → Vector 2

7. Vector

A vector is a list of numerical values representing an embedding.

Our current embedding output has:

3072 dimensions

The numbers are used to compare semantic similarity.

8. Vector Store

A vector store stores vectors together with their associated documents
and metadata and provides similarity-search functionality.

We use Chroma:

from langchain_chroma import Chroma

During ingestion:

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)

General flow:

Chunks
   ↓
Embedding Model
   ↓
Vectors
   ↓
Chroma

The local vector store is stored in:

chroma_db/

9. Why We Persist Chroma

We don't want to recreate embeddings every time the API starts.

Ingestion/build step

Markdown files
      ↓
Chunks
      ↓
Embeddings
      ↓
Chroma
      ↓
chroma_db/

Query time

Application starts
       ↓
Load existing Chroma
       ↓
Ready for retrieval

This avoids unnecessarily regenerating embeddings on every startup.

10. Similarity Search

Similarity search finds documents whose embeddings are semantically
close to the query embedding.

results = vector_store.similarity_search(
    "What is FastAPI?",
    k=2,
)

k=2 means return the two most relevant chunks.

Conceptually:

"What is FastAPI?"
        ↓
Query Embedding
        ↓
Compare with stored vectors
        ↓
Find similar vectors
        ↓
Return relevant chunks

11. Retriever

A Retriever retrieves relevant documents for a query.

We convert the Chroma vector store into a Retriever:

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

Then:

documents = retriever.invoke(
    question
)

The Retriever returns LangChain Document objects.

Vector Store vs Retriever

Vector Store

Stores vectors

Stores documents

Performs similarity search

Retriever

Provides a standard retrieval interface

Relationship:

Chroma Vector Store
        ↓
   as_retriever()
        ↓
     Retriever

12. RAG

RAG means:

Retrieval-Augmented Generation

The process is:

Receive a question.

Retrieve relevant external information.

Put that information into the prompt.

Send the prompt to the LLM.

Generate the answer.

Our flow:

User Question
      ↓
Retriever
      ↓
Relevant Chunks
      ↓
Context
      ↓
Prompt
      ↓
Gemini
      ↓
Generated Answer

13. Context

Retrieved documents are combined into a context string:

context = "\n\n".join(
    document.page_content
    for document in documents
)

The context is then supplied to Gemini.

14. Prompt

Our RAG prompt tells Gemini to use the retrieved context:

prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

If the answer is not present in the context,
say "I don't know based on the provided knowledge."
"""

The important idea is:

Question + Retrieved Context
             ↓
           Gemini

15. rag_service.py

The main RAG logic is:

def ask_rag(question: str) -> str:

    documents = retriever.invoke(question)

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

If the answer is not present in the context,
say "I don't know based on the provided knowledge."
"""

    response = llm.invoke(prompt)

    return response.text

This connects:

Retriever
    ↓
Context
    ↓
Prompt
    ↓
LLM

16. FastAPI

FastAPI exposes the RAG system through an HTTP API.

Endpoint:

POST /chat

Request:

{
    "message": "What is FastAPI?"
}

Response:

{
    "answer": "FastAPI is a modern Python web framework for building APIs..."
}

17. Pydantic Schema

We use Pydantic to validate the incoming request:

class ChatRequest(BaseModel):
    message: str

FastAPI therefore expects:

{
    "message": "What is FastAPI?"
}

18. FastAPI Endpoint

@app.post("/chat")
def chat(request: ChatRequest):
    answer = ask_rag(request.message)

    return {
        "answer": answer
    }

Flow:

HTTP Request
     ↓
FastAPI
     ↓
Pydantic Validation
     ↓
ChatRequest
     ↓
ask_rag()
     ↓
Retriever
     ↓
Chroma
     ↓
Context
     ↓
Gemini
     ↓
HTTP Response

19. Complete Architecture

Knowledge ingestion

python.md ───────┐
fastapi.md ──────┼──→ Document Loader
langchain.md ────┘
                         ↓
                    Documents
                         ↓
                    Text Splitter
                         ↓
                       Chunks
                         ↓
                  Gemini Embeddings
                         ↓
                       Vectors
                         ↓
                       Chroma
                         ↓
                    chroma_db/

Query / RAG

User
  │
  │ POST /chat
  ↓
FastAPI
  ↓
Pydantic
  ↓
ask_rag()
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
Gemini LLM
  ↓
Answer
  ↓
FastAPI Response

20. Ingestion vs Query Time

This distinction is important.

Ingestion time

Run when knowledge files are added or changed:

python app/services/embedding_service.py

Flow:

Files
 ↓
Load
 ↓
Split
 ↓
Embed
 ↓
Store in Chroma

Query time

Run the API:

uvicorn app.main:app --reload

Flow:

Question
 ↓
Retriever
 ↓
Chroma
 ↓
Relevant chunks
 ↓
Gemini
 ↓
Answer

21. Why Use LangChain?

Without LangChain, we could manually implement:

Read files
 ↓
Call embedding API
 ↓
Store vectors
 ↓
Calculate similarity
 ↓
Retrieve text
 ↓
Build prompt
 ↓
Call Gemini

LangChain provides reusable abstractions for:

Document
Text Splitter
Embedding
Vector Store
Retriever
Prompt
LLM

This makes it easier to build and change LLM applications.

22. Important Interview Definitions

LangChain

A framework for building applications powered by LLMs by connecting
models, prompts, documents, retrieval, tools, and workflows.

Document Loader

Loads external data into LangChain Document objects.

Document

A piece of content containing page_content and metadata.

Text Splitter

Splits large documents into smaller chunks.

Embedding

A numerical representation of text that captures semantic information.

Vector

A list of numerical values representing an embedding.

Vector Store

Stores embeddings and associated documents and supports similarity
search.

Retriever

Retrieves relevant documents for a query.

RAG

Retrieves relevant external information and provides it to an LLM before
generating an answer.

LLM

A Large Language Model that understands and generates natural language.

Chroma

A vector store/database used to store and search embeddings locally.

23. Environment Variables

The Gemini API key is stored in .env:

GOOGLE_API_KEY=your_api_key

Load it with:

from dotenv import load_dotenv

load_dotenv()

Never commit the real .env file to Git.

.gitignore:

.env
venv/
__pycache__/
*.pyc
chroma_db/

24. Running the Project

Activate the virtual environment:

.\venv\Scripts\Activate.ps1

Build/update the vector store when knowledge changes:

python app/services/embedding_service.py

Start FastAPI:

uvicorn app.main:app --reload

Open Swagger:

http://127.0.0.1:8000/docs

Test:

{
    "message": "What is FastAPI?"
}

25. Example Questions

What is FastAPI?

What is Python?

What is LangChain?

What Python frameworks are mentioned in the knowledge base?

For information outside the knowledge base, the application is
instructed to respond:

I don't know based on the provided knowledge.

26. What We Learned

This project covered the fundamental LangChain RAG pipeline:

LangChain

Document Loaders

Documents

Text Splitting

Chunks

Embeddings

Vectors

Vector Stores

Chroma

Similarity Search

Retrievers

Context

Prompts

LLMs

RAG

FastAPI integration

Pydantic request validation

Core concept

Knowledge
   ↓
Chunks
   ↓
Embeddings
   ↓
Vector Store
   ↓
Retriever
   ↓
Relevant Context
   ↓
LLM
   ↓
Answer