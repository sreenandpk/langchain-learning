from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)

load_dotenv()


# Embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")


# Load existing Chroma vector store
vector_store = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)


# Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 2})


# Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-3.8-flash")


def ask_rag(question: str) -> str:

    # Retrieve relevant documents
    documents = retriever.invoke(question)

    # Combine retrieved documents
    context = "\n\n".join(document.page_content for document in documents)

    # Create prompt
    prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

If the answer is not present in the context,
say "I don't know based on the provided knowledge."
"""

    # Send prompt to Gemini
    response = llm.invoke(prompt)

    # Return generated text
    return response.text
