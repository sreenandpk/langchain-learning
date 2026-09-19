from document_service import load_documents, split_documents
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")


documents = load_documents()

chunks = split_documents(documents)


vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db",
)


print(f"Created vector store with {len(chunks)} chunks.")
