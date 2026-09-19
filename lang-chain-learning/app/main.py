from fastapi import FastAPI

from app.schemas import ChatRequest
from app.services.rag_service import ask_rag

app = FastAPI(
    title="LangChain Learning API",
)


@app.get("/")
def root():
    return {"message": "LangChain Learning API"}


@app.post("/chat")
def chat(request: ChatRequest):
    answer = ask_rag(request.message)

    return {"answer": answer}
