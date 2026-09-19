from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents():
    loader = DirectoryLoader(
        "app/knowledge",
        glob="*.md",
        loader_cls=TextLoader,
    )

    return loader.load()


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50,
    )

    return splitter.split_documents(documents)
