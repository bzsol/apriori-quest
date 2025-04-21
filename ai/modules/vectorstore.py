import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

def get_or_create_vectorstore(documents, persist_path, model_name="llama3"):
    embeddings = OllamaEmbeddings(model=model_name)

    if os.path.exists(persist_path):
        return Chroma(persist_directory=persist_path, embedding_function=embeddings)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_path
    )
    vectorstore.persist()
    return vectorstore
