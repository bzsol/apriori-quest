# Updated imports using current non-deprecated packages
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader

# 1. Document Loading and Processing
# Specify the directory containing your PDF files
directory_path = "./docs"   # For cross-platform compatibility

# Initialize the directory loader
directory_loader = PyPDFDirectoryLoader(directory_path)

# Load all PDFs in the directory
documents = directory_loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separator="\n"
)
docs = text_splitter.split_documents(documents)

# 2. Vector Store Setup
embeddings = OllamaEmbeddings(model="llama3")  # Better for retrieval
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 3. Retriever Setup
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 4. LLM Setup with modern configuration
llm = OllamaLLM(
    model="llama3",
    temperature=0.7,
    top_p=0.9,
    repeat_penalty=1.1,
    num_ctx=4096
)

# 5. Modern RAG Prompt Template
template = """
You are a bot that formats JSON files from books and generates trivia questions.

Context:
{context}

Question: {question}

Provide detailed and easy questions in the same language. The JSON should have 4 answers, with one marked as correct:
{{
  "question": "Your trivia question here?",
  "answers": [
    {{ "text": "Answer option 1", "is_correct": false }},
    {{ "text": "Answer option 2", "is_correct": false }},
    {{ "text": "Answer option 3", "is_correct": true }},
    {{ "text": "Answer option 4", "is_correct": false }}
  ]
}}
"""

prompt = ChatPromptTemplate.from_template(template)

# 6. RAG Chain Construction
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])


rag_chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)


# 7. Enhanced Query Interface
def chat_interface():
    print("AI Assistant initialized. Type 'quit' to exit.")
    while True:
        try:
            user_input = input("\nYour question: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            if not user_input.strip():
                continue

            response = rag_chain.invoke(user_input)
            print(f"\nAssistant: {response}")

        except KeyboardInterrupt:
            print("\nSession ended by user.")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            continue


if __name__ == "__main__":
    chat_interface()