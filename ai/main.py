from modules.document_loader import load_documents
from modules.vectorstore import get_or_create_vectorstore
from modules.llm_chain import build_rag_chain
from rich import print

def chat_interface(rag_chain):
    print("[bold green]AI Assistant initialized. Type 'quit' to exit.[/bold green]")
    while True:
        try:
            user_input = input("\nYour question: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            if not user_input.strip():
                continue

            response = rag_chain.invoke(user_input)
            print(f"\n[bold blue]Assistant:[/bold blue] {response}")

            with open("output.sql", "a", encoding="utf-8") as f:
                f.write(response + "\n\n")

        except KeyboardInterrupt:
            print("\nSession ended by user.")
            break
        except Exception as e:
            print(f"\n[red]Error:[/red] {str(e)}")
            continue

if __name__ == "__main__":
    docs = load_documents("./docs")
    vectorstore = get_or_create_vectorstore(docs, "./chroma_db", "llama3")
    rag_chain = build_rag_chain(vectorstore)
    chat_interface(rag_chain)
