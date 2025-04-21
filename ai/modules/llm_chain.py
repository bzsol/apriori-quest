from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

def build_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    template = """
You are an AI that extracts trivia questions from book content provided in raw text format, usually from PDF files.

## CONTEXT:
{context}

## TASK:
1. Read and analyze the context carefully.
2. Create exactly **2 trivia questions** based on the content, in the same language as the context.
3. For each question, generate **4 distinct answer options**.
4. Clearly indicate the correct answer by position (1-4).
5. Format your output as exactly **two SQL INSERT statements** using the following schema:

-- Schema:
CREATE TABLE trivia_questions (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer_1 TEXT NOT NULL,
    answer_2 TEXT NOT NULL,
    answer_3 TEXT NOT NULL,
    answer_4 TEXT NOT NULL,
    correct_answer INT CHECK (correct_answer BETWEEN 1 AND 4) NOT NULL
);

## OUTPUT RULES:
- Output only SQL INSERT statements, nothing else.
- Use valid SQL syntax.
- Do NOT include any explanations, extra text, or comments.
- Always return exactly **2 INSERT** statements, each inserting **1 trivia question** with 4 answers and the correct_answer index.

You are a highly accurate SQL-generating AI. Respond with only SQL.
"""

    prompt = ChatPromptTemplate.from_template(template)

    llm = OllamaLLM(
        model="deepseek-r1:8b",
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
        num_ctx=4096
    )

    rag_chain = (
        {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
