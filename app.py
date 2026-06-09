from langchain_ollama import ChatOllama

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter

from tools import get_current_time




llm = ChatOllama(
    model="llama3",
    temperature=0
)



with open(
    "knowledge_base.txt",
    "r",
    encoding="utf-8"
) as f:

    text = f.read()




splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)




embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



vectorstore = FAISS.from_texts(
    chunks,
    embedding_model
)



chat_history = []


def rewrite_question(user_question):

    if len(chat_history) == 0:
        return user_question

    history = ""

    for msg in chat_history[-6:]:

        history += (
            f"{msg['role']}: "
            f"{msg['content']}\n"
        )

    prompt = f"""
You are a query rewriter.

Conversation History:
{history}

Latest Question:
{user_question}

Rewrite it as a standalone question.

Return ONLY the rewritten question.
"""

    response = llm.invoke(prompt)

    return response.content.strip()


def retrieve_context(question):

    docs = vectorstore.similarity_search_with_score(
        question,
        k=2
    )

    context = ""

    for doc, score in docs:

        context += (
            doc.page_content + "\n"
        )

    best_score = docs[0][1]

    return context, best_score


def needs_tool(question):

    keywords = [
        "time",
        "current time",
        "what time"
    ]

    question = question.lower()

    for word in keywords:

        if word in question:
            return True

    return False

print("\n======================")
print("Conversational RAG")
print("======================\n")

while True:

    user_question = input("You: ")

    if user_question.lower() == "exit":
        break

    standalone_question = rewrite_question(
        user_question
    )

    print(
        "\nStandalone Question:",
        standalone_question
    )


    context, score = retrieve_context(
        standalone_question
    )

    if score < 1.0:

        print("\nUsing RAG\n")

        prompt = f"""
Use ONLY the context below.

Context:
{context}

Question:
{standalone_question}

Answer clearly.
"""

        answer = llm.invoke(
            prompt
        ).content

    elif needs_tool(user_question):

        print("\nUsing Tool\n")

        tool_result = get_current_time()

        prompt = f"""
User Question:
{user_question}

Tool Result:
{tool_result}

Generate answer.
"""

        answer = llm.invoke(
            prompt
        ).content

    else:

        answer = (
            "No relevant document "
            "or tool found."
        )

    print("\nAssistant:")
    print(answer)

    chat_history.append(
        {
            "role": "user",
            "content": user_question
        }
    )

    chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    print("\n---------------------\n")
