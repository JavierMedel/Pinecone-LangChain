import os
import openai
from dotenv import load_dotenv

load_dotenv()

# LangChain imports
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# --------------------------------------------------------
# 1. Load API keys
# --------------------------------------------------------
openai.api_key = os.environ["OPENAI_API_KEY"]

# --------------------------------------------------------
# 2. Embeddings + Pinecone VectorStore
# --------------------------------------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

pinecone_index_name = "langchain-embeddings-demo"
vector_store = PineconeVectorStore(
    index_name=pinecone_index_name,
    embedding=embeddings
)

retriever = vector_store.as_retriever(search_kwargs={"k": 1})

# --------------------------------------------------------
# 3. LLM Initialization (GPT-4)
# --------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.7
)

# --------------------------------------------------------
# 4. Prompt Template
# --------------------------------------------------------
prompt_template = PromptTemplate(
    template="""
You are a helpful AI assistant. Continue the conversation using memory.

Conversation History:
{chat_history}

Context:
{context}

Question:
{question}

Answer:
""",
    input_variables=["chat_history", "context", "question"]
)

# --------------------------------------------------------
# 5. Runnable Chain using "|"
# --------------------------------------------------------
chain = (
    RunnableMap({
        "context": lambda x: "\n\n".join(
            doc.page_content for doc in retriever.invoke(x["question"])
        ),
        "question": lambda x: x["question"],
        "chat_history": lambda x: x.get("chat_history", "")
    })
    | prompt_template
    | llm
    | StrOutputParser()
)

# --------------------------------------------------------
# 6. Memory with RunnableWithMessageHistory
# --------------------------------------------------------
store = {}

def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="question",
    history_messages_key="chat_history",
)

# --------------------------------------------------------
# 7. Conversation LOOP
# --------------------------------------------------------
print("\n--- RAG + MEMORY CHATBOT ---")
print("Type 'exit' to stop.\n")

config = {"configurable": {"session_id": "my-session-1"}}

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit", "q"]:
        print("Goodbye!")
        break

    # Invoke chain with memory
    answer = chain_with_memory.invoke(
        {"question": user_input},
        config=config
    )

    print("\nAssistant:", answer, "\n")
