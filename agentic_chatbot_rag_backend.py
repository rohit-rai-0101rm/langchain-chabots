from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

import os
import sqlite3
from dotenv import load_dotenv

load_dotenv()


GROK_API_KEY = os.getenv("GROK_API_KEY")

llm = ChatOpenAI(
    model="qwen/qwen3.8-27b",
    temperature=0,
    api_key=GROK_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    # Groq free tier caps OUTPUT at 1000 tokens/minute (OTPM). Keep every
    # single response under that so a call isn't rejected before it runs.
    max_tokens=900,
)


# ========================= RAG: PDF ingestion + retrieval =========================

# Local, offline embedding model so no extra API key is needed for embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# FAISS index is persisted here so uploaded PDFs survive an app restart
FAISS_INDEX_DIR = "faiss_pdf_index"


def _load_vector_store():
    if os.path.exists(FAISS_INDEX_DIR):
        return FAISS.load_local(
            FAISS_INDEX_DIR,
            embeddings,
            allow_dangerous_deserialization=True,
        )
    return None


def ingest_rag_document(file_path: str) -> int:
    """Load a PDF, split it into chunks, and add it to the FAISS index.

    Returns the number of chunks that were ingested.
    """

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    chunks = splitter.split_documents(documents)

    vector_store = _load_vector_store()

    if vector_store is None:
        vector_store = FAISS.from_documents(chunks, embeddings)
    else:
        vector_store.add_documents(chunks)

    vector_store.save_local(FAISS_INDEX_DIR)

    return len(chunks)


@tool
def search_uploaded_pdfs(query: str) -> str:
    """Search the user's uploaded PDF documents for information relevant to the query.

    Use this whenever the user asks a question that could be answered by a PDF
    they attached in the chat.
    """

    vector_store = _load_vector_store()

    if vector_store is None:
        return "No PDF documents have been uploaded yet."

    results = vector_store.similarity_search(query, k=4)

    if not results:
        return "No relevant information was found in the uploaded PDFs."

    return "\n\n---\n\n".join(doc.page_content for doc in results)


tools = [search_uploaded_pdfs]
llm_with_tools = llm.bind_tools(tools)


# ========================= LangGraph chatbot =========================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def chat_node(state: ChatState):
    messages = state["messages"]

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


conn = sqlite3.connect(database="chatbaot.db", check_same_thread=False)
checkpoint = SqliteSaver(conn)


graph = StateGraph(ChatState)

# add nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", ToolNode(tools))

# add edges
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpoint)


def get_all_threads():
    all_threads = set()

    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config["configurable"]["thread_id"])
    return list(all_threads)
