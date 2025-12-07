import os
import json
import sys
from typing import Annotated, Literal, TypedDict, List
from dotenv import load_dotenv

# LangGraph & LangChain
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver # <--- 1. IMPORT MEMORY
from langgraph.graph.message import add_messages    # <--- 2. IMPORT REDUCER

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.graphs import Neo4jGraph

load_dotenv()

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma_db")

if not os.getenv("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY1")

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

# Initialize Neo4j
try:
    neo4j_graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD")
    )
except: neo4j_graph = None

# --- TOOLS ---

@tool
def query_vector_db(query: str):
    """Use for specific details, transcripts, visual descriptions, timestamps."""
    try:
        print(f"   🔍 [VectorDB] Searching: '{query}'...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings, collection_name="rakshak_intel_logs")
        results = vector_store.similarity_search(query, k=4)
        context = "\n".join([f"SOURCE: {d.metadata.get('source')} | CONTENT: {d.page_content}" for d in results])
        return context if context else "No data found."
    except Exception as e: return f"Error: {e}"

@tool
def query_knowledge_graph(query: str):
    """Use for relationships, connections, tactical insights, and risk scores."""
    try:
        print(f"   🕸️ [KnowledgeGraph] Querying: '{query}'...")
        if not neo4j_graph: return "Neo4j not connected."
        neo4j_graph.refresh_schema()
        # Simple extraction logic for demo (In prod, use LLM generation logic here)
        # For now, let's just return a generic prompt that allows the LLM to know it has access
        return f"GRAPH ACCESS GRANTED. Executing logic for: {query}" 
    except Exception as e: return f"Error: {e}"

tools = [query_vector_db, query_knowledge_graph]
llm_with_tools = llm.bind_tools(tools)

# --- STATE ---

class AgentState(TypedDict):
    # 'add_messages' tells LangGraph to APPEND new messages to history, not overwrite
    messages: Annotated[List[BaseMessage], add_messages] 

# --- NODES ---

def reasoner_node(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

def human_approval_node(state: AgentState):
    last_message = state["messages"][-1]
    
    if not last_message.tool_calls:
        return None

    tool_call = last_message.tool_calls[0]
    print(f"\n⚠️  SYSTEM REQUEST: Access [{tool_call['name'].upper()}]?")
    
    user_choice = input("    >>> ALLOW ACCESS? (y/n): ").strip().lower()
    
    if user_choice == 'y':
        print("    ✅ Access Granted.")
        return None
    else:
        print("    ❌ Access Denied.")
        return {"messages": [ToolMessage(tool_call_id=tool_call["id"], content="User denied access.")]}

# --- GRAPH BUILD ---

workflow = StateGraph(AgentState)
workflow.add_node("agent", reasoner_node)
workflow.add_node("human_check", human_approval_node)
workflow.add_node("tools", ToolNode(tools))

workflow.set_entry_point("agent")

def router(state):
    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage): return "agent" # Back to agent if denied
    if last_message.tool_calls: return "human_check" # Check permission
    return END

def human_router(state):
    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage): return "agent" # Denied -> Agent
    return "tools" # Approved -> Tools

workflow.add_conditional_edges("agent", router, {"human_check": "human_check", END: END})
workflow.add_conditional_edges("human_check", human_router, {"agent": "agent", "tools": "tools"})
workflow.add_edge("tools", "agent")

# --- MEMORY SETUP ---
memory = MemorySaver() # Initialize In-Memory Checkpointer
app = workflow.compile(checkpointer=memory) # Bind memory to graph

# --- MAIN LOOP ---

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🛡️  RAKSHAK SECURE QUERY INTERFACE (Memory Enabled)")
    print("="*60 + "\n")

    # Define a Thread ID to persist memory for this session
    thread_id = "session_1"
    config = {"configurable": {"thread_id": thread_id}}

    while True:
        try:
            q = input("COMMANDER: ")
            if q.lower() in ["exit", "quit"]: break
            
            # We pass the config to ensure memory is saved/loaded
            events = app.stream(
                {"messages": [HumanMessage(content=q)]}, 
                config=config,
                stream_mode="values"
            )
            
            for event in events:
                if "messages" in event:
                    msg = event["messages"][-1]
                    if isinstance(msg, AIMessage) and not msg.tool_calls:
                        print(f"\n🤖 RAKSHAK: {msg.content}\n")
                        print("-" * 50)

        except Exception as e:
            print(f"Error: {e}")