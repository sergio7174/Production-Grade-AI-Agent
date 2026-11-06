# my_langgraph_agent01/tests/test_agent.py
import os
import sys
# Automatically add the project root directory to Python's path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dotenv import load_dotenv
from src.agent import graph, store

load_dotenv(".env")

def test_agent_memory_and_persistence():
    print("\n========================================")
    print("🚀 STARTING AGENT MEMORY & PERSISTENCE TEST")
    print("========================================")
    
    user_id = "sergio_user"
    thread_id = "thread_session_01"
    namespace = ("users", user_id)
    
    # 1. Seed a long-term memory into the pgvector store
    print("\n[Setup] Storing a long-term fact into pgvector...")
    # so the store.put method is used to save a long-term memory into the pgvector database. The namespace ("users", user_id) ensures that the memory is associated with the specific user, and "bio_fact" serves as a unique key for this particular piece of information. The value being stored is a dictionary containing a text description of the user's identity and current project, which can be retrieved later by the agent to provide context-aware responses.  
    store.put(
        namespace,
        # what it means "bio_fact" is a key that identifies the type of memory being stored, and the associated value is a dictionary containing the actual memory content. In this case, it stores a fact about the user, Sergio, indicating that he is a full-stack web developer working on stateful AI agents with LangGraph and PostgreSQL.
        # "bio_fact" is a key that identifies the type of memory being stored, what kind of memory is this? It is a long-term memory that provides context about the user, Sergio, and his current project. This information can be retrieved later by the agent to personalize responses and maintain continuity in the conversation.
        # Long-Term Memory (PostgresStore): Lives independently of chat threads under the user's namespace (("users", "sergio_user")). Even if you close the application, delete the thread, or start a conversation months later, the fact stored under "bio_fact" remains permanently available for the agent to search and retrieve.
        "bio_fact",
        {"text": "The user is Sergio, a full-stack web developer building stateful AI agents with LangGraph and PostgreSQL."}
    )
    print("✓ Long-term memory saved successfully.")
    
    # Configuration bundle for LangGraph (passing both thread and user ID)
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    
    # 2. Turn 1: Ask a question that triggers long-term memory retrieval
    print("\n--- Turn 1: Asking agent about stored memory ---")
    message_1 = {"messages": [("user", "Hi! Can you tell me what you know about me and my current project?")]}
    
    response_1 = graph.invoke(message_1, config=config)
    print(f"\nUser: {message_1['messages'][0][1]}")
    print(f"Agent: {response_1['messages'][-1].content}")
    
    # 3. Turn 2: Continue conversation on the same thread (testing short-term checkpointer history)
    print("\n--- Turn 2: Continuing the same thread ---")
    message_2 = {"messages": [("user", "What database stack am I using for this?")]}
    
    response_2 = graph.invoke(message_2, config=config)
    print(f"\nUser: {message_2['messages'][0][1]}")
    print(f"Agent: {response_2['messages'][-1].content}")
    
    print("\n========================================")
    print("✨ TEST COMPLETED SUCCESSFULLY!")
    print("========================================")

if __name__ == "__main__":
    test_agent_memory_and_persistence()