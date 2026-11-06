# my_langgraph_agent01/src/agent.py
import os
import sys
from langchain_openai import ChatOpenAI

# Ensure the root directory is in Python's path so 'src.' imports always work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# MessagesState is a stateful representation of a conversation thread, storing the sequence of messages exchanged between the user and the agent.
from langgraph.graph import StateGraph, START, END, MessagesState

# get_checkpointer and get_vector_store are utility functions that provide access to the Postgres checkpointer and vector store, respectively, for managing long-term memory and persistence.
#from database import get_checkpointer, get_vector_store
from src.database import get_checkpointer, get_vector_store
from dotenv import load_dotenv
load_dotenv(".env")
api_key = os.getenv("OPENROUTER_API_KEY")
print(f"DEBUG OPENROUTER_API_KEY: {api_key[:10]}..." if api_key else "DEBUG OPENROUTER_API_KEY: NOT FOUND!")
# Initialize Google's free-tier Gemini model
#model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
model = ChatOpenAI(
    model="meta-llama/llama-3.1-8b-instruct",
    api_key=api_key,
    temperature=0,
    base_url="https://openrouter.ai/api/v1",
    max_tokens=2000,
    
    )

# the var config is a dictionary that can hold configuration parameters for the agent, such as user ID and thread ID, which are used to manage state and memory retrieval.
# the var store is an instance of PostgresStore that provides access to the pgvector database for storing and retrieving long-term memory associated with user interactions.
def call_model(state: MessagesState, config, *, store):
    """
    Core agent node that queries pgvector memory and prompts Gemini.
    """
    user_id = config.get("configurable", {}).get("user_id", "default_user")
    print(f"\n 📧 Line 36 - def call_model - user_id:",{user_id})
    namespace = ("users", user_id)
    print(f"\n 📧 Line 38 - def call_model - namespace:",{namespace})
    latest_message = state["messages"][-1].content
    print(f"\n 📧 Line 40 - def call_model - latest_message:",{latest_message})
    
    # Semantic search via pgvector
    # store.search retrieves the most relevant memories from the pgvector database based on the latest user message, allowing the agent to incorporate past interactions into its response.
    memories = store.search(namespace, query=latest_message, limit=3)
    memory_context = "\n".join([f"- {m.value.get('text')}" for m in memories]) if memories else "No prior memories found."

    system_prompt = (
        "You are a helpful, stateful AI assistant with long-term memory. "
        "Use the following retrieved user memories when relevant to personalize your response:\n"
        f"{memory_context}"
    )

    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    
    response = model.invoke(messages)
    return {"messages": [response]}

# Build the workflow graph
# StateGraph is a directed graph that defines the flow of the agent's operations, where nodes represent processing steps (like calling the model) and edges define the sequence of execution from start to end.
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)

workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

# Initialize checkpointer and vector store
# Hooks up PostgreSQL (PostgresSaver) to save the short-term conversation thread history (so chat history survives across app reboots).
# comes from the database.py file, which sets up the PostgresSaver and PostgresStore instances using the connection pool. The checkpointer is responsible for saving the state of the conversation thread to the Postgres database, allowing the agent to maintain context across sessions and recover from interruptions.
checkpointer = get_checkpointer()

# store is an instance of PostgresStore that provides access to the pgvector database for storing and retrieving long-term memory associated with user interactions. It is used by the agent to save and retrieve user memories across sessions.
store = get_vector_store()

# The checkpointer is responsible for saving the state of the conversation thread to the Postgres database, allowing the agent to maintain context across sessions and recover from interruptions.
graph = workflow.compile(checkpointer=checkpointer, store=store)

def save_graph_image(output_filename="my_langgraph_agent01.png"):
    """Generates and saves the LangGraph architecture diagram as a PNG image in project root."""
    try:
        print("\n-- Generando imagen del grafo ---")
        # Obtiene la ruta del archivo actual (src/) y sube un nivel a la raíz del proyecto
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.dirname(current_dir)
        output_path = os.path.join(root_dir, output_filename)
        
        # Obtiene los bytes de la imagen desde LangGraph
        image_bytes = graph.get_graph().draw_mermaid_png()
        
        # Escribe el archivo en el disco duro
        with open(output_path, "wb") as f:
            f.write(image_bytes)
            
        print(f"¡Éxito! Archivo guardado en: {output_path}")
        print(f"Busca el archivo '{output_filename}' en el panel izquierdo de tu editor y hazle clic.")

    except Exception as e:
        print(f"\n[ERROR] No se pudo guardar la imagen debido a: {e}")
        print("Asegúrate de haber instalado: pip install grandalf pillow")

# If you run `python src/agent.py` directly, it will build the image automatically!
if __name__ == "__main__":
    save_graph_image()

# Test invoking the graph directly to see prints
    print("\n--- Line 100 - Testing Agent Invocation ---")
    config = {"configurable": {"user_id": "sergio", "thread_id": "test-1"}}
    response = graph.invoke({"messages": [{"role": "user", "content": "Hello!"}]}, config=config)
    print("Agent Response:", response["messages"][-1].content)