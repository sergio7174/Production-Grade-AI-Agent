# my_langgraph_agent01/src/generate_graph.py
import os
from src.agent import graph

def generate_graph_image():
    """Generates and saves the LangGraph architecture diagram as a PNG image."""
    try:
        print("\n-- Generando imagen del grafo ---")
        # Obtiene la ruta de la carpeta raíz del proyecto
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(current_dir, "langraph_101.png")
        
        # Obtiene los bytes de la imagen desde LangGraph
        image_bytes = graph.get_graph().draw_mermaid_png()
        
        # Escribe el archivo en el disco duro
        with open(output_path, "wb") as f:
            f.write(image_bytes)
            
        print(f"¡Éxito! Archivo guardado en: {output_path}")
        print("Busca el archivo 'langraph_101.png' en el panel izquierdo de tu editor y hazle clic para verlo.")

    except Exception as e:
        print(f"\n[ERROR] No se pudo guardar la imagen debido a: {e}")
        print("Asegúrate de tener conexión a internet (para la renderización) o haber instalado: pip install grandalf pillow")

if __name__ == "__main__":
    generate_graph_image()