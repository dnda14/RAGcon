import os
import sys

# Forzar codificación UTF-8 para evitar problemas de caracteres en consola de Windows
sys.stdout.reconfigure(encoding='utf-8')

# Agregar el directorio raíz al path para importar correctamente
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_src.rag_search import DagRagSearchEngine

def main():
    print("=" * 60)
    print("  PROBADOR INTERACTIVO DE BÚSQUEDA HÍBRIDA (DAG-RAG) ")
    print("=" * 60)
    
    preguntas_ejemplo = {
        "1": "Who dominated Europe after World War I?",
        "2": "What was the role of the U.S. Army in World War II?",
        "3": "Who led the Nazi Party in Germany in 1933?",
        "4": "What happened at OMAHA Beach in Normandy?",
        "5": "Who was Mussolini and what alliance did he announce?"
    }

    # 1. Preguntas de sugerencia
    print("\nPreguntas de ejemplo sobre la Segunda Guerra Mundial:")
    for num, texto in preguntas_ejemplo.items():
        print(f" {num}. {texto}")
    
    # 2. Capturar entrada del usuario
    pregunta_input = input("\nIntroduce tu pregunta (o un número del 1 al 5, o Enter para el 1): ").strip()
    
    if not pregunta_input:
        pregunta = preguntas_ejemplo["1"]
    elif pregunta_input in preguntas_ejemplo:
        pregunta = preguntas_ejemplo[pregunta_input]
    else:
        pregunta = pregunta_input
        
    try:
        k = int(input("Define la profundidad ciega 'k' (ej. 1 o 2): ").strip())
    except ValueError:
        k = 1
        print("  -> Valor no válido. Usando k = 1 por defecto.")
        
    try:
        theta = float(input("Define el umbral de filtrado 'theta' (entre 0.0 y 1.0, ej. 0.5): ").strip())
        if not (0.0 <= theta <= 1.0):
            raise ValueError
    except ValueError:
        theta = 0.5
        print("  -> Valor no válido. Usando theta = 0.5 por defecto.")
        
    print("\nIniciando motor de búsqueda...")
    try:
        engine = DagRagSearchEngine()
    except Exception as e:
        print(f"\n[Error de Conexión] No se pudo inicializar el motor: {e}")
        print("Por favor, asegúrate de que Neo4j esté corriendo en localhost:7687.")
        return

    print("\n" + "-"*40)
    print(f"Pregunta: '{pregunta}'")
    print(f"Configuración: k = {k} | theta = {theta}")
    print("-"*40)
    
    # 1. Buscar el Nodo Semilla en ChromaDB
    print("\n[Paso 1] Buscando nodo semilla en ChromaDB...")
    seed_node, query_emb = engine.find_seed_node(pregunta)
    
    if not seed_node:
        print("  [X] No se encontró ningún nodo semilla relevante en ChromaDB.")
        engine.close()
        return
        
    print(f"  [OK] Nodo semilla encontrado: '{seed_node}'")
    
    # 2. Ejecutar el recorrido adaptativo en Neo4j
    print(f"\n[Paso 2] Ejecutando recorrido adaptativo BFS (Nivel {k} e inteligente Nivel {k+1}) en Neo4j...")
    contexto = engine.run_adaptive_bfs(seed_node, query_emb, k, theta)
    
    # 3. Mostrar resultados
    print("\n" + "="*50)
    print(" CONTEXTO RECUPERADO ")
    print("="*50)
    if not contexto.strip():
        print("  El recorrido no recuperó ninguna oración. Prueba aumentando 'k' o 'theta'.")
    else:
        # Formatear el texto recuperado para que sea legible separándolo por oraciones
        oraciones = contexto.split(". ")
        for i, oracion in enumerate(oraciones):
            oracion_clean = oracion.strip()
            if oracion_clean:
                if not oracion_clean.endswith("."):
                    oracion_clean += "."
                print(f" [{i+1}] {oracion_clean}\n")
                
        print("-"*50)
        print(f"Total de caracteres recuperados: {len(contexto)}")
        print(f"Total de hechos/oraciones conectadas: {len(oraciones)}")
    print("="*50)
    
    engine.close()
    print("\nBúsqueda finalizada.")

if __name__ == "__main__":
    main()
