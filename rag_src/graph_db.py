import json
import os
import re
from neo4j import GraphDatabase
import ollama
from cypher_queries import CREATE_TRIPLET_QUERY

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(BASE_DIR, 'dummy_corpus.json')

# Configuración de Neo4j (Modifica las credenciales según tu base de datos)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

# Configuración del LLM
OLLAMA_MODEL = "llama3" # O el modelo que tengas descargado: 'mistral', 'phi3', etc.

EXTRACTION_PROMPT = """
Eres un experto en extracción de grafos de conocimiento.
Analiza el siguiente texto y extrae las entidades principales y la relación que las une.
Debes devolver ÚNICAMENTE un arreglo JSON válido donde cada elemento sea un arreglo de 3 strings: [Sujeto, Relacion, Objeto].
No agregues explicaciones, ni texto antes o después del JSON. Si no encuentras relaciones, devuelve [].

Ejemplo:
[["Alan Turing", "trabajó en", "Bletchley Park"], ["Bletchley Park", "ubicado en", "Reino Unido"]]

Texto a analizar:
{text}
"""

def load_corpus(filepath):
    """Carga el corpus desde un archivo JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_triplets_with_ollama(text):
    """Usa Ollama local para extraer tripletas del texto."""
    prompt = EXTRACTION_PROMPT.format(text=text)
    
    try:
        response = ollama.chat(model=OLLAMA_MODEL, messages=[
            {
                'role': 'user',
                'content': prompt
            }
        ])
        
        # Intentar limpiar la respuesta en caso de que el LLM añada texto markdown
        raw_output = response['message']['content'].strip()
        # Buscar el bloque JSON si está envuelto en ```json ... ```
        match = re.search(r'\[\s*\[.*?\]\s*\]', raw_output, re.DOTALL)
        if match:
            json_str = match.group(0)
        else:
            json_str = raw_output
            
        triplets = json.loads(json_str)
        return triplets
    except json.JSONDecodeError:
        print(f"Error al decodificar JSON. Respuesta cruda: {raw_output}")
        return []
    except Exception as e:
        print(f"Error al conectar con Ollama: {e}")
        print(f"Asegúrate de que Ollama está corriendo y el modelo '{OLLAMA_MODEL}' está instalado.")
        return []

def insert_triplets_to_neo4j(driver, triplets):
    """Inserta una lista de tripletas en Neo4j."""
    with driver.session() as session:
        for triplet in triplets:
            if len(triplet) == 3:
                sujeto, relacion, objeto = triplet
                # Sanitizamos un poco
                sujeto = str(sujeto).strip().title()
                objeto = str(objeto).strip().title()
                relacion = str(relacion).strip().upper().replace(" ", "_")
                
                print(f"  -> Insertando: ({sujeto}) -[{relacion}]-> ({objeto})")
                
                session.run(
                    CREATE_TRIPLET_QUERY,
                    sujeto=sujeto,
                    relacion=relacion,
                    objeto=objeto
                )

def build_graph_db():
    print("Iniciando la construcción del Grafo de Conocimiento (Neo4j)...")
    
    if not os.path.exists(CORPUS_PATH):
        print(f"Error: No se encontró el corpus en {CORPUS_PATH}")
        return
        
    corpus = load_corpus(CORPUS_PATH)
    
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("Conexión a Neo4j establecida correctamente.")
    except Exception as e:
        print(f"Error de conexión a Neo4j: {e}")
        print("Asegúrate de que Neo4j Desktop o Server esté ejecutándose.")
        return

    print(f"Procesando {len(corpus)} documentos con Ollama (Modelo: {OLLAMA_MODEL})...")
    
    for i, item in enumerate(corpus):
        print(f"\n--- Documento {i+1}/{len(corpus)} ---")
        text = item['text']
        print(f"Texto: {text[:100]}...")
        
        # 1. Extracción con LLM Local
        triplets = extract_triplets_with_ollama(text)
        
        if not triplets:
            print("  Ninguna tripleta extraída o hubo un error.")
            continue
            
        print(f"  Extraídas {len(triplets)} tripletas.")
        
        # 2. Inserción en Grafo
        insert_triplets_to_neo4j(driver, triplets)
        
    driver.close()
    print("\n¡Proceso de construcción de grafo finalizado!")

if __name__ == "__main__":
    build_graph_db()
