import json
import os
import chromadb
from sentence_transformers import SentenceTransformer

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(BASE_DIR, 'dummy_corpus.json')
DB_DIR = os.path.join(BASE_DIR, 'chroma_db')

def load_corpus(filepath):
    """Carga el corpus desde un archivo JSON."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def build_vector_db():
    print("Iniciando la construcción de la Base de Datos Vectorial...")
    
    # 1. Cargar corpus
    if not os.path.exists(CORPUS_PATH):
        print(f"Error: No se encontró el corpus en {CORPUS_PATH}")
        return
        
    corpus = load_corpus(CORPUS_PATH)
    print(f"Corpus cargado con {len(corpus)} documentos.")
    
    # 2. Inicializar Modelo de Embedding Local
    # Usamos un modelo ligero y eficiente en español/multilingüe o general.
    # 'all-MiniLM-L6-v2' es muy rápido. Para mejor soporte en español podríamos usar 'paraphrase-multilingual-MiniLM-L12-v2'
    model_name = 'all-MiniLM-L6-v2'
    print(f"Cargando modelo de embeddings local: {model_name}...")
    model = SentenceTransformer(model_name)
    
    # 3. Inicializar ChromaDB de forma persistente
    print(f"Inicializando ChromaDB en el directorio local: {DB_DIR}")
    client = chromadb.PersistentClient(path=DB_DIR)
    
    # Crear o recuperar la colección
    collection_name = "mi_corpus_rag"
    collection = client.get_or_create_collection(name=collection_name)
    
    # Preparar datos para inserción por lotes
    ids = []
    documents = []
    embeddings = []
    
    print("Vectorizando textos...")
    for item in corpus:
        doc_id = item['id']
        text = item['text']
        
        ids.append(doc_id)
        documents.append(text)
        # Convertir a lista nativa para chromadb
        emb = model.encode(text).tolist()
        embeddings.append(emb)
        
    # 4. Ingestar en la Base de Datos
    print("Insertando documentos en ChromaDB...")
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings
    )
    
    print(f"¡Éxito! Se han guardado {collection.count()} documentos en la colección '{collection_name}'.")

def test_query(query_text):
    """Prueba rápida para verificar que la búsqueda semántica funciona."""
    print(f"\n--- Probando búsqueda semántica ---")
    print(f"Consulta: '{query_text}'")
    
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection("mi_corpus_rag")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    query_emb = model.encode(query_text).tolist()
    
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=2
    )
    
    for i in range(len(results['ids'][0])):
        print(f"\nResultado {i+1} (ID: {results['ids'][0][i]}, Distancia: {results['distances'][0][i]:.4f}):")
        print(f"Texto: {results['documents'][0][i]}")

if __name__ == "__main__":
    build_vector_db()
    # Ejecutamos una prueba sencilla
    test_query("¿Qué es la arquitectura RAG?")
