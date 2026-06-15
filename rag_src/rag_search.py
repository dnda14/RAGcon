import json
import os
import math
import numpy as np
from neo4j import GraphDatabase
import chromadb
from sentence_transformers import SentenceTransformer

import sys
sys.stdout.reconfigure(encoding='utf-8')

# Configuración de Rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QA_FILE = os.path.join(BASE_DIR, 'qa_ww2.json')
OUTPUT_FILE = os.path.join(BASE_DIR, 'contextos_recuperados.json')
CHROMA_DB_DIR = os.path.join(BASE_DIR, 'chroma_db')

# Credenciales de Neo4j
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

def coseno_similitud(v1, v2):
    """Calcula la similitud coseno matemática entre dos vectores numpy."""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
    return dot_product / (norm_v1 * norm_v2)

class MotorBusqueda:
    def __init__(self):
        print("Cargando modelo de Embeddings (all-MiniLM-L6-v2)...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print("Conectando a ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        self.collection = self.chroma_client.get_collection(name="coleccion-sujetos")
        
        print("Conectando a Neo4j...")
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.driver.verify_connectivity()

    def close(self):
        self.driver.close()

    def find_nodos_semilla(self, query_text, n_semillas=2):
        """Paso 1: Busca múltiples nodos de entrada en el espacio semántico de ChromaDB."""
        pregunta_emb = self.model.encode(query_text).tolist()
        results = self.collection.query(
            query_embeddings=[pregunta_emb],
            n_results=n_semillas
        )
        semillas = []
        if results['metadatas'] and len(results['metadatas'][0]) > 0:
            for meta in results['metadatas'][0]:
                if meta and 'nombre_sujeto' in meta:
                    semillas.append(meta['nombre_sujeto'])
        return semillas, pregunta_emb
    def run_adaptive_bfs(self, seed_node, pregunta_emb, k, theta):
        """Pasos 2 a 5: Ejecuta el recorrido topológico adaptativo en Neo4j."""
        core_edges_ids = set()
        contexto_sent = set()
        
        with self.driver.session() as session:
            # PASO 2: Expansión Ciega (Niveles 1 a k)
            # Búsqueda BIDIRECCIONAL: seguir aristas en ambas direcciones
            query_core = f"""
            MATCH p=(n:Entidad {{nombre: $seed}})-[*1..{k}]-(m:Entidad)
            UNWIND relationships(p) AS r
            RETURN DISTINCT elementId(r) AS rel_id, r.descripcion AS desc
            """
            
            result_core = session.run(query_core, seed=seed_node)
            for record in result_core:
                r_id = record["rel_id"]
                desc = record["desc"]
                if desc:
                    core_edges_ids.add(r_id)
                    contexto_sent.add(desc)
                    
            # PASO 3 y 4: Evaluación de Frontera (Nivel k+1) y Descarte
            if theta > 0.0:
                query_bound = f"""
                MATCH p=(n:Entidad {{nombre: $seed}})-[*{k+1}]-(m:Entidad)
                UNWIND relationships(p) AS r
                RETURN DISTINCT elementId(r) AS rel_id, r.descripcion AS desc, r.embedding AS emb
                """
                result_bound = session.run(query_bound, seed=seed_node)
                
                frontera = {} # rel_id -> (desc, score)
                
                for record in result_bound:
                    r_id = record["rel_id"]
                    
                    if r_id not in core_edges_ids and r_id not in frontera:
                        emb = record["emb"]
                        desc = record["desc"]
                        if emb and desc:
                            score = coseno_similitud(pregunta_emb, np.array(emb))
                            frontera[r_id] = (desc, float(score))
                            
                if frontera:
                    frontera_ordenda = sorted(frontera.values(), key=lambda x: x[1], reverse=True)
                    contador_kee = int(math.ceil(len(frontera_ordenda) * theta))
                    
                    for i in range(contador_kee):
                        contexto_sent.add(frontera_ordenda[i][0])
                        
        # PASO 5: Serialización
        contexto_final = " ".join(list(contexto_sent))
        return contexto_final

def main():
    print("=== Generador de Conjunto de Entrenamiento (Fase 2) ===")
    
    if not os.path.exists(QA_FILE):
        print(f"Error: No se encontró el archivo de preguntas {QA_FILE}")
        return
        
    with open(QA_FILE, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
        
    engine = MotorBusqueda()
    
    # Parámetros de la Malla (Reducidos para prueba rápida)
    k_values = [1, 2,3,4]
    theta_values = [0.2, 0.5, 0.8, 1.0]
    
    resultados = []
    
    print(f"\nIniciando Grid Search sobre {len(qa_data)} consultas...")
    
    with open('llm_entities.json', 'r', encoding='utf-8') as f:
        llm_entities = json.load(f)

    # Solo procesaremos las primeras 10 preguntas para no hacer el JSON gigantesco en la prueba
    # Puedes quitar el [:10] después si quieres correr todo el dataset
    for i, item in enumerate(qa_data):
        q_id = item.get("id", f"q_{i}")
        pregunta = item.get("question", "")
        print(f"\n[Q {i+1}] {pregunta}")
        
        pregunta_emb = engine.model.encode(pregunta).tolist()
        seed_nodes = set()
        
        # 1A. Buscar semillas usando la pregunta completa (para contextos largos)
        res_full = engine.collection.query(query_embeddings=[pregunta_emb], n_results=2)
        if res_full['metadatas'] and len(res_full['metadatas'][0]) > 0:
            for meta in res_full['metadatas'][0]:
                if meta and 'nombre_sujeto' in meta:
                    seed_nodes.add(meta['nombre_sujeto'])
        
        # 1B. Buscar semillas usando las entidades extraídas por el LLM (para francotirador específico)
        entities = llm_entities.get(q_id, [])
        for ent in entities:
            ent_emb = engine.model.encode(ent).tolist()
            res_ent = engine.collection.query(query_embeddings=[ent_emb], n_results=1)
            if res_ent['metadatas'] and len(res_ent['metadatas'][0]) > 0:
                for meta in res_ent['metadatas'][0]:
                    if meta and 'nombre_sujeto' in meta:
                        seed_nodes.add(meta['nombre_sujeto'])
                        
        seed_nodes = list(seed_nodes)
        
        if not seed_nodes:
            print("  -> No se encontraron nodos semilla en ChromaDB. Saltando...")
            continue
            
        print(f"  -> Nodos Semilla: {seed_nodes}")
        
        # 2. Iterar la malla
        for k in k_values:
            for theta in theta_values:
                contextos_combinados = set()
                for seed_node in seed_nodes:
                    contexto_parcial = engine.run_adaptive_bfs(seed_node, pregunta_emb, k, theta)
                    # Evitar duplicar oraciones
                    if contexto_parcial:
                        for sentence in contexto_parcial.split(". "):
                            if sentence.strip():
                                contextos_combinados.add(sentence.strip() + ".")
                
                contexto_final = " ".join(list(contextos_combinados))
                
                # Guardar resultado crudo (antes del LLM)
                resultados.append({
                    "query_id": q_id,
                    "pregunta": pregunta,
                    "nodo_semilla": ", ".join(seed_nodes),
                    "k": k,
                    "theta": theta,
                    "longitud_contexto": len(contexto_final),
                    "contexto_recuperado": contexto_final
                })
                
    engine.close()
    
    # Exportar a JSON
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(resultados, f, ensure_ascii=False, indent=4)
        
    print(f"\n¡Proceso finalizado! Se guardaron {len(resultados)} simulaciones en: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
