import os
import ollama
from dotenv import load_dotenv
from rag_search import DagRagSearchEngine
from gemini_extractor import extract_entities_gemini

# Cargar variables de entorno desde el archivo .env
load_dotenv()

# Configuración del LLM para generación de respuestas
OLLAMA_MODEL = "gemma2:2b"
# Leer la API Key del archivo .env
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Prompt final de generación (Anti-Alucinación Estricto)
GENERATION_PROMPT = """
Eres un asistente experto en historia, extremadamente estricto y preciso.
Tu única tarea es responder la [PREGUNTA] basándote EXCLUSIVAMENTE en el [CONTEXTO] proporcionado.

REGLA CRÍTICA: Tienes estrictamente prohibido usar tu conocimiento pre-entrenado o agregar cualquier dato que no esté explícitamente escrito en el [CONTEXTO].
Si agregas nombres, fechas, lugares o detalles que no aparecen en el contexto, tu respuesta será considerada incorrecta.
Si el [CONTEXTO] no contiene información suficiente para responder, debes responder ÚNICAMENTE: "No tengo suficiente información para responder".

[CONTEXTO]:
{context}

[PREGUNTA]:
{question}
"""

class RAGOnlinePipeline:
    def __init__(self):
        print("Inicializando Pipeline Online (End-to-End)...")
        self.engine = DagRagSearchEngine()

    def query(self, question: str):
        print(f"\n{'='*60}\nNUEVA CONSULTA: '{question}'\n{'='*60}")
        
        # ==========================================
        # PASO 1: Extracción de Entidades (Gemini)
        # ==========================================
        print("\n[Paso 1] Extrayendo entidades con Gemini...")
        entities = extract_entities_gemini(question, GEMINI_API_KEY)
        print(f" -> Entidades detectadas: {entities}")
        
        # ==========================================
        # PASO 2: Búsqueda Híbrida de Semillas
        # ==========================================
        print("\n[Paso 2] Buscando nodos semilla en ChromaDB (Búsqueda Híbrida)...")
        query_emb = self.engine.model.encode(question).tolist()
        seed_nodes = set()
        
        # 2A. Búsqueda por pregunta completa
        res_full = self.engine.collection.query(query_embeddings=[query_emb], n_results=2)
        if res_full['metadatas'] and len(res_full['metadatas'][0]) > 0:
            for meta in res_full['metadatas'][0]:
                if meta and 'nombre_sujeto' in meta:
                    seed_nodes.add(meta['nombre_sujeto'])
                    
        # 2B. Búsqueda por entidades
        for ent in entities:
            ent_emb = self.engine.model.encode(ent).tolist()
            res_ent = self.engine.collection.query(query_embeddings=[ent_emb], n_results=1)
            if res_ent['metadatas'] and len(res_ent['metadatas'][0]) > 0:
                for meta in res_ent['metadatas'][0]:
                    if meta and 'nombre_sujeto' in meta:
                        seed_nodes.add(meta['nombre_sujeto'])
                        
        seed_nodes = list(seed_nodes)
        
        if not seed_nodes:
            print(" -> No se encontraron nodos semilla. Abortando búsqueda.")
            return "No encontré información relacionada en mi base de conocimiento."
            
        print(f" -> Nodos semilla finales: {seed_nodes}")

        # ==========================================
        # PASO 3: Expansión BFS y Poda (Neo4j)
        # ==========================================
        print("\n[Paso 3] Recorriendo el Grafo y Aplicando Poda...")
        # Parámetros fijos que validamos
        k = 1
        theta = 0.8
        
        contextos_combinados = set()
        for seed_node in seed_nodes:
            contexto_parcial = self.engine.run_adaptive_bfs(seed_node, query_emb, k, theta)
            if contexto_parcial:
                # contexto_parcial es un string con varias oraciones. Lo agregamos completo.
                contextos_combinados.add(contexto_parcial)
                
        if not contextos_combinados:
            print(" -> El recorrido no recuperó contexto relevante.")
            return "No encontré relaciones válidas para tu consulta."
            
        contexto_final = " ".join(list(contextos_combinados))
        print(f" -> Contexto recuperado: {len(contexto_final)} caracteres.")
        
        # ==========================================
        # PASO 4: Generación (LLM Local - Gemma)
        # ==========================================
        print(f"\n[Paso 4] Generando respuesta con Ollama ({OLLAMA_MODEL})...")
        final_prompt = GENERATION_PROMPT.format(context=contexto_final, question=question)
        
        try:
            response = ollama.chat(model=OLLAMA_MODEL, messages=[
                {'role': 'user', 'content': final_prompt}
            ])
            answer = response['message']['content'].strip()
            print(f"\n{'='*60}\nRESPUESTA DEL SISTEMA:\n{answer}\n{'='*60}")
            return answer
            
        except Exception as e:
            print(f"[Error] Falló la generación con Ollama: {e}")
            return "Error al generar la respuesta."

if __name__ == "__main__":
    pipeline = RAGOnlinePipeline()
    try:
        # Pregunta de prueba
        pregunta = "Who commanded the German troops sent to aid the Italians in North Africa?"
        pipeline.query(pregunta)
    finally:
        pipeline.engine.close()
