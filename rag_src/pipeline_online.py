import os
import math
import pickle
import ollama
import spacy
from sentence_transformers import SentenceTransformer, util
from neo4j import GraphDatabase

from feature_engineering import extract_features_inference, MODELS_DIR

# Rutas y Modelos
XGB_MODEL_PATH = os.path.join(MODELS_DIR, 'xgb_router.pkl')
TFIDF_MODEL_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')

# Credenciales de Neo4j
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

# Configuración del LLM
OLLAMA_MODEL = "llama3"

# Prompt final de generación
GENERATION_PROMPT = """
Responde la [PREGUNTA] del usuario utilizando ÚNICAMENTE la información provista en el siguiente [CONTEXTO].
Si el contexto no contiene la respuesta, responde "No tengo suficiente información para responder".

[CONTEXTO]:
{context}

[PREGUNTA]:
{question}
"""

class RAGOnlinePipeline:
    def __init__(self):
        print("Inicializando Pipeline Online...")
        
        # 1. Cargar Modelos de ML
        print("  -> Cargando Enrutador XGBoost y TF-IDF...")
        with open(XGB_MODEL_PATH, 'rb') as f:
            self.xgb_model = pickle.load(f)
        with open(TFIDF_MODEL_PATH, 'rb') as f:
            self.tfidf_model = pickle.load(f)
            
        # 2. Cargar Modelos NLP
        print("  -> Cargando spaCy y SentenceTransformers...")
        self.nlp = spacy.load("es_core_news_sm")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 3. Conexión a Base de Datos
        print("  -> Conectando a Neo4j...")
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        self.driver.close()

    def _extract_entities(self, text):
        doc = self.nlp(text)
        # Extraemos entidades y sustantivos propios como fallback
        entities = [ent.text.title() for ent in doc.ents]
        if not entities:
            entities = [token.text.title() for token in doc if token.pos_ == "PROPN"]
        return entities

    def query(self, question):
        print(f"\n{'='*50}\nNUEVA CONSULTA: '{question}'\n{'='*50}")
        
        # ==========================================
        # PASO 1 y 2: Recepción y Enrutamiento (Routing)
        # ==========================================
        print("\n[Paso 1 y 2] Analizando complejidad y prediciendo profundidad...")
        features = extract_features_inference(question, self.tfidf_model)
        d_pred = self.xgb_model.predict(features)[0]
        
        # Como es una regresión, aseguramos que d no sea negativo ni menor que 1.
        d_pred = max(1.0, d_pred)
        d_int = int(math.floor(d_pred))
        d_frac = d_pred - d_int
        
        print(f" -> Valor d calculado: {d_pred:.4f}")
        print(f" -> Saltos en Grafo: {d_int}")
        print(f" -> Porcentaje de Poda: {d_frac*100:.1f}%")

        # ==========================================
        # PASO 3: Búsqueda y Caminata en el Grafo
        # ==========================================
        print("\n[Paso 3] Extrayendo entidades y buscando en Neo4j...")
        entities = self._extract_entities(question)
        print(f" -> Entidades detectadas: {entities}")
        
        if not entities:
            return "No pude detectar entidades claras en tu pregunta para buscar en el grafo."

        # Consulta Cypher dinámica para Random Walk (limitado por d_int)
        # Retornamos las relaciones directas que estén a d_int saltos
        query = f"""
        MATCH path=(n:Entidad)-[*1..{d_int}]-(m:Entidad)
        WHERE n.nombre IN $entidades
        UNWIND relationships(path) AS r
        RETURN startNode(r).nombre AS sujeto, r.tipo AS relacion, endNode(r).nombre AS objeto
        """
        
        triplets_text = set()
        with self.driver.session() as session:
            result = session.run(query, entidades=entities)
            for record in result:
                sujeto = record["sujeto"]
                relacion = str(record["relacion"]).replace("_", " ").lower()
                objeto = record["objeto"]
                sentence = f"{sujeto} {relacion} {objeto}."
                triplets_text.add(sentence)
                
        candidatos = list(triplets_text)
        print(f" -> Nodos/Relaciones candidatas encontradas: {len(candidatos)}")
        
        if not candidatos:
            return "No encontré información relacionada a esas entidades en mi base de conocimiento."

        # ==========================================
        # PASO 4: Poda Vectorial (Top-K)
        # ==========================================
        print("\n[Paso 4] Aplicando Poda Vectorial...")
        question_emb = self.embedder.encode(question, convert_to_tensor=True)
        candidatos_emb = self.embedder.encode(candidatos, convert_to_tensor=True)
        
        # Similitud Coseno
        cos_scores = util.cos_sim(question_emb, candidatos_emb)[0]
        
        # Ordenar de mayor a menor similitud
        scored_candidates = list(zip(candidatos, cos_scores.tolist()))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Poda: "Retener el X% de los nodos" dictado por la parte fraccional
        num_retener = max(1, int(len(candidatos) * d_frac))
        
        # Si la parte fraccional es muy baja o cero, conservamos al menos el mejor resultado
        ganadores = scored_candidates[:num_retener]
        
        print(f" -> Reteniendo {num_retener} candidato(s) más relevantes.")
        for txt, score in ganadores:
            print(f"    [Score: {score:.3f}] {txt}")
            
        contexto_final = " ".join([txt for txt, score in ganadores])

        # ==========================================
        # PASO 5: Generación (LLM)
        # ==========================================
        print("\n[Paso 5] Generando respuesta con Ollama...")
        final_prompt = GENERATION_PROMPT.format(context=contexto_final, question=question)
        
        try:
            response = ollama.chat(model=OLLAMA_MODEL, messages=[
                {'role': 'user', 'content': final_prompt}
            ])
            answer = response['message']['content'].strip()
            print(f"\n{'='*50}\nRESPUESTA DEL SISTEMA:\n{answer}\n{'='*50}")
            return answer
            
        except Exception as e:
            print(f"Error de generación con Ollama: {e}")
            return "Error al generar la respuesta."

if __name__ == "__main__":
    pipeline = RAGOnlinePipeline()
    
    try:
        # Pregunta de prueba integral
        pregunta = "¿Qué relación tiene Alan Turing con Bletchley Park?"
        pipeline.query(pregunta)
    finally:
        pipeline.close()
