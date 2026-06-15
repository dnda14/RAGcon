import os
import json
import ollama
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import xgboost as xgb
from rag_busqueda import MotorBusqueda
from spacy_extractor import extraer_entidades_spacy
from online_feature_extractor import OnlineFeatureExtractor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv()

OLLAMA_MODEL = "gemma2:2b"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GENERATION_PROMPT = """
You are an expert history assistant.
Your task is to answer the [QUESTION] based ONLY on the provided [CONTEXT].

CRITICAL RULES:
1. Do not use outside knowledge. 
2. If the exact answer is in the [CONTEXT], provide it clearly and concisely.
3. If the [CONTEXT] does not contain the answer, say exactly: "I do not have enough information to answer."

[CONTEXT]:
{context}

[QUESTION]:
{pregunta}
"""

class RAGOnlinePipeline:
    def __init__(self):
        print("Inicializando Pipeline Online (End-to-End)...")
        self.motor = MotorBusqueda()
        
        # Cargar Modelos ML
        print("Cargando Modelos de Machine Learning (XGBoost Continuo y SpaCy)...")
        self.model_reg = xgb.XGBRegressor()
        self.model_reg.load_model(os.path.join(BASE_DIR, "xgb_modelo_combinado_embedding.json"))
        
        self.feature_extractor = OnlineFeatureExtractor(BASE_DIR)

    def query(self, pregunta: str):
        print(f"\n{'='*60}\nNUEVA CONSULTA: '{pregunta}'\n{'='*60}")
        
       #EXTRAER ENTIDADES
        print("\n[Paso 1] Extrayendo entidades con SpaCy...")
        entidades = extraer_entidades_spacy(pregunta)
        print(f" -> Entidades detectadas: {entidades}")
        
        
        print("\n[Paso 2] Buscando nodos semilla en ChromaDB (Búsqueda Híbrida)...")
        pregunta_emb = self.motor.modelo.encode(pregunta).tolist()
        nodos_semillas = set()
        semillas_pregunta = []
        # 2A. Búsqueda por pregunta 
        res_full = self.motor.collection.query(query_embeddings=[pregunta_emb], n_results=2)
        if res_full['metadatas'] and len(res_full['metadatas'][0]) > 0:
            for meta in res_full['metadatas'][0]:
                if meta and 'nombre_sujeto' in meta:
                    nodos_semillas.add(meta['nombre_sujeto'])
                    semillas_pregunta.append(meta['nombre_sujeto'])
        print(f"Semillas por pregunta: {semillas_pregunta}")
        
        for ent in entidades:
            ent_emb = self.motor.modelo.encode(ent).tolist()
            res_ent = self.motor.collection.query(query_embeddings=[ent_emb], n_results=1)
            if res_ent['metadatas'] and len(res_ent['metadatas'][0]) > 0:
                for meta in res_ent['metadatas'][0]:
                    if meta and 'nombre_sujeto' in meta:
                        nodos_semillas.add(meta['nombre_sujeto'])
                        
        nodos_semillas = list(nodos_semillas)
        
        if not nodos_semillas:
            print(" -> No se encontraron nodos semilla. Abortando búsqueda.")
            return "No encontré información relacionada en mi base de conocimiento."
            
        print(f" -> Nodos semilla finales: {nodos_semillas}")

        # PASO 3: Expansión BFS y Poda con XGBoost Continuo
        import math
        print("\n[Paso 3] Prediciendo Parámetros con XGBoost Continuo...")
        df_features = self.feature_extractor.extraer(pregunta, pregunta_emb)
        d_pred = float(self.model_reg.predict(df_features)[0])
        
        # 1. k: Aseguramos los saltos redondeando hacia arriba (Mínimo 1, Máximo 5)
        k = math.ceil(d_pred)
        k = max(1, min(k, 5))
        
        # 2. theta: Extraemos la precisión decimal para la poda
        pred_theta = d_pred - math.floor(d_pred)
        theta = max(0.1, min(0.9, pred_theta))
        print(f" -> Predicción: k = {k}, theta = {theta:.3f}")
        
        print("\n[Paso 3.5] Recorriendo el Grafo y Aplicando Poda...")
        
        contextos_combinados = set()
        for nodo_semilla in nodos_semillas:
            contexto_parcial = self.motor.run_bfs(nodo_semilla, pregunta_emb, k, theta)
            if contexto_parcial:
                # contexto_parcial es un string con varias oraciones
                contextos_combinados.add(contexto_parcial)
                
        if not contextos_combinados:
            print(" -> El recorrido no recuperó contexto relevante.")
            return "No encontré relaciones válidas para tu consulta."
        
        for i in contextos_combinados:
            print(i)
        contexto_final = " ".join(list(contextos_combinados))
        print(f" -> Contexto recuperado: {len(contexto_final)} caracteres.")
        
        # PASO 4: Generación 
        print(f"\n[Paso 4] Generando respuesta con  ({OLLAMA_MODEL})...")
        final_prompt = GENERATION_PROMPT.format(context=contexto_final, pregunta=pregunta)
        
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
        #pregunta = "What was the connection between Churchill, Eisenhower, and the Allied invasion of North Africa?"
        #pregunta = "When Messi was born?"
        #pregunta ="What diplomatic result allowed French troops in Africa to join the Allied cause?"
        pregunta = "What did the Axis commanders hope to achieve by sending forces through the passes in Tunisia on February 14, 1943?"
        pipeline.query(pregunta)
    finally:
        pipeline.motor.close()


#intro
#trabajos relacionados
#propuesta explicada
#marco teorico
#firmar la propuesta otra vez

#docuemnto y codigo subir
