import os
import json
import math
import string
import pandas as pd
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer, util
import spacy

# Configuración
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QA_PATH = os.path.join(BASE_DIR, 'qa_real.json')
OUTPUT_CSV = os.path.join(BASE_DIR, 'training_data_real.csv')

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

def normalize_text(text):
    """Normaliza texto para Exact Match"""
    text = str(text).lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return " ".join(text.split())

def exact_match(answer, context):
    """Verifica si la respuesta está en el contexto"""
    norm_ans = normalize_text(answer)
    norm_ctx = normalize_text(context)
    return norm_ans in norm_ctx

class GridSearchGenerator:
    def __init__(self):
        print("Inicializando Grid Search Generator...")
        self.nlp = spacy.load("en_core_web_sm")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        
    def close(self):
        self.driver.close()

    def _extract_entities(self, text):
        doc = self.nlp(text)
        entities = [ent.text.title() for ent in doc.ents]
        if not entities:
            entities = [token.text.title() for token in doc if token.pos_ == "PROPN"]
        return entities

    def get_triplets_up_to_depth(self, entities, max_depth):
        query = f"""
        MATCH path=(n:Entidad)-[*1..{max_depth}]-(m:Entidad)
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
        return list(triplets_text)

    def process_query(self, question, expected_answer):
        entities = self._extract_entities(question)
        if not entities:
            return None # Fallo al extraer entidades
            
        question_emb = self.embedder.encode(question, convert_to_tensor=True)
        
        best_d = None
        
        # Grid Search
        # Iteramos profundidad desde 1 hasta 5
        for d_int in range(1, 6):
            candidatos = self.get_triplets_up_to_depth(entities, d_int)
            if not candidatos:
                continue
                
            candidatos_emb = self.embedder.encode(candidatos, convert_to_tensor=True)
            cos_scores = util.cos_sim(question_emb, candidatos_emb)[0]
            
            scored_candidates = list(zip(candidatos, cos_scores.tolist()))
            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            
            # Iteramos poda fraccional de 0.1 a 0.9 (y 1.0)
            frac_found = False
            for frac_step in range(1, 11):
                frac = frac_step / 10.0
                num_retener = max(1, int(len(candidatos) * frac))
                
                ganadores = scored_candidates[:num_retener]
                contexto_final = " ".join([txt for txt, score in ganadores])
                
                if exact_match(expected_answer, contexto_final):
                    # Si frac == 1.0, lo tomaremos como d_int exacto (ej. 1.0, 2.0). 
                    # Si frac es 0.1, será d_int + 0.1
                    # En la lógica original d_frac era d_pred - d_int.
                    # Asumimos d_pred = d_int + (frac si frac < 1.0 else 0)
                    if frac == 1.0:
                        best_d = float(d_int)
                    else:
                        best_d = d_int + frac
                        
                    frac_found = True
                    break # Encontramos la fracción mínima exitosa para este d_int
                    
            if frac_found:
                break # Encontramos la profundidad mínima exitosa

        return best_d

def generate_data():
    if not os.path.exists(QA_PATH):
        print(f"Error: No se encontró {QA_PATH}. Ejecuta download_datasets.py primero.")
        return
        
    with open(QA_PATH, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
        
    generator = GridSearchGenerator()
    results = []
    
    print(f"Iniciando Grid Search sobre {len(qa_data)} preguntas...")
    
    success_count = 0
    for i, item in enumerate(qa_data):
        print(f"Procesando [{i+1}/{len(qa_data)}]: {item['question']}")
        
        ideal_d = generator.process_query(item['question'], item['answer'])
        
        if ideal_d is not None:
            results.append({
                "Pregunta": item['question'],
                "Valor_d_Ideal": round(ideal_d, 2)
            })
            success_count += 1
            print(f"  -> Éxito! Valor_d_Ideal: {ideal_d}")
        else:
            print(f"  -> Falló (la respuesta no se encontró en el grafo o no hubo entidades)")
            
    generator.close()
    
    if results:
        df = pd.DataFrame(results)
        df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8')
        print(f"\nGeneración completada. Guardados {success_count} ejemplos en {OUTPUT_CSV}")
    else:
        print("\nNinguna pregunta tuvo éxito. No se guardó el CSV.")

if __name__ == "__main__":
    generate_data()
