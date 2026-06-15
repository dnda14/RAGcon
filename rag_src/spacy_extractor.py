import spacy
import subprocess

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Descargando el modelo de SpaCy (en_core_web_sm)...")
    subprocess.check_call(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

def extra_entidades_spacy(pregunta):
    """
    Extrae entidades nombradas y sustantivos principales de una pregunta.
    Es determinista y ligero. Funciona offline.
    """
    doc = nlp(pregunta)
    
    # 1. Extraer Entidades Nombradas
    entidades = [ent.text for ent in doc.ents]
    
    # 2. Extraer Sustantivos y Nombres Propios
    sustantivos = [token.text for token in doc if token.pos_ in ["PROPN", "NOUN"]]
    
    # Unir ambas listas, convertir a minúsculas para unificación simple, y quitar duplicados
    resultados_crudos = list(set([r.lower() for r in (entidades + sustantivos)]))
    
    # Filtrar palabras comunes de interrogación o irrelevantes (stopwords genéricas)
    stopwords = {"connection", "role", "what", "who", "which", "where", "how", "way", "decision", "series", "events"}
    resultados = [r for r in resultados_crudos if r not in stopwords]
    
    return resultados
