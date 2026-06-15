import pandas as pd
import numpy as np
import spacy
import os
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("Cargando el dataset...")
input_path = os.path.join(BASE_DIR, 'optimal_params_ALL.csv')
df = pd.read_csv(input_path)

print("Cargando modelo SpaCy...")
nlp = spacy.load("en_core_web_sm")

print("Cargando modelo de Embeddings...")
modelo_emdedding = SentenceTransformer('all-MiniLM-L6-v2')

print("Calculando TF-IDF del corpus...")
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matriz = vectorizer.fit_transform(df['question'].tolist())
feature_nombres = vectorizer.get_feature_nombres_out()

# Extraer características
print("Extrayendo características lingüísticas y semánticas...")

features_lista = []
embeddings_list = []

for idx, row in df.iterrows():
    q_text = str(row['question'])
    
    # --- 1. SpaCy Features ---
    doc = nlp(q_text)
    num_nouns = sum(1 for token in doc if token.pos_ == "NOUN")
    num_propn = sum(1 for token in doc if token.pos_ == "PROPN")
    num_verbs = sum(1 for token in doc if token.pos_ == "VERB")
    num_adj = sum(1 for token in doc if token.pos_ == "ADJ")
    num_ents = len(doc.ents)
    
    # --- 2. Estructural Features ---
    char_len = len(q_text)
    word_len = len(q_text.split())
    avg_word_len = char_len / word_len if word_len > 0 else 0
    
    # --- 3. Intent/Wh-words ---
    q_lower = q_text.lower()
    is_who = 1 if q_lower.startswith("who") else 0
    is_what = 1 if q_lower.startswith("what") else 0
    is_where = 1 if q_lower.startswith("where") else 0
    is_when = 1 if q_lower.startswith("when") else 0
    is_why = 1 if q_lower.startswith("why") else 0
    is_how = 1 if q_lower.startswith("how") else 0
    
    # --- 4. TF-IDF Rareness ---
    # Obtenemos el vector tfidf de esta pregunta
    tfidf_row = tfidf_matriz[idx].toarray()[0]
    # La máxima puntuación tfidf indica qué tan "rara" es la palabra más única de esta pregunta
    max_tfidf = np.max(tfidf_row) if np.max(tfidf_row) > 0 else 0
    avg_tfidf = np.mean(tfidf_row[tfidf_row > 0]) if len(tfidf_row[tfidf_row > 0]) > 0 else 0

    features_lista.append({
        'q_id': row['q_id'],
        'k': row['k'],
        'theta': row['theta'],
        'num_nouns': num_nouns,
        'num_propn': num_propn,
        'num_verbs': num_verbs,
        'num_adj': num_adj,
        'num_ents': num_ents,
        'char_len': char_len,
        'word_len': word_len,
        'avg_word_len': avg_word_len,
        'is_who': is_who,
        'is_what': is_what,
        'is_where': is_where,
        'is_when': is_when,
        'is_why': is_why,
        'is_how': is_how,
        'max_tfidf': max_tfidf,
        'avg_tfidf': avg_tfidf
    })

print("Extrayendo Embeddings de todas las preguntas...")
# Extraer embeddings en bloque para mayor velocidad
embeddings = modelo_emdedding.encode(df['question'].tolist(), show_progress_bar=True)

print("Ensamblando Dataset Final...")
# Crear dataframe de características heurísticas
df_features = pd.DataFrame(features_lista)

# Crear dataframe de embeddings
emb_cols = [f"emb_{i}" for i in range(embeddings.shape[1])]
df_embeddings = pd.DataFrame(embeddings, columns=emb_cols)

# Combinar todo
df_final = pd.concat([df_features, df_embeddings], axis=1)

output_file = os.path.join(BASE_DIR, 'ml_dataset.csv')
df_final.to_csv(output_file, index=False)
print(f"\n¡Proceso Terminado! Dataset para ML guardado en {output_file}")
print(f"Filas: {df_final.shape[0]}, Columnas: {df_final.shape[1]}")
