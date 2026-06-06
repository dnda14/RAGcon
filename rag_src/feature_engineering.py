import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    print("Error: No se encontró el modelo de spaCy. Ejecuta: python -m spacy download es_core_news_sm")
    nlp = None

def get_spacy_features(text):
    """Extrae características numéricas de un texto usando spaCy."""
    if not nlp:
        return 0, 0, 0
    
    doc = nlp(text)
    len_chars = len(text)
    num_entities = len(doc.ents)
    num_verbs = sum(1 for token in doc if token.pos_ == "VERB")
    
    return len_chars, num_entities, num_verbs

def extract_features_training(df, text_column='Pregunta'):
    """Extrae características para entrenamiento y ajusta el TF-IDF."""
    print("Extrayendo características lingüísticas (spaCy)...")
    
    # 1. Características manuales
    df['len_chars'], df['num_entities'], df['num_verbs'] = zip(*df[text_column].apply(get_spacy_features))
    
    # 2. Vectorización TF-IDF
    print("Vectorizando texto (TF-IDF)...")
    tfidf = TfidfVectorizer(max_features=50, stop_words=None) # Reducido a 50 para el corpus pequeño
    tfidf_matrix = tfidf.fit_transform(df[text_column]).toarray()
    
    # Guardar el vectorizador TF-IDF para usarlo en inferencia luego
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    with open(os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf, f)
        
    # 3. Combinar todo
    tfidf_df = pd.DataFrame(tfidf_matrix, columns=[f"tfidf_{i}" for i in range(tfidf_matrix.shape[1])])
    
    # Unir las características manuales con los vectores TF-IDF
    X = pd.concat([df[['len_chars', 'num_entities', 'num_verbs']], tfidf_df], axis=1)
    
    return X, tfidf

def extract_features_inference(text, tfidf_model):
    """Extrae características de una sola pregunta usando el modelo TF-IDF ya entrenado."""
    len_chars, num_entities, num_verbs = get_spacy_features(text)
    
    tfidf_vector = tfidf_model.transform([text]).toarray()
    tfidf_df = pd.DataFrame(tfidf_vector, columns=[f"tfidf_{i}" for i in range(tfidf_vector.shape[1])])
    
    manual_features = pd.DataFrame([[len_chars, num_entities, num_verbs]], columns=['len_chars', 'num_entities', 'num_verbs'])
    X = pd.concat([manual_features, tfidf_df], axis=1)
    
    return X

if __name__ == "__main__":
    # Prueba rápida
    print("Probando módulo de Feature Engineering...")
    df_test = pd.DataFrame([{"Pregunta": "¿Quién inventó la máquina Enigma?"}])
    X_test, _ = extract_features_training(df_test)
    print("Características extraídas:\n", X_test.head())
