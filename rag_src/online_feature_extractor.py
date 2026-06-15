import os
import json
import spacy
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

class OnlineFeatureExtractor:
    def __init__(self, base_dir: str):
        """Inicializa SpaCy y ajusta el vectorizador TF-IDF con el corpus."""
        print("Cargando SpaCy para extracción de features lingüísticas...")
        self.nlp = spacy.load("en_core_web_sm")
        
        print("Ajustando TF-IDF Vectorizer para preguntas en vivo...")
        qa_path = os.path.join(base_dir, "qa_ww2_cleaned.json")
        with open(qa_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        corpus = [item['question'] for item in qa_data]
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.vectorizer.fit(corpus)

    def extraer(self, pregunta: str, pregunta_emb: list) -> pd.DataFrame:
        """Extrae todas las features lingüísticas, TF-IDF y el embedding para XGBoost."""
        doc = self.nlp(pregunta)
        num_nouns = sum(1 for token in doc if token.pos_ == "NOUN")
        num_propn = sum(1 for token in doc if token.pos_ == "PROPN")
        num_verbs = sum(1 for token in doc if token.pos_ == "VERB")
        num_adj = sum(1 for token in doc if token.pos_ == "ADJ")
        num_ents = len(doc.ents)
        
        char_len = len(pregunta)
        word_len = len(pregunta.split())
        avg_word_len = char_len / word_len if word_len > 0 else 0
        
        q_lower = pregunta.lower()
        is_who = 1 if q_lower.startswith("who") else 0
        is_what = 1 if q_lower.startswith("what") else 0
        is_where = 1 if q_lower.startswith("where") else 0
        is_when = 1 if q_lower.startswith("when") else 0
        is_why = 1 if q_lower.startswith("why") else 0
        is_how = 1 if q_lower.startswith("how") else 0
        
        tfidf_row = self.vectorizer.transform([pregunta]).toarray()[0]
        max_tfidf = np.max(tfidf_row) if np.max(tfidf_row) > 0 else 0
        avg_tfidf = np.mean(tfidf_row[tfidf_row > 0]) if len(tfidf_row[tfidf_row > 0]) > 0 else 0
        
        features_dict = {
            'num_nouns': num_nouns, 'num_propn': num_propn, 'num_verbs': num_verbs,
            'num_adj': num_adj, 'num_ents': num_ents, 'char_len': char_len,
            'word_len': word_len, 'avg_word_len': avg_word_len, 'is_who': is_who,
            'is_what': is_what, 'is_where': is_where, 'is_when': is_when,
            'is_why': is_why, 'is_how': is_how, 'max_tfidf': max_tfidf,
            'avg_tfidf': avg_tfidf
        }
        
        # Agregar el vector del embedding directamente como features
        for i in range(384):
            features_dict[f"emb_{i}"] = pregunta_emb[i]
            
        return pd.DataFrame([features_dict])
