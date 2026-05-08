import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(BASE_DIR, 'training_data.csv')

def simulate_data():
    """
    Genera un dataset sintético para entrenar el modelo XGBoost.
    En la fase real, este script usaría fuerza bruta sobre HotpotQA
    para encontrar el valor d óptimo.
    """
    print("Generando datos de entrenamiento simulados...")
    
    # Lista de preguntas sintéticas y sus supuestos valores d ideales
    # d bajo (ej. 1.x) = un salto, pregunta directa
    # d alto (ej. 2.x, 3.x) = múltiples saltos, pregunta compleja/compuesta
    data = [
        {"Pregunta": "¿Quién fue Alan Turing?", "Valor_d_Ideal": 1.1},
        {"Pregunta": "¿Dónde trabajó Turing durante la Segunda Guerra Mundial?", "Valor_d_Ideal": 1.2},
        {"Pregunta": "¿Qué dirigió Alan Turing en Bletchley Park?", "Valor_d_Ideal": 2.1},
        {"Pregunta": "¿Qué es la arquitectura RAG y cómo mejora los modelos de lenguaje?", "Valor_d_Ideal": 2.5},
        {"Pregunta": "¿En qué año se propuso el Test de Turing?", "Valor_d_Ideal": 1.0},
        {"Pregunta": "¿Qué máquina ayudó a descifrar los mensajes de la máquina Enigma utilizada por los nazis?", "Valor_d_Ideal": 2.8},
        {"Pregunta": "¿Cuál es la nacionalidad del creador de la máquina Bombe?", "Valor_d_Ideal": 3.2}, # Compleja: Creador de Bombe -> Turing -> Nacionalidad
        {"Pregunta": "¿Qué es el Test de Turing?", "Valor_d_Ideal": 1.1},
        {"Pregunta": "¿Qué buscaba descifrar Hut 8?", "Valor_d_Ideal": 2.4},
        {"Pregunta": "¿Quién es considerado el padre de la ciencia de la computación?", "Valor_d_Ideal": 1.0}
    ]
    
    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_PATH, index=False, encoding='utf-8')
    print(f"Dataset simulado guardado en: {OUTPUT_PATH}")
    print(df.head())

if __name__ == "__main__":
    simulate_data()
