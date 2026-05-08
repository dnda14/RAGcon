import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import pickle

from simulate_training_data import simulate_data, OUTPUT_PATH as CSV_PATH
from feature_engineering import extract_features_training, extract_features_inference, MODELS_DIR

XGB_MODEL_PATH = os.path.join(MODELS_DIR, 'xgb_router.pkl')

def train_model():
    print("Iniciando Fase 2: Entrenamiento del Enrutador XGBoost\n")
    
    # 1. Asegurar que tenemos datos
    if not os.path.exists(CSV_PATH):
        simulate_data()
        
    df = pd.read_csv(CSV_PATH)
    print(f"Datos cargados: {len(df)} ejemplos.")
    
    # 2. Ingeniería de Características
    X, tfidf_model = extract_features_training(df, 'Pregunta')
    y = df['Valor_d_Ideal']
    
    # 3. División de datos (Train / Test)
    # Como tenemos muy pocos datos (10), usaremos un test_size muy pequeño solo para probar la lógica
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"\nEntrenando XGBoost Regressor con {len(X_train)} ejemplos...")
    
    # 4. Instanciar y Entrenar XGBoost
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # 5. Evaluación
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Evaluación del Modelo - Error Cuadrático Medio (MSE): {mse:.4f}")
    
    # 6. Guardar el modelo
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    with open(XGB_MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Modelo guardado exitosamente en: {XGB_MODEL_PATH}")
    
    return model, tfidf_model

def predict_depth(question, xgb_model, tfidf_model):
    """Prueba rápida de inferencia para una pregunta nueva."""
    print(f"\n--- Probando Inferencia del Enrutador ---")
    print(f"Pregunta del usuario: '{question}'")
    
    X_new = extract_features_inference(question, tfidf_model)
    
    # Predecir
    d_pred = xgb_model.predict(X_new)[0]
    
    # Interpretar d
    saltos = int(d_pred)
    poda = d_pred - saltos
    
    print(f"Valor 'd' predicho: {d_pred:.4f}")
    print(f" -> Saltos en el Grafo (Profundidad): {saltos if saltos > 0 else 1}")
    print(f" -> Porcentaje de Nodos a Retener (Poda): {poda * 100:.1f}%")

if __name__ == "__main__":
    xgb_model, tfidf_model = train_model()
    
    # Probamos con una pregunta que no estaba en el dataset
    pregunta_nueva = "¿Quién inventó la máquina Bombe para los nazis?"
    predict_depth(pregunta_nueva, xgb_model, tfidf_model)
