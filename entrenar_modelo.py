import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("1. Cargando ml_dataset.csv...")
df = pd.read_csv('ml_dataset.csv')

# ---------------------------------------------------------
# PREPARACIÓN DE DATOS
# ---------------------------------------------------------
print("2. Preparando variables (CON Embeddings y Objetivo Combinado)...")

# Crear el objetivo combinado (número decimal completo: k + theta)
y = df['k'] + df['theta']

# Eliminar SOLO las columnas que no son características (MANTENEMOS los embeddings)
cols_eliminar = ['q_id', 'k', 'theta']

# Matriz de características X
X = df.drop(columns=cols_eliminar)

print(f"Dimensiones de X (con embeddings): {X.shape}")

# Split de los datos (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------------------------------------
# ENTRENAMIENTO DEL MODELO DE REGRESIÓN
# ---------------------------------------------------------
print("\n3. Entrenando XGBRegressor para predecir el decimal completo (k.theta)...")
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    max_depth=5,
    learning_rate=0.05,
    n_estimators=150,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------------------------------------
# EVALUACIÓN
# ---------------------------------------------------------
print("\n4. Evaluando el modelo...")
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R2 Score: {r2:.4f}")

print("\nEjemplo de Predicciones:")
ejemplos = pd.DataFrame({'Real': y_test.values[:5], 'Predicción': y_pred[:5]})
print(ejemplos)

print("\n5. Importancia de Variables (Top 10)...")
features_importa = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(features_importa.head(10))

model.save_model("xgb_model_saved.json")
print("\nModelo guardado exitosamente como: xgb_model_saved.json")
