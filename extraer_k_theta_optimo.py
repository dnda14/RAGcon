import pandas as pd
import json

# 1. Cargar las preguntas originales
with open('qa_ww2_cleaned.json', 'r', encoding='utf-8') as f:
    qa_data = json.load(f)

# Crear diccionario para buscar rapido
qa_dict = {item['id']: item['question'] for item in qa_data}

# 2. Cargar los resultados exhaustivos (híbrido)
df = pd.read_csv('exhaustive_results_900_to_950.csv')

# 3. Filtrar solo los casos exitosos
df_found = df[df['found'] == True].copy()

# 4. Ordenar por k (ascendente) y luego theta (ascendente)
# Así, la primera fila de cada grupo será la que tenga menor k y menor theta
df_sorted = df_found.sort_values(by=['q_id', 'k', 'theta'], ascending=[True, True, True])

# 5. Tomar el primer registro por cada q_id
df_optimal = df_sorted.drop_duplicates(subset=['q_id'], keep='first').copy()

# 6. Agregar el texto de la pregunta
df_optimal['question'] = df_optimal['q_id'].map(qa_dict)

# Reordenar columnas para que sea más legible
cols = ['q_id', 'question', 'k', 'theta', 'context_len']
df_final = df_optimal[cols]

# 7. Guardar en el nuevo CSV
output_file = 'optimal_params_900_to_950.csv'
df_final.to_csv(output_file, index=False)

print(f"Se procesaron los resultados exitosos y se guardaron en: {output_file}")
print(f"Total de preguntas encontradas: {len(df_final)}")
