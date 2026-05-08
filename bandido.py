import numpy as np
import random

brazos = [0.1, 0.8, 0.5] 
q_estimado = [0, 0, 0]  # Lo que creemos que da cada brazo
conteo_brazos = [0, 0, 0]
epsilon = 0.0003  # 10% de exploración

for i in range(1000):
    if random.random() < epsilon:
        accion = random.randint(0, 2)
    else:
        accion = np.argmax(q_estimado) 

    recompensa = 1 if random.random() < brazos[accion] else 0
    
    conteo_brazos[accion] += 1
    q_estimado[accion] += (recompensa - q_estimado[accion]) / conteo_brazos[accion]

print(f"Probabilidades estimadas: {q_estimado}")
print(f"Brazos : {conteo_brazos}")
