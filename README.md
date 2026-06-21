# RAGcon: GraphRAG Híbrido con Poda Semántica Dinámica

RAGcon es un sistema de Generación Aumentada por Recuperación (RAG) avanzado que combina bases de datos vectoriales y grafos de conocimiento. Su innovación principal radica en la predicción continua de la profundidad de búsqueda mediante un modelo de Machine Learning (XGBoost), lo que permite adaptar el recorrido en el grafo y aplicar poda semántica dinámica (filtrado de ruido) basándose en la complejidad inherente de la consulta del usuario.

## Arquitectura del Sistema

El *pipeline* del sistema se divide en cuatro fases principales:

### Fase 1: Ingesta y Grafo de Conocimiento (Offline)
- **Ingesta:** Los documentos fuente (PDFs) se extraen y segmentan usando `pypdf` con solapamiento semántico (`ingestar_pdf.py`).
- **Extracción de Conocimiento:** Se usa un modelo LLM (ChatGPT) para extraer manualmente tripletas estructuradas (Sujeto, Relación, Objeto) de los fragmentos, resolviendo implícitamente las correferencias.
- **Base de Datos Híbrida:** La información se vectoriza mediante `all-MiniLM-L6-v2`. Las entidades (sujetos) se guardan en **ChromaDB** para la búsqueda vectorial inicial, mientras que la red completa de tripletas (y el *embedding* de su contexto original) se aloja en **Neo4j** como un grafo dirigido (`construir_bds.py`).

### Fase 2: Entrenamiento Predictivo (Offline)
- Se genera un conjunto de datos de entrenamiento mediante búsqueda exhaustiva (Grid Search) simulando múltiples parámetros de búsqueda (`rag_busqueda.py`, `generate_training_data.py`).
- Se extraen características numéricas de las preguntas (`feature_engineering.py`).
- Se entrena un regresor **XGBoost** capaz de predecir el factor de profundidad óptima continua ($d$) dada una nueva consulta (`entrenar_xgboost.py`).

### Fases 3 y 4: Inferencia y Generación (Online en tiempo real)
- Al ingresar una consulta, se extraen entidades clave con `spaCy` y se vectorizan para encontrar los **nodos semilla múltiples** en ChromaDB.
- Paralelamente, el extractor en línea (`online_feature_extractor.py`) alimenta a XGBoost para predecir $d$.
- Se ejecuta un algoritmo adaptativo (BFS) en Neo4j: la profundidad máxima ($k = \lceil d \rceil$) define el alcance, y la parte fraccionaria ($\theta = d - \lfloor d \rfloor$) actúa como umbral para descartar aristas ruidosas en la frontera del grafo.
- El subgrafo recuperado y serializado se envía al LLM local (ej. `gemma2:2b` mediante **Ollama**) para redactar la respuesta final (`pipeline_online.py`).

## Estructura de Archivos Principales (`rag_src/`)

- `ingestar_pdf.py`: Lectura y segmentación (*chunking*) de documentos PDF.
- `construir_bds.py`: Inserción de entidades en ChromaDB y relaciones/aristas en Neo4j.
- `cypher_queries.py`: Consultas Cypher fundamentales para Neo4j.
- `rag_busqueda.py`: Motor central de búsqueda con BFS bidireccional y filtrado por similitud coseno.
- `feature_engineering.py` / `online_feature_extractor.py`: Extracción de características lingüísticas y estructurales (TF-IDF, NER, etc.) de las preguntas.
- `generate_training_data.py` / `entrenar_xgboost.py`: Scripts para generar el dataset empírico y entrenar el modelo de ruteo/profundidad.
- `pipeline_online.py`: Script principal (*end-to-end*) que recibe la consulta del usuario y retorna la respuesta generada.

## Tecnologías Utilizadas
- **Grafos:** Neo4j, lenguaje Cypher.
- **Bases Vectoriales y Embeddings:** ChromaDB, `sentence-transformers` (`all-MiniLM-L6-v2`).
- **Procesamiento de Lenguaje (NLP):** `spaCy`.
- **Machine Learning Predictivo:** `xgboost`, `scikit-learn`, `pandas`.
- **LLM de Generación:** Ollama (`gemma2:2b`).
- **Utilidades:** `pypdf`, `dotenv`.

## Ejecución del Pipeline Online
Asegúrate de tener en ejecución las bases de datos (Neo4j y ChromaDB locales) y el servicio de Ollama.
```bash
python rag_src/pipeline_online.py
```
El script pedirá la consulta, buscará semillas híbridas, predecirá la profundidad $d$ con XGBoost, extraerá el contexto de Neo4j y generará la respuesta en consola.
