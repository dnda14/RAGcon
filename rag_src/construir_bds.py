import json
import os
import sys
import time
from neo4j import GraphDatabase
import chromadb
from sentence_transformers import SentenceTransformer
from cypher_queries import CREATE_TRIPLET_QUERY

#sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILE_TRIPLETAS = os.path.join(BASE_DIR, 'tripleta_guerra.json') 
CHROMA_DB_DIR = os.path.join(BASE_DIR, 'chroma_db')

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

LIMPIAR_DB = True
COLECCION_S = "coleccion-sujetos"

def init_modelo_embeddings():
    print("Cargando modelo de Embeddings (all-MiniLM-L6-v2)...")
    modelo_embedding = SentenceTransformer('all-MiniLM-L6-v2')
    return modelo_embedding

def crear_coleccion_vector():
    print("Conectando a ChromaDB...")
    cliente = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    if LIMPIAR_DB:
        try:
            cliente.delete_collection(COLECCION_S)
            print(f"Colección '{COLECCION_S}' eliminada para recreación.")
        except Exception:
            pass
            
    coleccion = cliente.get_or_create_collection(
        name=COLECCION_S,
        metadata={"hnsw:space": "cosine"}
    )
    return coleccion

def load_json(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No existe el archivo {filepath}")
        
    with open(filepath, 'r', encoding='utf-8') as f:
        texto = f.read().strip()
    
    decoder = json.JSONDecoder()
    pos = 0
    objs = []
    while pos < len(texto):
        while pos < len(texto) and texto[pos].isspace():
            pos += 1
        if pos >= len(texto):
            break
        try:
            obj, new_pos = decoder.raw_decode(texto, pos)
            objs.append(obj)
            pos = new_pos
        except json.JSONDecodeError as e:
            print(f"Error de JSON en la posición {pos}: {e}")
            sgt_llave = texto.find('{', pos + 1)
            if sgt_llave == -1:
                break
            pos = sgt_llave
    print(f"Cargados {len(objs)} fragmentos ")
    return objs

def insertar_datos(driver, coleccion_vector_db, modelo_embedding, fragmentos):
    sujetos_procesados = set()
    
    # 1. Recolectar datos
    sujetos_a_insertar = {}
    tripletas_a_insertar = []
    
    for f_idx, frag in enumerate(fragmentos):
        fragmento_id = str(frag.get("fragmento_id", f"doc_{f_idx}"))
        entidades = frag.get("entidades", [])
        tripletas = frag.get("tripletas", [])
        
        entidad_map = {}
        for ent in entidades:
            canon = str(ent.get("entidad_canonica", "")).strip()
            if canon:
                for menc in ent.get("menciones", []):
                    entidad_map[str(menc).lower().strip()] = canon
                    
        for tripleta in tripletas:
            sujeto_orig = str(tripleta.get("sujeto", "")).strip()
            relacion = str(tripleta.get("relacion", "")).strip().upper().replace(" ", "_")
            objeto_orig = str(tripleta.get("objeto", "")).strip()
            descripcion = str(tripleta.get("oracion_estructurada", "")).strip()
            
            if not sujeto_orig or not objeto_orig or not relacion:
                continue
            
            sujeto = entidad_map.get(sujeto_orig.lower(), sujeto_orig)
            objeto = entidad_map.get(objeto_orig.lower(), objeto_orig)
            
            if sujeto not in sujetos_procesados and sujeto not in sujetos_a_insertar:
                sujetos_a_insertar[sujeto] = fragmento_id
            if objeto not in sujetos_procesados and objeto not in sujetos_a_insertar:
                sujetos_a_insertar[objeto] = fragmento_id
            tripletas_a_insertar.append({
                "sujeto": sujeto, "relacion": relacion, "objeto": objeto,
                "descripcion": descripcion, "fragmento_id": fragmento_id
            })
            
    # Función de ayuda para dividir en pedazos pequeños
    def chunker(seq, size):
        return (seq[pos:pos + size] for pos in range(0, len(seq), size))

    # 2. Procesar sujetos en ChromaDB (bloques pequeños de 32 para no congelar la GPU)
    if sujetos_a_insertar:
        nombres_sujetos = list(sujetos_a_insertar.keys())
        print(f"Generando embeddings de {len(nombres_sujetos)} sujetos en bloques de 32 (para no congelar la PC)...")
        
        for chunk_nombres in chunker(nombres_sujetos, 32):
            #----------------------------------------------------------------------------------------
            emb_sujetos = modelo_embedding.encode(chunk_nombres, batch_size=8).tolist()
            #----------------------------------------------------------------------------------------
            
            ids = []
            metadatas = []
            for suj in chunk_nombres:
                suj_id = suj.lower().replace(" ", "_").replace("/", "_")
                ids.append(suj_id)
                metadatas.append({"nombre_sujeto": suj, "fragmento_id": sujetos_a_insertar[suj]})
                sujetos_procesados.add(suj)
            #----------------------------------------------------------------------------------------
            
            try:
                coleccion_vector_db.add(ids=ids, embeddings=emb_sujetos, metadatas=metadatas, documents=chunk_nombres)
            except Exception as e:
                print(f"[Chroma Error] No se guardaron los sujetos: {e}")
            #----------------------------------------------------------------------------------------
            
            # PAUSA de 0.1 segundos para que Windows refresque la pantalla y no tire pantallazo azul
            time.sleep(0.1)
                
    # 3. Procesar descripciones y Neo4j (bloques pequeños de 32)
    if tripletas_a_insertar:
        print(f"Procesando e insertando {len(tripletas_a_insertar)} tripletas en bloques de 32...")
        
        with driver.session() as sesion:
            for chunk_idx, chunk_tripletas in enumerate(chunker(tripletas_a_insertar, 32)):
                print(f"  -> Procesando bloque {chunk_idx + 1}...")
                descripciones = [t["descripcion"] for t in chunk_tripletas]
            #----------------------------------------------------------------------------------------

                emb_descripciones = modelo_embedding.encode(descripciones, batch_size=8).tolist()
            #----------------------------------------------------------------------------------------
            #----------------------------------------------------------------------------------------
            
                # Transacción pequeña por bloque
                with sesion.begin_transaction() as tx:
                    for idx, t in enumerate(chunk_tripletas):
                        
                        try:
                            tx.run(
                                CREATE_TRIPLET_QUERY,
                                sujeto=t["sujeto"], relacion=t["relacion"], objeto=t["objeto"],
                                descripcion=t["descripcion"], embedding=emb_descripciones[idx],
                                fragmento_id=t["fragmento_id"]
                            )
                        except Exception as e:
                            pass
            #----------------------------------------------------------------------------------------
             
                # PAUSA de 0.1 segundos para que la GPU respire y la pantalla no se congele
                time.sleep(0.1)
                    

def main():
    
    if not os.path.exists(FILE_TRIPLETAS):
        print(f"Error: No se encontró el archivo de tripletas {FILE_TRIPLETAS}")
        return
        
    modelo_embedding = init_modelo_embeddings()
    coleccion_vector_db = crear_coleccion_vector()
    
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        driver.verify_connectivity()
        print("Conectado a Neo4j")
    except Exception as e:
        print(f"Error de conexión : {e}")
        return

    if LIMPIAR_DB:
        print("Limpiando  Neo4j ")
        with driver.session() as sesion:
            sesion.run("MATCH (n) DETACH DELETE n")
        print("Base de datos de Neo4j limpia.")

    try:
        fragmentos = load_json(FILE_TRIPLETAS)
        insertar_datos(driver, coleccion_vector_db, modelo_embedding, fragmentos)
    except Exception as e:
        print(f"Error durante la ingesta: {e}")
    finally:
        driver.close()
        
    print("\nProceso terminado.")

if __name__ == "__main__":
    main()
