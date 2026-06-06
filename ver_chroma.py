import os
import chromadb

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, 'chroma_db')

def inspeccionar_chroma():
    
    client = chromadb.PersistentClient(path=DB_DIR)
    
    colecciones = client.list_collections()
    if not colecciones:
        print("La base de datos está vacía (no hay colecciones).")
        return

    print("Colecciones encontradas:")
    for col in colecciones:
        print(f" -> Nombre: '{col.name}' | Total de documentos: {col.count()}")
    
    print("\n" + "="*50)
    
    print(" colección 'coleccion-sujetos':")
    try:
        coleccion = client.get_collection("coleccion-sujetos")
        resultados = coleccion.get(limit=3, include=["documents", "metadatas"])
        
        for i in range(len(resultados['ids'])):
            print(f"\nDocumento {i+1}:")
            print(f" - ID: {resultados['ids'][i]}")
            print(f" - Texto (Sujeto): {resultados['documents'][i]}")
            print(f" - Metadatos: {resultados['metadatas'][i]}")
            
    except ValueError:
        print("La colección 'coleccion-sujetos' no existe aún.")

if __name__ == "__main__":
    inspeccionar_chroma()
