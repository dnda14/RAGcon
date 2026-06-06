import json
import os
import pypdf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(BASE_DIR, 'libro_guerra.pdf')
OUTPUT_JSON = os.path.join(BASE_DIR, 'corpus_ww2.json')

def chunk_texto(texto, chunk_sz=1000, solape=200):
    chunks = []
    inicio = 0
    while inicio < len(texto):
        end = inicio + chunk_sz
        chunks.append(texto[inicio:end])
        if end >= len(texto):
            break
        inicio += chunk_sz - solape
    return chunks

def procesar_pdf():
    print("cargando pdf")
    try:
        reader = pypdf.PdfReader(PDF_PATH)
        total_pags = len(reader.pages)
        print(f"Total de pags: {total_pags}")
    except Exception as e:
        print(f"Error al cargar el PDF: {e}")
        return

    texto_completo = ""
    for i in range(total_pags):
        pag = reader.pages[i]
        texto = pag.extract_text()
        if texto:
            texto_completo += texto + "\n"

    texto_completo = texto_completo.replace('\n', ' ').strip()
    texto_completo = " ".join(texto_completo.split())

    chunks = chunk_texto(texto_completo, chunk_sz=1000, solape=200)
    print(f" generaron {len(chunks)} chunks.")

    texto_json = []
    for i, chunk in enumerate(chunks):
        texto_json.append({
            "id": f"doc_ww2_{i}",
            "texto": chunk
        })

    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(texto_json, f, ensure_ascii=False, indent=4)
        
    print("\n fin")

if __name__ == "__main__":
    procesar_pdf()
