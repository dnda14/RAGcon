import os
import json
from datasets import load_dataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(BASE_DIR, 'corpus_real.json')
QA_PATH = os.path.join(BASE_DIR, 'qa_real.json')

NUM_SAMPLES_PER_DATASET = 20  # Limitamos a 20 por dataset para no demorar la creación del grafo

def fetch_hotpot_qa():
    print("Descargando muestra de HotpotQA...")
    dataset = load_dataset('hotpot_qa', 'distractor', split='train', streaming=True)
    qa_list = []
    lista_contxt = []
    for i, item in enumerate(dataset):
        if i >= NUM_SAMPLES_PER_DATASET: break
        q_id = f"hotpot_{item['id']}"
        question = item['question']
        answer = item['answer']
        
        full_ctxt = ""
        for title, sentences in zip(item['context']['title'], item['context']['sentences']):
            full_ctxt += f"{title}: " + " ".join(sentences) + "\n"
            
        qa_list.append({"id": q_id, "question": question, "answer": answer, "dataset": "HotpotQA"})
        lista_contxt.append({"id": q_id, "text": full_ctxt.strip()})
    return qa_list, lista_contxt

def fetch_squad():
    print("Descargando muestra de SQuAD...")
    try:
        dataset = load_dataset('squad', split='train', streaming=True)
        qa_list = []
        lista_contxt = []
        for i, item in enumerate(dataset):
            if i >= NUM_SAMPLES_PER_DATASET: break
            q_id = f"squad_{item['id']}"
            question = item['question']
            answer = item['answers']['text'][0] if item['answers']['text'] else ""
            
            full_ctxt = item['context']
                
            qa_list.append({"id": q_id, "question": question, "answer": answer, "dataset": "SQuAD"})
            lista_contxt.append({"id": q_id, "text": full_ctxt.strip()})
        return qa_list, lista_contxt
    except Exception as e:
        print(f"Error cargando SQuAD: {e}")
        return [], []

def fetch_nq():
    print("Descargando muestra de NQ (vía MRQA)...")
    try:
        # Usamos MRQA para obtener un formato más limpio de Natural Questions
        dataset = load_dataset('mrqa', split='train', streaming=True)
        qa_list = []
        lista_contxt = []
        count = 0
        for item in dataset:
            if item['subset'] != 'NaturalQuestionsShort': continue
            if count >= NUM_SAMPLES_PER_DATASET: break
            
            q_id = f"nq_{item['qid']}"
            question = item['question']
            answer = item['answers'][0] if item['answers'] else ""
            context = item['context']
            
            qa_list.append({"id": q_id, "question": question, "answer": answer, "dataset": "NQ"})
            lista_contxt.append({"id": q_id, "text": context.strip()})
            count += 1
        return qa_list, lista_contxt
    except Exception as e:
        print(f"Error cargando NQ: {e}")
        return [], []

def fetch_adversarial_qa():
    print("Descargando muestra de AdversarialQA...")
    try:
        dataset = load_dataset('adversarial_qa', 'adversarialQA', split='train', streaming=True)
        qa_list = []
        lista_contxt = []
        for i, item in enumerate(dataset):
            if i >= NUM_SAMPLES_PER_DATASET: break
            q_id = f"advqa_{item['id']}"
            question = item['question']
            answer = item['answers']['text'][0] if item['answers']['text'] else ""
            
            full_ctxt = item['context']
                
            qa_list.append({"id": q_id, "question": question, "answer": answer, "dataset": "AdversarialQA"})
            lista_contxt.append({"id": q_id, "text": full_ctxt.strip()})
        return qa_list, lista_contxt
    except Exception as e:
        print(f"Error cargando AdversarialQA: {e}")
        return [], []

def main():
    print("Iniciando descarga de datasets de prueba...")
    
    all_qa = []
    all_context = []
    
    qa, ctx = fetch_hotpot_qa()
    all_qa.extend(qa)
    all_context.extend(ctx)
    
    qa, ctx = fetch_squad()
    all_qa.extend(qa)
    all_context.extend(ctx)
    
    qa, ctx = fetch_nq()
    all_qa.extend(qa)
    all_context.extend(ctx)
    
    qa, ctx = fetch_adversarial_qa()
    all_qa.extend(qa)
    all_context.extend(ctx)
    
    print(f"\nSe recolectaron {len(all_qa)} preguntas y {len(all_context)} contextos en total.")
    
    with open(CORPUS_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_context, f, ensure_ascii=False, indent=2)
    print(f"Corpus guardado en {CORPUS_PATH}")
    
    with open(QA_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_qa, f, ensure_ascii=False, indent=2)
    print(f"Preguntas de prueba guardadas en {QA_PATH}")

if __name__ == "__main__":
    main()
