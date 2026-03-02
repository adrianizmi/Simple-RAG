import os
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np

# Definimos el modelo
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Funciones
def extract_text_pdf(path):
    reader = PdfReader(path)
    complete_text = ""
    for page in reader.pages:
        complete_text += page.extract_text() + "\n"
    return complete_text

def load_and_vectorize(directory="data"):
    texts = []
    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        
        if file.endswith(".txt"):
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
        
        elif file.endswith(".pdf"):
            print(f"Leyendo PDF: {file}...")
            texts.append(extract_text_pdf(path))
    
    print(f"Generando embeddings para {len(texts)} documentos...")
    embeddings = embed_model.encode(texts)
    
    return texts, embeddings

def search_best_context(question, texts, embeddings):

    # 1. We convert the user's question into a vector (list of numbers)
    question_vec = embed_model.encode([question])
    
    # 2. We calculate the similarity (Pure Mathematics: Dot Product)
    # We compare the question vector against all vectors in the database
    # np.dot performs the calculation and .flatten() converts it to a simple list
    verisimilitude = np.dot(embeddings, question_vec.T).flatten()
    
    # 3. We look for the index with the highest number (the closest match).
    winner_index = np.argmax(verisimilitude)
    
    # 4. We return the original text that corresponds to that index
    return texts[winner_index]