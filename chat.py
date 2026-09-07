import ollama
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("database.index")

chunks = np.load(
    "chunks.npy",
    allow_pickle=True
)

question = input("Pregunta: ")

query_embedding = embedding_model.encode([question])
query_embedding = np.array(query_embedding).astype("float32")

distance, indices = index.search(
    query_embedding,
    2
)

context = ""

for i in indices[0]:
    context += chunks[i]
    context += "\n"

prompt = f"""
Usa solamente la información del contexto.

Contexto:

{context}

Pregunta:

{question}

Respuesta:
"""

response = ollama.chat(
    model="llama3.2",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nRespuesta:")
print(response["message"]["content"])