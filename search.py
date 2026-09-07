from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("database.index")

chunks = np.load(
    "chunks.npy",
    allow_pickle=True
)

question = input("Pregunta: ")

query_embedding = model.encode([question])

query_embedding = np.array(
    query_embedding
).astype("float32")

distance, indices = index.search(
    query_embedding,
    2
)

for i in indices[0]:
    print("\nResultado:")
    print(chunks[i])