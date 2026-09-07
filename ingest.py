from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

with open(
    "data/documento.txt",
    "r",
    encoding="utf-8"
) as f:
    text = f.read()

chunks = text.split("\n\n")

print("Chunks:")

for c in chunks:
    print(c)
    print()

embeddings = model.encode(chunks)

embeddings = np.array(
    embeddings
).astype("float32")

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

faiss.write_index(
    index,
    "database.index"
)

np.save(
    "chunks.npy",
    chunks
)

print("Base creada")
