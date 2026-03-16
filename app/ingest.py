from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Sample documents
documents = [
    "Azure AI provides cloud-based machine learning services.",
    "RAG stands for Retrieval Augmented Generation.",
    "FAISS is used for efficient similarity search."
]

# Convert documents to embeddings
embeddings = model.encode(documents)
embeddings = np.array(embeddings)

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("Embeddings stored in FAISS successfully.")

# -------------------------------
# Query Part
# -------------------------------

query = "What is RAG?"
query_embedding = model.encode([query])
query_embedding = np.array(query_embedding)

# Search top 1 similar document
k = 1
distances, indices = index.search(query_embedding, k)

print("\nUser Question:", query)
print("Most Relevant Document:", documents[indices[0][0]])