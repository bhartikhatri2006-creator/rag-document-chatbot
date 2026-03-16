from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import faiss
import numpy as np
import torch

# -----------------------------
# 1. Load Embedding Model
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# -----------------------------
# 2. Sample Documents
# -----------------------------
documents = [
    "Azure AI provides cloud-based machine learning services for building intelligent applications.",
    "Retrieval Augmented Generation (RAG) is an AI technique where a system retrieves relevant documents from a knowledge base and then uses a language model to generate answers using that information.",
    "FAISS is a vector similarity search library developed by Facebook for fast retrieval of embeddings."
]

# -----------------------------
# 3. Create Embeddings + FAISS Index
# -----------------------------
embeddings = embedding_model.encode(documents)
embeddings = np.array(embeddings)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# -----------------------------
# 4. User Query
# -----------------------------
query = "Explain RAG in simple words."

query_embedding = embedding_model.encode([query])
query_embedding = np.array(query_embedding)

k = 1
distances, indices = index.search(query_embedding, k)
retrieved_doc = documents[indices[0][0]]

print("Retrieved Context:", retrieved_doc)

# -----------------------------
# 5. Load FLAN-T5 Model Properly
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# -----------------------------
# 6. Generate Final Answer
# -----------------------------
prompt = f"""
Use the context below to answer the question clearly.

Context:
{retrieved_doc}

Question:
{query}

Answer:
"""

inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False
)

final_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nFinal Answer:")
print(final_answer)