from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pypdf import PdfReader
import faiss
import numpy as np

# --------------------------
# Load embedding model
# --------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# --------------------------
# Read PDF
# --------------------------
reader = PdfReader("data/sample.pdf")

text = ""
for page in reader.pages:
    text += page.extract_text()

# --------------------------
# Chunk text
# --------------------------
chunk_size = 300
chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

print(f"Total chunks created: {len(chunks)}")

# --------------------------
# Create embeddings
# --------------------------
embeddings = embedding_model.encode(chunks)
embeddings = np.array(embeddings)

# --------------------------
# Create FAISS index
# --------------------------
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# --------------------------
# Load LLM
# --------------------------
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

print("\nRAG system ready. Ask a question.\n")

# --------------------------
# Question loop
# --------------------------
while True:

    query = input("Your Question: ")

    if query.lower() == "exit":
        break

    query_embedding = embedding_model.encode([query])
    query_embedding = np.array(query_embedding)

    distances, indices = index.search(query_embedding, 2)

    context = ""
    for idx in indices[0]:
        context += chunks[idx] + "\n"

    prompt = f"""
    Use the context below to answer the question.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=120
    )

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nAnswer:", answer)
    print("\n--------------------------------\n")