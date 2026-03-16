## Architecture

![RAG Architecture](architecture.png)

# Retrieval-Augmented Generation (RAG) System

This project implements a simple Retrieval-Augmented Generation (RAG) pipeline for document-based question answering.

The system retrieves relevant document chunks using vector similarity search and then generates answers using a language model.

## Architecture

User Question  
↓  
SentenceTransformer Embeddings  
↓  
FAISS Vector Search  
↓  
Retrieve Relevant Context  
↓  
FLAN-T5 Language Model  
↓  
Generated Answer

## Technologies Used

- Python
- SentenceTransformers
- FAISS (Vector Database)
- HuggingFace Transformers
- FLAN-T5
- PyPDF

## Features

- Reads PDF documents
- Splits documents into chunks
- Generates embeddings for semantic search
- Retrieves relevant document context
- Uses LLM to generate answers grounded in retrieved documents

## Setup

1. Clone the repository
git clone <repo_url>

2. Create virtual environment
python -m venv venv

3. Activate environment
venv\Scripts\activate

4. Install dependencies
pip install -r requirements.txt

5. Put your document file as sample.pdf in data folder as follows
project
│
├── app
├── data
├── README.md
├── requirements.txt

6.## Run the project
python app/rag.py

7.Then ask questions based on the document.

Type `exit` to stop the program.

## Example

Your Question: when will balance CL lapse?

Answer: 28 February


## Future Improvements

- Add FastAPI API layer
- Build web chatbot interface
- Persist FAISS index
- Add support for multiple documents