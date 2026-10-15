import os
import glob
import urllib.request
import zipfile
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from transformers import pipeline
 
# Reading all extracted text files
documents = []
extract_dir = "asia_documents"
for filepath in glob.glob(f"{extract_dir}/**/*.txt", recursive=True):
    with open(filepath, 'r', encoding='utf-8') as file:
        documents.append(file.read())
 
print(f"Loaded {len(documents)} documents.")
# If the zip contained a CSV instead, you'd use pd.read_csv() here.

print(documents[0])  # Print the first document to verify content

# chunking the documents into smaller, semantic pieces
def chunk_text(text, chunk_size=25, overlap=5):
    """Split the text into chunks of specified size with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

all_chunks = []
for doc in documents:
    chunks = chunk_text(doc)
    all_chunks.extend(chunks)

print(f"Total chunks created: {len(all_chunks)}")

for chunk in all_chunks[:1]:  # Print the first chunk to verify content
    print(chunk)
    print("---")


# Initializing the embedding model (runs locally on Colab's free CPU/GPU)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
 
# Generating embeddings for all text chunks
print("Generating embeddings...")
chunk_embeddings = embedding_model.encode(all_chunks, show_progress_bar=True)

print(f"Generated embeddings for {len(chunk_embeddings)} chunks.")
print(f"Embedding dimension: {chunk_embeddings.shape[1]}")

# defining retriever class, which mimics the behavior of a vector database
class SklearnRetriever:
    def __init__(self, n_neighbors=3):
        # Using 'cosine' metric because LLM embeddings measure semantic similarity based on angles between vectors
        self.retriever = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', algorithm='brute')
        self.chunks = []
 
    def fit(self, chunks, embeddings):
        """Indexes the documents."""
        self.chunks = chunks
        self.retriever.fit(embeddings)
        print("Vector index built successfully!")
 
    def retrieve(self, query_embedding):
        """Returns the most relevant chunks for a given query."""
        distances, indices = self.retriever.kneighbors([query_embedding])
        
        # Fetching the actual text strings using the retrieved indices
        results = [self.chunks[idx] for idx in indices[0]]
        return results
 
# Instantiating and fitting our custom retriever
vector_db = SklearnRetriever(n_neighbors=3)
vector_db.fit(all_chunks, chunk_embeddings)

# generator

# Loading a lightweight causal LLM built for the "text-generation" pipeline
llm = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct", device_map="auto", clean_up_tokenization_spaces=False)
 
def generate_rag_answer(query, retrieved_context):
    """Combines the prompt, context, and query for the LLM using chat formatting."""
    context_str = " --- ".join(retrieved_context)
    
    messages = [
        {
            "role": "system", 
            "content": (
                "You are a helpful assistant. Answer the user's question using ONLY the provided context. "
                "If the answer is not in the context, say 'I cannot find the answer in the provided documents.'"
                ),
        },
        {
            "role": "user", 
            "content": f"Context:\n{context_str}\n\nQuestion: {query}"
        }
    ]
    
    # Apply the model's chat template
    prompt = llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Generating the response
    response = llm(
        prompt, 
        max_new_tokens=150, 
        do_sample=False, 
        return_full_text=True,
    )
    
    # Extracting only the newly generated text past the prompt
    generated_text = response[0]['generated_text']
    answer = generated_text[len(prompt):].strip()
    return answer

# wiring a function to handle the entire RAG process
def ask_rag_pipeline(question):
    print(f"Question: {question}")
    
    # 1. Embedding the query
    query_vector = embedding_model.encode(question)
    
    # 2. Retrieving context via Scikit-Learn
    relevant_chunks = vector_db.retrieve(query_vector)
    
    # 3. Generating Answer
    answer = generate_rag_answer(question, relevant_chunks)
    
    print(f"\nRAG Answer: {answer}")
    print("\n--- Sources used ---")
    for i, chunk in enumerate(relevant_chunks):
        print(f"{i+1}. {chunk[:100]}...") # Truncated for readability
 
# Testing it out
test_question = "Which Asian countries are best known for islands and beaches?"
ask_rag_pipeline(test_question)