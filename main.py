from src.loader import load_documents
from src.chunker import chunk_documents
from src.retriever import SklearnRetriever
from src.generator import QwenGenerator

def main():
    # 1. Load data
    docs = load_documents("asia_documents")
    print(f"Loaded {len(docs)} documents.")

    # 2. Chunk
    chunks = chunk_documents(docs, chunk_size=25, overlap=5)
    print(f"Total chunks created: {len(chunks)}")

    # 3. Index & retrieve
    retriever = SklearnRetriever(n_neighbors=3)
    retriever.fit(chunks)

    # 4. Generator
    generator = QwenGenerator()

    # 5. Query
    question = "Which Asian countries are best known for islands and beaches?"
    print(f"\nQuestion: {question}")

    relevant_chunks = retriever.retrieve(question)
    answer = generator.generate(question, relevant_chunks)

    print(f"\nRAG Answer: {answer}\n")
    print("--- Sources used ---")
    for i, chunk in enumerate(relevant_chunks):
        print(f"{i + 1}. {chunk[:100]}...")

if __name__ == "__main__":
    main()