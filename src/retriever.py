import numpy as np
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer

class SklearnRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", n_neighbors: int = 3):
        self.embedding_model = SentenceTransformer(model_name)
        self.retriever = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
        self.chunks = []

    def fit(self, chunks: list[str]):
        """Generates embeddings and builds the Scikit-Learn nearest neighbor index."""
        self.chunks = chunks
        print("Generating embeddings...")
        embeddings = self.embedding_model.encode(chunks, show_progress_bar=True)
        self.retriever.fit(embeddings)
        print("Vector index built successfully!")

    def retrieve(self, query: str) -> list[str]:
        """Encodes query and returns top matching text chunks."""
        query_vector = self.embedding_model.encode(query)
        _, indices = self.retriever.kneighbors([query_vector])
        return [self.chunks[idx] for idx in indices[0]]