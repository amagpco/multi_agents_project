import cohere
import numpy as np
import os

class EmbeddingRanker:
    def __init__(self):
        self.client = cohere.Client()

    def get_embeddings(self, texts):
        try:
            response = self.client.embed(texts=texts)
            return np.array(response.embeddings)
        except Exception as e:
            return np.array([])

    def rank_chunks(self, query_embedding, chunk_embeddings):
        try:
            scores = np.dot(chunk_embeddings, query_embedding.T)
            ranked_indices = np.argsort(scores)[::-1]
            return ranked_indices
        except Exception as e:
            return []

class ChunkRanker:
    def __init__(self):
        self.ranker = EmbeddingRanker()

    def rank_chunks(self, query, all_chunks):
        query_embedding = self.ranker.get_embeddings([query])[0]
        chunk_embeddings = self.ranker.get_embeddings(all_chunks)
        ranked_indices = self.ranker.rank_chunks(query_embedding, chunk_embeddings)
        top_chunks = [all_chunks[i] for i in ranked_indices[:5]]
        return top_chunks