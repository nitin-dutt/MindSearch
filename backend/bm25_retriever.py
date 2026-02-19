import pickle
import os
from rank_bm25 import BM25Okapi
from typing import List
import nltk
from nltk.tokenize import word_tokenize

# Ensure extensions are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.chunks = []

    def build_index(self, chunks: List[str]):
        """
        Tokenize chunks and build BM25 index.
        """
        self.chunks = chunks
        tokenized_corpus = [word_tokenize(doc.lower()) for doc in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def save_index(self, path: str):
        """
        Save the BM25 object and chunks to a pickle file.
        """
        if self.bm25 is None:
            raise ValueError("Index not built yet.")
        
        with open(path, 'wb') as f:
            pickle.dump({'bm25': self.bm25, 'chunks': self.chunks}, f)

    def load_index(self, path: str):
        """
        Load the BM25 object and chunks from a pickle file.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Index file {path} not found.")
            
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.bm25 = data['bm25']
            self.chunks = data['chunks']

    def search(self, query: str, k: int = 5):
        """
        Return top-k chunks and their scores.
        Returns: List of (chunk, score, index)
        """
        if self.bm25 is None:
            return []

        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_n = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        results = []
        for i in top_n:
            results.append({
                'chunk': self.chunks[i],
                'score': scores[i],
                'index': i
            })
            
        return results
