from sentence_transformers import CrossEncoder
import torch

class ReRanker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the CrossEncoder model.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(model_name, device=device)

    def rerank(self, query, chunks, top_k=5):
        """
        Re-rank a list of chunks based on the query.
        Returns top_k chunks with their scores.
        
        Output format: List of {'chunk': str, 'score': float, 'original_index': int}
        """
        if not chunks:
            return []

        # Create pairs for the model
        pairs = [[query, chunk] for chunk in chunks]
        
        # Predict scores
        scores = self.model.predict(pairs)
        
        # Combine chunks with scores
        results = []
        for i, score in enumerate(scores):
            results.append({
                'chunk': chunks[i],
                'score': float(score),
                'original_index': i
            })
            
        # Sort by score descending
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:top_k]
