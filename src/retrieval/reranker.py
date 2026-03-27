import logging
from typing import List, Dict, Any

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)


class CrossEncoderReRanker:
    
    def __init__(self,
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 batch_size: int = 32):
        
        self.model = CrossEncoder(model_name)
        self.batch_size = batch_size
        logger.info(f"✓ Cross-encoder reranker loaded: {model_name}")
        logger.info(f"  Max length: {self.model.max_length}")

    def rerank(self,
               query: str,
               documents: List[Dict[str, Any]],
               top_k: int = 5,
               score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        
        if not documents:
            logger.warning("No documents provided for reranking")
            return []
        
        try:
            pairs = [(query, doc.get("content", "")) for doc in documents]
            scores = self.model.predict(pairs, batch_size=self.batch_size)
            
            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)
            
            ranked = sorted(
                documents,
                key=lambda x: x["rerank_score"],
                reverse=True
            )
            
            filtered = [doc for doc in ranked if doc["rerank_score"] >= score_threshold][:top_k]
            
            return filtered
            
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            raise
