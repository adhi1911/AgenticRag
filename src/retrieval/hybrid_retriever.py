import logging
from typing import List, Dict, Any

import numpy as np
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class HybridRetriever:
    
    def __init__(self,
                 dense_search_fn,
                 documents: List[Document],
                 dense_weight: float = 0.6,
                 sparse_weight: float = 0.4):
        
        self.dense_search_fn = dense_search_fn
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.documents = documents
        
        self.metadata_to_doc = {
            f"{doc.metadata.get('source', 'unknown')}_{doc.metadata.get('page', 0)}_{doc.metadata.get('chunk_index', 0)}": doc
            for doc in documents
        }
        
        corpus = [doc.page_content.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(corpus)
        logger.info(f"Hybrid retriever initialized with {len(documents)} documents")
        logger.info(f"Dense weight: {dense_weight}, Sparse weight: {sparse_weight}")

    def _dense_search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        try:
            results = self.dense_search_fn(query_embedding, top_k)
            return results
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []

    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        try:
            tokens = query.lower().split()
            scores = self.bm25.get_scores(tokens)
            top_indices = np.argsort(scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.documents):
                    doc = self.documents[idx]
                    results.append({
                        "chunk_id": f"bm25_{idx}",
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "bm25_score": float(scores[idx])
                    })
            
            return results
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []

    # reciprocal rank fusion
    def _rrf(self,
             dense_results: List[Dict[str, Any]],
             sparse_results: List[Dict[str, Any]],
             top_k: int = 10) -> List[Dict[str, Any]]:
        
        scores = {}
        
        for rank, result in enumerate(dense_results):
            chunk_id = result.get("chunk_id", "")
            score = 1 / (60 + rank)
            scores[chunk_id] = scores.get(chunk_id, 0) + self.dense_weight * score
        
        for rank, result in enumerate(sparse_results):
            chunk_id = result.get("chunk_id", "")
            score = 1 / (60 + rank)
            scores[chunk_id] = scores.get(chunk_id, 0) + self.sparse_weight * score
        
        combined = {}
        for result in dense_results + sparse_results:
            chunk_id = result.get("chunk_id", "")
            if chunk_id not in combined:
                combined[chunk_id] = result
        
        ranked = sorted(combined.items(), key=lambda x: scores.get(x[0], 0), reverse=True)
        
        final_results = []
        for chunk_id, result in ranked[:top_k]:
            result["combined_score"] = scores.get(chunk_id, 0)
            final_results.append(result)
        
        return final_results

    def search(self, query_embedding: np.ndarray, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        dense_results = self._dense_search(query_embedding, top_k=top_k * 5)
        sparse_results = self._bm25_search(query, top_k=top_k * 5)
        
        results = self._rrf(dense_results, sparse_results, top_k=top_k)
        
        return results
