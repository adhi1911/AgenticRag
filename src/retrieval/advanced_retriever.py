import logging
from typing import List, Dict, Any

import numpy as np
from langchain_core.documents import Document

from src.embeddings.embedding_manager import EmbeddingManager
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReRanker
from config.settings import settings

logger = logging.getLogger(__name__)


class AdvancedRetriever:
    
    def __init__(self,
                 embedding_manager: EmbeddingManager,
                 documents: List[Document],
                 dense_weight: float = 0.6,
                 sparse_weight: float = 0.4,
                 rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 enable_reranking: bool = True):
        
        self.embedding_manager = embedding_manager
        self.documents = documents
        self.enable_reranking = enable_reranking
        
        logger.info("Initializing AdvancedRetriever...")
        
        self.hybrid_retriever = HybridRetriever(
            dense_search_fn=self._dense_search_milvus,
            documents=documents,
            dense_weight=dense_weight,
            sparse_weight=sparse_weight
        )
        
        if enable_reranking:
            self.reranker = CrossEncoderReRanker(model_name=rerank_model)
        else:
            self.reranker = None
        
        logger.info("✓ AdvancedRetriever initialized successfully")

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        try:
            logger.info(f"Searching: '{query[:50]}...' (top_k={top_k})")
            
            query_embedding = self.embedding_manager.model.encode(query, convert_to_numpy=True)
            
            results = self.hybrid_retriever.search(
                query_embedding=query_embedding,
                query=query,
                top_k=top_k * 2
            )
            
            if self.enable_reranking and results:
                logger.info(f"Reranking {len(results)} results...")
                results = self.reranker.rerank(
                    query=query,
                    documents=results,
                    top_k=top_k
                )
            else:
                results = results[:top_k]
            
            logger.info(f"✓ Retrieved {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise

    def search_with_metadata(self, query: str, top_k: int = 5, filters: Dict = None) -> Dict[str, Any]:
        results = self.search(query, top_k=top_k)
        
        return {
            "query": query,
            "top_k": top_k,
            "results": results,
            "count": len(results),
            "filters_applied": filters
        }

    def get_collection_stats(self) -> Dict[str, Any]:
        return {
            "total_documents": len(self.documents),
            "retriever_type": "hybrid_with_reranking" if self.enable_reranking else "hybrid",
            "dense_weight": self.hybrid_retriever.dense_weight,
            "sparse_weight": self.hybrid_retriever.sparse_weight,
            "embedding_model": self.embedding_manager.embedding_model_name
        }

    def _dense_search_milvus(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        try:
            results = self.embedding_manager.search_by_embedding(query_embedding, top_k)
            return results
        except Exception as e:
            logger.error(f"Milvus dense search failed: {e}")
            return []
