import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from config.settings import settings

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)


class EmbeddingManager:
    def __init__(self,
                 embedding_model: str = settings.EMBEDDING_MODEL,
                 chroma_db_path: str = settings.CHROMA_DB_PATH,
                 collection_name: str = settings.CHROMA_COLLECTION_NAME):

        self.embedding_model_name = embedding_model
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name
        self.model = None
        self.chroma_client = None
        self.collection = None

        logger.info(f"Initializing EmbeddingManager with model: {embedding_model}")
        self._load_embedding_model()
        self._connect_chroma()
        logger.info("EmbeddingManager initialized successfully")

    def _load_embedding_model(self) -> None:
        try:
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise

    def _connect_chroma(self) -> None:
        try:
            logger.info(f"Connecting to ChromaDB at: {self.chroma_db_path}")
            self.chroma_client = chromadb.PersistentClient(
                path=self.chroma_db_path,
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            logger.info("✓ Connected to ChromaDB")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {str(e)}")
            raise

    def create_collection(self,
                          collection_name: str = settings.CHROMA_COLLECTION_NAME,
                          drop_existing: bool = False
                    ):
        """
        Create or get ChromaDB collection
        Args:
            collection_name: name of the collection to create or get
            drop_existing: if True, drops existing collection with same name before creating new one
        Returns:
            ChromaDB Collection object
        """
        try:
            if drop_existing:
                try:
                    self.chroma_client.delete_collection(collection_name)
                    logger.info(f"Dropped existing collection: {collection_name}")
                except:
                    pass  # collectoion might not exist, ignore

            # get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # use cosine similarity
            )
            logger.info(f"✓ ChromaDB collection '{collection_name}' ready")
            return self.collection

        except Exception as e:
            logger.error(f"Failed to create/get collection '{collection_name}': {str(e)}")
            raise

    def insert_vectors(self,
                       documents: List[Dict],
                       collection_name: str = settings.CHROMA_COLLECTION_NAME
                    ) -> int:
        """
        Inserts documents with embeddings into ChromaDB collection
        Args:
            documents: list of dicts with keys: chunk_id, source, source_type, chunk_index, total_chunks, page_content, metadata_json
            collection_name: name of the ChromaDB collection to insert into
        Returns:
            Number of vectors inserted
        """
        try:
            if not documents:
                logger.warning("No documents to insert")
                return 0

            # get or create collection
            collection = self.chroma_client.get_or_create_collection(collection_name)

            # extract texts and generate embeddings
            texts = [doc["page_content"] for doc in documents]
            embeddings = self.generate_embeddings(texts)

            # prepare data for ChromaDB
            ids = [doc["metadata"].get("chunk_id", f"chunk_{i}") for i, doc in enumerate(documents)]
            metadatas = []
            for doc in documents:
                metadata = doc["metadata"].copy()
                metadata["source"] = doc["metadata"].get("source", "unknown")
                metadata["source_type"] = doc["metadata"].get("source_type", "unknown")
                metadata["chunk_index"] = doc["metadata"].get("chunk_index", 0)
                metadata["total_chunks"] = doc["metadata"].get("total_chunks", 1)
                metadatas.append(metadata)

            # insert into ChromaDB
            collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Inserted {len(documents)} vectors into '{collection_name}'")
            return len(documents)

        except Exception as e:
            logger.error(f"Failed to insert vectors: {str(e)}")
            raise

    def search_similar(self,
                      query: str,
                      top_k: int = settings.TOP_K_RESULTS,
                      collection_name: str = settings.CHROMA_COLLECTION_NAME
                     ) -> List[Dict]:
        """
        Search for similar documents using vector similarity
        Args:
            query: search query string
            top_k: number of results to return
            collection_name: name of the collection to search in
        Returns:
            List of similar documents with scores
        """
        try:
            # generate embedding for query
            query_embedding = self.generate_embeddings([query])[0]

            # getting the collection
            collection = self.chroma_client.get_collection(collection_name)

            
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=['documents', 'metadatas', 'distances']
            )

            
            formatted_results = []
            if results['documents'] and len(results['documents'][0]) > 0:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    formatted_results.append({
                        'content': doc,
                        'metadata': metadata,
                        'score': 1 - distance,  # distance -> similarity score 
                        'rank': i + 1
                    })

            logger.info(f"Found {len(formatted_results)} similar documents for query: '{query}'")
            return formatted_results

        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise

    def get_collection_stats(self, collection_name: str = settings.CHROMA_COLLECTION_NAME) -> Dict:
        """
        Get statistics about the collection
        """
        try:
            collection = self.chroma_client.get_collection(collection_name)
            count = collection.count()
            return {
                'exists': True,
                'collection_name': collection_name,
                'num_entities': count,
                'embedding_dim': self.model.get_sentence_embedding_dimension(),
                'metric_type': 'cosine'
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {
                'exists': False,
                'error': str(e)
            }

    ### Generating embeddings
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts...")
            embeddings = self.model.encode(
                texts,
                batch_size=settings.BATCH_SIZE,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            logger.info(f"Embeddings generated successfully")

            return embeddings
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise