import logging
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
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
                       documents: List,
                       collection_name: str = settings.CHROMA_COLLECTION_NAME
                    ) -> int:
        """
        Inserts documents with embeddings into ChromaDB collection
        Args:
            documents: list of LangChain Document objects or dicts with keys: page_content, metadata
            collection_name: name of the ChromaDB collection to insert into
        Returns:
            Number of vectors inserted
        """
        try:
            if not documents:
                logger.warning("No documents to insert")
                return 0

            logger.info(f"Processing {len(documents)} documents for insertion")

            # Normalize all documents to dict format first
            normalized_docs = []
            for i, doc in enumerate(documents):
                try:
                    if hasattr(doc, 'page_content') and hasattr(doc, 'metadata'):
                        # LangChain Document object - convert to dict
                        logger.debug(f"Document {i}: Converting LangChain Document to dict")
                        normalized_docs.append({
                            'page_content': doc.page_content,
                            'metadata': dict(doc.metadata) if hasattr(doc.metadata, '__iter__') else doc.metadata
                        })
                    elif isinstance(doc, dict) and 'page_content' in doc and 'metadata' in doc:
                        # Already a proper dict
                        logger.debug(f"Document {i}: Already a dict")
                        normalized_docs.append(doc)
                    else:
                        logger.error(f"Document {i}: Invalid format - type={type(doc)}, has page_content={hasattr(doc, 'page_content')}")
                        raise ValueError(f"Document {i}: Invalid format. Expected Document object or dict with page_content and metadata")
                except Exception as e:
                    logger.error(f"Error normalizing document {i}: {str(e)}", exc_info=True)
                    raise

            logger.info(f"Normalized {len(normalized_docs)} documents to dict format")

            # get or create collection
            collection = self.chroma_client.get_or_create_collection(collection_name)

            # extract texts and generate embeddings
            texts = []
            metadatas = []
            ids = []

            for i, doc_dict in enumerate(normalized_docs):
                texts.append(doc_dict['page_content'])
                metadata = doc_dict['metadata'].copy() if isinstance(doc_dict['metadata'], dict) else dict(doc_dict['metadata'])
                # Ensure required metadata fields
                metadata["source"] = metadata.get("source", "unknown")
                metadata["source_type"] = metadata.get("source_type", "unknown")
                metadata["chunk_index"] = metadata.get("chunk_index", 0)
                metadata["total_chunks"] = metadata.get("total_chunks", 1)
                metadatas.append(metadata)
                ids.append(metadata.get("chunk_id", f"chunk_{i}"))

            # Check for duplicate IDs within this batch and fix them
            logger.info(f"Checking for duplicate IDs in batch of {len(ids)} documents...")
            seen_ids = {}
            fixed_count = 0
            for i, id_str in enumerate(ids):
                if id_str in seen_ids:
                    # Found duplicate - append index to make it unique
                    original_id = id_str
                    new_id = f"{id_str}__{i}"
                    logger.warning(f"⚠️ Duplicate ID detected: '{original_id}' at index {i}, renaming to '{new_id}'")
                    ids[i] = new_id
                    fixed_count += 1
                else:
                    seen_ids[id_str] = i

            if fixed_count > 0:
                logger.warning(f"⚠️ Fixed {fixed_count} duplicate IDs in this batch")
            else:
                logger.info(f"✓ All {len(ids)} IDs are unique")

            logger.info(f"Extracted {len(texts)} texts for embedding")
            logger.debug(f"First 3 IDs: {ids[:3]}")
            embeddings = self.generate_embeddings(texts)
            logger.info(f"Generated embeddings for {len(embeddings)} documents")

            # Before inserting, handle any existing IDs to avoid conflicts
            logger.info(f"Preparing for insertion into ChromaDB...")
            ids_to_add = ids.copy()
            
            try:
                # Try to get existing documents and check for conflicts
                collection_data = collection.get(include=[])
                existing_ids = set(collection_data.get('ids', []))
                
                # Find IDs that already exist
                conflicting_ids = [id for id in ids_to_add if id in existing_ids]
                
                if conflicting_ids:
                    logger.info(f"Found {len(conflicting_ids)} IDs already in collection, deleting them first...")
                    collection.delete(ids=conflicting_ids)
                    logger.info(f"✓ Deleted {len(conflicting_ids)} existing IDs")
            except Exception as e:
                logger.debug(f"Could not pre-check for existing IDs (collection may be empty): {str(e)}")

            # Now add the documents
            logger.info(f"Adding {len(ids_to_add)} documents to ChromaDB collection '{collection_name}'...")
            collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids_to_add
            )

            logger.info(f"✓ Successfully inserted {len(normalized_docs)} vectors into '{collection_name}'")
            return len(normalized_docs)

        except Exception as e:
            logger.error(f"Failed to insert vectors: {str(e)}", exc_info=True)
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

    def search_by_embedding(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search by pre-computed embedding vector
        Args:
            query_embedding: numpy array of embedding
            top_k: number of results to return
        Returns:
            List of similar documents with scores
        """
        try:
            collection = self.chroma_client.get_collection(settings.CHROMA_COLLECTION_NAME)
            
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
                        'chunk_id': f"dense_{i}",
                        'content': doc,
                        'metadata': metadata,
                        'score': 1 - distance,  # distance -> similarity score
                        'rank': i + 1
                    })
            
            logger.info(f"Found {len(formatted_results)} similar documents via embedding search")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Embedding search failed: {str(e)}")
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