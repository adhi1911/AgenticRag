import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from enum import Enum
from dataclasses import dataclass
from langchain_core.documents import Document

from config.settings import settings
from src.ingestion.processor import IngestionProcessor, load_documents_from_json, save_documents_to_json
from src.embeddings.embedding_manager import EmbeddingManager
from src.retrieval.advanced_retriever import AdvancedRetriever
from src.generation.response_generator import ResponseGenerator

logger = logging.getLogger(__name__)


class ActionType(str, Enum):
    """Types of actions the orchestrator can take"""
    INGEST = "ingest"
    RETRIEVE = "retrieve"
    GENERATE = "generate"
    UPLOAD = "upload"
    STATUS = "status"
    HELP = "help"


class QueryIntent(str, Enum):
    """Detected intent of user query"""
    INGEST_DOCUMENTS = "ingest_documents"
    ASK_QUESTION = "ask_question"
    UPLOAD_FILES = "upload_files"
    CHECK_STATUS = "check_status"
    GET_HELP = "get_help"
    UNKNOWN = "unknown"


@dataclass
class OrchestratorConfig:
    """Configuration for orchestrator behavior"""
    enable_auto_ingest: bool = True
    enable_file_upload: bool = True
    max_upload_size_mb: int = settings.MAX_PDF_SIZE_MB
    confidence_threshold: float = 0.7
    enable_reasoning: bool = True


class AgenticOrchestrator:
    """
    Intelligent orchestrator that routes user requests to appropriate modules.

    This class analyzes user input to determine intent and route to:
    - Document ingestion for new content
    - RAG pipeline for questions
    - File upload handling
    - Status reporting
    """

    def __init__(self, config: OrchestratorConfig = None):
        self.config = config or OrchestratorConfig()

        # initialzing components
        self.document_processor = IngestionProcessor()
        self.embedding_manager = EmbeddingManager()

    
        self.documents = self._load_existing_documents()

        self.retriever = AdvancedRetriever(self.embedding_manager, self.documents)
        self.generator = ResponseGenerator(self.retriever)

        #keyword lists for intent detection
        self.ingest_keywords = [
            'ingest', 'process', 'load', 'add', 'import', 'upload',
            'pdf', 'document', 'file', 'folder', 'directory'
        ]

        self.status_keywords = [
            'status', 'stats', 'info', 'summary', 'overview', 'health'
        ]

        self.upload_keywords = [
            'upload', 'attach', 'send file', 'submit', 'provide file'
        ]

        self.help_keywords = [
            'help', 'commands', 'what can you do', 'usage', 'guide'
        ]

        logger.info("AgenticOrchestrator initialized")

    def _load_existing_documents(self) -> List[Document]:
        """Load existing processed documents from storage."""
        try:
            processed_file = settings.PROCESSED_DIR / "processed_documents.json"
            if processed_file.exists():
                documents = load_documents_from_json(str(processed_file))
                logger.info(f"✓ Loaded {len(documents)} existing documents")
                return documents
            else:
                logger.info("No existing documents found, starting with empty knowledge base")
                return []
        except Exception as e:
            logger.error(f"Error loading existing documents: {e}")
            return []

    def analyze_intent(self, user_input: str, files: List[Path] = None) -> QueryIntent:
        """
        Analyze user input to determine intent.

        Args:
            user_input: The user's text input
            files: Any attached files

        Returns:
            Detected QueryIntent
        """
        input_lower = user_input.lower().strip()

        # Check for file uploads
        if files and len(files) > 0:
            return QueryIntent.UPLOAD_FILES

        # Check for upload commands (even without files)
        if any(keyword in input_lower for keyword in self.upload_keywords):
            return QueryIntent.UPLOAD_FILES

        # Check for status requests
        if any(keyword in input_lower for keyword in self.status_keywords):
            return QueryIntent.CHECK_STATUS

        # Check for help requests
        if any(keyword in input_lower for keyword in self.help_keywords):
            return QueryIntent.GET_HELP

        # Check for ingestion commands
        if any(keyword in input_lower for keyword in self.ingest_keywords):
            return QueryIntent.INGEST_DOCUMENTS

        # Check for question patterns
        question_words = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'can you', 'explain']
        if any(word in input_lower for word in question_words) or input_lower.endswith('?'):
            return QueryIntent.ASK_QUESTION

        # Default to question if we can't determine
        return QueryIntent.ASK_QUESTION

    def route_request(self, user_input: str, files: List[Path] = None, **kwargs) -> Dict[str, Any]:
        """
        Route user request to appropriate handler based on intent.

        Args:
            user_input: User's text input
            files: List of file paths to process
            **kwargs: Additional parameters

        Returns:
            Response dictionary with action taken and results
        """
        intent = self.analyze_intent(user_input, files)

        logger.info(f"Detected intent: {intent.value}")

        try:
            if intent == QueryIntent.UPLOAD_FILES:
                return self._handle_file_upload(files or [])

            elif intent == QueryIntent.INGEST_DOCUMENTS:
                return self._handle_ingest_request(user_input)

            elif intent == QueryIntent.ASK_QUESTION:
                return self._handle_question(user_input, **kwargs)

            elif intent == QueryIntent.CHECK_STATUS:
                return self._handle_status_request()

            elif intent == QueryIntent.GET_HELP:
                return self._handle_help_request()

            else:
                return self._handle_unknown_intent(user_input)

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                "action": ActionType.GENERATE.value,
                "intent": intent.value,
                "success": False,
                "error": str(e),
                "response": f"I encountered an error: {str(e)}. Please try again."
            }

    def _handle_file_upload(self, files: List[Path]) -> Dict[str, Any]:
        """Handle file upload requests."""
        if not files:
            return {
                "action": ActionType.UPLOAD.value,
                "success": False,
                "error": "No files provided",
                "response": "Please provide files to upload."
            }

        processed_files = []
        errors = []

        for file_path in files:
            try:
                # Validate file size
                if file_path.stat().st_size > self.config.max_upload_size_mb * 1024 * 1024:
                    errors.append(f"File {file_path.name} exceeds size limit")
                    continue

                # Process the file
                result = self.document_processor.process_file(str(file_path))
                
                # Check if processing was successful
                if result.get("success", False):
                    processed_files.append({
                        "filename": file_path.name,
                        "path": str(file_path),
                        "chunks": result.get("chunks", []),
                        "chunks_created": len(result.get("chunks", [])),
                        "metadata": result.get("metadata", {})
                    })
                else:
                    errors.append(f"{file_path.name}: {result.get('error', 'Unknown error')}")
                    logger.error(f"Failed to process {file_path.name}: {result.get('error', 'Unknown error')}")

            except Exception as e:
                errors.append(f"Error processing {file_path.name}: {str(e)}")
                logger.error(f"Exception in file upload: {str(e)}")

        # Auto-ingest if enabled
        if self.config.enable_auto_ingest and processed_files:
            try:
                self._ingest_processed_documents(processed_files)
            except Exception as e:
                logger.error(f"Auto-ingestion failed: {e}")

        return {
            "action": ActionType.UPLOAD.value,
            "success": len(processed_files) > 0,
            "processed_files": processed_files,
            "errors": errors,
            "auto_ingested": self.config.enable_auto_ingest,
            "response": f"Processed {len(processed_files)} files successfully. "
                       f"{'Auto-ingested into knowledge base.' if self.config.enable_auto_ingest else ''}"
        }

    def _handle_ingest_request(self, user_input: str) -> Dict[str, Any]:
        """Handle document ingestion requests."""
        # Extract paths from user input
        paths = self._extract_paths_from_input(user_input)

        if not paths:
            return {
                "action": ActionType.INGEST.value,
                "success": False,
                "error": "No valid paths found",
                "response": "Please specify file or folder paths to ingest. Example: 'ingest /path/to/documents'"
            }

        all_processed = []
        errors = []

        for path_str in paths:
            path = Path(path_str)
            try:
                if path.is_file():
                    result = self.document_processor.process_file(path_str)
                    all_processed.append(result)
                elif path.is_dir():
                    results = self.document_processor.process_directory(path_str)
                    # process_directory returns tuple (documents, summary)
                    if isinstance(results, tuple) and len(results) == 2:
                        documents, summary = results
                        all_processed.append({
                            "chunks": documents,
                            "metadata": summary,
                            "success": True
                        })
                    else:
                        all_processed.append(results)
                else:
                    errors.append(f"Path not found: {path_str}")

            except Exception as e:
                errors.append(f"Error processing {path_str}: {str(e)}")

        # Ingest into vector database
        if all_processed:
            try:
                self._ingest_processed_documents(all_processed)
            except Exception as e:
                errors.append(f"Ingest error: {str(e)}")

        return {
            "action": ActionType.INGEST.value,
            "success": len(all_processed) > 0,
            "processed_documents": len(all_processed),
            "errors": errors,
            "response": f"Ingested {len(all_processed)} documents into knowledge base."
        }

    def _handle_question(self, user_input: str, **kwargs) -> Dict[str, Any]:
        """Handle question-answering requests using RAG pipeline."""
        try:
            result = self.generator.generate_response(user_input, **kwargs)

            return {
                "action": ActionType.GENERATE.value,
                "success": True,
                "query": user_input,
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []),
                "metadata": {
                    "num_sources": result.get("num_sources", 0),
                    "response_format": result.get("response_format", "unknown"),
                    "model": result.get("model", "unknown")
                },
                "response": result.get("answer", "No answer generated")
            }

        except Exception as e:
            return {
                "action": ActionType.GENERATE.value,
                "success": False,
                "error": str(e),
                "response": f"Error generating answer: {str(e)}"
            }

    def _handle_status_request(self) -> Dict[str, Any]:
        """Handle status and statistics requests."""
        try:
            # Get basic stats
            stats = {
                "total_documents": len(self.documents),
                "embedding_model": self.embedding_manager.model_name if hasattr(self.embedding_manager, 'model_name') else 'sentence-transformers/all-MiniLM-L6-v2',
                "vector_db": "ChromaDB",
                "generation_model": self.generator.groq_generator.model_name,
                "modules_status": {
                    "ingestion": "✓ Available",
                    "embedding": "✓ Available",
                    "retrieval": "✓ Available",
                    "generation": "✓ Available"
                }
            }

            response_text = f"""
                    System Status:
                    • Documents in knowledge base: {stats['total_documents']}
                    • Embedding model: {stats['embedding_model']}
                    • Vector database: {stats['vector_db']}
                    • Generation model: {stats['generation_model']}

                    All modules are operational and ready to use.
                    """

            return {
                "action": ActionType.STATUS.value,
                "success": True,
                "stats": stats,
                "response": response_text.strip()
            }

        except Exception as e:
            return {
                "action": ActionType.STATUS.value,
                "success": False,
                "error": str(e),
                "response": f"Error getting status: {str(e)}"
            }

    def _handle_help_request(self) -> Dict[str, Any]:
        """Handle help and usage requests."""
        help_text = """
        Available Commands:

        Document Management:
        • Upload files: Attach files to your message
        • Ingest documents: "ingest /path/to/documents"
        • Process folder: "ingest /path/to/folder"

        Question Answering:
        • Ask anything: "What are the key concepts in AI?" or "How does machine learning work?"
        • The system will automatically retrieve relevant information and generate answers

        System Status:
        • Check status: "status" or "system info"
        • Get statistics: "stats" or "overview"

        Help:
        • Get help: "help" or "commands"

        The system uses advanced RAG (Retrieval-Augmented Generation) with:
        • Hybrid search (dense + sparse retrieval)
        • Cross-encoder reranking
        • Query transformation (multi-query expansion)
        • Flexible response formats with citations
        """

        return {
            "action": ActionType.HELP.value,
            "success": True,
            "response": help_text.strip()
        }

    def _handle_unknown_intent(self, user_input: str) -> Dict[str, Any]:
        """Handle unknown or unclear intents."""
        return {
            "action": ActionType.HELP.value,
            "intent": QueryIntent.UNKNOWN.value,
            "success": True,
            "response": f"I couldn't determine what you want to do with: '{user_input}'\n\n" +
                       "Try asking a question, uploading files, or type 'help' for available commands."
        }

    def _extract_paths_from_input(self, user_input: str) -> List[str]:
        """Extract file/folder paths from user input."""
        # Simple extraction - look for quoted strings or words that look like paths
        import re

        # Find quoted paths
        quoted_paths = re.findall(r'["\']([^"\']+)["\']', user_input)

        # Find unquoted paths (basic heuristic)
        words = user_input.split()
        potential_paths = [word for word in words if '/' in word or '\\' in word]

        return quoted_paths + potential_paths

    def _ingest_processed_documents(self, processed_docs: List[Dict]) -> None:
        """Ingest processed documents into the vector database and save to disk."""
        all_chunks = []
        for doc_result in processed_docs:
            chunks = doc_result.get("chunks", [])
            logger.info(f"Extracting chunks from doc_result: got {len(chunks)} chunks")
            if chunks:
                logger.debug(f"First chunk type: {type(chunks[0])}")
            all_chunks.extend(chunks)

        logger.info(f"Total chunks to ingest: {len(all_chunks)}")
        if all_chunks:
            logger.debug(f"First chunk in all_chunks: type={type(all_chunks[0])}, has page_content={hasattr(all_chunks[0], 'page_content')}")
            
        if all_chunks:
            # Save chunks to processed directory (append to existing documents)
            processed_file = settings.PROCESSED_DIR / "processed_documents.json"
            
            # Load existing documents
            existing_docs = []
            if processed_file.exists():
                try:
                    existing_docs = load_documents_from_json(str(processed_file))
                    logger.info(f"✓ Loaded {len(existing_docs)} existing documents")
                except Exception as e:
                    logger.warning(f"Could not load existing documents, starting fresh: {str(e)}")
                    existing_docs = []
            
            # Combine existing and new documents
            combined_docs = existing_docs + all_chunks
            logger.info(f"Combining {len(existing_docs)} existing + {len(all_chunks)} new = {len(combined_docs)} total documents")
            
            # Save combined documents
            save_documents_to_json(combined_docs, str(processed_file))
            logger.info(f"✓ Saved {len(combined_docs)} total chunks to {processed_file}")
            
            # Ingest new chunks into vector database
            self.embedding_manager.insert_vectors(all_chunks)
            logger.info(f"✓ Ingested {len(all_chunks)} new chunks into vector database")

    def get_capabilities(self) -> Dict[str, Any]:
        """Get system capabilities and supported operations."""
        return {
            "supported_actions": [action.value for action in ActionType],
            "supported_intents": [intent.value for intent in QueryIntent],
            "file_types": [".pdf", ".txt", ".md", ".json"],
            "max_file_size_mb": self.config.max_upload_size_mb,
            "features": [
                "Document ingestion and processing",
                "Hybrid retrieval with reranking",
                "Query transformation (multi-query)",
                "Flexible answer generation",
                "Citation support",
                "File upload handling"
            ]
        }