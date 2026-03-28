

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from datetime import datetime
from dataclasses import dataclass
import hashlib
import re

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass 
class ChunkMetadata:
    """Metadata for each chunk"""
    chunk_id: str
    source_file: str
    source_type: str  
    chunk_index: int
    total_chunks: int
    timestamp: str
    file_hash: str
    page_number: Optional[int] = None
    section: Optional[str] = None


class IngestionProcessor: 

    def __init__(self, chunk_size: int= 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # LangChain text splitter (intelligent chunking)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        logger.info(f"IngestionProcessor initialized: chunk_size={chunk_size}, overlap={chunk_overlap}")
    

    
    @staticmethod
    def _generate_file_hash(file_path: str) -> str:
        """Generate SHA256 hash of file for tracking duplicates"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()[:16]    

    @staticmethod
    def _generate_chunk_id(source_file: str, chunk_index: int, file_hash: str) -> str:
        """Generate unique chunk ID"""
        file_name = Path(source_file).stem
        return f"{file_name}_{file_hash}_chunk_{chunk_index}"
    

    ### processing 
    # pdfs
    def parse_pdf(self, file_path: str) -> Dict[str, any]:
        """
            Processing pdf files.
            Args: 
                file_path: path to the pdf file
            Returns:
                Dict with keys: source_file, source_type, full_text, page_data, num_pages
        """
        logger.info(f"Parsing PDF: {file_path}")
        try: 
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"PDF not found: {file_path}")
            
            loader = PyPDFLoader(str(file_path))
            documents = loader.load()
            
            if not documents:
                logger.warning(f"PDF {file_path.name} returned no documents")
                raise ValueError("No documents extracted from PDF")
            
            logger.info(f"✓ Loaded {len(documents)} pages from PDF")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {str(e)}")
            raise
    
    # text file
    
    def load_text_file(self, file_path: str) -> List[Document]:
        """
        Load text file using LangChain's TextLoader
        
        Args:
            file_path: Path to text file
            
        Returns:
            List of LangChain Document objects
        """
        logger.info(f"Loading text file: {file_path}")
        
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
            
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents = loader.load()
            
            if not documents:
                logger.warning(f"Text file {file_path.name} is empty")
                raise ValueError("File contains no text")
            
            logger.info(f"✓ Loaded {len(documents)} document(s) from text file")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading text file {file_path}: {str(e)}")
            raise
    
    # log files (custom processing)
    
    def load_log_file(self, file_path: str, log_type: str = "auto") -> List[Document]:
        """
        Load and parse log files with structured extraction
        
        Args:
            file_path: Path to log file
            log_type: Type of log - "error", "debug", "application", or "auto" (detects from filename)
            
        Returns:
            List of LangChain Document objects with parsed log entries
        """
        logger.info(f"Loading log file: {file_path}")
        
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Log file not found: {file_path}")
            
            # auto-dtect log type from file name.
            detected_type = log_type
            if log_type == "auto":
                filename_lower = file_path.name.lower()
                if "error" in filename_lower:
                    detected_type = "error"
                elif "debug" in filename_lower:
                    detected_type = "debug"
                else:
                    detected_type = "application"
            
            with open(file_path, "r", encoding="utf-8") as f:
                log_content = f.read()
            
            if not log_content.strip():
                logger.warning(f"Log file {file_path.name} is empty")
                raise ValueError("Log file contains no data")
            
            # document object 
            doc = Document(
                page_content=log_content,
                metadata={
                    "source": str(file_path),
                    "source_type": "log",
                    "log_type": detected_type,
                    "timestamp": datetime.now().isoformat(),
                    "original_filename": file_path.name,
                }
            )
            
            logger.info(f"✓ Loaded {len(log_content)} chars from {detected_type} log file")
            return [doc]
            
        except Exception as e:
            logger.error(f"Error loading log file {file_path}: {str(e)}")
            raise
    
    # transcript files (audio/video)
    
    def load_transcript(
        self, 
        file_path: str, 
        transcript_type: str = "auto"
    ) -> List[Document]:
        """
        Load audio or video transcript files
        
        Args:
            file_path: Path to transcript file (.txt)
            transcript_type: Type of transcript - "audio", "video", or "auto" (detects from filename)
            
        Returns:
            List of LangChain Document objects
        """
        logger.info(f"Loading transcript: {file_path}")
        
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"Transcript not found: {file_path}")
            
            # auto detection if not specified
            detected_type = transcript_type
            if transcript_type == "auto":
                filename_lower = file_path.name.lower()
                if "audio" in filename_lower:
                    detected_type = "audio"
                elif "video" in filename_lower:
                    detected_type = "video"
                else:
                    detected_type = "transcript"
            
            with open(file_path, "r", encoding="utf-8") as f:
                transcript_text = f.read()
            
            if not transcript_text.strip():
                logger.warning(f"Transcript {file_path.name} is empty")
                raise ValueError("Transcript contains no text")
            
            doc = Document(
                page_content=transcript_text,
                metadata={
                    "source": str(file_path),
                    "source_type": detected_type,
                    "transcript_type": detected_type,
                    "timestamp": datetime.now().isoformat(),
                    "original_filename": file_path.name,
                }
            )
            
            logger.info(f"✓ Loaded {len(transcript_text)} chars from {detected_type} transcript")
            return [doc]
            
        except Exception as e:
            logger.error(f"Error loading transcript {file_path}: {str(e)}")
            raise
    
    # chunking and metadata
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk long documents and preserve metadata
        
        Args:
            documents: List of Document objects from loaders
            
        Returns:
            List of Document objects with chunks
        """
        logger.info(f"Chunking {len(documents)} documents")
        
        try:
            chunked_docs = []
            
            for doc_idx, doc in enumerate(documents):
                # skipping short docs.
                # if len(doc.page_content) < self.chunk_size:
                #     chunked_docs.append(doc)
                #     continue
                
                
                chunks = self.text_splitter.split_text(doc.page_content)
                
                file_hash = self._generate_file_hash(doc.metadata.get("source", ""))
                
                for chunk_idx, chunk_text in enumerate(chunks):
                    chunk_id = self._generate_chunk_id(
                        doc.metadata.get("source", "unknown"),
                        chunk_idx,
                        file_hash
                    )
                    
                    # preserving original metadata and adding chunk-specific metadata
                    chunk_metadata = doc.metadata.copy()
                    chunk_metadata.update({
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                        "file_hash": file_hash,
                    })
                    
                    chunked_doc = Document(
                        page_content=chunk_text,
                        metadata=chunk_metadata
                    )
                    chunked_docs.append(chunked_doc)
            
            logger.info(f"Created {len(chunked_docs)} chunked documents")
            return chunked_docs
            
        except Exception as e:
            logger.error(f"Error chunking documents: {str(e)}")
            raise
    
    # main orchestrator
    
    def process_document(
        self, 
        file_path: str, 
        transcript_type: str = "auto",
        log_type: str = "auto"
    ) -> List[Document]:
        """
        Load and process a single document end-to-end
        
        Args:
            file_path: Path to document (PDF, TXT, transcript, or log)
            transcript_type: For transcripts - "audio", "video", or "auto"
            log_type: For logs - "error", "debug", "application", or "auto"
            
        Returns:
            List of LangChain Document objects (chunked and processed)
        """
        file_path = Path(file_path)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing document: {file_path.name}")
        logger.info(f"{'='*60}")
        
        try:
            documents = []
            
            # loading document based on file type
            if file_path.suffix.lower() == ".pdf":
                documents = self.parse_pdf(str(file_path))
                for doc in documents:
                    doc.metadata["source_type"] = "pdf"
                    
            elif file_path.suffix.lower() == ".log":
                documents = self.load_log_file(str(file_path), log_type)
                    
            elif file_path.suffix.lower() in [".txt"]:
                filename_lower = file_path.name.lower()
                
                if any(keyword in filename_lower for keyword in ["transcript", "audio", "video"]):
                    documents = self.load_transcript(str(file_path), transcript_type)
                elif any(keyword in filename_lower for keyword in ["log", "error", "debug"]):
                    documents = self.load_log_file(str(file_path), log_type)
                else:
                    documents = self.load_text_file(str(file_path))
                    for doc in documents:
                        doc.metadata["source_type"] = "txt"
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            # chunking documents
            chunked_documents = self.chunk_documents(documents)
            
            logger.info(f"Document processing complete: {len(chunked_documents)} documents created\n")
            return chunked_documents
            
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {str(e)}\n")
            raise

    def process_file(self, file_path: str) -> Dict[str, any]:
        """
        Wrapper method to process a file and return orchestrator-friendly dict format
        
        Args:
            file_path: Path to file to process
            
        Returns:
            Dict with keys: chunks, metadata, success, error
        """
        try:
            chunks = self.process_document(file_path)
            return {
                "success": True,
                "chunks": chunks,
                "metadata": {
                    "source": file_path,
                    "source_type": Path(file_path).suffix.lower(),
                    "num_chunks": len(chunks)
                }
            }
        except Exception as e:
            logger.error(f"Error in process_file for {file_path}: {str(e)}")
            return {
                "success": False,
                "chunks": [],
                "error": str(e),
                "metadata": {"source": file_path}
            }
    
    # batch processing for directories
    
    def process_documents_batch(self, file_paths: List[str]) -> Tuple[List[Document], dict]:
        """
        Process multiple documents using DirectoryLoader for efficiency
        
        Args:
            file_paths: List of file paths to process
            
        Returns:
            Tuple of (all_documents, processing_summary)
        """
        logger.info(f"\nProcessing batch of {len(file_paths)} documents...")
        
        all_documents = []
        successful = 0
        failed = 0
        errors = []
        
        for file_path in tqdm(file_paths, desc="Processing documents"):
            try:
                documents = self.process_document(file_path)
                all_documents.extend(documents)
                successful += 1
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                failed += 1
                errors.append({"file": str(file_path), "error": str(e)})
        
        summary = {
            "total_files": len(file_paths),
            "successful": successful,
            "failed": failed,
            "total_documents": len(all_documents),
            "errors": errors,
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Batch processing summary:")
        logger.info(f"  Files processed: {successful}/{len(file_paths)}")
        logger.info(f"  Total documents created: {len(all_documents)}")
        logger.info(f"  Failed files: {failed}")
        logger.info(f"{'='*60}\n")
        
        return all_documents, summary
    
    def process_directory(
        self, 
        directory_path: str,
        file_pattern: str = "**/*",
        exclude_patterns: Optional[List[str]] = None
    ) -> Tuple[List[Document], dict]:
        """
        Process all documents in a directory using LangChain's DirectoryLoader
        
        Args:
            directory_path: Path to directory containing documents
            file_pattern: Glob pattern for files to process (default: **/* for all files)
            exclude_patterns: List of patterns to exclude
            
        Returns:
            Tuple of (all_documents, processing_summary)
        """
        logger.info(f"\nProcessing directory: {directory_path}")
        
        try:
            dir_path = Path(directory_path)
            if not dir_path.is_dir():
                raise ValueError(f"{directory_path} is not a valid directory")
            
            # collect matching files.
            file_paths = []
            for pattern in ["**/*.pdf", "**/*.txt"]:
                file_paths.extend([str(f) for f in dir_path.glob(pattern)])
            
            if not file_paths:
                logger.warning(f"No supported files found in {directory_path}")
                return [], {"total_files": 0, "successful": 0, "failed": 0, "total_documents": 0}
            
            logger.info(f"Found {len(file_paths)} files to process")
            return self.process_documents_batch(file_paths)
            
        except Exception as e:
            logger.error(f"Error processing directory {directory_path}: {str(e)}")
            raise




def save_documents_to_json(documents: List[Document], output_file: str) -> None:
    """
    Save LangChain Documents to JSON for serialization
    
    Args:
        documents: List of Document objects
        output_file: Output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # convert documents to serializable format
    docs_data = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in documents
    ]
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(docs_data, f, indent=2, default=str)
    
    logger.info(f"✓ Saved {len(documents)} documents to {output_file}")


def load_documents_from_json(input_file: str) -> List[Document]:
    """
    Load Documents from JSON file
    
    Args:
        input_file: Path to JSON file
        
    Returns:
        List of Document objects
    """
    with open(input_file, "r", encoding="utf-8") as f:
        docs_data = json.load(f)
    
    documents = [
        Document(page_content=doc["page_content"], metadata=doc["metadata"])
        for doc in docs_data
    ]
    
    logger.info(f"✓ Loaded {len(documents)} documents from {input_file}")
    return documents
