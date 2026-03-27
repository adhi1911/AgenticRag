import logging
from typing import List, Dict, Any

from src.generation.query_transformer import QueryTransformer
from src.generation.groq_generator import GroqGenerator, GenerationConfig, ResponseFormat
from src.retrieval import AdvancedRetriever
from config.settings import settings

logger = logging.getLogger(__name__)


class ResponseGenerator:

    def __init__(self,
                 retriever: AdvancedRetriever,
                 use_query_transformation: bool = True,
                 use_hyde: bool = False,
                 use_multi_query: bool = True,
                 num_multi_queries: int = 3):

        self.retriever = retriever
        self.use_query_transformation = use_query_transformation
        self.use_hyde = use_hyde
        self.use_multi_query = use_multi_query
        self.num_multi_queries = num_multi_queries

        # Initialize components
        self.query_transformer = QueryTransformer() if use_query_transformation else None
        self.groq_generator = GroqGenerator()

        logger.info("✓ ResponseGenerator initialized")
        logger.info(f"  Query transformation: {use_query_transformation}")
        logger.info(f"  HyDE: {use_hyde}, Multi-query: {use_multi_query} ({num_multi_queries} queries)")

    def _retrieve_with_transformation(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve documents using query transformation if enabled"""
        if not self.use_query_transformation:
            # Simple retrieval
            return self.retriever.search(query, top_k=settings.TOP_K_RESULTS)

        # Transform query
        transformed = self.query_transformer.transform(
            query=query,
            use_hyde=self.use_hyde,
            use_multi=self.use_multi_query,
            num_multi_queries=self.num_multi_queries
        )

        all_results = []

        # Retrieve for each transformed query
        for transformed_query in transformed["transformed_queries"]:
            try:
                if self.use_hyde and transformed_query == transformed["transformed_queries"][-1]:
                    # This is the HyDE document, use it directly for retrieval
                    results = self.retriever.search(transformed_query, top_k=settings.TOP_K_RESULTS // 2)
                else:
                    # Regular query
                    results = self.retriever.search(transformed_query, top_k=settings.TOP_K_RESULTS)

                all_results.extend(results)
                logger.debug(f"Retrieved {len(results)} docs for transformed query")

            except Exception as e:
                logger.error(f"Error retrieving for transformed query: {e}")
                continue

        # Remove duplicates based on chunk_id and keep highest scoring
        seen_chunks = {}
        for result in all_results:
            chunk_id = result.get("chunk_id", "")
            score = result.get("rerank_score", result.get("combined_score", 0))

            if chunk_id not in seen_chunks or score > seen_chunks[chunk_id]["score"]:
                seen_chunks[chunk_id] = {"result": result, "score": score}

        # Sort by score and return top results
        unique_results = [item["result"] for item in seen_chunks.values()]
        unique_results.sort(key=lambda x: x.get("rerank_score", x.get("combined_score", 0)), reverse=True)

        return unique_results[:settings.TOP_K_RESULTS]

    def generate_response(self,
                         query: str,
                         response_format: ResponseFormat | str = ResponseFormat.NARRATIVE,
                         temperature: float = settings.GROQ_TEMPERATURE,
                         max_tokens: int = settings.GROQ_MAX_TOKENS,
                         citation_style: str = "inline",
                         include_reasoning: bool = False) -> Dict[str, Any]:
        """Generate complete response with retrieval and generation"""

        try:
            logger.info(f"Generating response for: '{query[:50]}...'")

            # Convert string response_format to enum if needed
            if isinstance(response_format, str):
                try:
                    response_format = ResponseFormat(response_format.lower())
                except ValueError:
                    logger.warning(f"Invalid response_format '{response_format}', using default")
                    response_format = ResponseFormat.NARRATIVE

            # Step 1: Retrieve relevant documents
            documents = self._retrieve_with_transformation(query)
            logger.info(f"✓ Retrieved {len(documents)} relevant documents")

            if not documents:
                return {
                    "answer": "I couldn't find relevant information to answer your question. Please try rephrasing or check if your documents contain the information you're looking for.",
                    "sources": [],
                    "query": query,
                    "num_sources": 0,
                    "error": "No relevant documents found"
                }

            # Step 2: Configure generation
            config = GenerationConfig(
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
                citation_style=citation_style,
                include_reasoning=include_reasoning
            )

            # Step 3: Generate answer
            result = self.groq_generator.generate_with_metadata(query, documents, config)

            logger.info("✓ Response generated successfully")
            return result

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "answer": f"Error generating response: {str(e)}",
                "sources": [],
                "query": query,
                "num_sources": 0,
                "error": str(e)
            }

    def generate_simple(self, query: str) -> str:
        """Simple generation without metadata"""
        result = self.generate_response(query)
        return result.get("answer", "Error generating response")

    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        return {
            "query_transformation_enabled": self.use_query_transformation,
            "hyde_enabled": self.use_hyde,
            "multi_query_enabled": self.use_multi_query,
            "num_multi_queries": self.num_multi_queries,
            "retriever_stats": self.retriever.get_collection_stats(),
            "model": self.groq_generator.model_name
        }
