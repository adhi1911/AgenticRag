"""
Query Evaluator - Assesses retrieval and generation quality.
Used by agents to decide if more iterations or reformulations are needed.
"""

import logging
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class RetrievalQuality:
    """Metrics for assessing retrieval quality"""
    num_results: int
    avg_relevance_score: float
    diversification_score: float  # 0-1, how diverse the results are
    is_sufficient: bool
    confidence: float  # 0-1, confidence in the retrieval


class QueryEvaluator:
    """
    Evaluates the quality of retrieval and generation results.
    Helps determine if the agent needs to ask follow-up questions or reformulate queries.
    """

    def __init__(
        self,
        min_results_threshold: int = 3,  # Increased from 2
        min_relevance_score: float = 0.7,  # Increased from 0.5
        min_confidence_threshold: float = 0.6
    ):
        """
        Initialize evaluator with quality thresholds.
        
        Args:
            min_results_threshold: Minimum number of results needed
            min_relevance_score: Minimum average relevance score (0-1)
            min_confidence_threshold: Minimum confidence for answer adequacy
        """
        self.min_results_threshold = min_results_threshold
        self.min_relevance_score = min_relevance_score
        self.min_confidence_threshold = min_confidence_threshold
        logger.info("QueryEvaluator initialized")

    def evaluate_retrieval(self, sources: List[Dict[str, Any]]) -> RetrievalQuality:
        """
        Evaluate quality of retrieved documents.
        
        Args:
            sources: List of retrieved documents with rerank_score
            
        Returns:
            RetrievalQuality object with assessment
        """
        num_results = len(sources)
        
        # Calculate average relevance score
        if sources:
            scores = [s.get('rerank_score', 0) for s in sources if isinstance(s.get('rerank_score'), (int, float))]
            avg_relevance = sum(scores) / len(scores) if scores else 0
        else:
            avg_relevance = 0
        
        # Diversification: check if sources span different documents
        unique_sources = set(s.get('source', 'unknown') for s in sources)
        diversification = len(unique_sources) / max(num_results, 1)
        
        # Determine if retrieval is sufficient
        is_sufficient = (
            num_results >= self.min_results_threshold and
            avg_relevance >= self.min_relevance_score and
            diversification >= 0.5  # At least 50% unique docs
        )
        
        # Confidence calculation - more realistic scoring
        # Results factor: 3+ results = good, but not perfect
        results_score = min(num_results / 5.0, 1.0) * 0.4  # Max 0.4 for results
        
        # Relevance factor: need high relevance scores
        relevance_score = min(avg_relevance / 0.8, 1.0) * 0.4  # Max 0.4 for relevance
        
        # Diversity factor: prefer diverse sources
        diversity_score = diversification * 0.2  # Max 0.2 for diversity
        
        confidence = results_score + relevance_score + diversity_score
        
        # Cap at realistic maximum - even perfect retrieval isn't 100% confidence
        confidence = min(confidence, 0.85)
        
        logger.debug(
            f"Retrieval eval: {num_results} results, "
            f"avg_relevance={avg_relevance:.2f}, diversity={diversification:.2f}, "
            f"sufficient={is_sufficient}, confidence={confidence:.2f}"
        )
        
        return RetrievalQuality(
            num_results=num_results,
            avg_relevance_score=avg_relevance,
            diversification_score=diversification,
            is_sufficient=is_sufficient,
            confidence=confidence
        )

    def evaluate_answer_quality(
        self,
        question: str,
        answer: str,
        retrieval_quality: RetrievalQuality
    ) -> Tuple[float, str]:
        """
        Evaluate overall answer quality based on retrieval and answer characteristics.
        
        Args:
            question: Original user question
            answer: Generated answer
            retrieval_quality: Result of retrieval evaluation
            
        Returns:
            Tuple of (confidence_score: float, reasoning: str)
        """
        confidence = 0.0
        reasons = []
        
        # Factor 1: Retrieval quality (60% weight)
        retrieval_confidence = retrieval_quality.confidence * 0.6
        confidence += retrieval_confidence
        
        if retrieval_quality.is_sufficient:
            reasons.append("✓ Sufficient relevant documents found")
        else:
            reasons.append("⚠ Limited relevant documents")
        
        # Factor 2: Answer length (need substantial content)
        answer_length = len(answer.split())
        if answer_length > 100:  # Need much more content
            confidence += 0.15
            reasons.append(f"✓ Comprehensive answer ({answer_length} words)")
        elif answer_length > 50:
            confidence += 0.08
            reasons.append(f"⚠ Moderate answer ({answer_length} words)")
        else:
            reasons.append(f"✗ Insufficient detail ({answer_length} words)")
        
        # Factor 3: Answer contains citations (more important)
        if "[" in answer and any(str(i) in answer for i in range(10)):  # Actual numbered citations
            confidence += 0.08
            reasons.append("✓ Answer includes numbered citations")
        elif "[" in answer:
            confidence += 0.04
            reasons.append("⚠ Generic citations")
        else:
            reasons.append("✗ No citations in answer")
        
        # Factor 4: Check if answer addresses the question specifically
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        word_overlap = len(question_words & answer_words) / max(len(question_words), 1)
        
        if word_overlap > 0.3:  # Good overlap
            confidence += 0.07
            reasons.append(f"✓ Good question-answer alignment ({word_overlap:.1%} overlap)")
        elif word_overlap > 0.1:
            reasons.append(f"⚠ Partial alignment ({word_overlap:.1%} overlap)")
        else:
            reasons.append(f"✗ Poor alignment ({word_overlap:.1%} overlap)")
        
        if word_overlap > 0.3:
            confidence += 0.1
            reasons.append("✓ Answer addresses question")
        else:
            reasons.append("⚠ Low relevance to question")
        
        confidence = min(confidence, 1.0)
        
        logger.debug(f"Answer quality: confidence={confidence:.2f}, reasons={reasons}")
        
        return confidence, " | ".join(reasons)

    def should_reformulate(self, retrieval_quality: RetrievalQuality) -> bool:
        """
        Determine if the query should be reformulated based on retrieval quality.
        
        Args:
            retrieval_quality: Result of retrieval evaluation
            
        Returns:
            True if reformulation is recommended
        """
        should_reformulate = (
            retrieval_quality.num_results < self.min_results_threshold or
            retrieval_quality.avg_relevance_score < self.min_relevance_score or
            retrieval_quality.confidence < 0.4
        )
        
        if should_reformulate:
            logger.info(
                f"Query reformulation recommended: "
                f"results={retrieval_quality.num_results}, "
                f"relevance={retrieval_quality.avg_relevance_score:.2f}, "
                f"confidence={retrieval_quality.confidence:.2f}"
            )
        
        return should_reformulate
