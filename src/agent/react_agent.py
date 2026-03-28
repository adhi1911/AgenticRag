"""
ReAct Agent - Implements Reasoning + Acting loop for agentic behavior.
Iterates on queries if confidence is low, reformulates if needed.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from src.agent.query_evaluator import QueryEvaluator, RetrievalQuality
from src.generation.response_generator import ResponseGenerator

logger = logging.getLogger(__name__)


@dataclass
class AgentThought:
    """Represents a single thought/reasoning step by the agent"""
    step: int
    action: str  # "think", "retrieve", "evaluate", "reformulate", "generate"
    reasoning: str
    result: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentTrace:
    """Complete trace of agent's reasoning process"""
    query: str
    thoughts: List[AgentThought] = field(default_factory=list)
    iterations: int = 0
    final_answer: str = ""
    final_confidence: float = 0.0
    reformulations: int = 0
    success: bool = False
    total_time_ms: float = 0.0


class ReActAgent:
    """
    Implements ReAct (Reasoning + Acting) pattern for agentic RAG.
    
    The agent:
    1. THINKS about the question
    2. ACTS by retrieving documents
    3. OBSERVES the quality of results
    4. DECIDES - either answer or reformulate and retry
    """

    def __init__(
        self,
        response_generator: ResponseGenerator,
        max_iterations: int = 3,
        confidence_threshold: float = 0.75  # Increased from 0.6
    ):
        """
        Initialize the ReAct agent.
        
        Args:
            response_generator: Generator for creating answers
            max_iterations: Maximum number of reasoning iterations
            confidence_threshold: Confidence needed to finalize answer
        """
        self.generator = response_generator
        self.evaluator = QueryEvaluator()
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        logger.info(f"ReActAgent initialized (max_iterations={max_iterations})")

    def think(self, query: str) -> str:
        """
        THINK step: Analyze the query to understand what's being asked.
        
        Args:
            query: User's question
            
        Returns:
            Analysis of the question
        """
        # Simple NLP-style analysis
        question_type = "general"
        if any(word in query.lower() for word in ['how', 'explain', 'describe']):
            question_type = "explanatory"
        elif any(word in query.lower() for word in ['what is', 'define', 'meaning']):
            question_type = "definitional"
        elif any(word in query.lower() for word in ['why', 'reason', 'cause']):
            question_type = "causal"
        elif any(word in query.lower() for word in ['list', 'enumerate', 'give', 'examples']):
            question_type = "listing"
        
        analysis = (
            f"Question type: {question_type}. "
            f"Query length: {len(query.split())} words. "
            f"Looking for: {question_type.lower()} information."
        )
        return analysis

    def decide_reformulation(
        self,
        query: str,
        retrieval_quality: RetrievalQuality,
        iteration_count: int
    ) -> Tuple[bool, str]:
        """
        DECIDE step: Determine if query needs reformulation.
        
        Args:
            query: Current query
            retrieval_quality: Quality assessment of retrieval
            iteration_count: Current iteration number
            
        Returns:
            Tuple of (should_reformulate: bool, new_query_or_reason: str)
        """
        if not self.evaluator.should_reformulate(retrieval_quality):
            return False, "Retrieval quality sufficient"
        
        if iteration_count >= self.max_iterations - 1:
            return False, f"Max iterations ({self.max_iterations}) reached"
        
        # Generate reformulation
        reformulation = self._generate_reformulation(query, retrieval_quality)
        return True, reformulation

    def _generate_reformulation(
        self,
        original_query: str,
        quality: RetrievalQuality
    ) -> str:
        """
        Generate a reformulated version of the query based on quality issues.
        
        Args:
            original_query: Original user query
            quality: Quality metrics of current retrieval
            
        Returns:
            Reformulated query
        """
        if quality.num_results == 0:
            # No results - try more general version
            return f"{original_query} overview"
        elif quality.avg_relevance_score < 0.3:
            # Low relevance - try synonym expansion
            return f"what is {original_query} detailed explanation"
        else:
            # Moderate quality - add context request
            return f"{original_query} in detail with background"

    def process_query(
        self,
        query: str,
        **generator_kwargs
    ) -> Tuple[Dict[str, Any], AgentTrace]:
        """
        Process a query using the ReAct loop with iteration.
        
        Args:
            query: User question
            **generator_kwargs: Arguments to pass to response generator
            
        Returns:
            Tuple of (result_dict, agent_trace)
        """
        import time
        start_time = time.time()
        
        trace = AgentTrace(query=query)
        current_query = query
        final_sources = []  # Capture sources from the final iteration
        final_metadata = {}  # Capture response metadata
        
        # Main ReAct loop
        for iteration in range(self.max_iterations):
            trace.iterations = iteration + 1
            
            # THINK: Analyze the query
            thinking = self.think(current_query)
            trace.thoughts.append(AgentThought(
                step=len(trace.thoughts) + 1,
                action="think",
                reasoning=f"Analyzing question: {current_query}",
                result=thinking
            ))
            logger.info(f"[Iteration {iteration + 1}] THINK: {thinking}")
            
            # ACT: Generate response (includes retrieval)
            try:
                response = self.generator.generate_response(current_query, **generator_kwargs)
                answer = response.get("answer", "")
                sources = response.get("sources", [])
                
                trace.thoughts.append(AgentThought(
                    step=len(trace.thoughts) + 1,
                    action="retrieve",
                    reasoning=f"Retrieved {len(sources)} documents",
                    result=f"{len(sources)} sources found"
                ))
                logger.info(f"[Iteration {iteration + 1}] ACT: Retrieved {len(sources)} sources")
                
            except Exception as e:
                logger.error(f"Error in generation: {e}")
                trace.thoughts.append(AgentThought(
                    step=len(trace.thoughts) + 1,
                    action="retrieve",
                    reasoning="Error during retrieval",
                    result=str(e)
                ))
                continue
            
            # OBSERVE: Evaluate quality
            retrieval_quality = self.evaluator.evaluate_retrieval(sources)
            answer_confidence, quality_reasons = self.evaluator.evaluate_answer_quality(
                current_query, answer, retrieval_quality
            )
            
            trace.thoughts.append(AgentThought(
                step=len(trace.thoughts) + 1,
                action="evaluate",
                reasoning=f"Evaluating answer quality (confidence: {answer_confidence:.2%})",
                result=quality_reasons
            ))
            logger.info(f"[Iteration {iteration + 1}] OBSERVE: Confidence={answer_confidence:.2%}")
            
            # DECIDE: Continue or finalize?
            if answer_confidence >= self.confidence_threshold:
                # Answer is good enough - capture sources and metadata
                final_sources = sources
                # Capture all relevant metadata from generator response
                final_metadata = {
                    "temperature": response.get("temperature", 0.3),
                    "response_format": response.get("response_format", "unknown"),
                    "citation_style": response.get("citation_style", "inline"),
                    "model": response.get("model", "unknown"),
                    "num_sources": response.get("num_sources", 0)
                }
                trace.thoughts.append(AgentThought(
                    step=len(trace.thoughts) + 1,
                    action="finalize",
                    reasoning=f"Answer confidence {answer_confidence:.2%} exceeds threshold {self.confidence_threshold:.2%}",
                    result="Finalizing answer"
                ))
                trace.final_answer = answer
                trace.final_confidence = answer_confidence
                trace.success = True
                logger.info(f"[Iteration {iteration + 1}] DECIDE: Answer is good enough. Finalizing.")
                break
            
            # Not confident enough - try reformulation
            should_reformulate, reason = self.decide_reformulation(
                current_query, retrieval_quality, iteration
            )
            
            if should_reformulate:
                trace.reformulations += 1
                current_query = reason
                trace.thoughts.append(AgentThought(
                    step=len(trace.thoughts) + 1,
                    action="reformulate",
                    reasoning=f"Low confidence ({answer_confidence:.2%}), reformulating query",
                    result=f"New query: {reason}"
                ))
                logger.info(f"[Iteration {iteration + 1}] DECIDE: Reformulating query to: {reason}")
            else:
                # Can't reformulate, use best answer - capture sources and metadata
                final_sources = sources
                final_metadata = {
                    "temperature": response.get("temperature", 0.3),
                    "response_format": response.get("response_format", "unknown"),
                    "citation_style": response.get("citation_style", "inline"),
                    "model": response.get("model", "unknown"),
                    "num_sources": response.get("num_sources", 0)
                }
                trace.final_answer = answer
                trace.final_confidence = answer_confidence
                trace.success = True
                trace.thoughts.append(AgentThought(
                    step=len(trace.thoughts) + 1,
                    action="finalize",
                    reasoning=f"Cannot reformulate further: {reason}",
                    result="Using best available answer"
                ))
                logger.info(f"[Iteration {iteration + 1}] DECIDE: Using best answer (reason: {reason})")
                break
        
        # Time tracking
        trace.total_time_ms = (time.time() - start_time) * 1000
        
        # Safety check: if final_metadata wasn't set (shouldn't happen, but just in case)
        if not final_metadata:
            final_metadata = {
                "temperature": 0.3,
                "response_format": "unknown",
                "citation_style": "inline",
                "model": "unknown",
                "num_sources": len(final_sources)
            }
        
        # Build response dictionary
        result = {
            "success": trace.success,
            "answer": trace.final_answer,
            "confidence": trace.final_confidence,
            "iterations": trace.iterations,
            "reformulations": trace.reformulations,
            "sources": final_sources,  # Include sources for UI display
            "metadata": {
                **final_metadata,  # Include response generator metadata (temperature, etc)
                "mode": "agentic",
                "agent_reasoning": [
                    {
                        "step": t.step,
                        "action": t.action,
                        "reasoning": t.reasoning,
                        "result": t.result
                    }
                    for t in trace.thoughts
                ]
            }
        }
        
        logger.info(
            f"Agent completed: iterations={trace.iterations}, "
            f"reformulations={trace.reformulations}, "
            f"confidence={trace.final_confidence:.2%}, "
            f"time={trace.total_time_ms:.0f}ms"
        )
        
        return result, trace
