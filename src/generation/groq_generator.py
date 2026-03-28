import logging
from typing import List, Dict, Any, Literal
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import groq
from config.settings import settings

logger = logging.getLogger(__name__)


class ResponseFormat(str, Enum):
    NARRATIVE = "narrative"
    STRUCTURED = "structured"
    CONCISE = "concise"
    RESEARCH = "research"


@dataclass
class GenerationConfig:
    """Configuration for answer generation behavior"""
    system_message: str | None = None
    response_format: ResponseFormat = ResponseFormat.NARRATIVE
    temperature: float = settings.GROQ_TEMPERATURE
    max_tokens: int = settings.GROQ_MAX_TOKENS
    include_reasoning: bool = False
    citation_style: Literal["inline", "footnote", "numbered"] = "inline"


class GroqGenerator:

    def __init__(self, api_key: str = settings.GROQ_API_KEY, model_name: str = settings.GROQ_MODEL):
        self.client = groq.Groq(api_key=api_key)
        self.model_name = model_name

        logger.info(f"✓ GroqGenerator initialized with model: {model_name}")

        self.system_presets = {
            "legal": "You are an expert legal analyst. Focus on applicable laws, precedents, and nuanced interpretations. Be precise and thorough.",
            "technical": "You are a technical expert. Explain concepts clearly with practical examples. Focus on implementation details and trade-offs.",
            "researcher": "You are a research synthesizer. Integrate multiple perspectives, highlight key findings, and identify gaps or contradictions.",
            "simple": "Answer clearly and concisely. Use simple language. Focus on the most important points.",
            "general_expert": "You are an expert assistant providing accurate, practical information about various topics. Provide comprehensive and helpful responses."
        }

    def _format_context(self, documents: List[Dict[str, Any]], citation_style: Literal["inline", "footnote", "numbered"] = "inline") -> str:
        """Format documents into context string with citation style."""
        context_parts = []

        for i, doc in enumerate(documents, 1):
            source = doc.get('metadata', {}).get('source', 'unknown')
            # Extract only filename for security (don't show full path)
            source_filename = Path(source).name if source != 'unknown' else 'unknown'
            page = doc.get('metadata', {}).get('page', 'unknown')
            content = doc.get('content', '')
            chunk_id = doc.get('chunk_id', '')

            # Format based on citation style
            if citation_style == "inline":
                context_parts.append(
                    f"[{i}] {content}\n"
                    f"    (Source: {source_filename}, Page: {page})"
                )

            elif citation_style == "footnote":
                context_parts.append(
                    f"{content} [{i}]\n"
                    f"Footnote {i}: Source: {source_filename}, Page: {page}"
                )

            elif citation_style == "numbered":
                context_parts.append(content)
                if i == len(documents):
                    # Add all sources at the end
                    sources_info = "\n".join(
                        [f"{j}. Source: {Path(d.get('metadata', {}).get('source', 'unknown')).name}, Page: {d.get('metadata', {}).get('page', 'unknown')}"
                         for j, d in enumerate(documents, 1)]
                    )
                    context_parts.append(f"\nSources:\n{sources_info}")

        return "\n\n".join(context_parts)

    def _get_system_message(self, config: GenerationConfig) -> str:
        """Determine system message based on config."""
        if config.system_message:
            return config.system_message

        format_defaults = {
            ResponseFormat.NARRATIVE: "Provide a comprehensive answer with citations. You are an expert assistant providing detailed, accurate information.",
            ResponseFormat.STRUCTURED: "Organize your answer with clear sections and bullet points. Focus on key concepts and practical applications.",
            ResponseFormat.CONCISE: "Provide a brief, direct answer with essential sources. Be specific and clear in your response.",
            ResponseFormat.RESEARCH: "Synthesize information from multiple sources. Show analysis and reasoning about the topic.",
        }
        return format_defaults.get(config.response_format, format_defaults[ResponseFormat.NARRATIVE])

    def _build_prompt(self, query: str, documents: List[Dict], config: GenerationConfig) -> str:
        """Build prompt dynamically based on config."""
        context = self._format_context(documents, config.citation_style)

        if config.include_reasoning:
            answer_instruction = "First, explain your reasoning step by step. Then provide the final answer."
        else:
            answer_instruction = "Provide a clear, accurate answer based on the provided context."

        prompt = f"""{self._get_system_message(config)}

Context Information:
{context}

Question: {query}

Instructions:
- Answer based ONLY on the provided context
- If the context doesn't contain enough information, say so clearly
- Use the citation style specified
- {answer_instruction}

Answer:"""
        return prompt

    def generate(self, query: str, documents: List[Dict[str, Any]], config: GenerationConfig | None = None) -> str:
        """Generate answer with flexible configuration"""

        if config is None:
            config = GenerationConfig()

        prompt = self._build_prompt(query, documents, config)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )

            answer = response.choices[0].message.content.strip()
            logger.info(f"✓ Generated answer ({len(answer)} chars) for query: '{query[:50]}...'")
            return answer

        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating response: {str(e)}"

    def generate_with_metadata(self, query: str, documents: List[Dict[str, Any]], config: GenerationConfig | None = None) -> Dict[str, Any]:
        """Generate answer and return with source metadata."""
        if config is None:
            config = GenerationConfig()

        answer = self.generate(query, documents, config)

        sources = []
        for doc in documents:
            sources.append({
                "source": doc.get('metadata', {}).get('source', 'unknown'),
                "page": doc.get('metadata', {}).get('page', 'unknown'),
                "rerank_score": doc.get('rerank_score', doc.get('score', None)),
                "chunk_id": doc.get('chunk_id', ''),
                "preview": doc.get('content', '')[:100] + "..." if len(doc.get('content', '')) > 100 else doc.get('content', '')
            })

        return {
            "answer": answer,
            "sources": sources,
            "query": query,
            "num_sources": len(sources),
            "response_format": config.response_format.value,
            "temperature": config.temperature,
            "citation_style": config.citation_style,
            "model": self.model_name
        }

    def get_system_preset(self, name: str) -> str:
        """Get a predefined system message preset"""
        return self.system_presets.get(name, "")

    def set_system_preset(self, name: str, message: str):
        """Set a custom system message preset"""
        self.system_presets[name] = message
