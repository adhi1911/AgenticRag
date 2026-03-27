import logging
from typing import List, Dict, Any

import groq
from config.settings import settings

logger = logging.getLogger(__name__)


class QueryTransformer:

    def __init__(self, api_key: str = settings.GROQ_API_KEY, model: str = settings.GROQ_MODEL):
        self.client = groq.Groq(api_key=api_key)
        self.model = model
        logger.info(f"✓ QueryTransformer initialized with model: {model}")

    def hyde(self, query: str) -> str:
        """Generate a HyDE-style synthetic document from the query."""
        prompt = f"""Write a detailed answer to the following question.
The answer should be comprehensive and specific.

Question: {query}

Hypothetical Answer:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=512
            )
            answer = response.choices[0].message.content
            logger.debug(f"Generated HyDE answer ({len(answer)} chars)")
            return answer
        except Exception as e:
            logger.error(f"Error generating HyDE: {e}")
            return ""

    def multi_query(self, query: str, num_queries: int = 3) -> List[str]:
        """Generate multiple diverse query phrasings"""
        prompt = f"""Generate {num_queries} alternative ways to phrase the following question.
Each should be a complete question and explore different aspects of the original.
Return as a numbered list.

Original Question: {query}

Alternative Phrasings:"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=512
            )

            content = response.choices[0].message.content
            lines = content.split('\n')
            queries = []

            for line in lines:
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith("-")):
                    q = line.lstrip("0123456789.- ").strip()
                    if q:
                        queries.append(q)

            return queries[:num_queries]

        except Exception as e:
            logger.error(f"Error generating multi-queries: {e}")
            return []

    def transform(self, query: str, use_hyde: bool = False, use_multi: bool = False, num_multi_queries: int = 3) -> Dict[str, Any]:
        """Transform query using HyDE and/or multi-query techniques"""
        queries = [query]  # Always include original

        if use_multi:
            alternatives = self.multi_query(query, num_multi_queries)
            queries.extend(alternatives)
            logger.info(f"Generated {len(alternatives)} multi-queries")

        if use_hyde:
            hyde_doc = self.hyde(query)
            if hyde_doc:
                queries.append(hyde_doc)
                logger.info("Generated HyDE document")

        return {
            "original_query": query,
            "transformed_queries": queries,
            "use_hyde": use_hyde,
            "use_multi": use_multi,
            "num_multi_queries": num_multi_queries,
            "total_queries": len(queries)
        }
