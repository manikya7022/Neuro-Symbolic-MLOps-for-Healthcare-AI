"""Query Complexity Estimation for Adaptive Routing.

Estimates complexity of incoming queries to determine optimal routing:
- Low complexity: Fast neural-only path
- Medium complexity: Neural with optional verification
- High complexity: Hybrid neural-symbolic path
- Safety-critical: Symbolic-first path

Uses embedding-based features combined with heuristics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class ComplexityResult:
    """Result of complexity estimation."""
    
    score: float  # 0.0 (simple) to 1.0 (complex)
    category: str  # "low", "medium", "high", "critical"
    is_safety_critical: bool
    requires_symbolic: bool
    details: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_score(
        cls,
        score: float,
        is_safety_critical: bool = False,
        details: dict[str, Any] | None = None,
    ) -> ComplexityResult:
        """Create result from complexity score."""
        config = settings.router.complexity
        
        if is_safety_critical:
            category = "critical"
            requires_symbolic = True
        elif score >= config.high_threshold:
            category = "high"
            requires_symbolic = True
        elif score >= config.low_threshold:
            category = "medium"
            requires_symbolic = False
        else:
            category = "low"
            requires_symbolic = False
        
        return cls(
            score=min(max(score, 0.0), 1.0),
            category=category,
            is_safety_critical=is_safety_critical,
            requires_symbolic=requires_symbolic,
            details=details or {},
        )


class ComplexityEstimator:
    """Estimates query complexity for routing decisions.
    
    Uses a combination of:
    1. Embedding-based features (semantic complexity)
    2. Heuristic features (keywords, structure)
    3. Domain-specific patterns (healthcare)
    
    Example:
        ```python
        estimator = ComplexityEstimator()
        result = estimator.estimate("Can I take aspirin with my blood thinner?")
        
        if result.is_safety_critical:
            # Route to symbolic first
            ...
        elif result.category == "high":
            # Use hybrid approach
            ...
        ```
    """
    
    # Safety-critical keywords that trigger symbolic verification
    SAFETY_KEYWORDS = {
        "critical": [
            "overdose", "toxic", "lethal", "fatal", "death",
            "emergency", "urgent", "allergic reaction", "anaphylaxis",
            "contraindicated", "never take", "dangerous",
        ],
        "high_risk": [
            "interaction", "blood thinner", "anticoagulant", "warfarin",
            "pregnancy", "pregnant", "breastfeeding", "child", "infant",
            "kidney", "liver", "heart failure", "bleeding",
        ],
        "medical_action": [
            "dosage", "dose", "how much", "maximum", "minimum",
            "can i take", "should i take", "is it safe", "safe to",
            "combine", "together with", "mix", "while taking",
        ],
    }
    
    # Question patterns that indicate higher complexity
    COMPLEX_PATTERNS = [
        r"if\s+.+\s+then",  # Conditional logic
        r"what\s+if",  # Hypothetical
        r"(and|or)\s+.+\s+(and|or)",  # Multiple conditions
        r"compared\s+to",  # Comparison
        r"difference\s+between",  # Comparison
        r"(should|would|could)\s+i",  # Decision support
        r"\d+\s*(mg|ml|mcg|g)",  # Specific dosages
        r"(multiple|several|various)\s+(medications?|drugs?)",  # Multi-drug
    ]
    
    def __init__(self) -> None:
        """Initialize complexity estimator."""
        config = settings.router.complexity
        self.low_threshold = config.low_threshold
        self.high_threshold = config.high_threshold
        self.embedding_weight = config.embedding_weight
        self.heuristic_weight = config.heuristic_weight
        
        # Cache for embeddings
        self._embedding_cache: dict[str, np.ndarray] = {}
        
        # Embedding model (lazy loaded)
        self._embedder = None
    
    def estimate(self, query: str, context: dict[str, Any] | None = None) -> ComplexityResult:
        """Estimate query complexity.
        
        Args:
            query: User query text
            context: Optional context (patient info, current meds, etc.)
        
        Returns:
            ComplexityResult with score and category
        """
        context = context or {}
        details: dict[str, Any] = {"query_length": len(query)}
        
        # 1. Check for safety-critical keywords
        safety_score, safety_details = self._check_safety_keywords(query)
        details["safety"] = safety_details
        is_safety_critical = safety_score >= 0.8
        
        # 2. Calculate heuristic complexity
        heuristic_score, heuristic_details = self._calculate_heuristic_complexity(query)
        details["heuristic"] = heuristic_details
        
        # 3. Calculate embedding-based complexity (if embedder available)
        embedding_score = self._calculate_embedding_complexity(query)
        details["embedding_score"] = embedding_score
        
        # 4. Combine scores
        if embedding_score is not None:
            combined_score = (
                self.embedding_weight * embedding_score +
                self.heuristic_weight * heuristic_score
            )
        else:
            combined_score = heuristic_score
        
        # Safety critical queries override complexity score
        if is_safety_critical:
            combined_score = max(combined_score, 0.8)
        
        # Context-based adjustments
        if context.get("patient"):
            if context["patient"].get("conditions"):
                combined_score += 0.1 * len(context["patient"]["conditions"])
            if context["patient"].get("current_medications"):
                combined_score += 0.05 * len(context["patient"]["current_medications"])
        
        result = ComplexityResult.from_score(
            score=min(combined_score, 1.0),
            is_safety_critical=is_safety_critical,
            details=details,
        )
        
        logger.debug(
            "complexity_estimated",
            query_preview=query[:50],
            score=result.score,
            category=result.category,
            is_safety_critical=is_safety_critical,
        )
        
        return result
    
    def _check_safety_keywords(self, query: str) -> tuple[float, dict[str, Any]]:
        """Check for safety-critical keywords.
        
        Args:
            query: User query
        
        Returns:
            Tuple of (safety_score, details)
        """
        query_lower = query.lower()
        found_keywords: dict[str, list[str]] = {}
        
        for category, keywords in self.SAFETY_KEYWORDS.items():
            matches = [kw for kw in keywords if kw in query_lower]
            if matches:
                found_keywords[category] = matches
        
        # Calculate safety score
        if "critical" in found_keywords:
            score = 1.0
        elif "high_risk" in found_keywords and "medical_action" in found_keywords:
            score = 0.8
        elif "high_risk" in found_keywords:
            score = 0.6
        elif "medical_action" in found_keywords:
            score = 0.4
        else:
            score = 0.0
        
        return score, {"found_keywords": found_keywords}
    
    def _calculate_heuristic_complexity(self, query: str) -> tuple[float, dict[str, Any]]:
        """Calculate complexity using heuristic rules.
        
        Args:
            query: User query
        
        Returns:
            Tuple of (complexity_score, details)
        """
        score = 0.3  # Base score
        details: dict[str, Any] = {}
        
        # Query length
        length = len(query)
        details["length"] = length
        if length > 200:
            score += 0.15
        elif length > 100:
            score += 0.1
        elif length < 20:
            score -= 0.1
        
        # Word count
        words = query.split()
        word_count = len(words)
        details["word_count"] = word_count
        if word_count > 30:
            score += 0.1
        
        # Question complexity (multiple questions)
        question_marks = query.count("?")
        details["question_count"] = question_marks
        if question_marks > 1:
            score += 0.1 * (question_marks - 1)
        
        # Check for complex patterns
        pattern_matches = []
        for pattern in self.COMPLEX_PATTERNS:
            if re.search(pattern, query, re.IGNORECASE):
                pattern_matches.append(pattern)
                score += 0.1
        details["pattern_matches"] = len(pattern_matches)
        
        # Check for specific entities (drug names, conditions)
        # Simple heuristic: capitalized words that aren't sentence starters
        potential_entities = re.findall(r"(?<!\. )\b[A-Z][a-z]+\b", query)
        details["potential_entities"] = len(potential_entities)
        score += min(0.15, len(potential_entities) * 0.05)
        
        # Negation complexity
        negations = len(re.findall(r"\b(not|no|never|without|don't|doesn't|can't)\b", query, re.I))
        details["negations"] = negations
        score += min(0.1, negations * 0.03)
        
        return min(max(score, 0.0), 1.0), details
    
    def _calculate_embedding_complexity(self, query: str) -> float | None:
        """Calculate complexity using embeddings.
        
        Uses embedding model to estimate semantic complexity.
        Returns None if embedder not available.
        
        Args:
            query: User query
        
        Returns:
            Embedding-based complexity score or None
        """
        try:
            # Check cache
            if query in self._embedding_cache:
                embedding = self._embedding_cache[query]
            else:
                # Lazy load embedder
                if self._embedder is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                        model_name = settings.neural.embeddings.model
                        self._embedder = SentenceTransformer(model_name)
                    except ImportError:
                        logger.warning("sentence_transformers_not_available")
                        return None
                
                embedding = self._embedder.encode(query)
                self._embedding_cache[query] = embedding
            
            # Simple complexity heuristic: embedding magnitude variance
            # More complex queries tend to have higher variance
            magnitude = np.linalg.norm(embedding)
            variance = np.var(embedding)
            
            # Normalize to 0-1 range (heuristic bounds)
            complexity = min(1.0, (variance * 100 + magnitude / 10) / 2)
            
            return float(complexity)
            
        except Exception as e:
            logger.warning("embedding_complexity_failed", error=str(e))
            return None
    
    def batch_estimate(
        self,
        queries: list[str],
        contexts: list[dict[str, Any]] | None = None,
    ) -> list[ComplexityResult]:
        """Estimate complexity for multiple queries.
        
        Args:
            queries: List of queries
            contexts: Optional list of contexts
        
        Returns:
            List of complexity results
        """
        contexts = contexts or [{}] * len(queries)
        return [
            self.estimate(query, context)
            for query, context in zip(queries, contexts)
        ]
