"""Confidence Scoring for Neural Responses.

Provides comprehensive confidence scoring that combines multiple signals:
- Uncertainty quantification scores
- Response quality metrics  
- Domain-specific validation (healthcare)
- Safety-critical content detection

Used by the router to decide when to trigger symbolic verification.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.config import settings
from src.neural.llm_client import LLMResponse
from src.neural.uncertainty import UncertaintyQuantifier, UncertaintyResult

logger = structlog.get_logger(__name__)


@dataclass
class ConfidenceResult:
    """Comprehensive confidence assessment of a neural response."""
    
    # Overall confidence (0-1)
    overall: float
    
    # Component scores
    uncertainty_score: float  # From UQ module (inverted: high = confident)
    quality_score: float  # Response quality metrics
    safety_score: float  # Safety-critical content detection
    domain_score: float  # Healthcare domain relevance
    
    # Recommendations
    requires_verification: bool
    requires_human_review: bool
    safe_to_return: bool
    
    # Details for debugging/monitoring
    details: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    
    @property
    def grade(self) -> str:
        """Letter grade for confidence level."""
        if self.overall >= 0.9:
            return "A"
        elif self.overall >= 0.8:
            return "B"
        elif self.overall >= 0.7:
            return "C"
        elif self.overall >= 0.6:
            return "D"
        else:
            return "F"


class ConfidenceScorer:
    """Scores confidence of neural responses for routing decisions.
    
    Combines multiple signals to produce a comprehensive confidence
    assessment that guides the adaptive router.
    
    Example:
        ```python
        scorer = ConfidenceScorer(uncertainty_quantifier)
        result = await scorer.score(query, response, uncertainty_result)
        
        if result.requires_verification:
            # Route to symbolic solver
            symbolic_result = await solver.verify(response)
        ```
    """
    
    # Healthcare-specific danger terms that require extra caution
    HIGH_RISK_TERMS = [
        "fatal", "lethal", "death", "die", "deadly",
        "overdose", "toxic", "poison",
        "hemorrhage", "bleeding", "stroke",
        "anaphylaxis", "allergic shock",
        "black box warning", "contraindicated",
        "pregnancy category x",
    ]
    
    # Terms that indicate uncertainty
    UNCERTAINTY_MARKERS = [
        "i'm not sure", "i think", "possibly", "might be",
        "could be", "uncertain", "don't know", "not certain",
        "consult a doctor", "seek medical advice",
        "may or may not", "it depends", "varies by",
    ]
    
    # Expected medical content markers
    MEDICAL_CONTENT_MARKERS = [
        "mg", "dose", "dosage", "medication", "drug",
        "side effect", "interaction", "contraindication",
        "symptom", "treatment", "diagnosis", "condition",
        "prescription", "otc", "over-the-counter",
    ]
    
    def __init__(
        self,
        uncertainty_quantifier: UncertaintyQuantifier | None = None,
        confidence_threshold: float | None = None,
        safety_threshold: float | None = None,
    ) -> None:
        """Initialize confidence scorer.
        
        Args:
            uncertainty_quantifier: UQ module for uncertainty scores
            confidence_threshold: Threshold for requiring verification
            safety_threshold: Threshold for requiring human review
        """
        self.uq = uncertainty_quantifier
        
        config = settings.neural.uncertainty
        self.confidence_threshold = confidence_threshold or config.threshold
        self.safety_threshold = safety_threshold or 0.5  # More strict for safety
        
        # Weights for combining scores
        self.weights = {
            "uncertainty": 0.35,
            "quality": 0.25,
            "safety": 0.25,
            "domain": 0.15,
        }
    
    async def score(
        self,
        query: str,
        response: LLMResponse,
        uncertainty_result: UncertaintyResult | None = None,
    ) -> ConfidenceResult:
        """Score confidence of a neural response.
        
        Args:
            query: Original user query
            response: LLM response to score
            uncertainty_result: Pre-computed uncertainty (optional)
        
        Returns:
            ConfidenceResult with comprehensive assessment
        """
        warnings: list[str] = []
        details: dict[str, Any] = {}
        
        # 1. Get uncertainty score (invert to confidence)
        if uncertainty_result:
            uncertainty_score = uncertainty_result.confidence
            details["uncertainty_method"] = uncertainty_result.method.value
        else:
            # Default moderate confidence if no UQ available
            uncertainty_score = 0.6
            details["uncertainty_method"] = "default"
        
        # 2. Calculate response quality score
        quality_score, quality_details = self._score_quality(response.text)
        details["quality"] = quality_details
        
        # 3. Calculate safety score
        safety_score, safety_details = self._score_safety(query, response.text)
        details["safety"] = safety_details
        if safety_details.get("high_risk_terms"):
            warnings.append(f"High-risk terms detected: {safety_details['high_risk_terms']}")
        
        # 4. Calculate domain relevance score
        domain_score, domain_details = self._score_domain_relevance(query, response.text)
        details["domain"] = domain_details
        
        # 5. Combine scores with weights
        overall = (
            self.weights["uncertainty"] * uncertainty_score +
            self.weights["quality"] * quality_score +
            self.weights["safety"] * safety_score +
            self.weights["domain"] * domain_score
        )
        
        # 6. Determine recommendations
        requires_verification = (
            overall < self.confidence_threshold or
            safety_score < self.safety_threshold or
            (uncertainty_result and uncertainty_result.needs_symbolic_verification)
        )
        
        requires_human_review = (
            safety_score < 0.3 or
            any(term in response.text.lower() for term in ["consult a doctor", "seek medical attention"])
        )
        
        safe_to_return = (
            overall >= self.confidence_threshold and
            safety_score >= self.safety_threshold and
            not requires_human_review
        )
        
        if requires_verification:
            warnings.append("Response requires symbolic verification")
        if requires_human_review:
            warnings.append("Response requires human review")
        
        logger.debug(
            "confidence_scored",
            overall=overall,
            uncertainty=uncertainty_score,
            quality=quality_score,
            safety=safety_score,
            domain=domain_score,
            requires_verification=requires_verification,
        )
        
        return ConfidenceResult(
            overall=overall,
            uncertainty_score=uncertainty_score,
            quality_score=quality_score,
            safety_score=safety_score,
            domain_score=domain_score,
            requires_verification=requires_verification,
            requires_human_review=requires_human_review,
            safe_to_return=safe_to_return,
            details=details,
            warnings=warnings,
        )
    
    def _score_quality(self, text: str) -> tuple[float, dict[str, Any]]:
        """Score response quality based on text characteristics.
        
        Args:
            text: Response text
        
        Returns:
            Tuple of (score, details)
        """
        score = 0.7  # Base score
        details: dict[str, Any] = {}
        
        # Length checks
        length = len(text)
        details["length"] = length
        
        if length < 20:
            score -= 0.3
            details["length_issue"] = "too_short"
        elif length < 50:
            score -= 0.1
            details["length_issue"] = "short"
        elif length > 2000:
            score -= 0.1
            details["length_issue"] = "verbose"
        else:
            score += 0.1
        
        # Sentence structure
        sentences = text.split(".")
        details["sentence_count"] = len(sentences)
        
        if len(sentences) < 2:
            score -= 0.1
        elif len(sentences) > 20:
            score -= 0.05
        
        # Check for complete sentences
        if text and not text.strip().endswith((".", "!", "?")):
            score -= 0.15
            details["incomplete_ending"] = True
        
        # Check for repetition (sign of LLM issues)
        words = text.lower().split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            details["unique_word_ratio"] = unique_ratio
            if unique_ratio < 0.3:
                score -= 0.2
                details["repetition_issue"] = True
        
        # Check for formatting
        has_structure = bool(re.search(r"(\n-|\n\d\.|\n•)", text))
        details["has_structure"] = has_structure
        if has_structure:
            score += 0.05
        
        return max(min(score, 1.0), 0.0), details
    
    def _score_safety(self, query: str, text: str) -> tuple[float, dict[str, Any]]:
        """Score safety of the response content.
        
        Args:
            query: Original query
            text: Response text
        
        Returns:
            Tuple of (score, details)
        """
        score = 0.8  # Base score
        details: dict[str, Any] = {}
        
        text_lower = text.lower()
        query_lower = query.lower()
        
        # Check for high-risk terms
        found_high_risk = [term for term in self.HIGH_RISK_TERMS if term in text_lower]
        details["high_risk_terms"] = found_high_risk
        
        if found_high_risk:
            # Reduce score based on number of high-risk terms
            score -= min(0.3, len(found_high_risk) * 0.1)
        
        # Check for uncertainty markers
        found_uncertainty = [term for term in self.UNCERTAINTY_MARKERS if term in text_lower]
        details["uncertainty_markers"] = found_uncertainty
        
        if found_uncertainty:
            # Uncertainty is actually good for safety (model is appropriately cautious)
            score += min(0.1, len(found_uncertainty) * 0.02)
        
        # Check if response includes appropriate disclaimers for medical content
        has_disclaimer = any(phrase in text_lower for phrase in [
            "consult", "healthcare provider", "doctor", "physician",
            "medical professional", "should not be", "not a substitute",
        ])
        details["has_disclaimer"] = has_disclaimer
        
        # Query is about sensitive topic but no disclaimer
        sensitive_query = any(term in query_lower for term in [
            "safe", "dangerous", "overdose", "pregnancy", "child", "infant",
        ])
        
        if sensitive_query and not has_disclaimer:
            score -= 0.15
            details["missing_disclaimer_for_sensitive"] = True
        elif has_disclaimer:
            score += 0.1
        
        # Check for dosage information without warnings
        has_dosage = bool(re.search(r"\d+\s*(mg|ml|mcg|g|units?)", text_lower))
        details["has_dosage_info"] = has_dosage
        
        if has_dosage:
            # Dosage info requires extra verification
            score -= 0.1
            details["dosage_needs_verification"] = True
        
        return max(min(score, 1.0), 0.0), details
    
    def _score_domain_relevance(self, query: str, text: str) -> tuple[float, dict[str, Any]]:
        """Score how relevant the response is to healthcare domain.
        
        Args:
            query: Original query
            text: Response text
        
        Returns:
            Tuple of (score, details)
        """
        score = 0.5  # Base score
        details: dict[str, Any] = {}
        
        text_lower = text.lower()
        
        # Check for medical content markers
        found_markers = [term for term in self.MEDICAL_CONTENT_MARKERS if term in text_lower]
        details["medical_markers"] = found_markers
        
        if found_markers:
            # More medical content = more domain relevant
            score += min(0.3, len(found_markers) * 0.05)
        
        # Check if response addresses the query topic
        query_words = set(query.lower().split())
        response_words = set(text_lower.split())
        query_coverage = len(query_words & response_words) / max(len(query_words), 1)
        details["query_coverage"] = query_coverage
        
        if query_coverage > 0.3:
            score += 0.2
        elif query_coverage < 0.1:
            score -= 0.2
            details["low_query_relevance"] = True
        
        # Check for off-topic content
        off_topic_indicators = [
            "i cannot", "i'm an ai", "as an ai",
            "i don't have access", "not sure what you mean",
        ]
        
        is_off_topic = any(phrase in text_lower for phrase in off_topic_indicators)
        details["off_topic_indicators"] = is_off_topic
        
        if is_off_topic:
            score -= 0.3
        
        # Check for specific condition/drug mentions (indicates specificity)
        # Simple heuristic: capitalized words that might be drug/condition names
        potential_entities = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", text)
        details["potential_entities"] = len(potential_entities)
        
        if potential_entities:
            score += min(0.1, len(potential_entities) * 0.02)
        
        return max(min(score, 1.0), 0.0), details


def confidence_to_action(result: ConfidenceResult) -> str:
    """Map confidence result to recommended action.
    
    Args:
        result: Confidence scoring result
    
    Returns:
        Action recommendation string
    """
    if result.requires_human_review:
        return "ESCALATE_HUMAN"
    elif result.requires_verification:
        return "VERIFY_SYMBOLIC"
    elif result.safe_to_return:
        return "RETURN_DIRECT"
    else:
        return "VERIFY_SYMBOLIC"
