"""Uncertainty Quantification for LLM outputs.

Implements multiple methods to quantify uncertainty in neural responses:
1. Token Entropy: Based on log probabilities of generated tokens
2. Semantic Consistency: Multiple generations compared for agreement
3. Conformal Prediction: Statistical coverage guarantees

High uncertainty triggers symbolic fallback for safety-critical queries.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import structlog

from src.config import settings
from src.neural.llm_client import LLMClient, LLMResponse

logger = structlog.get_logger(__name__)


class UncertaintyMethod(str, Enum):
    """Uncertainty quantification methods."""
    
    TOKEN_ENTROPY = "token_entropy"
    SEMANTIC_CONSISTENCY = "semantic_consistency"
    CONFORMAL = "conformal"
    ENSEMBLE = "ensemble"


@dataclass
class UncertaintyResult:
    """Result of uncertainty quantification."""
    
    score: float  # 0.0 (certain) to 1.0 (uncertain)
    confidence: float  # 1.0 - score (for convenience)
    method: UncertaintyMethod
    needs_symbolic_verification: bool
    details: dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_score(
        cls,
        score: float,
        method: UncertaintyMethod,
        threshold: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> UncertaintyResult:
        """Create result from uncertainty score."""
        threshold = threshold or settings.neural.uncertainty.threshold
        return cls(
            score=min(max(score, 0.0), 1.0),
            confidence=1.0 - min(max(score, 0.0), 1.0),
            method=method,
            needs_symbolic_verification=score > (1.0 - threshold),
            details=details or {},
        )


class UncertaintyQuantifier:
    """Quantifies uncertainty in LLM responses.
    
    Provides multiple methods for uncertainty estimation, each with
    different trade-offs between accuracy and compute cost:
    
    - Token Entropy: Fast, uses log probabilities from single generation
    - Semantic Consistency: More accurate, requires multiple generations
    - Conformal: Provides statistical guarantees, requires calibration data
    - Ensemble: Combines multiple methods for robust estimation
    
    Example:
        ```python
        uq = UncertaintyQuantifier(client)
        result = await uq.quantify("Is aspirin safe with warfarin?", response)
        if result.needs_symbolic_verification:
            # Route to symbolic solver
            pass
        ```
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        method: UncertaintyMethod | str | None = None,
        threshold: float | None = None,
    ) -> None:
        """Initialize uncertainty quantifier.
        
        Args:
            llm_client: LLM client for additional generations
            method: Uncertainty method to use
            threshold: Confidence threshold for symbolic fallback
        """
        self.client = llm_client
        
        config = settings.neural.uncertainty
        self.method = UncertaintyMethod(method or config.method)
        self.threshold = threshold or config.threshold
        self.mc_samples = config.mc_samples
        self.conformal_alpha = config.conformal_alpha
        
        # Calibration data for conformal prediction
        self._calibration_scores: list[float] = []
        self._is_calibrated = False
    
    async def quantify(
        self,
        prompt: str,
        response: LLMResponse,
        method: UncertaintyMethod | None = None,
    ) -> UncertaintyResult:
        """Quantify uncertainty of an LLM response.
        
        Args:
            prompt: Original prompt
            response: LLM response to evaluate
            method: Override default method
        
        Returns:
            UncertaintyResult with score and recommendation
        """
        method = method or self.method
        
        if method == UncertaintyMethod.TOKEN_ENTROPY:
            return await self._token_entropy(response)
        elif method == UncertaintyMethod.SEMANTIC_CONSISTENCY:
            return await self._semantic_consistency(prompt, response)
        elif method == UncertaintyMethod.CONFORMAL:
            return await self._conformal_prediction(prompt, response)
        elif method == UncertaintyMethod.ENSEMBLE:
            return await self._ensemble(prompt, response)
        else:
            raise ValueError(f"Unknown uncertainty method: {method}")
    
    async def _token_entropy(self, response: LLMResponse) -> UncertaintyResult:
        """Calculate uncertainty from token log probabilities.
        
        Uses normalized entropy of token probabilities. High entropy
        indicates the model was uncertain about token choices.
        
        Args:
            response: LLM response with token info
        
        Returns:
            UncertaintyResult based on entropy
        """
        tokens = response.tokens
        
        # If no log probs available, use heuristic
        if not tokens or all(t.log_prob is None for t in tokens):
            # Fallback: estimate from response characteristics
            uncertainty = self._heuristic_uncertainty(response.text)
            return UncertaintyResult.from_score(
                score=uncertainty,
                method=UncertaintyMethod.TOKEN_ENTROPY,
                threshold=self.threshold,
                details={"fallback": "heuristic", "reason": "no_logprobs"},
            )
        
        # Calculate entropy from log probabilities
        log_probs = [t.log_prob for t in tokens if t.log_prob is not None]
        
        if not log_probs:
            return UncertaintyResult.from_score(
                score=0.5,
                method=UncertaintyMethod.TOKEN_ENTROPY,
                threshold=self.threshold,
                details={"fallback": "default"},
            )
        
        # Convert log probs to probabilities
        probs = [math.exp(lp) for lp in log_probs]
        
        # Normalize (in case of numerical issues)
        total = sum(probs)
        if total > 0:
            probs = [p / total for p in probs]
        
        # Calculate entropy (normalized by max entropy)
        entropy = -sum(p * math.log(p + 1e-10) for p in probs if p > 0)
        max_entropy = math.log(len(probs))  # Uniform distribution entropy
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # Also consider minimum probability (most uncertain token)
        min_prob = min(probs) if probs else 0
        min_prob_uncertainty = 1.0 - min_prob
        
        # Combine metrics
        uncertainty = 0.7 * normalized_entropy + 0.3 * min_prob_uncertainty
        
        logger.debug(
            "token_entropy_computed",
            entropy=normalized_entropy,
            min_prob=min_prob,
            uncertainty=uncertainty,
            num_tokens=len(log_probs),
        )
        
        return UncertaintyResult.from_score(
            score=uncertainty,
            method=UncertaintyMethod.TOKEN_ENTROPY,
            threshold=self.threshold,
            details={
                "normalized_entropy": normalized_entropy,
                "min_probability": min_prob,
                "num_tokens": len(log_probs),
                "avg_log_prob": response.avg_log_prob,
            },
        )
    
    async def _semantic_consistency(
        self,
        prompt: str,
        response: LLMResponse,
    ) -> UncertaintyResult:
        """Measure uncertainty via semantic consistency across samples.
        
        Generates multiple responses and measures agreement. High variance
        in responses indicates uncertainty.
        
        Args:
            prompt: Original prompt
            response: Initial response
        
        Returns:
            UncertaintyResult based on consistency
        """
        # Generate additional samples with higher temperature
        additional_responses: list[LLMResponse] = []
        
        tasks = [
            self.client.generate(
                prompt=prompt,
                temperature=0.9,  # Higher temperature for diversity
                max_tokens=256,  # Shorter for efficiency
            )
            for _ in range(self.mc_samples - 1)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, LLMResponse):
                additional_responses.append(result)
        
        if not additional_responses:
            # Fallback if all samples failed
            return await self._token_entropy(response)
        
        # Calculate semantic similarity between responses
        all_responses = [response.text] + [r.text for r in additional_responses]
        consistency_score = self._calculate_text_consistency(all_responses)
        
        # Lower consistency = higher uncertainty
        uncertainty = 1.0 - consistency_score
        
        logger.debug(
            "semantic_consistency_computed",
            num_samples=len(all_responses),
            consistency=consistency_score,
            uncertainty=uncertainty,
        )
        
        return UncertaintyResult.from_score(
            score=uncertainty,
            method=UncertaintyMethod.SEMANTIC_CONSISTENCY,
            threshold=self.threshold,
            details={
                "num_samples": len(all_responses),
                "consistency_score": consistency_score,
                "sample_lengths": [len(r) for r in all_responses],
            },
        )
    
    async def _conformal_prediction(
        self,
        prompt: str,
        response: LLMResponse,
    ) -> UncertaintyResult:
        """Use conformal prediction for uncertainty estimation.
        
        Provides statistical coverage guarantees based on calibration data.
        Requires prior calibration with held-out examples.
        
        Args:
            prompt: Original prompt
            response: LLM response
        
        Returns:
            UncertaintyResult with conformal score
        """
        if not self._is_calibrated:
            logger.warning("conformal_not_calibrated_using_fallback")
            return await self._token_entropy(response)
        
        # Compute conformity score for this response
        base_result = await self._token_entropy(response)
        conformity_score = 1.0 - base_result.score
        
        # Calculate p-value from calibration data
        n_calibration = len(self._calibration_scores)
        n_larger = sum(1 for s in self._calibration_scores if s <= conformity_score)
        p_value = (n_larger + 1) / (n_calibration + 1)
        
        # High p-value indicates typical (confident) response
        # Low p-value indicates atypical (uncertain) response
        uncertainty = 1.0 - p_value
        
        return UncertaintyResult.from_score(
            score=uncertainty,
            method=UncertaintyMethod.CONFORMAL,
            threshold=self.threshold,
            details={
                "conformity_score": conformity_score,
                "p_value": p_value,
                "calibration_size": n_calibration,
                "alpha": self.conformal_alpha,
            },
        )
    
    async def _ensemble(
        self,
        prompt: str,
        response: LLMResponse,
    ) -> UncertaintyResult:
        """Combine multiple uncertainty methods.
        
        Uses weighted average of different methods for robust estimation.
        
        Args:
            prompt: Original prompt
            response: LLM response
        
        Returns:
            UncertaintyResult from ensemble
        """
        # Get results from different methods
        results = await asyncio.gather(
            self._token_entropy(response),
            self._semantic_consistency(prompt, response),
            return_exceptions=True,
        )
        
        scores = []
        weights = []
        details: dict[str, Any] = {}
        
        for i, result in enumerate(results):
            if isinstance(result, UncertaintyResult):
                method_name = result.method.value
                scores.append(result.score)
                # Weight token entropy lower (fast but less accurate)
                weights.append(0.4 if result.method == UncertaintyMethod.TOKEN_ENTROPY else 0.6)
                details[method_name] = result.score
        
        if not scores:
            # All methods failed
            return UncertaintyResult.from_score(
                score=0.5,
                method=UncertaintyMethod.ENSEMBLE,
                threshold=self.threshold,
                details={"fallback": "all_methods_failed"},
            )
        
        # Weighted average
        total_weight = sum(weights)
        ensemble_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        
        logger.debug(
            "ensemble_uncertainty_computed",
            scores=scores,
            weights=weights,
            ensemble=ensemble_score,
        )
        
        return UncertaintyResult.from_score(
            score=ensemble_score,
            method=UncertaintyMethod.ENSEMBLE,
            threshold=self.threshold,
            details={
                "component_scores": details,
                "weights": {m: w for m, w in zip(details.keys(), weights)},
            },
        )
    
    def calibrate(self, calibration_scores: list[float]) -> None:
        """Calibrate conformal prediction with held-out data.
        
        Args:
            calibration_scores: Conformity scores from calibration set
        """
        self._calibration_scores = sorted(calibration_scores)
        self._is_calibrated = True
        logger.info("uncertainty_quantifier_calibrated", n_samples=len(calibration_scores))
    
    def _heuristic_uncertainty(self, text: str) -> float:
        """Heuristic uncertainty based on response characteristics.
        
        Used as fallback when log probabilities are not available.
        
        Args:
            text: Response text
        
        Returns:
            Heuristic uncertainty score
        """
        uncertainty = 0.5  # Base uncertainty
        
        text_lower = text.lower()
        
        # Hedging language increases uncertainty
        hedging_phrases = [
            "i'm not sure",
            "i think",
            "possibly",
            "might be",
            "could be",
            "uncertain",
            "don't know",
            "not certain",
            "may or may not",
            "it depends",
        ]
        
        for phrase in hedging_phrases:
            if phrase in text_lower:
                uncertainty += 0.1
        
        # Confident language decreases uncertainty
        confident_phrases = [
            "definitely",
            "certainly",
            "absolutely",
            "without doubt",
            "clearly",
            "it is known",
        ]
        
        for phrase in confident_phrases:
            if phrase in text_lower:
                uncertainty -= 0.05
        
        # Very short responses are uncertain
        if len(text) < 50:
            uncertainty += 0.1
        
        # Very long responses might be rambling
        if len(text) > 1000:
            uncertainty += 0.05
        
        return min(max(uncertainty, 0.0), 1.0)
    
    def _calculate_text_consistency(self, texts: list[str]) -> float:
        """Calculate consistency score between multiple texts.
        
        Uses simple lexical overlap as a proxy for semantic similarity.
        For production, use embedding-based similarity.
        
        Args:
            texts: List of response texts
        
        Returns:
            Consistency score (0-1)
        """
        if len(texts) < 2:
            return 1.0
        
        # Tokenize (simple word-based)
        def tokenize(text: str) -> set[str]:
            return set(text.lower().split())
        
        token_sets = [tokenize(t) for t in texts]
        
        # Calculate pairwise Jaccard similarity
        similarities = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                intersection = len(token_sets[i] & token_sets[j])
                union = len(token_sets[i] | token_sets[j])
                if union > 0:
                    similarities.append(intersection / union)
        
        return float(np.mean(similarities)) if similarities else 1.0
