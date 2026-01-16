"""Routing Decision Engine for Neural-Symbolic Query Routing.

Core routing logic that determines how to process each query:
- Neural only (fast path)
- Symbolic only (guaranteed correctness)
- Hybrid (neural + symbolic verification)
- Neural with verification (neural + contract checking)
- Symbolic first (for safety-critical queries)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from src.config import settings
from src.router.complexity_estimator import ComplexityEstimator, ComplexityResult
from src.router.latency_manager import LatencyBudget, LatencyManager

logger = structlog.get_logger(__name__)


class RouteType(str, Enum):
    """Types of routing paths."""
    
    NEURAL_ONLY = "neural_only"
    SYMBOLIC_ONLY = "symbolic_only"
    HYBRID = "hybrid"
    NEURAL_WITH_VERIFICATION = "neural_with_verification"
    SYMBOLIC_FIRST = "symbolic_first"


class RoutingStrategy(str, Enum):
    """Routing strategies."""
    
    ADAPTIVE = "adaptive"  # Choose based on query analysis
    NEURAL_FIRST = "neural_first"  # Prefer neural, verify if uncertain
    SYMBOLIC_FIRST = "symbolic_first"  # Prefer symbolic for correctness
    HYBRID = "hybrid"  # Always use both
    FAST = "fast"  # Optimize for latency


@dataclass
class RoutingDecision:
    """Represents a routing decision."""
    
    route_type: RouteType
    strategy_used: RoutingStrategy
    complexity: ComplexityResult
    budget: LatencyBudget
    requires_streaming: bool = False
    fallback_route: RouteType | None = None
    confidence: float = 1.0
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    
    @property
    def involves_neural(self) -> bool:
        """Check if route uses neural component."""
        return self.route_type in {
            RouteType.NEURAL_ONLY,
            RouteType.HYBRID,
            RouteType.NEURAL_WITH_VERIFICATION,
        }
    
    @property
    def involves_symbolic(self) -> bool:
        """Check if route uses symbolic component."""
        return self.route_type in {
            RouteType.SYMBOLIC_ONLY,
            RouteType.HYBRID,
            RouteType.SYMBOLIC_FIRST,
            RouteType.NEURAL_WITH_VERIFICATION,
        }


class RoutingDecisionEngine:
    """Engine for making routing decisions.
    
    Analyzes queries and determines optimal routing path based on:
    - Query complexity
    - Safety requirements
    - Latency budget
    - System configuration
    
    Example:
        ```python
        engine = RoutingDecisionEngine()
        
        decision = engine.decide(
            query="Can I take aspirin with my warfarin?",
            context={"patient": patient_info}
        )
        
        if decision.route_type == RouteType.SYMBOLIC_FIRST:
            # Safety-critical - verify first
            result = await symbolic_solver.check(...)
        ```
    """
    
    def __init__(
        self,
        strategy: RoutingStrategy | str | None = None,
    ) -> None:
        """Initialize routing engine.
        
        Args:
            strategy: Default routing strategy
        """
        config = settings.router
        self.strategy = RoutingStrategy(strategy or config.strategy)
        self.safety_config = config.safety
        
        self.complexity_estimator = ComplexityEstimator()
        self.latency_manager = LatencyManager()
        
        # Statistics
        self._decisions: dict[RouteType, int] = {rt: 0 for rt in RouteType}
    
    def decide(
        self,
        query: str,
        context: dict[str, Any] | None = None,
        priority: str = "medium",
        custom_budget_ms: float | None = None,
    ) -> RoutingDecision:
        """Make a routing decision for a query.
        
        Args:
            query: User query
            context: Optional context (patient info, etc.)
            priority: Query priority level
            custom_budget_ms: Custom latency budget
        
        Returns:
            RoutingDecision with chosen route
        """
        context = context or {}
        
        # 1. Create latency budget
        budget = self.latency_manager.create_budget(
            priority=priority,
            custom_ms=custom_budget_ms,
        )
        
        # 2. Estimate complexity
        complexity = self.complexity_estimator.estimate(query, context)
        
        # 3. Determine route based on strategy
        if self.strategy == RoutingStrategy.ADAPTIVE:
            route_type, reason = self._adaptive_routing(query, complexity, budget, context)
        elif self.strategy == RoutingStrategy.NEURAL_FIRST:
            route_type, reason = self._neural_first_routing(complexity, budget)
        elif self.strategy == RoutingStrategy.SYMBOLIC_FIRST:
            route_type, reason = self._symbolic_first_routing(complexity, budget)
        elif self.strategy == RoutingStrategy.HYBRID:
            route_type, reason = RouteType.HYBRID, "Strategy forces hybrid"
        elif self.strategy == RoutingStrategy.FAST:
            route_type, reason = self._fast_routing(complexity, budget)
        else:
            route_type, reason = RouteType.NEURAL_WITH_VERIFICATION, "Default route"
        
        # 4. Allocate budget for chosen route
        self.latency_manager.allocate_for_route(budget, route_type.value)
        
        # 5. Determine if streaming should be used
        requires_streaming = self.latency_manager.should_use_streaming(budget)
        
        # 6. Set fallback route
        fallback_route = self._determine_fallback(route_type)
        
        # 7. Calculate confidence in routing decision
        confidence = self._calculate_routing_confidence(complexity, route_type)
        
        decision = RoutingDecision(
            route_type=route_type,
            strategy_used=self.strategy,
            complexity=complexity,
            budget=budget,
            requires_streaming=requires_streaming,
            fallback_route=fallback_route,
            confidence=confidence,
            reason=reason,
            details={
                "query_preview": query[:100],
                "context_keys": list(context.keys()),
            },
        )
        
        # Update statistics
        self._decisions[route_type] += 1
        
        logger.info(
            "routing_decision",
            route_type=route_type.value,
            complexity_category=complexity.category,
            is_safety_critical=complexity.is_safety_critical,
            confidence=confidence,
            reason=reason,
        )
        
        return decision
    
    def _adaptive_routing(
        self,
        query: str,
        complexity: ComplexityResult,
        budget: LatencyBudget,
        context: dict[str, Any],
    ) -> tuple[RouteType, str]:
        """Adaptive routing based on query analysis.
        
        Args:
            query: User query
            complexity: Complexity result
            budget: Latency budget
            context: Query context
        
        Returns:
            Tuple of (route_type, reason)
        """
        # Safety-critical queries always go symbolic first
        if complexity.is_safety_critical:
            return RouteType.SYMBOLIC_FIRST, "Safety-critical query detected"
        
        # Check for explicit safety keywords
        query_lower = query.lower()
        for keyword in self.safety_config.critical_keywords:
            if keyword in query_lower:
                if self.safety_config.always_verify:
                    return RouteType.NEURAL_WITH_VERIFICATION, f"Safety keyword '{keyword}' detected"
        
        # Check context for high-risk factors
        if context.get("patient"):
            patient = context["patient"]
            if patient.get("pregnant") or patient.get("breastfeeding"):
                return RouteType.SYMBOLIC_FIRST, "Pregnancy/breastfeeding context"
            if patient.get("age", 100) < 18:
                return RouteType.NEURAL_WITH_VERIFICATION, "Pediatric patient"
            if len(patient.get("current_medications", [])) >= 5:
                return RouteType.HYBRID, "Polypharmacy context"
        
        # Route based on complexity
        if complexity.category == "critical":
            return RouteType.SYMBOLIC_FIRST, "Critical complexity"
        elif complexity.category == "high":
            return RouteType.HYBRID, "High complexity"
        elif complexity.category == "medium":
            # Check budget - if tight, go neural only
            if budget.total_ms < 2000:
                return RouteType.NEURAL_ONLY, "Medium complexity with tight budget"
            return RouteType.NEURAL_WITH_VERIFICATION, "Medium complexity"
        else:
            # Low complexity - fast neural path
            return RouteType.NEURAL_ONLY, "Low complexity query"
    
    def _neural_first_routing(
        self,
        complexity: ComplexityResult,
        budget: LatencyBudget,
    ) -> tuple[RouteType, str]:
        """Neural-first routing strategy."""
        if complexity.is_safety_critical:
            return RouteType.NEURAL_WITH_VERIFICATION, "Neural-first with safety verification"
        elif complexity.requires_symbolic:
            return RouteType.NEURAL_WITH_VERIFICATION, "Neural-first with symbolic verification"
        else:
            return RouteType.NEURAL_ONLY, "Neural-first fast path"
    
    def _symbolic_first_routing(
        self,
        complexity: ComplexityResult,
        budget: LatencyBudget,
    ) -> tuple[RouteType, str]:
        """Symbolic-first routing strategy."""
        if complexity.category == "low" and budget.total_ms < 2000:
            return RouteType.SYMBOLIC_ONLY, "Symbolic-first with simple query"
        else:
            return RouteType.SYMBOLIC_FIRST, "Symbolic-first strategy"
    
    def _fast_routing(
        self,
        complexity: ComplexityResult,
        budget: LatencyBudget,
    ) -> tuple[RouteType, str]:
        """Fast routing strategy - optimize for latency."""
        if complexity.is_safety_critical:
            # Even in fast mode, don't skip safety checks
            return RouteType.NEURAL_WITH_VERIFICATION, "Fast mode with safety check"
        else:
            return RouteType.NEURAL_ONLY, "Fast mode - neural only"
    
    def _determine_fallback(self, primary_route: RouteType) -> RouteType | None:
        """Determine fallback route if primary fails."""
        fallbacks = {
            RouteType.NEURAL_ONLY: RouteType.NEURAL_WITH_VERIFICATION,
            RouteType.NEURAL_WITH_VERIFICATION: RouteType.HYBRID,
            RouteType.HYBRID: RouteType.SYMBOLIC_ONLY,
            RouteType.SYMBOLIC_FIRST: RouteType.HYBRID,
            RouteType.SYMBOLIC_ONLY: RouteType.NEURAL_WITH_VERIFICATION,
        }
        return fallbacks.get(primary_route)
    
    def _calculate_routing_confidence(
        self,
        complexity: ComplexityResult,
        route_type: RouteType,
    ) -> float:
        """Calculate confidence in routing decision."""
        base_confidence = 0.8
        
        # Higher confidence for clear-cut cases
        if complexity.is_safety_critical and route_type in {RouteType.SYMBOLIC_FIRST, RouteType.HYBRID}:
            base_confidence += 0.15
        elif complexity.category == "low" and route_type == RouteType.NEURAL_ONLY:
            base_confidence += 0.15
        
        # Lower confidence for edge cases
        if complexity.category == "medium":
            base_confidence -= 0.1
        
        return min(max(base_confidence, 0.5), 1.0)
    
    def get_statistics(self) -> dict[str, Any]:
        """Get routing statistics."""
        total = sum(self._decisions.values())
        
        return {
            "strategy": self.strategy.value,
            "total_decisions": total,
            "decisions_by_route": {rt.value: count for rt, count in self._decisions.items()},
            "latency_stats": self.latency_manager.get_statistics(),
        }
