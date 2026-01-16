"""Latency Budget Management for Real-Time Routing.

Manages latency budgets for query processing:
- Default budget allocation
- Dynamic budget adjustment based on query type
- SLA enforcement
- Timeout management
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


@dataclass
class LatencyBudget:
    """Represents a latency budget for a request."""
    
    total_ms: float
    start_time: float = field(default_factory=time.perf_counter)
    allocated: dict[str, float] = field(default_factory=dict)
    spent: dict[str, float] = field(default_factory=dict)
    
    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (time.perf_counter() - self.start_time) * 1000
    
    @property
    def remaining_ms(self) -> float:
        """Get remaining budget in milliseconds."""
        return max(0, self.total_ms - self.elapsed_ms)
    
    @property
    def is_expired(self) -> bool:
        """Check if budget is expired."""
        return self.remaining_ms <= 0
    
    @property
    def utilization(self) -> float:
        """Get budget utilization ratio (0-1)."""
        return min(1.0, self.elapsed_ms / self.total_ms)
    
    def allocate(self, component: str, ms: float) -> bool:
        """Allocate budget to a component.
        
        Args:
            component: Component name
            ms: Milliseconds to allocate
        
        Returns:
            True if allocation succeeded
        """
        if ms > self.remaining_ms:
            return False
        self.allocated[component] = ms
        return True
    
    def record_spent(self, component: str, ms: float) -> None:
        """Record time spent by a component.
        
        Args:
            component: Component name
            ms: Milliseconds spent
        """
        self.spent[component] = ms
    
    def get_allocation(self, component: str) -> float:
        """Get allocated budget for component."""
        return self.allocated.get(component, 0)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_ms": self.total_ms,
            "elapsed_ms": self.elapsed_ms,
            "remaining_ms": self.remaining_ms,
            "utilization": self.utilization,
            "allocated": self.allocated,
            "spent": self.spent,
        }


class LatencyManager:
    """Manages latency budgets for requests.
    
    Provides:
    - Budget creation with default values
    - Component allocation strategies
    - SLA monitoring
    - Budget recommendations based on query type
    
    Example:
        ```python
        manager = LatencyManager()
        budget = manager.create_budget(priority="high")
        
        # Allocate to components
        manager.allocate_for_neural(budget)
        manager.allocate_for_symbolic(budget)
        
        # Track usage
        start = time.perf_counter()
        # ... do work ...
        budget.record_spent("neural", (time.perf_counter() - start) * 1000)
        ```
    """
    
    # Priority-based default budgets (ms)
    PRIORITY_BUDGETS = {
        "critical": 10000,  # 10 seconds for critical decisions
        "high": 5000,  # 5 seconds default
        "medium": 3000,  # 3 seconds
        "low": 1000,  # 1 second for simple queries
    }
    
    # Component overhead estimates (ms)
    COMPONENT_OVERHEADS = {
        "neural": 500,  # LLM inference baseline
        "symbolic": 200,  # Solver baseline
        "verification": 100,  # Contract checking
        "routing": 50,  # Decision making
    }
    
    def __init__(self) -> None:
        """Initialize latency manager."""
        config = settings.router.latency
        self.default_budget_ms = config.default_budget_ms
        self.neural_overhead_ms = config.neural_overhead_ms
        self.symbolic_overhead_ms = config.symbolic_overhead_ms
        self.streaming_enabled = config.streaming_enabled
        
        # Statistics
        self._total_requests = 0
        self._sla_violations = 0
        self._latency_history: list[float] = []
    
    def create_budget(
        self,
        priority: str = "medium",
        custom_ms: float | None = None,
    ) -> LatencyBudget:
        """Create a new latency budget.
        
        Args:
            priority: Priority level (critical, high, medium, low)
            custom_ms: Custom budget in milliseconds
        
        Returns:
            New LatencyBudget
        """
        self._total_requests += 1
        
        if custom_ms is not None:
            total_ms = custom_ms
        else:
            total_ms = self.PRIORITY_BUDGETS.get(priority, self.default_budget_ms)
        
        budget = LatencyBudget(total_ms=total_ms)
        
        logger.debug(
            "budget_created",
            priority=priority,
            total_ms=total_ms,
        )
        
        return budget
    
    def allocate_for_route(
        self,
        budget: LatencyBudget,
        route_type: str,
    ) -> dict[str, float]:
        """Allocate budget based on route type.
        
        Args:
            budget: Latency budget
            route_type: Type of route (neural_only, symbolic_only, hybrid)
        
        Returns:
            Dictionary of component allocations
        """
        remaining = budget.remaining_ms
        allocations: dict[str, float] = {}
        
        # Reserve overhead for routing
        remaining -= self.COMPONENT_OVERHEADS["routing"]
        allocations["routing"] = self.COMPONENT_OVERHEADS["routing"]
        
        if route_type == "neural_only":
            # All remaining budget for neural
            allocations["neural"] = remaining
            
        elif route_type == "symbolic_only":
            # All remaining budget for symbolic
            allocations["symbolic"] = remaining
            
        elif route_type == "hybrid":
            # Split between neural and symbolic with verification
            verification_budget = self.COMPONENT_OVERHEADS["verification"]
            remaining -= verification_budget
            allocations["verification"] = verification_budget
            
            # 60/40 split favoring neural (faster for common cases)
            allocations["neural"] = remaining * 0.6
            allocations["symbolic"] = remaining * 0.4
            
        elif route_type == "neural_with_verification":
            # Primarily neural with verification overhead
            verification_budget = self.COMPONENT_OVERHEADS["verification"]
            remaining -= verification_budget
            allocations["verification"] = verification_budget
            allocations["neural"] = remaining
            
        elif route_type == "symbolic_first":
            # Symbolic first, then neural if time allows
            allocations["symbolic"] = remaining * 0.7
            allocations["neural"] = remaining * 0.3
        
        # Apply allocations
        for component, ms in allocations.items():
            budget.allocate(component, ms)
        
        logger.debug(
            "budget_allocated",
            route_type=route_type,
            allocations=allocations,
        )
        
        return allocations
    
    def should_use_streaming(self, budget: LatencyBudget) -> bool:
        """Determine if streaming should be used.
        
        Args:
            budget: Current latency budget
        
        Returns:
            True if streaming recommended
        """
        if not self.streaming_enabled:
            return False
        
        # Use streaming for longer operations
        return budget.total_ms >= 2000
    
    def check_sla(self, budget: LatencyBudget, sla_ms: float | None = None) -> bool:
        """Check if SLA was met.
        
        Args:
            budget: Completed budget
            sla_ms: SLA threshold (default: budget total)
        
        Returns:
            True if SLA was met
        """
        sla_ms = sla_ms or budget.total_ms
        met = budget.elapsed_ms <= sla_ms
        
        if not met:
            self._sla_violations += 1
            logger.warning(
                "sla_violation",
                elapsed_ms=budget.elapsed_ms,
                sla_ms=sla_ms,
            )
        
        self._latency_history.append(budget.elapsed_ms)
        
        # Keep history bounded
        if len(self._latency_history) > 1000:
            self._latency_history = self._latency_history[-1000:]
        
        return met
    
    def get_recommended_timeout(
        self,
        component: str,
        budget: LatencyBudget,
    ) -> float:
        """Get recommended timeout for a component.
        
        Args:
            component: Component name
            budget: Current budget
        
        Returns:
            Recommended timeout in seconds
        """
        allocated = budget.get_allocation(component)
        if allocated > 0:
            # Use allocation with some buffer
            return allocated / 1000 * 1.1
        
        # Fallback to remaining budget
        return max(0.1, budget.remaining_ms / 1000)
    
    def get_statistics(self) -> dict[str, Any]:
        """Get latency statistics.
        
        Returns:
            Dictionary of statistics
        """
        if not self._latency_history:
            avg_latency = 0
            p50 = 0
            p95 = 0
            p99 = 0
        else:
            import numpy as np
            avg_latency = np.mean(self._latency_history)
            p50 = np.percentile(self._latency_history, 50)
            p95 = np.percentile(self._latency_history, 95)
            p99 = np.percentile(self._latency_history, 99)
        
        sla_rate = (
            (self._total_requests - self._sla_violations) / self._total_requests
            if self._total_requests > 0
            else 1.0
        )
        
        return {
            "total_requests": self._total_requests,
            "sla_violations": self._sla_violations,
            "sla_compliance_rate": sla_rate,
            "avg_latency_ms": avg_latency,
            "p50_latency_ms": p50,
            "p95_latency_ms": p95,
            "p99_latency_ms": p99,
        }
