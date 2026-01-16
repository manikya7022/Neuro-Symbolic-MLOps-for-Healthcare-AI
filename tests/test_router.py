"""Tests for the routing module."""

import pytest
from src.router.complexity_estimator import ComplexityEstimator, ComplexityResult
from src.router.latency_manager import LatencyManager, LatencyBudget
from src.router.decision_engine import RoutingDecisionEngine, RouteType, RoutingStrategy


class TestComplexityEstimator:
    """Tests for ComplexityEstimator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.estimator = ComplexityEstimator()
    
    def test_simple_query_low_complexity(self):
        """Simple queries should have low complexity."""
        result = self.estimator.estimate("What is aspirin?")
        assert result.category in ["low", "medium"]
        assert result.score < 0.5
        assert not result.is_safety_critical
    
    def test_safety_critical_keywords(self):
        """Queries with safety keywords should be flagged."""
        result = self.estimator.estimate("Is this drug combination lethal?")
        assert result.is_safety_critical
        assert result.category == "critical"
    
    def test_drug_interaction_query(self):
        """Drug interaction queries should have higher complexity."""
        result = self.estimator.estimate(
            "Can I take aspirin with my blood thinner warfarin while pregnant?"
        )
        assert result.is_safety_critical or result.category in ["high", "critical"]
    
    def test_context_affects_complexity(self):
        """Patient context should increase complexity."""
        base_result = self.estimator.estimate("What is the dose for ibuprofen?")
        
        context = {
            "patient": {
                "conditions": ["kidney_disease", "heart_failure"],
                "current_medications": ["warfarin", "lisinopril", "metformin"],
            }
        }
        context_result = self.estimator.estimate("What is the dose for ibuprofen?", context)
        
        assert context_result.score >= base_result.score
    
    def test_batch_estimation(self):
        """Batch estimation should work correctly."""
        queries = [
            "What is aspirin?",
            "Is warfarin dangerous?",
            "What are the side effects?",
        ]
        results = self.estimator.batch_estimate(queries)
        
        assert len(results) == 3
        assert all(isinstance(r, ComplexityResult) for r in results)


class TestLatencyManager:
    """Tests for LatencyManager."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.manager = LatencyManager()
    
    def test_create_budget_default(self):
        """Default budget creation works."""
        budget = self.manager.create_budget()
        assert budget.total_ms > 0
        assert budget.remaining_ms <= budget.total_ms
    
    def test_create_budget_priority(self):
        """Priority affects budget size."""
        critical_budget = self.manager.create_budget(priority="critical")
        low_budget = self.manager.create_budget(priority="low")
        
        assert critical_budget.total_ms > low_budget.total_ms
    
    def test_create_budget_custom(self):
        """Custom budget overrides default."""
        budget = self.manager.create_budget(custom_ms=1234)
        assert budget.total_ms == 1234
    
    def test_budget_allocation(self):
        """Budget allocation works for different routes."""
        budget = self.manager.create_budget()
        
        allocations = self.manager.allocate_for_route(budget, "hybrid")
        
        assert "neural" in allocations
        assert "symbolic" in allocations
        assert "verification" in allocations
    
    def test_budget_expiration(self):
        """Budget expiration is tracked correctly."""
        budget = self.manager.create_budget(custom_ms=100)
        
        import time
        time.sleep(0.15)  # Wait longer than budget
        
        assert budget.is_expired
        assert budget.remaining_ms == 0


class TestRoutingDecisionEngine:
    """Tests for RoutingDecisionEngine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = RoutingDecisionEngine()
    
    def test_simple_query_routes_neural(self):
        """Simple queries should route to neural-only or neural-with-verification."""
        decision = self.engine.decide("What is aspirin?")
        
        assert decision.route_type in [
            RouteType.NEURAL_ONLY,
            RouteType.NEURAL_WITH_VERIFICATION,
        ]
    
    def test_safety_critical_routes_symbolic_first(self):
        """Safety-critical queries should route symbolic-first."""
        decision = self.engine.decide(
            "Is this lethal dosage of medication?",
            priority="critical",
        )
        
        assert decision.route_type in [
            RouteType.SYMBOLIC_FIRST,
            RouteType.HYBRID,
            RouteType.NEURAL_WITH_VERIFICATION,
        ]
    
    def test_pregnant_patient_context(self):
        """Pregnancy context should trigger higher verification."""
        decision = self.engine.decide(
            "What medication can I take for headache?",
            context={"patient": {"pregnant": True}},
        )
        
        assert decision.involves_symbolic
    
    def test_polypharmacy_context(self):
        """Multiple medications should trigger hybrid routing."""
        decision = self.engine.decide(
            "What can I take for pain?",
            context={
                "patient": {
                    "current_medications": [
                        "warfarin", "lisinopril", "metformin",
                        "atorvastatin", "omeprazole", "aspirin"
                    ]
                }
            },
        )
        
        assert decision.route_type in [RouteType.HYBRID, RouteType.NEURAL_WITH_VERIFICATION]
    
    def test_decision_has_budget(self):
        """Decision should include allocated budget."""
        decision = self.engine.decide("What is aspirin?")
        
        assert decision.budget is not None
        assert decision.budget.total_ms > 0
    
    def test_fallback_route_set(self):
        """Fallback route should be set for primary routes."""
        decision = self.engine.decide("What is aspirin?")
        
        # Most routes should have a fallback
        if decision.route_type != RouteType.HYBRID:
            assert decision.fallback_route is not None
    
    def test_statistics_tracking(self):
        """Engine should track routing statistics."""
        # Make some decisions
        self.engine.decide("Query 1")
        self.engine.decide("Query 2")
        self.engine.decide("Query 3")
        
        stats = self.engine.get_statistics()
        
        assert stats["total_decisions"] == 3


class TestRoutingStrategies:
    """Test different routing strategies."""
    
    def test_neural_first_strategy(self):
        """Neural-first strategy prefers neural path."""
        engine = RoutingDecisionEngine(strategy=RoutingStrategy.NEURAL_FIRST)
        decision = engine.decide("Simple question")
        
        assert decision.involves_neural
    
    def test_symbolic_first_strategy(self):
        """Symbolic-first strategy prefers symbolic path."""
        engine = RoutingDecisionEngine(strategy=RoutingStrategy.SYMBOLIC_FIRST)
        decision = engine.decide("Simple question")
        
        assert decision.involves_symbolic
    
    def test_fast_strategy(self):
        """Fast strategy optimizes for latency."""
        engine = RoutingDecisionEngine(strategy=RoutingStrategy.FAST)
        decision = engine.decide("Simple question")
        
        # Fast mode should use neural-only for simple queries
        assert decision.route_type == RouteType.NEURAL_ONLY
    
    def test_hybrid_strategy(self):
        """Hybrid strategy always uses both tracks."""
        engine = RoutingDecisionEngine(strategy=RoutingStrategy.HYBRID)
        decision = engine.decide("Any question")
        
        assert decision.route_type == RouteType.HYBRID
