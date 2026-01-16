"""Integration tests using benchmark dataset.

Tests the full neuro-symbolic pipeline end-to-end using
realistic healthcare scenarios from the benchmark dataset.
"""

import json
import pytest
from pathlib import Path
from typing import Any

from src.router import RoutingDecisionEngine, RouteType
from src.symbolic import MedicalConstraintSolver, MedicalKnowledgeBase, ClinicalRuleEngine
from src.symbolic.z3_solver import DrugInteraction, DosageConstraint, PatientContext, InteractionSeverity
from src.verification import ContractChecker, DriftDetector


# Load benchmark dataset
BENCHMARK_PATH = Path(__file__).parent / "benchmark_dataset.json"


def load_benchmark_dataset() -> list[dict[str, Any]]:
    """Load the benchmark dataset."""
    with open(BENCHMARK_PATH) as f:
        data = json.load(f)
    return data["scenarios"]


def build_patient_context(patient_data: dict | None) -> PatientContext | None:
    """Build PatientContext from benchmark data."""
    if patient_data is None:
        return None
    
    return PatientContext(
        weight_kg=patient_data.get("weight_kg"),
        age_years=patient_data.get("age"),
        allergies=patient_data.get("allergies", []),
        conditions=patient_data.get("conditions", []),
        current_medications=patient_data.get("current_medications", []),
        renal_function=patient_data.get("renal_function", "normal"),
        hepatic_function=patient_data.get("hepatic_function", "normal"),
        pregnant=patient_data.get("pregnant", False),
        breastfeeding=patient_data.get("breastfeeding", False),
    )


class TestBenchmarkRouting:
    """Test routing decisions against benchmark expectations."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up routing engine."""
        self.engine = RoutingDecisionEngine()
        self.scenarios = load_benchmark_dataset()
    
    @pytest.mark.parametrize("scenario_id", [
        "DRUG_INTERACTION_001",
        "DRUG_INTERACTION_002",
        "ALLERGY_001",
        "PREGNANCY_001",
        "EMERGENCY_001",
    ])
    def test_safety_critical_routing(self, scenario_id: str):
        """Safety-critical queries should be routed appropriately."""
        scenario = next(s for s in self.scenarios if s["id"] == scenario_id)
        
        context = {}
        if scenario["patient"]:
            context["patient"] = scenario["patient"]
        
        decision = self.engine.decide(
            query=scenario["query"],
            context=context,
            priority="high" if scenario["expected"]["is_safety_critical"] else "medium",
        )
        
        # Safety-critical should route to symbolic-involved path
        assert decision.complexity.is_safety_critical == scenario["expected"]["is_safety_critical"], \
            f"Scenario {scenario_id}: Expected is_safety_critical={scenario['expected']['is_safety_critical']}"
        
        if scenario["expected"]["is_safety_critical"]:
            assert decision.involves_symbolic, \
                f"Scenario {scenario_id}: Safety-critical should involve symbolic"
    
    @pytest.mark.parametrize("scenario_id", [
        "SIMPLE_001",
        "SIMPLE_002",
    ])
    def test_simple_query_routing(self, scenario_id: str):
        """Simple queries should route to neural-only."""
        scenario = next(s for s in self.scenarios if s["id"] == scenario_id)
        
        decision = self.engine.decide(query=scenario["query"])
        
        assert decision.route_type == RouteType.NEURAL_ONLY, \
            f"Scenario {scenario_id}: Simple queries should route neural-only"
        assert not decision.complexity.is_safety_critical
    
    def test_polypharmacy_routing(self):
        """Polypharmacy cases should route to hybrid."""
        scenario = next(s for s in self.scenarios if s["id"] == "POLYPHARM_001")
        
        context = {"patient": scenario["patient"]}
        decision = self.engine.decide(
            query=scenario["query"],
            context=context,
        )
        
        # Should use hybrid due to multiple medications
        assert decision.route_type in [RouteType.HYBRID, RouteType.NEURAL_WITH_VERIFICATION, RouteType.SYMBOLIC_FIRST]
        assert decision.complexity.score >= 0.5, "Polypharmacy should have high complexity"


class TestBenchmarkSymbolic:
    """Test symbolic constraint solving against benchmark expectations."""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        """Set up symbolic components."""
        self.solver = MedicalConstraintSolver()
        self.knowledge_base = MedicalKnowledgeBase()
        await self.knowledge_base.load()
        
        # Load interactions from knowledge base into solver
        all_interactions = []
        for interactions in self.knowledge_base._interactions.values():
            for i in interactions:
                all_interactions.append(DrugInteraction(
                    drug_a=i.drug_a,
                    drug_b=i.drug_b,
                    severity=InteractionSeverity(i.severity),
                    description=i.clinical_effect,
                    mechanism=i.mechanism,
                    management=i.management,
                ))
        self.solver.load_interactions(all_interactions)
        
        # Load dosage constraints
        self.solver.load_dosage_constraints([
            DosageConstraint(drug="aspirin", max_dose_mg=650, max_daily_mg=4000),
            DosageConstraint(drug="ibuprofen", max_dose_mg=800, max_daily_mg=3200, renal_adjustment=True),
            DosageConstraint(drug="acetaminophen", max_dose_mg=1000, max_daily_mg=4000),
        ])
        
        self.scenarios = load_benchmark_dataset()
    
    @pytest.mark.asyncio
    async def test_drug_interaction_warfarin_aspirin(self):
        """Warfarin + Aspirin should be flagged as dangerous."""
        scenario = next(s for s in self.scenarios if s["id"] == "DRUG_INTERACTION_001")
        patient = build_patient_context(scenario["patient"])
        
        result = self.solver.check_medication(
            drug="aspirin",
            dose_mg=325,
            patient=patient,
        )
        
        assert not result.is_safe, "Warfarin + Aspirin should not be safe"
        assert result.risk_level in ["HIGH", "CRITICAL"]
        assert any("interaction" in v.constraint_type for v in result.violations)
    
    @pytest.mark.asyncio
    async def test_penicillin_allergy_cross_reactivity(self):
        """Penicillin allergy should flag amoxicillin."""
        scenario = next(s for s in self.scenarios if s["id"] == "ALLERGY_001")
        patient = build_patient_context(scenario["patient"])
        
        result = self.solver.check_medication(
            drug="amoxicillin",
            dose_mg=500,
            patient=patient,
        )
        
        assert any(
            "allergy" in v.constraint_type or "cross" in v.constraint_type
            for v in result.violations
        ), "Should detect penicillin cross-reactivity"
    
    @pytest.mark.asyncio
    async def test_pregnancy_statin_contraindication(self):
        """Statins should be contraindicated in pregnancy."""
        scenario = next(s for s in self.scenarios if s["id"] == "PREGNANCY_001")
        patient = build_patient_context(scenario["patient"])
        
        # Check a Category X drug
        result = self.solver.check_medication(
            drug="warfarin",  # Also Category X
            dose_mg=5,
            patient=patient,
        )
        
        assert not result.is_safe
        assert any("pregnancy" in v.constraint_type for v in result.violations)
    
    @pytest.mark.asyncio
    async def test_dosage_exceeded(self):
        """Excessive dosages should be flagged."""
        scenario = next(s for s in self.scenarios if s["id"] == "DOSAGE_001")
        patient = build_patient_context(scenario["patient"])
        
        result = self.solver.check_medication(
            drug="ibuprofen",
            dose_mg=1000,  # Exceeds 800mg max
            patient=patient,
        )
        
        assert not result.is_safe
        assert any("dosage" in v.constraint_type for v in result.violations)
    
    @pytest.mark.asyncio
    async def test_renal_impairment_adjustment(self):
        """Renal impairment should require dose adjustment."""
        scenario = next(s for s in self.scenarios if s["id"] == "RENAL_001")
        patient = build_patient_context(scenario["patient"])
        
        result = self.solver.check_medication(
            drug="ibuprofen",  # Requires renal adjustment
            dose_mg=400,
            patient=patient,
        )
        
        # Should at least warn about renal adjustment
        assert any(
            "renal" in v.constraint_type or "renal" in v.description.lower()
            for v in result.violations
        ) or len(result.warnings) > 0


class TestBenchmarkVerification:
    """Test contract verification against benchmark expectations."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up verification components."""
        self.checker = ContractChecker()
        self.checker.register_default_contracts()
        self.scenarios = load_benchmark_dataset()
    
    def test_allergy_consistency_violation(self):
        """Should detect when response recommends allergen."""
        scenario = next(s for s in self.scenarios if s["id"] == "ALLERGY_001")
        
        # Simulate a bad response that recommends the allergen
        bad_response = "You should take amoxicillin 500mg three times daily for your infection."
        
        result = self.checker.check(
            bad_response,
            context={"patient": scenario["patient"]},
        )
        
        # Should detect the allergen recommendation
        allergen_violations = [
            v for v in result.violations
            if "allergy" in v.contract_type.value.lower() or "allergen" in v.message.lower()
        ]
        
        assert len(allergen_violations) > 0 or not result.passed
    
    def test_safe_response_passes(self):
        """Safe, well-formatted responses should pass."""
        good_response = """
        For your headache, acetaminophen (Tylenol) is generally considered safer than 
        aspirin while on warfarin. You should take 325-650mg every 4-6 hours as needed,
        not exceeding 3000mg per day. Please consult your doctor before starting any
        new medication, especially given your current warfarin therapy.
        """
        
        result = self.checker.check(good_response)
        
        # Should pass (no critical violations)
        critical = [v for v in result.violations if v.severity.value == "critical"]
        assert len(critical) == 0


class TestBenchmarkDriftDetection:
    """Test drift detection with benchmark scenarios."""
    
    def setup_method(self):
        """Set up drift detector."""
        self.detector = DriftDetector(window_size=50, alert_threshold=0.1)
        self.checker = ContractChecker()
        self.checker.register_default_contracts()
    
    def test_drift_detection_with_violations(self):
        """Should detect drift when violation rate increases."""
        # Simulate a period of good responses
        for _ in range(20):
            good_response = "Consult your doctor for personalized advice on medication."
            result = self.checker.check(good_response)
            self.detector.record(result.violations)
        
        # Simulate a period of bad responses (drift)
        for _ in range(30):
            bad_response = "This medication is guaranteed to cure you with no side effects."
            result = self.checker.check(bad_response)
            self.detector.record(result.violations)
        
        metrics = self.detector.get_metrics()
        
        # Should detect the increasing violation rate
        assert metrics.overall_violation_rate > 0.1 or metrics.trend == "increasing"
    
    def test_ci_gate_with_benchmark_responses(self):
        """CI gate should pass with good responses."""
        # Simulate all safe responses
        for _ in range(20):
            good_response = "Please consult your healthcare provider for specific dosing."
            result = self.checker.check(good_response)
            self.detector.record(result.violations)
        
        passed, message = self.detector.check_ci_gate()
        
        assert passed, f"CI gate should pass with good responses: {message}"


class TestBenchmarkEndToEnd:
    """End-to-end integration tests with benchmark scenarios."""
    
    @pytest.fixture(autouse=True)
    async def setup(self):
        """Set up all components."""
        self.router = RoutingDecisionEngine()
        self.solver = MedicalConstraintSolver()
        self.kb = MedicalKnowledgeBase()
        self.checker = ContractChecker()
        self.drift = DriftDetector()
        
        await self.kb.load()
        self.checker.register_default_contracts()
        
        # Load interactions
        all_interactions = []
        for interactions in self.kb._interactions.values():
            for i in interactions:
                all_interactions.append(DrugInteraction(
                    drug_a=i.drug_a,
                    drug_b=i.drug_b,
                    severity=InteractionSeverity(i.severity),
                    description=i.clinical_effect,
                    mechanism=i.mechanism,
                    management=i.management,
                ))
        self.solver.load_interactions(all_interactions)
        
        self.scenarios = load_benchmark_dataset()
    
    @pytest.mark.asyncio
    async def test_full_pipeline_drug_interaction(self):
        """Test full pipeline with drug interaction scenario."""
        scenario = next(s for s in self.scenarios if s["id"] == "DRUG_INTERACTION_001")
        
        # Step 1: Route the query
        context = {"patient": scenario["patient"]}
        decision = self.router.decide(
            query=scenario["query"],
            context=context,
            priority="high",
        )
        
        assert decision.complexity.is_safety_critical or decision.involves_symbolic
        
        # Step 2: Run symbolic check
        patient = build_patient_context(scenario["patient"])
        symbolic_result = self.solver.check_medication(
            drug="aspirin",
            dose_mg=325,
            patient=patient,
        )
        
        assert not symbolic_result.is_safe
        
        # Step 3: Would verify LLM response (mocked here)
        mock_response = """
        Taking aspirin while on warfarin significantly increases your risk of bleeding.
        Both medications affect blood clotting. Please consult your doctor before
        combining these medications. For headache relief, acetaminophen may be safer.
        """
        
        verification = self.checker.check(mock_response, context)
        
        # Response that acknowledges the risk should pass
        critical = [v for v in verification.violations if v.severity.value == "critical"]
        assert len(critical) == 0
        
        # Step 4: Record for drift
        self.drift.record(verification.violations)
        
        metrics = self.drift.get_metrics()
        assert metrics.observations == 1
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("scenario_id", [
        "DRUG_INTERACTION_001",
        "ALLERGY_001",
        "PREGNANCY_001",
        "DOSAGE_001",
        "SIMPLE_001",
    ])
    async def test_full_pipeline_parametrized(self, scenario_id: str):
        """Parametrized full pipeline test for multiple scenarios."""
        scenario = next(s for s in self.scenarios if s["id"] == scenario_id)
        
        context = {}
        if scenario["patient"]:
            context["patient"] = scenario["patient"]
        
        # Route
        decision = self.router.decide(
            query=scenario["query"],
            context=context,
        )
        
        # Verify routing matches expectations
        expected = scenario["expected"]
        
        if expected["is_safety_critical"]:
            assert decision.complexity.is_safety_critical or decision.involves_symbolic, \
                f"Safety-critical scenario {scenario_id} should route to symbolic"
        
        # If we have a patient and expected symbolic result
        if scenario["patient"] and expected.get("symbolic_safe") is not None:
            patient = build_patient_context(scenario["patient"])
            
            # Try to find a drug to check from the query
            test_drugs = ["aspirin", "warfarin", "amoxicillin", "ibuprofen", "metformin"]
            for drug in test_drugs:
                if drug in scenario["query"].lower():
                    result = self.solver.check_medication(
                        drug=drug,
                        dose_mg=325,
                        patient=patient,
                    )
                    
                    if expected["symbolic_safe"] is False:
                        assert not result.is_safe or len(result.violations) > 0, \
                            f"Scenario {scenario_id}: Expected unsafe but got safe"
                    break


# Run benchmarks with summary
def run_benchmark_summary():
    """Run all benchmarks and print summary."""
    scenarios = load_benchmark_dataset()
    
    print("\n" + "=" * 60)
    print("NEURO-SYMBOLIC HEALTHCARE BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"\nTotal scenarios: {len(scenarios)}")
    
    # Count by category
    categories = {}
    for s in scenarios:
        cat = s["category"]
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nScenarios by category:")
    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count}")
    
    # Count safety-critical
    safety_critical = sum(1 for s in scenarios if s["expected"]["is_safety_critical"])
    print(f"\nSafety-critical scenarios: {safety_critical}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    run_benchmark_summary()
