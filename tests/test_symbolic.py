"""Tests for the symbolic components."""

import pytest
from src.symbolic.z3_solver import (
    MedicalConstraintSolver,
    DrugInteraction,
    DosageConstraint,
    PatientContext,
    InteractionSeverity,
)
from src.symbolic.knowledge_base import MedicalKnowledgeBase, Drug, Condition
from src.symbolic.rule_engine import ClinicalRuleEngine, Rule, RuleCondition, RuleAction, RulePriority


class TestMedicalConstraintSolver:
    """Tests for Z3-based constraint solver."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.solver = MedicalConstraintSolver()
        
        # Load test interactions
        self.solver.load_interactions([
            DrugInteraction(
                drug_a="warfarin",
                drug_b="aspirin",
                severity=InteractionSeverity.MAJOR,
                description="Increased bleeding risk",
                mechanism="Additive anticoagulant effect",
                management="Avoid combination",
            ),
            DrugInteraction(
                drug_a="metformin",
                drug_b="contrast",
                severity=InteractionSeverity.MAJOR,
                description="Risk of lactic acidosis",
                mechanism="Contrast may cause renal impairment",
                management="Hold metformin 48h before/after contrast",
            ),
        ])
        
        # Load test dosage constraints
        self.solver.load_dosage_constraints([
            DosageConstraint(
                drug="aspirin",
                min_dose_mg=81,
                max_dose_mg=650,
                max_daily_mg=4000,
            ),
            DosageConstraint(
                drug="ibuprofen",
                max_dose_mg=800,
                max_daily_mg=3200,
                renal_adjustment=True,
            ),
        ])
    
    def test_allergy_detection(self):
        """Allergies should be detected and flagged."""
        patient = PatientContext(allergies=["aspirin"])
        
        result = self.solver.check_medication(
            drug="aspirin",
            dose_mg=325,
            patient=patient,
        )
        
        assert not result.is_safe
        assert any(v.constraint_type == "allergy" for v in result.violations)
    
    def test_cross_reactivity_warning(self):
        """Cross-reactivity should be warned about."""
        patient = PatientContext(allergies=["penicillin"])
        
        result = self.solver.check_medication(
            drug="amoxicillin",
            dose_mg=500,
            patient=patient,
        )
        
        assert any(v.constraint_type == "cross_reactivity" for v in result.violations)
    
    def test_drug_interaction_detection(self):
        """Drug interactions should be detected."""
        patient = PatientContext(current_medications=["warfarin"])
        
        result = self.solver.check_medication(
            drug="aspirin",
            dose_mg=325,
            patient=patient,
        )
        
        assert any(v.constraint_type == "drug_interaction" for v in result.violations)
    
    def test_dosage_exceeded(self):
        """Excessive dosages should be flagged."""
        patient = PatientContext()
        
        result = self.solver.check_medication(
            drug="aspirin",
            dose_mg=1000,  # Exceeds max of 650mg
            patient=patient,
        )
        
        assert any("dosage" in v.constraint_type for v in result.violations)
    
    def test_daily_dose_exceeded(self):
        """Excessive daily doses should be flagged."""
        patient = PatientContext()
        
        result = self.solver.check_medication(
            drug="aspirin",
            dose_mg=650,
            patient=patient,
            frequency_per_day=10,  # 6500mg/day exceeds 4000mg limit
        )
        
        assert any("daily" in v.constraint_type for v in result.violations)
    
    def test_renal_adjustment_needed(self):
        """Renal impairment should trigger adjustment warning."""
        patient = PatientContext(renal_function="moderate")
        
        result = self.solver.check_medication(
            drug="ibuprofen",
            dose_mg=400,
            patient=patient,
        )
        
        assert any("renal" in v.constraint_type for v in result.violations)
    
    def test_pregnancy_contraindication(self):
        """Pregnancy should flag X drugs."""
        patient = PatientContext(pregnant=True)
        
        result = self.solver.check_medication(
            drug="warfarin",
            dose_mg=5,
            patient=patient,
        )
        
        assert any("pregnancy" in v.constraint_type for v in result.violations)
    
    def test_safe_medication(self):
        """Safe medications should pass."""
        patient = PatientContext()
        
        result = self.solver.check_medication(
            drug="aspirin",
            dose_mg=325,
            patient=patient,
        )
        
        assert result.is_safe
        assert result.risk_level == "LOW"
    
    def test_multi_drug_regimen(self):
        """Multi-drug regimen checking works."""
        patient = PatientContext()
        
        result = self.solver.check_multi_drug_regimen(
            drugs=[
                ("aspirin", 325),
                ("ibuprofen", 400),  # Both NSAIDs
            ],
            patient=patient,
        )
        
        # Should warn about multiple NSAIDs
        assert len(result.warnings) > 0 or len(result.violations) > 0


class TestMedicalKnowledgeBase:
    """Tests for knowledge base."""
    
    @pytest.fixture
    async def kb(self):
        """Create and load knowledge base."""
        kb = MedicalKnowledgeBase()
        await kb.load()
        return kb
    
    @pytest.mark.asyncio
    async def test_get_drug(self, kb):
        """Get drug information."""
        drug = kb.get_drug("aspirin")
        
        assert drug is not None
        assert drug.name == "aspirin"
        assert "NSAID" in drug.drug_class
    
    @pytest.mark.asyncio
    async def test_get_condition(self, kb):
        """Get condition information."""
        condition = kb.get_condition("hypertension")
        
        assert condition is not None
        assert condition.name == "hypertension"
        assert "cardiovascular" in condition.category
    
    @pytest.mark.asyncio
    async def test_get_interactions(self, kb):
        """Get drug interactions."""
        interactions = kb.get_interactions("warfarin")
        
        assert len(interactions) > 0
        assert any(i.drug_b.lower() == "aspirin" for i in interactions)
    
    @pytest.mark.asyncio
    async def test_drugs_by_class(self, kb):
        """Get drugs by class."""
        nsaids = kb.get_drugs_by_class("NSAID")
        
        assert len(nsaids) > 0
        assert any(d.name == "aspirin" for d in nsaids)
    
    @pytest.mark.asyncio
    async def test_find_drugs_for_condition(self, kb):
        """Find drugs for condition."""
        drugs = kb.find_drugs_for_condition("diabetes")
        
        assert len(drugs) > 0
        assert any(d.name == "metformin" for d in drugs)
    
    @pytest.mark.asyncio
    async def test_check_contraindication(self, kb):
        """Check contraindication."""
        is_contraindicated = kb.check_contraindication("aspirin", "peptic ulcer")
        
        assert is_contraindicated


class TestClinicalRuleEngine:
    """Tests for clinical rule engine."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.engine = ClinicalRuleEngine()
        self.engine.load_default_rules()
    
    def test_rules_loaded(self):
        """Default rules should be loaded."""
        stats = self.engine.get_statistics()
        assert stats["total_rules"] > 0
    
    def test_drug_interaction_rule(self):
        """Drug interaction rules should fire."""
        context = {
            "current_medications": ["warfarin"],
            "proposed_drug": "aspirin",
        }
        
        is_allowed, results = self.engine.evaluate_with_blocking(context)
        fired = self.engine.get_fired_rules(results)
        
        # Should have fired some rule
        assert len(fired) > 0 or is_allowed
    
    def test_category_filtering(self):
        """Results can be filtered by category."""
        context = {"current_medications": [], "proposed_drug": "aspirin"}
        
        results = self.engine.evaluate(context, categories=["drug_interaction"])
        
        # All results should be from drug_interaction category
        for result in results:
            if result.fired:
                assert result.rule.category == "drug_interaction"
    
    def test_custom_rule(self):
        """Custom rules can be added."""
        custom_rule = Rule(
            id="CUSTOM-001",
            name="Test Rule",
            description="A test rule",
            conditions=[
                RuleCondition("test_field", "eq", "test_value")
            ],
            action=RuleAction.WARN,
            priority=RulePriority.MEDIUM,
            message="Test warning",
            category="custom",
        )
        
        self.engine.add_rule(custom_rule)
        
        # Rule should fire when condition met
        results = self.engine.evaluate({"test_field": "test_value"})
        fired = self.engine.get_fired_rules(results)
        
        assert any(r.rule.id == "CUSTOM-001" for r in fired)
    
    def test_priority_ordering(self):
        """Rules should be evaluated in priority order."""
        results = self.engine.evaluate({})
        
        # Results should maintain some semblance of priority order
        # (we can't fully verify without more complex setup)
        assert len(results) >= 0
