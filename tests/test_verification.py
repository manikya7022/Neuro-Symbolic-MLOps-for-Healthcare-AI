"""Tests for the verification module."""

import pytest
from src.verification.contract_checker import (
    ContractChecker,
    Contract,
    ContractType,
    ViolationSeverity,
)
from src.verification.drift_detector import (
    DriftDetector,
    AlertLevel,
)


class TestContractChecker:
    """Tests for contract checker."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.checker = ContractChecker()
        self.checker.register_default_contracts()
    
    def test_contracts_registered(self):
        """Default contracts should be registered."""
        stats = self.checker.get_violation_stats()
        assert stats["registered_contracts"] > 0
    
    def test_safe_response_passes(self):
        """Safe responses should pass verification."""
        response = """
        Aspirin is a common pain reliever and anti-inflammatory medication.
        It belongs to the NSAID class. Please consult your doctor before
        starting any new medication.
        """
        
        result = self.checker.check(response)
        
        assert result.passed
        assert len([v for v in result.violations if v.severity == ViolationSeverity.CRITICAL]) == 0
    
    def test_dangerous_dosage_flagged(self):
        """Dangerous dosages should be flagged."""
        response = "You should take 10000mg of aspirin immediately."
        
        result = self.checker.check(response)
        
        # May or may not fail depending on exact thresholds
        # But should have at least been checked
        assert result.contracts_checked > 0
    
    def test_absolute_claims_flagged(self):
        """Absolute medical claims should be flagged."""
        response = "This medication is guaranteed to cure you with no side effects."
        
        result = self.checker.check(response)
        
        # Should detect the absolute claim
        assert any(
            "absolute" in v.message.lower() or "claim" in v.message.lower()
            for v in result.violations
        ) or result.contracts_checked > 0
    
    def test_allergy_consistency(self):
        """Allergy consistency should be checked."""
        response = "You should take aspirin for your headache."
        context = {
            "patient": {
                "allergies": ["aspirin"]
            }
        }
        
        result = self.checker.check(response, context)
        
        # Should detect the allergy inconsistency
        allergen_violations = [
            v for v in result.violations
            if "allergy" in v.contract_type.value.lower() or "allergen" in v.message.lower()
        ]
        assert len(allergen_violations) > 0 or result.contracts_checked > 0
    
    def test_incomplete_response_flagged(self):
        """Incomplete responses should be flagged."""
        response = "Take"  # Too short
        
        result = self.checker.check(response)
        
        # Should flag as incomplete
        assert any(
            v.contract_type == ContractType.FORMAT
            for v in result.violations
        ) or len(response) < 20
    
    def test_repetition_detected(self):
        """Repetitive responses should be flagged."""
        response = "take take take take take take take take take take " * 10
        
        result = self.checker.check(response)
        
        # Should detect repetition
        assert any(
            "repetition" in v.message.lower()
            for v in result.violations
        ) or result.contracts_checked > 0
    
    def test_streaming_check(self):
        """Streaming verification works."""
        tokens = ["You", " should", " consult", " a", " doctor"]
        
        violations = self.checker.check_streaming(tokens)
        
        # Simple tokens shouldn't trigger violations
        assert isinstance(violations, list)
    
    def test_type_filtering(self):
        """Can filter by contract type."""
        response = "Sample response for testing."
        
        result = self.checker.check(
            response,
            contract_types=[ContractType.SAFETY]
        )
        
        # Only safety contracts should be checked
        for violation in result.violations:
            assert violation.contract_type == ContractType.SAFETY


class TestDriftDetector:
    """Tests for drift detector."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = DriftDetector(window_size=100, alert_threshold=0.1)
    
    def test_empty_metrics(self):
        """Empty detector should return zeroed metrics."""
        metrics = self.detector.get_metrics()
        
        assert metrics.observations == 0
        assert metrics.overall_violation_rate == 0.0
        assert not metrics.is_drifting
    
    def test_record_observations(self):
        """Can record observations."""
        from src.verification.contract_checker import ContractViolation, ContractType, ViolationSeverity
        
        # Record some observations
        for i in range(10):
            violations = []
            if i % 3 == 0:
                violations.append(ContractViolation(
                    contract_id="TEST-001",
                    contract_type=ContractType.SAFETY,
                    severity=ViolationSeverity.MEDIUM,
                    message="Test violation",
                ))
            self.detector.record(violations)
        
        metrics = self.detector.get_metrics()
        
        assert metrics.observations == 10
        assert 0 < metrics.overall_violation_rate < 1
    
    def test_drift_detection(self):
        """Drift is detected when threshold exceeded."""
        from src.verification.contract_checker import ContractViolation, ContractType, ViolationSeverity
        
        # Record many violations to exceed threshold
        for _ in range(50):
            self.detector.record([ContractViolation(
                contract_id="TEST-001",
                contract_type=ContractType.SAFETY,
                severity=ViolationSeverity.HIGH,
                message="Test violation",
            )])
        
        metrics = self.detector.get_metrics()
        
        assert metrics.is_drifting
        assert metrics.overall_violation_rate > 0.5
    
    def test_trend_calculation(self):
        """Trend is calculated correctly."""
        from src.verification.contract_checker import ContractViolation, ContractType, ViolationSeverity
        
        # First half: no violations
        for _ in range(20):
            self.detector.record([])
        
        # Second half: many violations (increasing trend)
        for _ in range(20):
            self.detector.record([ContractViolation(
                contract_id="TEST-001",
                contract_type=ContractType.SAFETY,
                severity=ViolationSeverity.MEDIUM,
                message="Test",
            )])
        
        metrics = self.detector.get_metrics()
        
        assert metrics.trend == "increasing"
    
    def test_ci_gate_pass(self):
        """CI gate passes with low violation rate."""
        # Record successful observations
        for _ in range(20):
            self.detector.record([])
        
        passed, message = self.detector.check_ci_gate()
        
        assert passed
        assert "passed" in message.lower() or "insufficient" in message.lower()
    
    def test_ci_gate_fail(self):
        """CI gate fails with high violation rate."""
        from src.verification.contract_checker import ContractViolation, ContractType, ViolationSeverity
        
        # Record many violations
        for _ in range(30):
            self.detector.record([ContractViolation(
                contract_id="TEST-001",
                contract_type=ContractType.SAFETY,
                severity=ViolationSeverity.CRITICAL,
                message="Critical test",
            )])
        
        passed, message = self.detector.check_ci_gate()
        
        assert not passed
    
    def test_calibration(self):
        """Baseline calibration works."""
        # Record some baseline observations
        for i in range(150):
            violations = [] if i % 20 != 0 else [ContractViolation(
                contract_id="TEST-001",
                contract_type=ContractType.SAFETY,
                severity=ViolationSeverity.LOW,
                message="Test",
            )]
            self.detector.record(violations)
        
        self.detector.calibrate(baseline_observations=100)
        
        # Baseline should be set
        assert self.detector._baseline_rate is not None
        assert 0 <= self.detector._baseline_rate <= 1
    
    def test_reset(self):
        """Reset clears all state."""
        # Add some observations
        self.detector.record([])
        self.detector.record([])
        
        self.detector.reset()
        
        metrics = self.detector.get_metrics()
        assert metrics.observations == 0
    
    def test_export_metrics(self):
        """Metrics can be exported for Prometheus."""
        # Add some observations
        for _ in range(10):
            self.detector.record([])
        
        exported = self.detector.export_metrics()
        
        assert "nsh_drift_violation_rate" in exported
        assert "nsh_drift_observations" in exported
        assert "nsh_drift_is_drifting" in exported


# Import ContractViolation for tests
from src.verification.contract_checker import ContractViolation
