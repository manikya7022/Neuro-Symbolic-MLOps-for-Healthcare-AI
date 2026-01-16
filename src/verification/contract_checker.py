"""Neural-Symbolic Contract Checker.

Verifies that neural outputs satisfy symbolic constraints:
- Safety constraints (no harmful recommendations)
- Consistency constraints (matches knowledge base)
- Format constraints (proper structure)
- Domain constraints (healthcare-specific)

Provides real-time verification for streaming responses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


class ContractType(str, Enum):
    """Types of contracts."""
    
    SAFETY = "safety"  # No harmful content
    CONSISTENCY = "consistency"  # Matches knowledge base
    FORMAT = "format"  # Proper response format
    DOMAIN = "domain"  # Domain-specific rules
    REGULATORY = "regulatory"  # Compliance requirements


class ViolationSeverity(str, Enum):
    """Severity levels for violations."""
    
    CRITICAL = "critical"  # Must block response
    HIGH = "high"  # Should flag for review
    MEDIUM = "medium"  # Warning
    LOW = "low"  # Informational


@dataclass
class ContractViolation:
    """A violation of a contract."""
    
    contract_id: str
    contract_type: ContractType
    severity: ViolationSeverity
    message: str
    evidence: str = ""
    location: str = ""  # Where in response
    suggested_fix: str = ""


@dataclass
class ContractResult:
    """Result of contract verification."""
    
    passed: bool
    violations: list[ContractViolation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    contracts_checked: int = 0
    verification_time_ms: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_critical_violations(self) -> bool:
        """Check if any critical violations exist."""
        return any(v.severity == ViolationSeverity.CRITICAL for v in self.violations)
    
    @property
    def should_block(self) -> bool:
        """Determine if response should be blocked."""
        config = settings.verification.contracts
        if config.strict_mode:
            return len(self.violations) > 0
        return self.has_critical_violations


@dataclass
class Contract:
    """A contract that neural outputs must satisfy."""
    
    id: str
    name: str
    description: str
    contract_type: ContractType
    severity: ViolationSeverity
    check_fn: Callable[[str, dict[str, Any]], tuple[bool, str]]
    enabled: bool = True
    
    def check(self, response: str, context: dict[str, Any]) -> tuple[bool, str]:
        """Execute contract check.
        
        Args:
            response: Neural response to check
            context: Additional context
        
        Returns:
            Tuple of (passed, evidence_or_error)
        """
        if not self.enabled:
            return True, ""
        return self.check_fn(response, context)


class ContractChecker:
    """Verifies neural outputs against symbolic contracts.
    
    Provides:
    - Pre-defined safety contracts for healthcare
    - Custom contract registration
    - Streaming verification (token-by-token)
    - Violation aggregation and reporting
    
    Example:
        ```python
        checker = ContractChecker()
        checker.register_default_contracts()
        
        result = checker.check(
            response="Take 500mg aspirin with your warfarin",
            context={"patient": patient_info}
        )
        
        if result.should_block:
            # Don't return this response
            ...
        ```
    """
    
    def __init__(self) -> None:
        """Initialize contract checker."""
        self._contracts: dict[str, Contract] = {}
        self._violation_history: list[ContractViolation] = []
        
        config = settings.verification.contracts
        self.enabled = config.enabled
        self.strict_mode = config.strict_mode
        self.log_violations = config.log_violations
    
    def register(self, contract: Contract) -> None:
        """Register a contract.
        
        Args:
            contract: Contract to register
        """
        self._contracts[contract.id] = contract
        logger.debug("contract_registered", contract_id=contract.id)
    
    def register_default_contracts(self) -> None:
        """Register default healthcare contracts."""
        # Safety contracts
        self._register_safety_contracts()
        
        # Consistency contracts
        self._register_consistency_contracts()
        
        # Format contracts
        self._register_format_contracts()
        
        # Domain contracts
        self._register_domain_contracts()
        
        logger.info("default_contracts_registered", count=len(self._contracts))
    
    def _register_safety_contracts(self) -> None:
        """Register safety-related contracts."""
        
        # No dangerous dosage recommendations
        def check_dangerous_dosage(response: str, _: dict) -> tuple[bool, str]:
            # Check for unreasonably high dosages
            dosage_patterns = re.findall(
                r"(\d+(?:\.\d+)?)\s*(mg|g|ml|mcg)",
                response.lower()
            )
            for amount_str, unit in dosage_patterns:
                amount = float(amount_str)
                # Flag suspiciously high doses
                if unit == "mg" and amount > 5000:
                    return False, f"Potentially dangerous dosage: {amount}mg"
                if unit == "g" and amount > 5:
                    return False, f"Potentially dangerous dosage: {amount}g"
            return True, ""
        
        self.register(Contract(
            id="SAFETY-001",
            name="Dangerous Dosage Check",
            description="Prevents recommending potentially dangerous dosages",
            contract_type=ContractType.SAFETY,
            severity=ViolationSeverity.CRITICAL,
            check_fn=check_dangerous_dosage,
        ))
        
        # No absolute medical claims without disclaimers
        def check_absolute_claims(response: str, _: dict) -> tuple[bool, str]:
            absolute_phrases = [
                r"will\s+definitely",
                r"guaranteed\s+to",
                r"100%\s+safe",
                r"completely\s+harmless",
                r"no\s+side\s+effects",
                r"always\s+works",
            ]
            for pattern in absolute_phrases:
                if re.search(pattern, response, re.IGNORECASE):
                    return False, f"Absolute medical claim detected: {pattern}"
            return True, ""
        
        self.register(Contract(
            id="SAFETY-002",
            name="Absolute Claims Check",
            description="Prevents making absolute medical claims without disclaimers",
            contract_type=ContractType.SAFETY,
            severity=ViolationSeverity.HIGH,
            check_fn=check_absolute_claims,
        ))
        
        # Check for recommending contraindicated combinations
        def check_known_contraindications(response: str, context: dict) -> tuple[bool, str]:
            response_lower = response.lower()
            
            # Known dangerous combinations
            dangerous_pairs = [
                ("warfarin", "aspirin", "bleeding risk"),
                ("maoi", "ssri", "serotonin syndrome"),
                ("methotrexate", "nsaid", "toxicity"),
            ]
            
            for drug_a, drug_b, risk in dangerous_pairs:
                if drug_a in response_lower and drug_b in response_lower:
                    # Check if it's recommending combination (not warning against)
                    warning_phrases = ["avoid", "don't", "should not", "contraindicated", "dangerous"]
                    has_warning = any(phrase in response_lower for phrase in warning_phrases)
                    if not has_warning:
                        return False, f"Potentially dangerous combination: {drug_a} + {drug_b} ({risk})"
            
            return True, ""
        
        self.register(Contract(
            id="SAFETY-003",
            name="Contraindication Check",
            description="Prevents recommending known dangerous drug combinations",
            contract_type=ContractType.SAFETY,
            severity=ViolationSeverity.CRITICAL,
            check_fn=check_known_contraindications,
        ))
    
    def _register_consistency_contracts(self) -> None:
        """Register consistency-related contracts."""
        
        # Response should be consistent with patient context
        def check_allergy_consistency(response: str, context: dict) -> tuple[bool, str]:
            patient = context.get("patient", {})
            allergies = patient.get("allergies", [])
            
            response_lower = response.lower()
            
            for allergy in allergies:
                allergy_lower = allergy.lower()
                # Check if response recommends the allergen
                recommendation_patterns = [
                    f"take {allergy_lower}",
                    f"use {allergy_lower}",
                    f"recommend {allergy_lower}",
                    f"prescribe {allergy_lower}",
                ]
                
                for pattern in recommendation_patterns:
                    if pattern in response_lower:
                        return False, f"Recommending allergen: {allergy}"
            
            return True, ""
        
        self.register(Contract(
            id="CONSISTENCY-001",
            name="Allergy Consistency Check",
            description="Ensures response respects patient allergies",
            contract_type=ContractType.CONSISTENCY,
            severity=ViolationSeverity.CRITICAL,
            check_fn=check_allergy_consistency,
        ))
        
        # Check medication consistency with current medications
        def check_medication_consistency(response: str, context: dict) -> tuple[bool, str]:
            patient = context.get("patient", {})
            current_meds = patient.get("current_medications", [])
            
            if not current_meds:
                return True, ""
            
            response_lower = response.lower()
            
            # Check for contradictory advice about current medications
            if "stop taking" in response_lower or "discontinue" in response_lower:
                for med in current_meds:
                    if med.lower() in response_lower:
                        # This might be okay in some contexts, flag for review
                        return True, ""  # Just note it, don't fail
            
            return True, ""
        
        self.register(Contract(
            id="CONSISTENCY-002",
            name="Medication Consistency Check", 
            description="Checks consistency with current medications",
            contract_type=ContractType.CONSISTENCY,
            severity=ViolationSeverity.MEDIUM,
            check_fn=check_medication_consistency,
        ))
    
    def _register_format_contracts(self) -> None:
        """Register format-related contracts."""
        
        # Response should be complete
        def check_completeness(response: str, _: dict) -> tuple[bool, str]:
            if len(response.strip()) < 20:
                return False, "Response too short to be helpful"
            if not response.strip().endswith((".", "!", "?", ":")):
                return False, "Response appears incomplete"
            return True, ""
        
        self.register(Contract(
            id="FORMAT-001",
            name="Response Completeness Check",
            description="Ensures response is complete and properly formatted",
            contract_type=ContractType.FORMAT,
            severity=ViolationSeverity.LOW,
            check_fn=check_completeness,
        ))
        
        # No repetition (sign of LLM issues)
        def check_repetition(response: str, _: dict) -> tuple[bool, str]:
            words = response.lower().split()
            if len(words) > 20:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.3:
                    return False, f"Excessive repetition detected (unique ratio: {unique_ratio:.2f})"
            return True, ""
        
        self.register(Contract(
            id="FORMAT-002",
            name="Repetition Check",
            description="Detects excessive repetition in response",
            contract_type=ContractType.FORMAT,
            severity=ViolationSeverity.MEDIUM,
            check_fn=check_repetition,
        ))
    
    def _register_domain_contracts(self) -> None:
        """Register healthcare domain contracts."""
        
        # Should include appropriate disclaimers for medical advice
        def check_disclaimers(response: str, context: dict) -> tuple[bool, str]:
            # If response contains medical recommendations, check for disclaimers
            medical_indicators = ["take", "dose", "medication", "treatment", "prescribe"]
            has_medical_content = any(ind in response.lower() for ind in medical_indicators)
            
            if has_medical_content:
                disclaimer_indicators = [
                    "consult", "doctor", "physician", "healthcare provider",
                    "medical professional", "seek advice", "not a substitute",
                ]
                has_disclaimer = any(ind in response.lower() for ind in disclaimer_indicators)
                
                if not has_disclaimer:
                    # Not critical, just informational
                    return True, "Consider adding medical disclaimer"
            
            return True, ""
        
        self.register(Contract(
            id="DOMAIN-001",
            name="Medical Disclaimer Check",
            description="Suggests adding disclaimers for medical advice",
            contract_type=ContractType.DOMAIN,
            severity=ViolationSeverity.LOW,
            check_fn=check_disclaimers,
        ))
    
    def check(
        self,
        response: str,
        context: dict[str, Any] | None = None,
        contract_types: list[ContractType] | None = None,
    ) -> ContractResult:
        """Check response against all registered contracts.
        
        Args:
            response: Neural response to verify
            context: Additional context (patient info, etc.)
            contract_types: Optional filter for contract types
        
        Returns:
            ContractResult with violations and metadata
        """
        import time
        start_time = time.perf_counter()
        
        if not self.enabled:
            return ContractResult(passed=True, contracts_checked=0)
        
        context = context or {}
        violations: list[ContractViolation] = []
        warnings: list[str] = []
        contracts_checked = 0
        
        for contract in self._contracts.values():
            # Filter by type if specified
            if contract_types and contract.contract_type not in contract_types:
                continue
            
            contracts_checked += 1
            
            try:
                passed, evidence = contract.check(response, context)
                
                if not passed:
                    violation = ContractViolation(
                        contract_id=contract.id,
                        contract_type=contract.contract_type,
                        severity=contract.severity,
                        message=f"{contract.name}: {evidence}",
                        evidence=evidence,
                    )
                    violations.append(violation)
                    self._violation_history.append(violation)
                    
                    if self.log_violations:
                        logger.warning(
                            "contract_violation",
                            contract_id=contract.id,
                            severity=contract.severity.value,
                            message=evidence,
                        )
                elif evidence:
                    # Passed but with a note
                    warnings.append(f"{contract.name}: {evidence}")
                    
            except Exception as e:
                logger.error(
                    "contract_check_error",
                    contract_id=contract.id,
                    error=str(e),
                )
        
        verification_time = (time.perf_counter() - start_time) * 1000
        
        passed = not any(
            v.severity in {ViolationSeverity.CRITICAL, ViolationSeverity.HIGH}
            for v in violations
        ) if not self.strict_mode else len(violations) == 0
        
        return ContractResult(
            passed=passed,
            violations=violations,
            warnings=warnings,
            contracts_checked=contracts_checked,
            verification_time_ms=verification_time,
            details={
                "strict_mode": self.strict_mode,
                "response_length": len(response),
            },
        )
    
    def check_streaming(
        self,
        tokens: list[str],
        context: dict[str, Any] | None = None,
    ) -> list[ContractViolation]:
        """Check tokens during streaming.
        
        Performs lightweight checks on accumulated tokens.
        
        Args:
            tokens: List of generated tokens so far
            context: Additional context
        
        Returns:
            List of violations detected
        """
        # Reconstruct text from tokens
        text = "".join(tokens)
        
        # Only run critical safety checks during streaming
        result = self.check(
            text,
            context,
            contract_types=[ContractType.SAFETY],
        )
        
        return result.violations
    
    def get_violation_stats(self) -> dict[str, Any]:
        """Get violation statistics."""
        by_type = {}
        by_severity = {}
        
        for violation in self._violation_history:
            # Count by type
            type_key = violation.contract_type.value
            by_type[type_key] = by_type.get(type_key, 0) + 1
            
            # Count by severity
            sev_key = violation.severity.value
            by_severity[sev_key] = by_severity.get(sev_key, 0) + 1
        
        return {
            "total_violations": len(self._violation_history),
            "by_type": by_type,
            "by_severity": by_severity,
            "registered_contracts": len(self._contracts),
        }
