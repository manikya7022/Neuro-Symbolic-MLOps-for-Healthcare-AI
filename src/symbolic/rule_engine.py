"""Clinical Rule Engine for Healthcare Decision Support.

Implements rule-based reasoning for:
- Clinical guideline enforcement
- Safety protocol verification
- Treatment pathway validation
- Regulatory compliance checking

Uses a declarative rule system with forward chaining inference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

import structlog

from src.config import settings

logger = structlog.get_logger(__name__)


class RulePriority(int, Enum):
    """Priority levels for rule execution."""
    
    CRITICAL = 100  # Safety-critical, always apply
    HIGH = 75
    MEDIUM = 50
    LOW = 25
    OPTIONAL = 10


class RuleAction(str, Enum):
    """Actions that rules can trigger."""
    
    BLOCK = "block"  # Stop and reject
    WARN = "warn"  # Add warning but continue
    REQUIRE = "require"  # Require additional action
    MODIFY = "modify"  # Modify the output
    LOG = "log"  # Just log for audit


@dataclass
class RuleCondition:
    """A condition that must be met for a rule to fire."""
    
    field: str  # Field to check (e.g., "drug", "dose", "patient.age")
    operator: str  # Comparison operator (eq, ne, gt, lt, gte, lte, in, contains, matches)
    value: Any  # Value to compare against
    
    def evaluate(self, context: dict[str, Any]) -> bool:
        """Evaluate condition against context.
        
        Args:
            context: Dictionary of values to check against
        
        Returns:
            True if condition is met
        """
        # Navigate nested fields
        actual_value = context
        for part in self.field.split("."):
            if isinstance(actual_value, dict):
                actual_value = actual_value.get(part)
            elif hasattr(actual_value, part):
                actual_value = getattr(actual_value, part)
            else:
                return False
            
            if actual_value is None:
                return False
        
        # Apply operator
        if self.operator == "eq":
            return actual_value == self.value
        elif self.operator == "ne":
            return actual_value != self.value
        elif self.operator == "gt":
            return actual_value > self.value
        elif self.operator == "lt":
            return actual_value < self.value
        elif self.operator == "gte":
            return actual_value >= self.value
        elif self.operator == "lte":
            return actual_value <= self.value
        elif self.operator == "in":
            return actual_value in self.value
        elif self.operator == "contains":
            if isinstance(actual_value, (list, tuple, set)):
                return self.value in actual_value
            elif isinstance(actual_value, str):
                return self.value.lower() in actual_value.lower()
            return False
        elif self.operator == "matches":
            return bool(re.search(self.value, str(actual_value), re.IGNORECASE))
        elif self.operator == "exists":
            return actual_value is not None
        else:
            logger.warning("unknown_operator", operator=self.operator)
            return False


@dataclass
class Rule:
    """A clinical rule that can be evaluated."""
    
    id: str
    name: str
    description: str
    conditions: list[RuleCondition]
    action: RuleAction
    priority: RulePriority = RulePriority.MEDIUM
    message: str = ""
    recommendation: str = ""
    evidence: str = ""
    category: str = "general"
    enabled: bool = True
    
    # Optional custom logic function
    custom_check: Callable[[dict[str, Any]], bool] | None = None
    
    def evaluate(self, context: dict[str, Any]) -> bool:
        """Check if all conditions are met.
        
        Args:
            context: Evaluation context
        
        Returns:
            True if all conditions match
        """
        if not self.enabled:
            return False
        
        # Check standard conditions
        for condition in self.conditions:
            if not condition.evaluate(context):
                return False
        
        # Check custom logic if provided
        if self.custom_check and not self.custom_check(context):
            return False
        
        return True


@dataclass
class RuleResult:
    """Result of rule evaluation."""
    
    rule: Rule
    fired: bool
    action: RuleAction
    message: str
    recommendation: str
    context_snapshot: dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyProtocol:
    """A safety protocol consisting of multiple rules."""
    
    id: str
    name: str
    description: str
    rules: list[Rule]
    category: str = "safety"
    mandatory: bool = True
    
    def get_critical_rules(self) -> list[Rule]:
        """Get all critical priority rules."""
        return [r for r in self.rules if r.priority == RulePriority.CRITICAL]


class ClinicalRuleEngine:
    """Engine for evaluating clinical rules.
    
    Implements forward-chaining rule evaluation with:
    - Priority-based execution order
    - Rule conflict resolution
    - Audit logging
    - Context propagation
    
    Example:
        ```python
        engine = ClinicalRuleEngine()
        engine.load_default_rules()
        
        context = {
            "drug": "warfarin",
            "patient": {"age": 75, "conditions": ["atrial fibrillation"]},
            "proposed_action": "prescribe aspirin"
        }
        
        results = engine.evaluate(context)
        for result in results:
            if result.fired:
                print(f"Rule {result.rule.name}: {result.message}")
        ```
    """
    
    def __init__(self) -> None:
        """Initialize the rule engine."""
        self._rules: dict[str, Rule] = {}
        self._protocols: dict[str, SafetyProtocol] = {}
        self._rule_categories: dict[str, list[str]] = {}
        
        # Execution statistics
        self._execution_count = 0
        self._rules_fired_count = 0
    
    def add_rule(self, rule: Rule) -> None:
        """Add a rule to the engine.
        
        Args:
            rule: Rule to add
        """
        self._rules[rule.id] = rule
        
        # Index by category
        if rule.category not in self._rule_categories:
            self._rule_categories[rule.category] = []
        self._rule_categories[rule.category].append(rule.id)
        
        logger.debug("rule_added", rule_id=rule.id, category=rule.category)
    
    def add_protocol(self, protocol: SafetyProtocol) -> None:
        """Add a safety protocol.
        
        Args:
            protocol: Protocol to add
        """
        self._protocols[protocol.id] = protocol
        
        # Add all rules from protocol
        for rule in protocol.rules:
            self.add_rule(rule)
        
        logger.info("protocol_added", protocol_id=protocol.id, rules=len(protocol.rules))
    
    def load_default_rules(self) -> None:
        """Load default clinical rules."""
        # Drug safety rules
        self._load_drug_safety_rules()
        
        # Patient safety rules
        self._load_patient_safety_rules()
        
        # Dosing rules
        self._load_dosing_rules()
        
        # Compliance rules
        self._load_compliance_rules()
        
        logger.info("default_rules_loaded", total_rules=len(self._rules))
    
    def _load_drug_safety_rules(self) -> None:
        """Load drug safety rules."""
        drug_rules = [
            Rule(
                id="DS001",
                name="Warfarin-NSAID Interaction",
                description="Prevent concurrent warfarin and NSAID use due to bleeding risk",
                conditions=[
                    RuleCondition("current_medications", "contains", "warfarin"),
                    RuleCondition("proposed_drug", "in", ["aspirin", "ibuprofen", "naproxen"]),
                ],
                action=RuleAction.BLOCK,
                priority=RulePriority.CRITICAL,
                message="CRITICAL: Concurrent warfarin and NSAID use significantly increases bleeding risk",
                recommendation="Use acetaminophen for pain. If NSAID absolutely required, use lowest dose for shortest duration with close monitoring",
                evidence="Multiple studies show 3-6x increased GI bleeding risk",
                category="drug_interaction",
            ),
            Rule(
                id="DS002",
                name="Aspirin Allergy Cross-Reactivity",
                description="Warn about cross-reactivity with other NSAIDs in aspirin-allergic patients",
                conditions=[
                    RuleCondition("patient.allergies", "contains", "aspirin"),
                    RuleCondition("proposed_drug_class", "eq", "NSAID"),
                ],
                action=RuleAction.WARN,
                priority=RulePriority.HIGH,
                message="Patient allergic to aspirin - risk of cross-reactivity with other NSAIDs",
                recommendation="Consider non-NSAID alternatives. If NSAID needed, use with caution in monitored setting",
                category="allergy",
            ),
            Rule(
                id="DS003",
                name="Metformin Renal Contraindication",
                description="Block metformin in severe renal impairment",
                conditions=[
                    RuleCondition("proposed_drug", "eq", "metformin"),
                    RuleCondition("patient.renal_function", "eq", "severe"),
                ],
                action=RuleAction.BLOCK,
                priority=RulePriority.CRITICAL,
                message="Metformin contraindicated in severe renal impairment due to lactic acidosis risk",
                recommendation="Consider alternative diabetes therapy (sulfonylureas, insulin)",
                category="contraindication",
            ),
            Rule(
                id="DS004",
                name="Statin-Pregnancy Contraindication",
                description="Block statins during pregnancy",
                conditions=[
                    RuleCondition("proposed_drug_class", "eq", "statin"),
                    RuleCondition("patient.pregnant", "eq", True),
                ],
                action=RuleAction.BLOCK,
                priority=RulePriority.CRITICAL,
                message="Statins are Category X - contraindicated during pregnancy",
                recommendation="Discontinue statin. Manage lipids with diet and exercise during pregnancy",
                category="pregnancy",
            ),
        ]
        
        for rule in drug_rules:
            self.add_rule(rule)
    
    def _load_patient_safety_rules(self) -> None:
        """Load patient safety rules."""
        patient_rules = [
            Rule(
                id="PS001",
                name="Elderly High-Risk Medication",
                description="Warn about high-risk medications in elderly patients",
                conditions=[
                    RuleCondition("patient.age", "gte", 65),
                    RuleCondition("proposed_drug", "in", ["benzodiazepines", "anticholinergics", "opioids"]),
                ],
                action=RuleAction.WARN,
                priority=RulePriority.HIGH,
                message="High-risk medication in elderly patient - increased fall and cognitive impairment risk",
                recommendation="Consider lower dose or alternative medication. Assess fall risk.",
                evidence="Beers Criteria",
                category="elderly_safety",
            ),
            Rule(
                id="PS002",
                name="Pediatric Weight-Based Dosing Required",
                description="Ensure weight-based dosing for pediatric patients",
                conditions=[
                    RuleCondition("patient.age", "lt", 18),
                    RuleCondition("dose_type", "ne", "weight_based"),
                ],
                action=RuleAction.REQUIRE,
                priority=RulePriority.HIGH,
                message="Pediatric patient requires weight-based dosing",
                recommendation="Calculate dose based on patient weight (mg/kg)",
                category="pediatric_safety",
            ),
            Rule(
                id="PS003",
                name="Renal Dose Adjustment Required",
                description="Flag medications requiring renal dose adjustment",
                conditions=[
                    RuleCondition("patient.renal_function", "in", ["moderate", "severe"]),
                    RuleCondition("drug.requires_renal_adjustment", "eq", True),
                ],
                action=RuleAction.REQUIRE,
                priority=RulePriority.HIGH,
                message="Dose adjustment required for renal impairment",
                recommendation="Adjust dose or interval based on creatinine clearance",
                category="renal_adjustment",
            ),
        ]
        
        for rule in patient_rules:
            self.add_rule(rule)
    
    def _load_dosing_rules(self) -> None:
        """Load dosing rules."""
        dosing_rules = [
            Rule(
                id="DO001",
                name="Maximum Daily Dose Exceeded",
                description="Block doses exceeding maximum daily limit",
                custom_check=lambda ctx: (
                    ctx.get("dose_mg", 0) * ctx.get("frequency", 1) >
                    ctx.get("drug", {}).get("max_daily_mg", float("inf"))
                ),
                conditions=[],
                action=RuleAction.BLOCK,
                priority=RulePriority.CRITICAL,
                message="Proposed dose exceeds maximum daily limit",
                recommendation="Reduce dose or frequency to stay within maximum daily limit",
                category="dosing",
            ),
            Rule(
                id="DO002",
                name="Subtherapeutic Dose Warning",
                description="Warn if dose may be subtherapeutic",
                custom_check=lambda ctx: (
                    ctx.get("dose_mg", float("inf")) <
                    ctx.get("drug", {}).get("min_therapeutic_mg", 0)
                ),
                conditions=[],
                action=RuleAction.WARN,
                priority=RulePriority.MEDIUM,
                message="Dose may be below therapeutic threshold",
                recommendation="Consider increasing dose if clinically appropriate",
                category="dosing",
            ),
        ]
        
        for rule in dosing_rules:
            self.add_rule(rule)
    
    def _load_compliance_rules(self) -> None:
        """Load compliance rules."""
        compliance_rules = [
            Rule(
                id="CO001",
                name="Controlled Substance Documentation",
                description="Require documentation for controlled substance prescriptions",
                conditions=[
                    RuleCondition("drug.controlled_substance", "eq", True),
                ],
                action=RuleAction.REQUIRE,
                priority=RulePriority.HIGH,
                message="Controlled substance requires additional documentation",
                recommendation="Document indication, prior treatments, and prescription monitoring program check",
                category="compliance",
            ),
            Rule(
                id="CO002",
                name="High-Risk Medication Consent",
                description="Require informed consent for high-risk medications",
                conditions=[
                    RuleCondition("drug.requires_consent", "eq", True),
                ],
                action=RuleAction.REQUIRE,
                priority=RulePriority.HIGH,
                message="High-risk medication requires documented informed consent",
                recommendation="Obtain and document patient informed consent before initiating therapy",
                category="compliance",
            ),
        ]
        
        for rule in compliance_rules:
            self.add_rule(rule)
    
    def evaluate(
        self,
        context: dict[str, Any],
        categories: list[str] | None = None,
    ) -> list[RuleResult]:
        """Evaluate all applicable rules against context.
        
        Args:
            context: Context dictionary with patient, drug, and action info
            categories: Optional list of categories to evaluate (default: all)
        
        Returns:
            List of rule results
        """
        self._execution_count += 1
        results: list[RuleResult] = []
        
        # Get rules to evaluate
        if categories:
            rule_ids = []
            for cat in categories:
                rule_ids.extend(self._rule_categories.get(cat, []))
            rules = [self._rules[rid] for rid in rule_ids if rid in self._rules]
        else:
            rules = list(self._rules.values())
        
        # Sort by priority (highest first)
        rules.sort(key=lambda r: r.priority.value, reverse=True)
        
        # Evaluate each rule
        for rule in rules:
            try:
                fired = rule.evaluate(context)
                
                if fired:
                    self._rules_fired_count += 1
                    logger.debug(
                        "rule_fired",
                        rule_id=rule.id,
                        rule_name=rule.name,
                        action=rule.action.value,
                    )
                
                results.append(RuleResult(
                    rule=rule,
                    fired=fired,
                    action=rule.action if fired else RuleAction.LOG,
                    message=rule.message if fired else "",
                    recommendation=rule.recommendation if fired else "",
                    context_snapshot=dict(context) if fired else {},
                ))
                
            except Exception as e:
                logger.error("rule_evaluation_error", rule_id=rule.id, error=str(e))
        
        return results
    
    def evaluate_with_blocking(
        self,
        context: dict[str, Any],
    ) -> tuple[bool, list[RuleResult]]:
        """Evaluate rules and check if any blocking rules fired.
        
        Args:
            context: Context dictionary
        
        Returns:
            Tuple of (is_allowed, results)
        """
        results = self.evaluate(context)
        
        blocking_results = [r for r in results if r.fired and r.action == RuleAction.BLOCK]
        is_allowed = len(blocking_results) == 0
        
        return is_allowed, results
    
    def get_fired_rules(self, results: list[RuleResult]) -> list[RuleResult]:
        """Filter to only fired rules.
        
        Args:
            results: List of rule results
        
        Returns:
            List of results where rule fired
        """
        return [r for r in results if r.fired]
    
    def get_statistics(self) -> dict[str, Any]:
        """Get execution statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            "total_rules": len(self._rules),
            "total_protocols": len(self._protocols),
            "categories": list(self._rule_categories.keys()),
            "rules_by_category": {k: len(v) for k, v in self._rule_categories.items()},
            "execution_count": self._execution_count,
            "rules_fired_count": self._rules_fired_count,
        }
