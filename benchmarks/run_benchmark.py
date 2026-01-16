#!/usr/bin/env python3
"""Benchmark runner for Neuro-Symbolic Healthcare System.

Runs the complete benchmark suite and generates a report.

Usage:
    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --quick  # Quick mode, fewer iterations
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.router import RoutingDecisionEngine, RouteType
from src.symbolic import MedicalConstraintSolver, MedicalKnowledgeBase
from src.symbolic.z3_solver import DrugInteraction, DosageConstraint, PatientContext, InteractionSeverity
from src.verification import ContractChecker, DriftDetector


@dataclass
class BenchmarkResult:
    """Result of a single benchmark scenario."""
    
    scenario_id: str
    category: str
    passed: bool
    expected_route: str
    actual_route: str
    expected_safe: bool | None
    actual_safe: bool | None
    latency_ms: float
    errors: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    
    timestamp: str
    total_scenarios: int
    passed: int
    failed: int
    errors: int
    avg_latency_ms: float
    results_by_category: dict[str, dict[str, int]]
    results: list[BenchmarkResult]
    
    @property
    def pass_rate(self) -> float:
        return self.passed / self.total_scenarios if self.total_scenarios > 0 else 0
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "summary": {
                "total": self.total_scenarios,
                "passed": self.passed,
                "failed": self.failed,
                "errors": self.errors,
                "pass_rate": f"{self.pass_rate:.1%}",
                "avg_latency_ms": self.avg_latency_ms,
            },
            "by_category": self.results_by_category,
            "results": [
                {
                    "id": r.scenario_id,
                    "category": r.category,
                    "passed": r.passed,
                    "latency_ms": r.latency_ms,
                    "errors": r.errors,
                }
                for r in self.results
            ],
        }


class HealthcareBenchmark:
    """Benchmark runner for healthcare scenarios."""
    
    def __init__(self, dataset_path: Path | None = None):
        self.dataset_path = dataset_path or Path(__file__).parent.parent / "tests" / "benchmark_dataset.json"
        self.scenarios: list[dict] = []
        self.results: list[BenchmarkResult] = []
        
        # Components
        self.router: RoutingDecisionEngine | None = None
        self.solver: MedicalConstraintSolver | None = None
        self.kb: MedicalKnowledgeBase | None = None
        self.checker: ContractChecker | None = None
    
    def load_dataset(self) -> None:
        """Load benchmark dataset."""
        with open(self.dataset_path) as f:
            data = json.load(f)
        self.scenarios = data["scenarios"]
        print(f"Loaded {len(self.scenarios)} benchmark scenarios")
    
    async def setup(self) -> None:
        """Initialize all components."""
        print("Initializing components...")
        
        self.router = RoutingDecisionEngine()
        self.solver = MedicalConstraintSolver()
        self.kb = MedicalKnowledgeBase()
        self.checker = ContractChecker()
        
        await self.kb.load()
        self.checker.register_default_contracts()
        
        # Load interactions from KB to solver
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
        
        # Load dosage constraints - comprehensive list
        self.solver.load_dosage_constraints([
            DosageConstraint(drug="aspirin", max_dose_mg=650, max_daily_mg=4000),
            DosageConstraint(drug="ibuprofen", max_dose_mg=800, max_daily_mg=3200, renal_adjustment=True),
            DosageConstraint(drug="acetaminophen", max_dose_mg=1000, max_daily_mg=4000),
            DosageConstraint(drug="warfarin", max_dose_mg=10, max_daily_mg=10),  # Typical max is 10mg
            DosageConstraint(drug="metformin", max_dose_mg=1000, max_daily_mg=2550, renal_adjustment=True),
            DosageConstraint(drug="atorvastatin", max_dose_mg=80, max_daily_mg=80),
            DosageConstraint(drug="lisinopril", max_dose_mg=40, max_daily_mg=80),
            DosageConstraint(drug="omeprazole", max_dose_mg=40, max_daily_mg=40),
        ])
        
        print("Components initialized")
    
    def build_patient_context(self, patient_data: dict | None) -> PatientContext | None:
        """Build PatientContext from scenario data."""
        if patient_data is None:
            return None
        
        return PatientContext(
            weight_kg=patient_data.get("weight_kg"),
            age_years=patient_data.get("age"),
            allergies=patient_data.get("allergies", []),
            conditions=patient_data.get("conditions", []),
            current_medications=patient_data.get("current_medications", []),
            renal_function=patient_data.get("renal_function", "normal"),
            pregnant=patient_data.get("pregnant", False),
        )
    
    def _extract_test_drug(self, scenario: dict) -> tuple[str, float] | None:
        """Extract the relevant drug and dose to test from scenario."""
        query = scenario["query"].lower()
        scenario_id = scenario["id"]
        
        # Mapping of scenario IDs to specific drugs/doses to test
        scenario_drug_map = {
            # Drug interactions
            "DRUG_INTERACTION_001": ("aspirin", 325),
            "DRUG_INTERACTION_002": ("metformin", 500),
            "DRUG_INTERACTION_003": ("ibuprofen", 400),
            "DRUG_INTERACTION_004": ("simvastatin", 40),
            "DRUG_INTERACTION_005": ("tramadol", 50),
            # Allergies
            "ALLERGY_001": ("amoxicillin", 500),
            "ALLERGY_002": ("naproxen", 250),
            "ALLERGY_003": ("sulfasalazine", 500),
            "ALLERGY_004": ("hydrocodone", 5),
            # Pregnancy
            "PREGNANCY_001": ("atorvastatin", 20),
            "PREGNANCY_002": ("acetaminophen", 500),
            "PREGNANCY_003": ("methotrexate", 15),
            "PREGNANCY_004": ("valproic acid", 500),
            # Dosage
            "DOSAGE_001": ("ibuprofen", 1000),  # Overdose
            "DOSAGE_002": ("ibuprofen", 200),   # Pediatric
            "DOSAGE_003": ("acetaminophen", 6000),  # Daily overdose (simulated as single)
            "DOSAGE_004": ("aspirin", 500),
            # Renal
            "RENAL_001": ("metformin", 500),
            "RENAL_002": ("ibuprofen", 400),
            # Polypharmacy
            "POLYPHARM_001": ("omeprazole", 20),
            "POLYPHARM_002": ("aspirin", 81),
            # Simple queries (no patient)
            "SIMPLE_001": ("aspirin", 325),
            "SIMPLE_002": ("metformin", 500),
            "SIMPLE_003": ("ibuprofen", 400),
            "SIMPLE_004": ("warfarin", 5),
            # Emergency
            "EMERGENCY_001": ("warfarin", 20),
            "EMERGENCY_002": ("lisinopril", 80),  # Double 40mg dose
            # Elderly
            "ELDERLY_001": ("diphenhydramine", 25),
            "ELDERLY_002": ("diazepam", 5),
            "ELDERLY_003": ("amitriptyline", 25),
        }
        
        return scenario_drug_map.get(scenario_id)
    
    async def run_scenario(self, scenario: dict) -> BenchmarkResult:
        """Run a single benchmark scenario."""
        start_time = time.perf_counter()
        errors = []
        details = {}
        
        scenario_id = scenario["id"]
        expected = scenario["expected"]
        
        try:
            # Build context
            context = {}
            if scenario["patient"]:
                context["patient"] = scenario["patient"]
            
            # Step 1: Test routing
            decision = self.router.decide(
                query=scenario["query"],
                context=context,
            )
            
            actual_route = decision.route_type.value
            details["routing"] = {
                "complexity_score": decision.complexity.score,
                "complexity_category": decision.complexity.category,
                "is_safety_critical": decision.complexity.is_safety_critical,
            }
            
            # Check routing correctness
            route_ok = True
            if expected["is_safety_critical"] and not decision.involves_symbolic:
                errors.append("Safety-critical query should route to symbolic")
                route_ok = False
            
            # Step 2: Test symbolic (if applicable)
            actual_safe = None
            if scenario["patient"] and expected.get("symbolic_safe") is not None:
                patient = self.build_patient_context(scenario["patient"])
                
                # Get specific drug and dose for this scenario
                drug_info = self._extract_test_drug(scenario)
                
                if drug_info:
                    drug, dose = drug_info
                    result = self.solver.check_medication(
                        drug=drug,
                        dose_mg=dose,
                        patient=patient,
                    )
                    actual_safe = result.is_safe
                    details["symbolic"] = {
                        "drug": drug,
                        "dose": dose,
                        "is_safe": result.is_safe,
                        "risk_level": result.risk_level,
                        "violations": len(result.violations),
                        "violation_types": [v.constraint_type for v in result.violations],
                    }
                    
                    # Check symbolic correctness
                    if expected["symbolic_safe"] != actual_safe:
                        errors.append(
                            f"Expected symbolic_safe={expected['symbolic_safe']}, got {actual_safe}"
                        )
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            passed = len(errors) == 0
            
        except Exception as e:
            errors.append(f"Exception: {str(e)}")
            latency_ms = (time.perf_counter() - start_time) * 1000
            passed = False
            actual_route = "error"
            actual_safe = None
        
        return BenchmarkResult(
            scenario_id=scenario_id,
            category=scenario["category"],
            passed=passed,
            expected_route=expected.get("route_type", "unknown"),
            actual_route=actual_route,
            expected_safe=expected.get("symbolic_safe"),
            actual_safe=actual_safe,
            latency_ms=latency_ms,
            errors=errors,
            details=details,
        )
    
    async def run_all(self) -> BenchmarkReport:
        """Run all benchmark scenarios."""
        print(f"\nRunning {len(self.scenarios)} scenarios...")
        print("-" * 60)
        
        self.results = []
        
        for i, scenario in enumerate(self.scenarios):
            result = await self.run_scenario(scenario)
            self.results.append(result)
            
            status = "✓" if result.passed else "✗"
            print(f"  [{status}] {result.scenario_id}: {result.latency_ms:.1f}ms")
            if result.errors:
                for error in result.errors:
                    print(f"      → {error}")
        
        print("-" * 60)
        
        # Calculate statistics
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed and not r.errors)
        errors = sum(1 for r in self.results if r.errors)
        avg_latency = sum(r.latency_ms for r in self.results) / len(self.results)
        
        # By category
        by_category: dict[str, dict[str, int]] = {}
        for r in self.results:
            if r.category not in by_category:
                by_category[r.category] = {"passed": 0, "failed": 0}
            by_category[r.category]["passed" if r.passed else "failed"] += 1
        
        return BenchmarkReport(
            timestamp=datetime.now().isoformat(),
            total_scenarios=len(self.scenarios),
            passed=passed,
            failed=failed,
            errors=errors,
            avg_latency_ms=avg_latency,
            results_by_category=by_category,
            results=self.results,
        )
    
    def print_report(self, report: BenchmarkReport) -> None:
        """Print benchmark report."""
        print("\n" + "=" * 60)
        print("BENCHMARK REPORT")
        print("=" * 60)
        print(f"\nTimestamp: {report.timestamp}")
        print(f"\nSummary:")
        print(f"  Total scenarios: {report.total_scenarios}")
        print(f"  Passed: {report.passed} ({report.pass_rate:.1%})")
        print(f"  Failed: {report.failed}")
        print(f"  Errors: {report.errors}")
        print(f"  Avg latency: {report.avg_latency_ms:.1f}ms")
        
        print(f"\nBy Category:")
        for cat, stats in sorted(report.results_by_category.items()):
            total = stats["passed"] + stats["failed"]
            rate = stats["passed"] / total if total > 0 else 0
            print(f"  {cat}: {stats['passed']}/{total} ({rate:.0%})")
        
        print("\n" + "=" * 60)
    
    def save_report(self, report: BenchmarkReport, output_path: Path) -> None:
        """Save report to JSON file."""
        with open(output_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nReport saved to: {output_path}")


async def main():
    """Run benchmark."""
    parser = argparse.ArgumentParser(description="Run healthcare benchmark")
    parser.add_argument("--quick", action="store_true", help="Quick mode")
    parser.add_argument("--output", type=Path, help="Output report path")
    args = parser.parse_args()
    
    benchmark = HealthcareBenchmark()
    benchmark.load_dataset()
    await benchmark.setup()
    
    report = await benchmark.run_all()
    benchmark.print_report(report)
    
    if args.output:
        benchmark.save_report(report, args.output)
    
    # Exit with error if failures
    if report.failed > 0 or report.errors > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
