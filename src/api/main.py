"""FastAPI main application for Neuro-Symbolic Healthcare System.

Production-ready API with:
- REST endpoints for healthcare queries
- WebSocket support for streaming
- Prometheus metrics integration
- Health checks and monitoring
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.config import settings
from src.api.schemas import (
    QueryRequest,
    QueryResponse,
    MedicationCheckRequest,
    MedicationCheckResponse,
    DrugInteractionRequest,
    DrugInteractionResponse,
    HealthCheckResponse,
    MetricsResponse,
    ErrorResponse,
    RouteInfo,
    VerificationInfo,
    SymbolicCheckInfo,
    ConfidenceInfo,
    PatientInfo,
)
from src.router import RoutingDecisionEngine, RouteType
from src.neural import LLMClient, UncertaintyQuantifier, ConfidenceScorer
from src.symbolic import MedicalConstraintSolver, MedicalKnowledgeBase, ClinicalRuleEngine
from src.verification import ContractChecker, DriftDetector

logger = structlog.get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "nsh_requests_total",
    "Total requests",
    ["endpoint", "status"]
)
REQUEST_LATENCY = Histogram(
    "nsh_request_latency_seconds",
    "Request latency",
    ["endpoint"]
)
ROUTE_COUNT = Counter(
    "nsh_route_total",
    "Routing decisions",
    ["route_type"]
)

# Global components (initialized on startup)
llm_client: LLMClient | None = None
router_engine: RoutingDecisionEngine | None = None
constraint_solver: MedicalConstraintSolver | None = None
knowledge_base: MedicalKnowledgeBase | None = None
rule_engine: ClinicalRuleEngine | None = None
contract_checker: ContractChecker | None = None
drift_detector: DriftDetector | None = None
uncertainty_quantifier: UncertaintyQuantifier | None = None
confidence_scorer: ConfidenceScorer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    global llm_client, router_engine, constraint_solver, knowledge_base
    global rule_engine, contract_checker, drift_detector
    global uncertainty_quantifier, confidence_scorer
    
    logger.info("starting_neuro_symbolic_healthcare_system")
    
    # Initialize components
    llm_client = LLMClient()
    router_engine = RoutingDecisionEngine()
    constraint_solver = MedicalConstraintSolver()
    knowledge_base = MedicalKnowledgeBase()
    rule_engine = ClinicalRuleEngine()
    contract_checker = ContractChecker()
    drift_detector = DriftDetector()
    
    # Initialize dependent components
    uncertainty_quantifier = UncertaintyQuantifier(llm_client)
    confidence_scorer = ConfidenceScorer(uncertainty_quantifier)
    
    # Load data
    await knowledge_base.load()
    rule_engine.load_default_rules()
    contract_checker.register_default_contracts()
    
    # Load drug interactions into solver
    interactions = knowledge_base.get_interactions("")
    from src.symbolic.z3_solver import DrugInteraction, InteractionSeverity
    solver_interactions = []
    for interaction in knowledge_base._interactions.values():
        for i in interaction:
            solver_interactions.append(DrugInteraction(
                drug_a=i.drug_a,
                drug_b=i.drug_b,
                severity=InteractionSeverity(i.severity),
                description=i.clinical_effect,
                mechanism=i.mechanism,
                management=i.management,
            ))
    constraint_solver.load_interactions(solver_interactions)
    
    logger.info("system_initialized")
    
    yield
    
    # Cleanup
    if llm_client:
        await llm_client.close()
    logger.info("system_shutdown")


# Create FastAPI app
app = FastAPI(
    title="Neuro-Symbolic Healthcare AI",
    description="Production-ready hybrid reasoning system for healthcare compliance checking",
    version=settings.neural.ollama.model,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error("unhandled_exception", error=str(exc), path=request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error=str(exc),
            error_code="INTERNAL_ERROR",
        ).model_dump(mode="json"),
    )


@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check endpoint."""
    components = {
        "api": "healthy",
        "llm": "unknown",
        "knowledge_base": "healthy" if knowledge_base and knowledge_base._loaded else "not_loaded",
    }
    
    # Check LLM availability
    if llm_client:
        try:
            if await llm_client.is_available():
                components["llm"] = "healthy"
            else:
                components["llm"] = "unavailable"
        except Exception:
            components["llm"] = "error"
    
    status = "healthy" if all(v == "healthy" for v in components.values()) else "degraded"
    
    return HealthCheckResponse(
        status=status,
        version="0.1.0",
        components=components,
    )


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


@app.get("/api/v1/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get system metrics."""
    return MetricsResponse(
        routing_stats=router_engine.get_statistics() if router_engine else {},
        verification_stats=contract_checker.get_violation_stats() if contract_checker else {},
        drift_metrics=drift_detector.get_metrics().to_dict() if drift_detector else {},
        latency_stats=router_engine.latency_manager.get_statistics() if router_engine else {},
    )


@app.post("/api/v1/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """Process a healthcare query.
    
    This is the main endpoint that:
    1. Routes the query (neural/symbolic/hybrid)
    2. Generates response via LLM
    3. Verifies against symbolic constraints
    4. Returns comprehensive result
    """
    start_time = time.perf_counter()
    
    try:
        # Build context from patient info
        context: dict[str, Any] = {}
        if request.patient:
            context["patient"] = request.patient.model_dump()
        
        # 1. Make routing decision
        decision = router_engine.decide(
            query=request.query,
            context=context,
            priority=request.priority.value,
            custom_budget_ms=request.custom_timeout_ms,
        )
        
        ROUTE_COUNT.labels(route_type=decision.route_type.value).inc()
        
        # 2. Generate response via LLM
        system_prompt = """You are a healthcare AI assistant. Provide accurate, helpful medical information.
Always include appropriate disclaimers for medical advice. Be specific about drug interactions and safety."""
        
        llm_response = await llm_client.generate(
            prompt=request.query,
            system_prompt=system_prompt,
        )
        
        # 3. Calculate uncertainty and confidence
        uncertainty_result = await uncertainty_quantifier.quantify(
            request.query, llm_response
        )
        confidence_result = await confidence_scorer.score(
            request.query, llm_response, uncertainty_result
        )
        
        # 4. Verify response against contracts
        verification_result = contract_checker.check(
            llm_response.text,
            context,
        )
        
        # Record for drift detection
        drift_detector.record(verification_result.violations)
        
        # 5. Run symbolic safety check if needed
        symbolic_check = None
        if decision.involves_symbolic and request.patient:
            # Extract any drug mentions from response for checking
            # This is a simplified version - production would use NER
            patient_context = _build_patient_context(request.patient)
            
            # For demo, check if any known drugs are mentioned
            for drug_name in knowledge_base._drugs.keys():
                if drug_name in llm_response.text.lower():
                    check_result = constraint_solver.check_medication(
                        drug=drug_name,
                        dose_mg=100,  # Default for demo
                        patient=patient_context,
                    )
                    symbolic_check = SymbolicCheckInfo(
                        is_safe=check_result.is_safe,
                        risk_level=check_result.risk_level,
                        violations=[v.__dict__ for v in check_result.violations],
                        warnings=check_result.warnings,
                        verified_constraints=check_result.verified_constraints,
                    )
                    break
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        # Build response
        response = QueryResponse(
            response=llm_response.text,
            route=RouteInfo(
                route_type=decision.route_type.value,
                strategy=decision.strategy_used.value,
                complexity_score=decision.complexity.score,
                complexity_category=decision.complexity.category,
                is_safety_critical=decision.complexity.is_safety_critical,
                latency_budget_ms=decision.budget.total_ms,
            ),
            verification=VerificationInfo(
                passed=verification_result.passed,
                contracts_checked=verification_result.contracts_checked,
                violations=[v.__dict__ for v in verification_result.violations],
                warnings=verification_result.warnings,
            ),
            symbolic_check=symbolic_check,
            confidence=ConfidenceInfo(
                overall=confidence_result.overall,
                uncertainty_score=confidence_result.uncertainty_score,
                quality_score=confidence_result.quality_score,
                safety_score=confidence_result.safety_score,
                grade=confidence_result.grade,
            ),
            latency_ms=latency_ms,
            warnings=confidence_result.warnings,
        )
        
        REQUEST_COUNT.labels(endpoint="/api/v1/query", status="success").inc()
        REQUEST_LATENCY.labels(endpoint="/api/v1/query").observe(latency_ms / 1000)
        
        return response
        
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/api/v1/query", status="error").inc()
        logger.error("query_processing_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/medication/check", response_model=MedicationCheckResponse)
async def check_medication(request: MedicationCheckRequest):
    """Check medication safety for a patient.
    
    Performs symbolic constraint checking for:
    - Drug interactions with current medications
    - Dosage limits
    - Contraindications based on conditions
    - Allergy checking
    """
    start_time = time.perf_counter()
    
    try:
        patient_context = _build_patient_context(request.patient)
        
        result = constraint_solver.check_medication(
            drug=request.drug,
            dose_mg=request.dose_mg,
            patient=patient_context,
            frequency_per_day=request.frequency_per_day,
        )
        
        # Generate recommendations
        recommendations = []
        for violation in result.violations:
            if violation.recommendation:
                recommendations.append(violation.recommendation)
        
        response = MedicationCheckResponse(
            drug=request.drug,
            is_safe=result.is_safe,
            risk_level=result.risk_level,
            violations=[{
                "type": v.constraint_type,
                "severity": v.severity.value,
                "description": v.description,
                "drugs": v.drugs_involved,
            } for v in result.violations],
            warnings=result.warnings,
            recommendations=recommendations,
            verified_constraints=result.verified_constraints,
            solver_time_ms=result.solver_time_ms,
        )
        
        REQUEST_COUNT.labels(endpoint="/api/v1/medication/check", status="success").inc()
        
        return response
        
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/api/v1/medication/check", status="error").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/interactions", response_model=DrugInteractionResponse)
async def check_interactions(request: DrugInteractionRequest):
    """Check for drug-drug interactions."""
    try:
        interactions = []
        overall_risk = "LOW"
        recommendations = []
        
        # Check all pairs
        for i, drug_a in enumerate(request.drugs):
            for drug_b in request.drugs[i + 1:]:
                kb_interactions = knowledge_base.get_interactions(drug_a)
                for interaction in kb_interactions:
                    if (interaction.drug_a.lower() == drug_b.lower() or 
                        interaction.drug_b.lower() == drug_b.lower()):
                        interactions.append({
                            "drug_a": interaction.drug_a,
                            "drug_b": interaction.drug_b,
                            "severity": interaction.severity,
                            "mechanism": interaction.mechanism,
                            "clinical_effect": interaction.clinical_effect,
                            "management": interaction.management,
                        })
                        recommendations.append(interaction.management)
                        
                        if interaction.severity in ["major", "contraindicated"]:
                            overall_risk = "HIGH"
                        elif interaction.severity == "moderate" and overall_risk != "HIGH":
                            overall_risk = "MEDIUM"
        
        return DrugInteractionResponse(
            interactions=interactions,
            overall_risk=overall_risk,
            recommendations=list(set(recommendations)),
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/drugs/{drug_name}")
async def get_drug_info(drug_name: str):
    """Get information about a drug."""
    drug = knowledge_base.get_drug(drug_name)
    if not drug:
        raise HTTPException(status_code=404, detail=f"Drug '{drug_name}' not found")
    return drug.__dict__


@app.get("/api/v1/conditions/{condition_name}")
async def get_condition_info(condition_name: str):
    """Get information about a condition."""
    condition = knowledge_base.get_condition(condition_name)
    if not condition:
        raise HTTPException(status_code=404, detail=f"Condition '{condition_name}' not found")
    return condition.__dict__


def _build_patient_context(patient_info: PatientInfo):
    """Build PatientContext from PatientInfo schema."""
    from src.symbolic.z3_solver import PatientContext
    
    return PatientContext(
        weight_kg=patient_info.weight_kg,
        age_years=patient_info.age,
        allergies=patient_info.allergies,
        conditions=patient_info.conditions,
        current_medications=patient_info.current_medications,
        renal_function=patient_info.renal_function,
        hepatic_function=patient_info.hepatic_function,
        pregnant=patient_info.pregnant,
        breastfeeding=patient_info.breastfeeding,
    )


def run_server():
    """Run the server (for CLI)."""
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=True,
    )


if __name__ == "__main__":
    run_server()
