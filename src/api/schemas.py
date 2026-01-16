"""Pydantic schemas for API request/response validation."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class PatientInfo(BaseModel):
    """Patient information for context."""
    
    age: int | None = Field(None, ge=0, le=150, description="Patient age in years")
    weight_kg: float | None = Field(None, ge=0, le=500, description="Patient weight in kg")
    allergies: list[str] = Field(default_factory=list, description="Known allergies")
    conditions: list[str] = Field(default_factory=list, description="Medical conditions")
    current_medications: list[str] = Field(default_factory=list, description="Current medications")
    renal_function: str = Field("normal", description="Renal function: normal, mild, moderate, severe")
    hepatic_function: str = Field("normal", description="Hepatic function: normal, mild, moderate, severe")
    pregnant: bool = Field(False, description="Pregnancy status")
    breastfeeding: bool = Field(False, description="Breastfeeding status")


class QueryPriority(str, Enum):
    """Query priority levels."""
    
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class QueryRequest(BaseModel):
    """Request for healthcare query."""
    
    query: str = Field(..., min_length=1, max_length=5000, description="User query")
    patient: PatientInfo | None = Field(None, description="Patient context")
    priority: QueryPriority = Field(QueryPriority.MEDIUM, description="Query priority")
    custom_timeout_ms: int | None = Field(None, ge=100, le=60000, description="Custom timeout in ms")
    include_reasoning: bool = Field(True, description="Include reasoning trace")
    stream: bool = Field(False, description="Stream response")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "Can I take aspirin with my blood thinner warfarin?",
                    "patient": {
                        "age": 65,
                        "current_medications": ["warfarin"],
                        "conditions": ["atrial fibrillation"]
                    },
                    "priority": "high"
                }
            ]
        }
    }


class RouteInfo(BaseModel):
    """Information about routing decision."""
    
    route_type: str = Field(..., description="Type of route used")
    strategy: str = Field(..., description="Routing strategy")
    complexity_score: float = Field(..., description="Query complexity score")
    complexity_category: str = Field(..., description="Complexity category")
    is_safety_critical: bool = Field(..., description="Whether query is safety-critical")
    latency_budget_ms: float = Field(..., description="Allocated latency budget")


class VerificationInfo(BaseModel):
    """Information about verification results."""
    
    passed: bool = Field(..., description="Whether verification passed")
    contracts_checked: int = Field(..., description="Number of contracts checked")
    violations: list[dict[str, Any]] = Field(default_factory=list, description="Contract violations")
    warnings: list[str] = Field(default_factory=list, description="Verification warnings")


class SymbolicCheckInfo(BaseModel):
    """Information about symbolic constraint checking."""
    
    is_safe: bool = Field(..., description="Whether medication is safe")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, HIGH, CRITICAL")
    violations: list[dict[str, Any]] = Field(default_factory=list, description="Constraint violations")
    warnings: list[str] = Field(default_factory=list, description="Safety warnings")
    verified_constraints: int = Field(..., description="Number of constraints verified")


class ConfidenceInfo(BaseModel):
    """Confidence scoring information."""
    
    overall: float = Field(..., ge=0, le=1, description="Overall confidence score")
    uncertainty_score: float = Field(..., description="Uncertainty score")
    quality_score: float = Field(..., description="Response quality score")
    safety_score: float = Field(..., description="Safety score")
    grade: str = Field(..., description="Letter grade")


class QueryResponse(BaseModel):
    """Response for healthcare query."""
    
    response: str = Field(..., description="Generated response")
    route: RouteInfo = Field(..., description="Routing information")
    verification: VerificationInfo | None = Field(None, description="Verification results")
    symbolic_check: SymbolicCheckInfo | None = Field(None, description="Symbolic safety check")
    confidence: ConfidenceInfo | None = Field(None, description="Confidence scores")
    latency_ms: float = Field(..., description="Total response latency")
    timestamp: datetime = Field(default_factory=datetime.now)
    warnings: list[str] = Field(default_factory=list, description="Any warnings")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "response": "You should consult with your doctor before taking aspirin with warfarin. Both medications affect blood clotting and combining them significantly increases bleeding risk. If pain relief is needed, acetaminophen (Tylenol) is generally safer with warfarin.",
                    "route": {
                        "route_type": "symbolic_first",
                        "strategy": "adaptive",
                        "complexity_score": 0.75,
                        "complexity_category": "high",
                        "is_safety_critical": True,
                        "latency_budget_ms": 5000
                    },
                    "verification": {
                        "passed": True,
                        "contracts_checked": 8,
                        "violations": [],
                        "warnings": []
                    },
                    "symbolic_check": {
                        "is_safe": False,
                        "risk_level": "HIGH",
                        "violations": [
                            {
                                "constraint_type": "drug_interaction",
                                "severity": "major",
                                "description": "Warfarin + Aspirin: Significantly increased bleeding risk"
                            }
                        ],
                        "warnings": [],
                        "verified_constraints": 3
                    },
                    "confidence": {
                        "overall": 0.85,
                        "uncertainty_score": 0.8,
                        "quality_score": 0.9,
                        "safety_score": 0.85,
                        "grade": "B"
                    },
                    "latency_ms": 1234.5,
                    "warnings": []
                }
            ]
        }
    }


class MedicationCheckRequest(BaseModel):
    """Request for medication safety check."""
    
    drug: str = Field(..., description="Drug name to check")
    dose_mg: float = Field(..., ge=0, description="Proposed dose in mg")
    frequency_per_day: int = Field(1, ge=1, le=24, description="Dosing frequency per day")
    patient: PatientInfo = Field(..., description="Patient information")


class MedicationCheckResponse(BaseModel):
    """Response for medication safety check."""
    
    drug: str = Field(..., description="Drug checked")
    is_safe: bool = Field(..., description="Overall safety determination")
    risk_level: str = Field(..., description="Risk level")
    violations: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    verified_constraints: int = Field(...)
    solver_time_ms: float = Field(...)


class DrugInteractionRequest(BaseModel):
    """Request for drug interaction check."""
    
    drugs: list[str] = Field(..., min_length=2, description="List of drugs to check")
    patient: PatientInfo | None = Field(None, description="Optional patient context")


class DrugInteractionResponse(BaseModel):
    """Response for drug interaction check."""
    
    interactions: list[dict[str, Any]] = Field(...)
    overall_risk: str = Field(...)
    recommendations: list[str] = Field(default_factory=list)


class HealthCheckResponse(BaseModel):
    """Health check response."""
    
    status: str = Field("healthy", description="Service status")
    version: str = Field(..., description="API version")
    components: dict[str, str] = Field(default_factory=dict, description="Component statuses")
    timestamp: datetime = Field(default_factory=datetime.now)


class MetricsResponse(BaseModel):
    """Metrics response for monitoring."""
    
    routing_stats: dict[str, Any] = Field(...)
    verification_stats: dict[str, Any] = Field(...)
    drift_metrics: dict[str, Any] = Field(...)
    latency_stats: dict[str, Any] = Field(...)


class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    details: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.now)
