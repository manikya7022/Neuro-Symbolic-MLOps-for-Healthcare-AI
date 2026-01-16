"""Configuration management for Neuro-Symbolic Healthcare System."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class OllamaConfig(BaseModel):
    """Ollama LLM configuration."""
    
    host: str = "http://localhost:11434"
    model: str = "llama3.2:3b"
    fallback_model: str = "mistral:7b"
    timeout: int = 60
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: int = 2048


class EmbeddingsConfig(BaseModel):
    """Embedding model configuration."""
    
    model: str = "all-MiniLM-L6-v2"
    cache_size: int = 10000
    batch_size: int = 32


class UncertaintyConfig(BaseModel):
    """Uncertainty quantification configuration."""
    
    method: str = "token_entropy"
    threshold: float = 0.7
    mc_samples: int = 5
    conformal_alpha: float = 0.1


class NeuralConfig(BaseModel):
    """Neural components configuration."""
    
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    uncertainty: UncertaintyConfig = Field(default_factory=UncertaintyConfig)


class Z3Config(BaseModel):
    """Z3 solver configuration."""
    
    timeout: int = 30000
    max_memory: int = 4096


class KnowledgeBaseConfig(BaseModel):
    """Knowledge base configuration."""
    
    path: str = "knowledge_bases/medical_ontology.owl"
    cache_enabled: bool = True
    reasoning_enabled: bool = True


class RulesConfig(BaseModel):
    """Symbolic rules configuration."""
    
    drug_interactions: str = "knowledge_bases/drug_interactions.json"
    clinical_guidelines: str = "knowledge_bases/clinical_guidelines.json"
    safety_protocols: str = "knowledge_bases/safety_protocols.json"


class SymbolicConfig(BaseModel):
    """Symbolic components configuration."""
    
    z3: Z3Config = Field(default_factory=Z3Config)
    knowledge_base: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)
    rules: RulesConfig = Field(default_factory=RulesConfig)


class ComplexityConfig(BaseModel):
    """Query complexity estimation configuration."""
    
    low_threshold: float = 0.3
    high_threshold: float = 0.7
    embedding_weight: float = 0.6
    heuristic_weight: float = 0.4


class LatencyConfig(BaseModel):
    """Latency budget configuration."""
    
    default_budget_ms: int = 5000
    neural_overhead_ms: int = 500
    symbolic_overhead_ms: int = 200
    streaming_enabled: bool = True


class SafetyConfig(BaseModel):
    """Safety-critical query detection configuration."""
    
    critical_keywords: list[str] = Field(default_factory=lambda: [
        "medication", "drug", "dosage", "allergy", "interaction",
        "contraindication", "side effect", "overdose", "pregnancy", "pediatric"
    ])
    always_verify: bool = True


class RouterConfig(BaseModel):
    """Adaptive router configuration."""
    
    strategy: str = "adaptive"
    complexity: ComplexityConfig = Field(default_factory=ComplexityConfig)
    latency: LatencyConfig = Field(default_factory=LatencyConfig)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)


class ContractsConfig(BaseModel):
    """Neural-symbolic contracts configuration."""
    
    enabled: bool = True
    strict_mode: bool = False
    log_violations: bool = True


class StreamingVerificationConfig(BaseModel):
    """Streaming verification configuration."""
    
    enabled: bool = True
    check_interval_tokens: int = 50
    buffer_size: int = 1000


class DriftDetectionConfig(BaseModel):
    """Logic drift detection configuration."""
    
    enabled: bool = True
    window_size: int = 1000
    alert_threshold: float = 0.05
    metrics_interval: int = 60


class VerificationConfig(BaseModel):
    """Verification module configuration."""
    
    contracts: ContractsConfig = Field(default_factory=ContractsConfig)
    streaming: StreamingVerificationConfig = Field(default_factory=StreamingVerificationConfig)
    drift_detection: DriftDetectionConfig = Field(default_factory=DriftDetectionConfig)


class MLflowConfig(BaseModel):
    """MLflow configuration."""
    
    tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "neuro-symbolic-healthcare"
    auto_log: bool = True


class NeuralVersioningConfig(BaseModel):
    """Neural model versioning configuration."""
    
    registry_name: str = "neural-models"
    auto_promote: bool = False


class SymbolicVersioningConfig(BaseModel):
    """Symbolic versioning configuration."""
    
    dvc_remote: str = "local"
    auto_commit: bool = True


class VersioningConfig(BaseModel):
    """Versioning configuration."""
    
    neural: NeuralVersioningConfig = Field(default_factory=NeuralVersioningConfig)
    symbolic: SymbolicVersioningConfig = Field(default_factory=SymbolicVersioningConfig)


class JointTestingConfig(BaseModel):
    """Joint testing configuration."""
    
    enabled: bool = True
    simulation_scenarios: list[str] = Field(default_factory=lambda: [
        "medication_safety", "drug_interaction", "dosage_verification"
    ])
    drift_gate_threshold: float = 0.02


class MLOpsConfig(BaseModel):
    """MLOps configuration."""
    
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    versioning: VersioningConfig = Field(default_factory=VersioningConfig)
    joint_testing: JointTestingConfig = Field(default_factory=JointTestingConfig)


class RedisConfig(BaseModel):
    """Redis cache configuration."""
    
    url: str = "redis://localhost:6379"
    default_ttl: int = 3600
    max_connections: int = 10


class LocalCacheConfig(BaseModel):
    """Local cache configuration."""
    
    enabled: bool = True
    max_size: int = 1000
    ttl: int = 600


class CacheConfig(BaseModel):
    """Cache configuration."""
    
    redis: RedisConfig = Field(default_factory=RedisConfig)
    local: LocalCacheConfig = Field(default_factory=LocalCacheConfig)


class APIConfig(BaseModel):
    """API configuration."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    cors_origins: list[str] = Field(default_factory=lambda: [
        "http://localhost:3000", "http://localhost:8080"
    ])


class Settings(BaseSettings):
    """Application settings loaded from environment and config files."""
    
    # Core settings
    app_name: str = "Neuro-Symbolic Healthcare AI"
    debug: bool = False
    log_level: str = "INFO"
    
    # Component configurations
    api: APIConfig = Field(default_factory=APIConfig)
    neural: NeuralConfig = Field(default_factory=NeuralConfig)
    symbolic: SymbolicConfig = Field(default_factory=SymbolicConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    mlops: MLOpsConfig = Field(default_factory=MLOpsConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    
    model_config = {
        "env_prefix": "NSH_",
        "env_nested_delimiter": "__",
        "extra": "ignore",  # Allow extra fields from YAML
    }


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from YAML file."""
    if not config_path.exists():
        return {}
    
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two configuration dictionaries."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings."""
    # Load base config
    config_dir = Path(__file__).parent.parent / "configs"
    base_config = load_yaml_config(config_dir / "base.yaml")
    
    # Load environment-specific config
    env = os.getenv("NSH_ENV", "development")
    env_config = load_yaml_config(config_dir / f"{env}.yaml")
    
    # Merge configs
    merged = merge_configs(base_config, env_config)
    
    # Create settings with merged config
    return Settings(**merged)


# Global settings instance
settings = get_settings()
