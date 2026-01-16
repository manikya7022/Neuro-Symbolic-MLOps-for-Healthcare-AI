# Neuro-Symbolic MLOps for Healthcare AI

A **production-ready hybrid reasoning system** that combines neural LLMs with symbolic solvers for healthcare compliance checking with real-time adaptive routing.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![Status: Research](https://img.shields.io/badge/Status-Research-orange.svg)

## 🎯 Overview

This system addresses a fundamental challenge in AI: **LLMs (neural) excel at flexibility but lack consistency; symbolic solvers are rigid but guaranteed.** Our solution dynamically orchestrates both approaches in real-time.

### Key Features

- **Adaptive Query Router**: Dynamically allocates queries to neural vs. symbolic components based on complexity, safety requirements, and latency budgets
- **Uncertainty Quantification**: Real-time confidence scoring to trigger symbolic fallbacks when neural confidence is low
- **Neural-Symbolic Contracts**: Formal verification of neural outputs against symbolic constraints
- **Logic Drift Detection**: Novel MLOps feature that detects when neural outputs increasingly violate symbolic constraints
- **Dual-Track Versioning**: MLflow for neural weights + DVC for knowledge bases

### 💡 Why This Matters

Modern healthcare AI faces a critical dilemma: **Large Language Models offer unprecedented flexibility in understanding natural language medical queries, but they hallucinate, lack consistency, and cannot guarantee safety-critical constraints.** A patient asking "Can I take aspirin with warfarin?" needs more than a fluent response—they need a *correct* one. Traditional symbolic systems (rule engines, constraint solvers) provide mathematical guarantees but are brittle and cannot handle the nuance of human language.

This project bridges that gap by creating a **hybrid architecture where neural and symbolic systems complement each other in real-time**. When an LLM is confident and the query is low-risk, it responds directly. When uncertainty is high or the query involves drug interactions, dosages, or contraindications, symbolic verification kicks in—checking formal constraints before any response reaches the patient. The result is an AI system that combines the *accessibility* of conversational AI with the *safety guarantees* of formal medical knowledge bases.

For healthcare organizations, researchers, and developers, this represents a paradigm shift: **you no longer have to choose between flexibility and safety.** The system adapts dynamically, routing simple educational queries through fast neural paths while escalating safety-critical decisions to verified symbolic reasoning. With integrated drift detection, you can monitor in production when your LLM starts violating medical constraints—catching model degradation before it impacts patients. This is not just an AI assistant; it's a framework for **trustworthy, auditable, and deployable healthcare AI**.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FastAPI Gateway (REST + WebSocket)                │
└─────────────────────────────────────┬───────────────────────────────────────┘
                                      │
┌─────────────────────────────────────▼───────────────────────────────────────┐
│                          ADAPTIVE QUERY ROUTER                               │
│   Complexity Estimator │ Latency Budget │ Uncertainty Threshold             │
└─────────────────────────────────────┬───────────────────────────────────────┘
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
          ┌─────────────────┐                 ┌─────────────────┐
          │  NEURAL TRACK   │                 │ SYMBOLIC TRACK  │
          │  • Ollama LLM   │                 │  • Z3 Solver    │
          │  • Uncertainty  │                 │  • Knowledge    │
          │    Quantifier   │                 │    Base         │
          │  • Confidence   │                 │  • Rule Engine  │
          └────────┬────────┘                 └────────┬────────┘
                   └──────────────┬───────────────────┘
                                  ▼
          ┌───────────────────────────────────────────────────────┐
          │           NEURAL-SYMBOLIC VERIFIER                     │
          │  Contract Checker │ Logic Drift Detector               │
          └───────────────────────────────────────────────────────┘
                                  │
          ┌───────────────────────▼───────────────────────────────┐
          │              DUAL-TRACK MLOps                          │
          │  MLflow (Weights) │ DVC (Knowledge Base)               │
          └───────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) (for local LLM inference)
- Docker & Docker Compose (optional, for full stack)

### Installation

```bash
# Clone the repository
cd neuro-symbolic-hybrid-reasoning

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -e ".[dev]"

# Pull Ollama model
ollama pull llama3.2:3b
```

### Run the API Server

```bash
# Start the server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for the interactive API documentation.

### Example Query

```python
import httpx

response = httpx.post("http://localhost:8000/api/v1/query", json={
    "query": "Can I take aspirin with my blood thinner warfarin?",
    "patient": {
        "age": 65,
        "current_medications": ["warfarin"],
        "conditions": ["atrial fibrillation"]
    },
    "priority": "high"
})

result = response.json()
print(f"Response: {result['response']}")
print(f"Route: {result['route']['route_type']}")
print(f"Safe: {result['symbolic_check']['is_safe']}")
print(f"Confidence: {result['confidence']['grade']}")
```

### Run with Docker Compose

```bash
# Start all services (Ollama, Redis, Prometheus, Grafana, MLflow)
docker-compose up -d

# Check health
curl http://localhost:8000/health
```

## 📁 Project Structure

```
neuro-symbolic-hybrid-reasoning/
├── src/
│   ├── router/             # Adaptive routing algorithms
│   │   ├── complexity_estimator.py
│   │   ├── latency_manager.py
│   │   └── decision_engine.py
│   ├── neural/             # LLM wrappers, uncertainty, confidence
│   │   ├── llm_client.py
│   │   ├── uncertainty.py
│   │   └── confidence.py
│   ├── symbolic/           # Z3 solver, knowledge base, rules
│   │   ├── z3_solver.py
│   │   ├── knowledge_base.py
│   │   └── rule_engine.py
│   ├── verification/       # Contract checking, drift detection
│   │   ├── contract_checker.py
│   │   └── drift_detector.py
│   └── api/                # FastAPI endpoints
│       ├── main.py
│       └── schemas.py
├── knowledge_bases/        # Medical ontologies and rules
├── configs/                # YAML configurations
├── tests/                  # Unit and integration tests
├── docker-compose.yml      # Local infrastructure
└── pyproject.toml          # Project configuration
```

## 🔬 Research Contributions

### 1. Adaptive Routing Algorithm

Query-aware routing that considers:
- **Complexity**: Embedding-based + heuristic estimation
- **Safety-criticality**: Keyword and context detection
- **Latency budget**: SLA-aware allocation

### 2. Neural-Symbolic Contracts

Formal specification of output constraints:
```
CONTRACT MedicationResponse:
    ENSURES no_contraindicated_drugs(patient_allergies, recommended_drugs)
    ENSURES dosage_within_limits(patient_weight, recommended_dosage)
    ON_VIOLATION: trigger_symbolic_recalculation()
```

### 3. Logic Drift Detection

Statistical tracking of constraint violations over time for CI/CD gates:
- Sliding window violation rate
- Per-constraint drift scores
- Automatic alerting

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/query` | POST | Main healthcare query endpoint |
| `/api/v1/medication/check` | POST | Direct medication safety check |
| `/api/v1/interactions` | POST | Drug-drug interaction check |
| `/api/v1/drugs/{name}` | GET | Get drug information |
| `/api/v1/conditions/{name}` | GET | Get condition information |
| `/api/v1/metrics` | GET | System metrics |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |



## 🛠️ Configuration

All configuration is in `configs/base.yaml`:

```yaml
router:
  strategy: "adaptive"  # adaptive, neural_first, symbolic_first, hybrid
  
neural:
  ollama:
    model: "llama3.2:3b"
    
symbolic:
  z3:
    timeout: 30000  # ms
    
verification:
  drift_detection:
    alert_threshold: 0.05  # 5% violation rate
```

## 📚 Further Reading

- [Implementation Plan](./docs/implementation_plan.md)
- [API Reference](http://localhost:8000/docs)
- [Architecture Deep Dive](./docs/architecture.md)



MIT License - see [LICENSE](LICENSE) for details.
