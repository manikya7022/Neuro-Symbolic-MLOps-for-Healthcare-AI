"""Pytest configuration and fixtures."""

import pytest
import asyncio


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_patient_context():
    """Sample patient context for testing."""
    return {
        "age": 65,
        "weight_kg": 70,
        "allergies": ["penicillin"],
        "conditions": ["hypertension", "type 2 diabetes"],
        "current_medications": ["lisinopril", "metformin"],
        "renal_function": "normal",
        "hepatic_function": "normal",
        "pregnant": False,
        "breastfeeding": False,
    }


@pytest.fixture
def sample_queries():
    """Sample queries for testing."""
    return [
        ("What is aspirin?", "low"),
        ("Can I take aspirin with warfarin?", "high"),
        ("Is 10000mg of ibuprofen safe?", "critical"),
        ("What are the side effects of metformin?", "medium"),
    ]
