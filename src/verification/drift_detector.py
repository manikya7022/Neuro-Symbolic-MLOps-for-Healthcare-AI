"""Logic Drift Detection for Neural-Symbolic Systems.

Detects when neural outputs increasingly violate symbolic constraints,
indicating model degradation or distribution shift.

Key features:
- Sliding window violation rate tracking
- Per-constraint drift scoring
- Statistical alerting with configurable thresholds
- CI/CD gate integration
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
import structlog

from src.config import settings
from src.verification.contract_checker import ContractViolation, ViolationSeverity

logger = structlog.get_logger(__name__)


class AlertLevel(str, Enum):
    """Alert severity levels."""
    
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DriftAlert:
    """An alert triggered by drift detection."""
    
    level: AlertLevel
    metric: str
    current_value: float
    threshold: float
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class DriftMetrics:
    """Current drift metrics."""
    
    overall_violation_rate: float
    violation_rates_by_contract: dict[str, float]
    violation_rates_by_severity: dict[str, float]
    drift_scores: dict[str, float]
    trend: str  # "stable", "increasing", "decreasing"
    window_size: int
    observations: int
    alerts: list[DriftAlert] = field(default_factory=list)
    
    @property
    def is_drifting(self) -> bool:
        """Check if significant drift is detected."""
        config = settings.verification.drift_detection
        return self.overall_violation_rate > config.alert_threshold
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "overall_violation_rate": self.overall_violation_rate,
            "violation_rates_by_contract": self.violation_rates_by_contract,
            "violation_rates_by_severity": self.violation_rates_by_severity,
            "drift_scores": self.drift_scores,
            "trend": self.trend,
            "window_size": self.window_size,
            "observations": self.observations,
            "is_drifting": self.is_drifting,
            "alert_count": len(self.alerts),
        }


class DriftDetector:
    """Detects logic drift in neural-symbolic systems.
    
    Tracks violations over time and detects when:
    - Overall violation rate exceeds threshold
    - Specific contract violations spike
    - Trend indicates increasing violations
    
    Designed for CI/CD integration - can be used as a deployment gate.
    
    Example:
        ```python
        detector = DriftDetector()
        
        # Record observations
        for result in contract_results:
            detector.record(result.violations)
        
        # Check for drift
        metrics = detector.get_metrics()
        if metrics.is_drifting:
            # Fail CI/CD gate
            raise DriftDetectedError(metrics)
        ```
    """
    
    def __init__(
        self,
        window_size: int | None = None,
        alert_threshold: float | None = None,
    ) -> None:
        """Initialize drift detector.
        
        Args:
            window_size: Sliding window size for metrics
            alert_threshold: Threshold for triggering alerts
        """
        config = settings.verification.drift_detection
        self.enabled = config.enabled
        self.window_size = window_size or config.window_size
        self.alert_threshold = alert_threshold or config.alert_threshold
        self.metrics_interval = config.metrics_interval
        
        # Sliding windows for observations
        self._observations: deque[bool] = deque(maxlen=self.window_size)
        self._violations: deque[ContractViolation] = deque(maxlen=self.window_size)
        self._timestamps: deque[float] = deque(maxlen=self.window_size)
        
        # Per-contract tracking
        self._contract_violations: dict[str, deque[bool]] = {}
        
        # Baseline (established during calibration)
        self._baseline_rate: float | None = None
        self._baseline_observations: int = 0
        
        # Alert history
        self._alerts: list[DriftAlert] = []
        self._last_metrics_time: float = 0
        
        # Callbacks for alerts
        self._alert_callbacks: list[Any] = []
    
    def record(
        self,
        violations: list[ContractViolation],
        had_violation: bool | None = None,
    ) -> None:
        """Record an observation.
        
        Args:
            violations: List of violations from contract checking
            had_violation: Whether this observation had any violation
                          (auto-detected if None)
        """
        if not self.enabled:
            return
        
        current_time = time.time()
        self._timestamps.append(current_time)
        
        # Record overall observation
        had_any = had_violation if had_violation is not None else len(violations) > 0
        self._observations.append(had_any)
        
        # Record individual violations
        for violation in violations:
            self._violations.append(violation)
            
            # Track per-contract
            contract_id = violation.contract_id
            if contract_id not in self._contract_violations:
                self._contract_violations[contract_id] = deque(maxlen=self.window_size)
            self._contract_violations[contract_id].append(True)
        
        # Pad per-contract tracking for contracts that didn't violate
        for contract_id in self._contract_violations:
            if not any(v.contract_id == contract_id for v in violations):
                self._contract_violations[contract_id].append(False)
        
        # Check for alerts periodically
        if current_time - self._last_metrics_time >= self.metrics_interval:
            self._check_and_alert()
            self._last_metrics_time = current_time
    
    def get_metrics(self) -> DriftMetrics:
        """Calculate current drift metrics.
        
        Returns:
            DriftMetrics with current state
        """
        observations = list(self._observations)
        violations = list(self._violations)
        
        if not observations:
            return DriftMetrics(
                overall_violation_rate=0.0,
                violation_rates_by_contract={},
                violation_rates_by_severity={},
                drift_scores={},
                trend="stable",
                window_size=self.window_size,
                observations=0,
                alerts=self._alerts[-10:],  # Last 10 alerts
            )
        
        # Overall violation rate
        overall_rate = sum(observations) / len(observations)
        
        # Per-contract violation rates
        contract_rates = {}
        for contract_id, contract_obs in self._contract_violations.items():
            if contract_obs:
                contract_rates[contract_id] = sum(contract_obs) / len(contract_obs)
        
        # Per-severity violation rates
        severity_counts: dict[str, int] = {}
        for v in violations:
            sev = v.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        total_violations = len(violations)
        severity_rates = {
            sev: count / total_violations if total_violations > 0 else 0.0
            for sev, count in severity_counts.items()
        }
        
        # Calculate drift scores (deviation from baseline)
        drift_scores = self._calculate_drift_scores(overall_rate, contract_rates)
        
        # Determine trend
        trend = self._calculate_trend()
        
        return DriftMetrics(
            overall_violation_rate=overall_rate,
            violation_rates_by_contract=contract_rates,
            violation_rates_by_severity=severity_rates,
            drift_scores=drift_scores,
            trend=trend,
            window_size=self.window_size,
            observations=len(observations),
            alerts=self._alerts[-10:],
        )
    
    def _calculate_drift_scores(
        self,
        current_rate: float,
        contract_rates: dict[str, float],
    ) -> dict[str, float]:
        """Calculate drift scores relative to baseline.
        
        Args:
            current_rate: Current overall violation rate
            contract_rates: Current per-contract rates
        
        Returns:
            Dictionary of drift scores
        """
        scores = {}
        
        if self._baseline_rate is not None:
            # Overall drift (how much higher than baseline)
            scores["overall"] = max(0, current_rate - self._baseline_rate)
        else:
            scores["overall"] = current_rate
        
        # Per-contract drift scores
        for contract_id, rate in contract_rates.items():
            # Simple scoring: higher rate = higher drift
            scores[contract_id] = rate
        
        return scores
    
    def _calculate_trend(self) -> str:
        """Calculate violation trend.
        
        Returns:
            "stable", "increasing", or "decreasing"
        """
        observations = list(self._observations)
        if len(observations) < 20:
            return "stable"
        
        # Compare first half to second half
        mid = len(observations) // 2
        first_half = observations[:mid]
        second_half = observations[mid:]
        
        first_rate = sum(first_half) / len(first_half)
        second_rate = sum(second_half) / len(second_half)
        
        diff = second_rate - first_rate
        
        if diff > 0.05:
            return "increasing"
        elif diff < -0.05:
            return "decreasing"
        return "stable"
    
    def _check_and_alert(self) -> None:
        """Check metrics and trigger alerts if needed."""
        metrics = self.get_metrics()
        
        # Check overall threshold
        if metrics.overall_violation_rate > self.alert_threshold:
            self._trigger_alert(
                level=AlertLevel.CRITICAL if metrics.overall_violation_rate > self.alert_threshold * 2 else AlertLevel.WARNING,
                metric="overall_violation_rate",
                current=metrics.overall_violation_rate,
                threshold=self.alert_threshold,
                message=f"Violation rate {metrics.overall_violation_rate:.2%} exceeds threshold {self.alert_threshold:.2%}",
            )
        
        # Check for critical severity spikes
        critical_rate = metrics.violation_rates_by_severity.get("critical", 0)
        if critical_rate > 0.01:  # More than 1% critical violations
            self._trigger_alert(
                level=AlertLevel.CRITICAL,
                metric="critical_violation_rate",
                current=critical_rate,
                threshold=0.01,
                message=f"Critical violation rate {critical_rate:.2%} detected",
            )
        
        # Check trend
        if metrics.trend == "increasing" and metrics.overall_violation_rate > self.alert_threshold / 2:
            self._trigger_alert(
                level=AlertLevel.WARNING,
                metric="trend",
                current=metrics.overall_violation_rate,
                threshold=self.alert_threshold,
                message="Violation rate showing increasing trend",
            )
    
    def _trigger_alert(
        self,
        level: AlertLevel,
        metric: str,
        current: float,
        threshold: float,
        message: str,
    ) -> None:
        """Trigger a drift alert.
        
        Args:
            level: Alert severity level
            metric: Metric that triggered alert
            current: Current value
            threshold: Threshold that was exceeded
            message: Alert message
        """
        alert = DriftAlert(
            level=level,
            metric=metric,
            current_value=current,
            threshold=threshold,
            message=message,
        )
        
        self._alerts.append(alert)
        
        logger.warning(
            "drift_alert",
            level=level.value,
            metric=metric,
            current=current,
            threshold=threshold,
            message=message,
        )
        
        # Call registered callbacks
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error("alert_callback_error", error=str(e))
    
    def calibrate(self, baseline_observations: int = 100) -> None:
        """Establish baseline from current observations.
        
        Args:
            baseline_observations: Number of observations to use
        """
        if len(self._observations) < baseline_observations:
            logger.warning(
                "insufficient_observations_for_calibration",
                current=len(self._observations),
                required=baseline_observations,
            )
            return
        
        observations = list(self._observations)[-baseline_observations:]
        self._baseline_rate = sum(observations) / len(observations)
        self._baseline_observations = len(observations)
        
        logger.info(
            "drift_detector_calibrated",
            baseline_rate=self._baseline_rate,
            observations=self._baseline_observations,
        )
    
    def reset(self) -> None:
        """Reset detector state."""
        self._observations.clear()
        self._violations.clear()
        self._timestamps.clear()
        self._contract_violations.clear()
        self._alerts.clear()
        self._baseline_rate = None
        logger.info("drift_detector_reset")
    
    def register_alert_callback(self, callback: Any) -> None:
        """Register a callback for alerts.
        
        Args:
            callback: Function to call on alerts
        """
        self._alert_callbacks.append(callback)
    
    def check_ci_gate(self, max_violation_rate: float | None = None) -> tuple[bool, str]:
        """Check if drift is acceptable for CI/CD gate.
        
        Args:
            max_violation_rate: Maximum acceptable violation rate
        
        Returns:
            Tuple of (passed, message)
        """
        max_rate = max_violation_rate or self.alert_threshold
        metrics = self.get_metrics()
        
        if metrics.observations < 10:
            return True, "Insufficient observations for gate check"
        
        if metrics.overall_violation_rate > max_rate:
            return False, f"Drift detected: {metrics.overall_violation_rate:.2%} > {max_rate:.2%}"
        
        if metrics.violation_rates_by_severity.get("critical", 0) > 0:
            return False, "Critical violations detected"
        
        if metrics.trend == "increasing" and metrics.overall_violation_rate > max_rate / 2:
            return False, "Increasing violation trend detected"
        
        return True, f"Gate passed: {metrics.overall_violation_rate:.2%} violation rate"
    
    def export_metrics(self) -> dict[str, Any]:
        """Export metrics for monitoring systems.
        
        Returns:
            Dictionary formatted for Prometheus/Grafana
        """
        metrics = self.get_metrics()
        
        return {
            "nsh_drift_violation_rate": metrics.overall_violation_rate,
            "nsh_drift_observations": metrics.observations,
            "nsh_drift_is_drifting": int(metrics.is_drifting),
            "nsh_drift_alert_count": len(metrics.alerts),
            "nsh_drift_trend": metrics.trend,
            **{
                f"nsh_drift_contract_{k}_rate": v
                for k, v in metrics.violation_rates_by_contract.items()
            },
            **{
                f"nsh_drift_severity_{k}_rate": v
                for k, v in metrics.violation_rates_by_severity.items()
            },
        }
