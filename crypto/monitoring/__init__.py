"""
Crypto Beta Monitoring System.

A suite of tools for adaptive beta monitoring in crypto statistical arbitrage.

Philosophy:
- ML for MEASUREMENT, not PREDICTION
- Validate static assumptions in real-time
- Provide early warning for relationship breakdown
- Adjust position sizing based on confidence

Components:
- correlation_stability: Track pair correlations across time windows (Priority 1)
- ensemble_beta: Multi-window beta estimates with confidence weighting (Priority 2)
- residual_anomaly: Detect moves that deviate from beta expectations (Priority 3)
- kalman_beta: Time-varying beta estimation via Kalman filter (Priority 4)
- dashboard: Unified interface combining all components

Session Origin: February 2, 2026
Catalyst: Feb 1-2 crypto selloff validated beta calculations while exposing
          the need for adaptive monitoring (SOL underperformed by 14%)
"""

# Priority 1: Correlation Stability
from .correlation_stability import (
    CorrelationStabilityTracker,
    CorrelationStabilityResult,
    quick_stability_check,
    calculate_position_size_multiplier,
    # Configuration
    SHORT_WINDOW,
    LONG_WINDOW,
    STABILITY_THRESHOLD,
)

# Priority 2: Ensemble Beta
from .ensemble_beta import (
    EnsembleBetaTracker,
    EnsembleBetaResult,
    BetaEstimate,
    quick_ensemble_beta,
    get_beta_adjustment_factor,
    # Configuration
    BETA_WINDOWS,
)

# Priority 3: Residual Anomaly Detection
from .residual_anomaly import (
    ResidualAnomalyDetector,
    ResidualAnomaly,
    AnomalyReport,
    quick_anomaly_check,
    validate_february_selloff,
    # Configuration
    ANOMALY_THRESHOLD_WARNING,
    ANOMALY_THRESHOLD_CRITICAL,
)

# Priority 4: Kalman Filter Beta
from .kalman_beta import (
    KalmanBetaFilter,
    KalmanBetaEstimate,
    KalmanBetaHistory,
    KalmanBetaTracker,
    quick_kalman_beta,
    estimate_kalman_parameters,
    validate_kalman_vs_static,
)

# Dashboard
from .dashboard import (
    BetaMonitoringDashboard,
    DashboardSnapshot,
    AssetMonitoringResult,
    PairMonitoringResult,
    create_dashboard,
    quick_check,
)

__all__ = [
    # Priority 1
    'CorrelationStabilityTracker',
    'CorrelationStabilityResult',
    'quick_stability_check',
    'calculate_position_size_multiplier',
    'SHORT_WINDOW',
    'LONG_WINDOW',
    'STABILITY_THRESHOLD',
    
    # Priority 2
    'EnsembleBetaTracker',
    'EnsembleBetaResult',
    'BetaEstimate',
    'quick_ensemble_beta',
    'get_beta_adjustment_factor',
    'BETA_WINDOWS',
    
    # Priority 3
    'ResidualAnomalyDetector',
    'ResidualAnomaly',
    'AnomalyReport',
    'quick_anomaly_check',
    'validate_february_selloff',
    'ANOMALY_THRESHOLD_WARNING',
    'ANOMALY_THRESHOLD_CRITICAL',
    
    # Priority 4
    'KalmanBetaFilter',
    'KalmanBetaEstimate',
    'KalmanBetaHistory',
    'KalmanBetaTracker',
    'quick_kalman_beta',
    'estimate_kalman_parameters',
    'validate_kalman_vs_static',
    
    # Dashboard
    'BetaMonitoringDashboard',
    'DashboardSnapshot',
    'AssetMonitoringResult',
    'PairMonitoringResult',
    'create_dashboard',
    'quick_check',
]

__version__ = '1.0.0'
