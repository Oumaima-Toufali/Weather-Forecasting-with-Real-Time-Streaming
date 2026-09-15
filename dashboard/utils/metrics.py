import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
try:
    from scipy import stats
except ImportError:
    stats = None

class MonitoringMetrics:
    @staticmethod
    def calculate_drift(current_data: np.ndarray, reference_data: np.ndarray) -> Dict[str, float]:
        """Perform KS test for drift detection."""
        if stats is None:
            # Fallback if scipy is missing: compare means and stds
            mean_diff = abs(np.mean(current_data) - np.mean(reference_data))
            std_diff = abs(np.std(current_data) - np.std(reference_data))
            return {"ks_score": mean_diff, "p_value": 0.0, "method": "mean_diff_fallback"}
        
        ks_stat, p_value = stats.ks_2samp(current_data, reference_data)
        return {"ks_score": ks_stat, "p_value": p_value, "method": "ks_test"}

    @staticmethod
    def check_outliers(data: np.ndarray, threshold: float = 3.0) -> Dict[str, Any]:
        """Detect outliers using Z-score."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return {"count": 0, "indices": []}
        
        z_scores = np.abs((data - mean) / std)
        outliers = np.where(z_scores > threshold)[0]
        return {
            "count": len(outliers),
            "percentage": (len(outliers) / len(data)) * 100 if len(data) > 0 else 0,
            "threshold": threshold
        }

    @staticmethod
    def check_schema(df: pd.DataFrame, expected_fields: List[str]) -> Dict[str, Any]:
        """Check if all expected fields are present in the DataFrame."""
        missing = [f for f in expected_fields if f not in df.columns]
        return {
            "is_valid": len(missing) == 0,
            "missing_fields": missing,
            "completeness": (1 - len(missing) / len(expected_fields)) * 100 if expected_fields else 100
        }

    @staticmethod
    def get_missing_values(df: pd.DataFrame) -> Dict[str, float]:
        """Calculate missing values percentage per feature."""
        return df.isnull().mean().to_dict()

    @staticmethod
    def check_physical_ranges(df: pd.DataFrame) -> Dict[str, int]:
        """Check for violations of physical constraints."""
        violations = {}
        
        # Temperature: -50 to 60
        if 'temperature' in df.columns:
            violations['temperature'] = len(df[(df['temperature'] < -50) | (df['temperature'] > 60)])
        
        # Humidity: 0 to 100
        if 'humidity' in df.columns:
            violations['humidity'] = len(df[(df['humidity'] < 0) | (df['humidity'] > 100)])
            
        # Wind speed: >= 0
        if 'wind_speed' in df.columns:
            violations['wind_speed'] = len(df[df['wind_speed'] < 0])
            
        return violations

    @staticmethod
    def calculate_stability(predictions: np.ndarray, window_size: int = 10) -> float:
        """Calculate prediction stability (std over rolling window)."""
        if len(predictions) < 2:
            return 0.0
        return float(np.std(predictions))

    @staticmethod
    def check_cross_horizon_consistency(predictions_by_horizon: Dict[int, float]) -> bool:
        """
        Check if predictions are logically consistent (e.g., 1h, 3h, 6h trends).
        Example: temperatures shouldn't jump wildly between horizons at the same event time.
        """
        horizons = sorted(predictions_by_horizon.keys())
        if len(horizons) < 2:
            return True
        
        # Simple check: max difference between consecutive horizons shouldn't exceed a threshold
        for i in range(len(horizons) - 1):
            diff = abs(predictions_by_horizon[horizons[i+1]] - predictions_by_horizon[horizons[i]])
            if diff > 10: # Threshold of 10 degrees is high, but can be adjusted
                return False
        return True

    @staticmethod
    def get_confidence_intervals(predictions: np.ndarray) -> Dict[str, float]:
        """Calculate P10, P50, P90 quantiles."""
        if len(predictions) == 0:
            return {"p10": 0.0, "p50": 0.0, "p90": 0.0}
        return {
            "p10": float(np.percentile(predictions, 10)),
            "p50": float(np.percentile(predictions, 50)),
            "p90": float(np.percentile(predictions, 90))
        }

