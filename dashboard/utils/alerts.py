from typing import List, Dict, Any, Optional

class AlertManager:
    def __init__(self, thresholds: Optional[Dict[str, Any]] = None):
        self.thresholds = thresholds or {
            "kafka_lag_critical": 1000,
            "kafka_lag_warning": 500,
            "no_message_timeout_min": 5,
            "drift_score_critical": 0.3,
            "error_rate_warning": 5.0,
            "completeness_warning": 98.0
        }

    def check_alerts(self, system_stats: Dict[str, Any], data_metrics: Dict[str, Any], model_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        alerts = []

        # 1. Kafka Alerts
        for topic, stats in system_stats.items():
            lag = stats.get("total_messages", 0)
            if lag > self.thresholds["kafka_lag_critical"]:
                alerts.append({
                    "level": "CRITICAL",
                    "category": "System",
                    "message": f"High Kafka lag on {topic}: {lag} messages",
                    "value": lag
                })
            elif lag > self.thresholds["kafka_lag_warning"]:
                alerts.append({
                    "level": "WARNING",
                    "category": "System",
                    "message": f"Increased Kafka lag on {topic}: {lag} messages",
                    "value": lag
                })

        # 2. Data Quality Alerts
        for feature, metrics in data_metrics.items():
            drift = metrics.get("drift", {}).get("ks_score", 0)
            if drift > self.thresholds["drift_score_critical"]:
                alerts.append({
                    "level": "CRITICAL",
                    "category": "Data Quality",
                    "message": f"Feature drift detected: {feature} (score: {drift:.2f})",
                    "value": drift
                })
            
            completeness = metrics.get("completeness", 100.0)
            if completeness < self.thresholds["completeness_warning"]:
                alerts.append({
                    "level": "INFO",
                    "category": "Data Quality",
                    "message": f"Low feature completeness: {feature} ({completeness:.1f}%)",
                    "value": completeness
                })

        # 3. Model Alerts
        stability = model_metrics.get("stability", 0)
        if stability > 5.0: # Example threshold for std
            alerts.append({
                "level": "WARNING",
                "category": "Model",
                "message": f"High prediction instability detected (std: {stability:.2f})",
                "value": stability
            })

        return alerts

