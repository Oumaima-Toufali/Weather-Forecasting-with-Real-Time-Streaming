# Path: backend/serving/predict_consumer.py
"""
Streaming prediction consumer.

Consumes feature vectors from Kafka, validates with Avro, runs RandomForest
models trained offline, and produces per-horizon predictions to Kafka.

CLI:
    python -m backend.serving.predict_consumer [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import pandas as pd
from confluent_kafka import Consumer, Producer

from backend.config.kafka_config import KAFKA_BOOTSTRAP_SERVERS, TOPICS
from backend.utils.schema_validator import load_avro_schema, validate_avro

# Reuse flattening logic from dataset builder to stay aligned with training
from backend.training.build_dataset import _flatten_feature_record  # type: ignore


class ModelBundle:
    """Hold a fitted model and its expected feature columns."""

    def __init__(self, horizon: int, model_path: Path, metadata_path: Path):
        self.horizon = horizon
        self.model = joblib.load(model_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.feature_columns: List[str] = metadata.get("feature_columns", [])
        if not self.feature_columns:
            raise ValueError("No feature_columns found in model metadata")

    def build_feature_vector(self, row: pd.Series):
        """
        Build the feature vector in the exact training order.

        Missing features are filled with 0.0 and listed for logging.
        """
        import numpy as np

        values = []
        missing: List[str] = []
        for col in self.feature_columns:
            if col in row and not pd.isna(row[col]):
                values.append(float(row[col]))
            else:
                missing.append(col)
                values.append(0.0)

        X = np.array(values, dtype=float).reshape(1, -1)
        return X, missing

    def predict_point(self, X) -> float:
        return float(self.model.predict(X)[0])

    def predict_quantiles(self, X):
        import numpy as np

        # Use distribution of individual trees to approximate quantiles
        tree_preds = np.array([est.predict(X)[0] for est in self.model.estimators_])
        p10, p50, p90 = np.percentile(tree_preds, [10, 50, 90]).tolist()
        return float(p10), float(p50), float(p90)


class PredictConsumer:
    """Consumes features and emits RandomForest predictions per horizon."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

        self.consumer = Consumer(
            {
                "bootstrap.servers": ",".join(KAFKA_BOOTSTRAP_SERVERS),
                "group.id": "weather-prediction-serving",
                "auto.offset.reset": "earliest",
                "enable.auto.commit": True,
            }
        )
        self.consumer.subscribe([TOPICS["features"]])

        self.producer = Producer({"bootstrap.servers": ",".join(KAFKA_BOOTSTRAP_SERVERS)})

        self.features_schema = load_avro_schema("backend/schemas/features_weather.avsc")
        self.pred_schema = load_avro_schema("backend/schemas/weather_prediction.avsc")
        self.pred_quant_schema = load_avro_schema("backend/schemas/weather_prediction_quantiles.avsc")

        models_dir = Path("models")
        metadata_path = models_dir / "model_metadata.json"
        self.models: Dict[int, ModelBundle] = {}
        for horizon in (1, 3, 6):
            model_path = models_dir / f"weather_model_{horizon}h.pkl"
            if not model_path.exists():
                print(f"[WARN] Model file missing: {model_path}")
                continue
            try:
                self.models[horizon] = ModelBundle(horizon, model_path, metadata_path)
            except Exception as exc:  # pragma: no cover
                print(f"[WARN] Could not load model {model_path}: {exc}")

        if not self.models:
            raise SystemExit("[ERROR] No models available; cannot start serving.")

    def _to_prediction_record(self, event_time: str, horizon: int, value: float) -> dict:
        now_iso = datetime.now(timezone.utc).isoformat()
        return {
            "event_time": event_time,
            "horizon_hours": horizon,
            "temperature_prediction": value,
            "model_name": "random_forest",
            "model_version": "v1",
            "created_at": now_iso,
        }

    def _to_quantile_record(self, event_time: str, horizon: int, p10: float, p50: float, p90: float) -> dict:
        now_iso = datetime.now(timezone.utc).isoformat()
        return {
            "event_time": event_time,
            "horizon_hours": horizon,
            "temperature_p10": p10,
            "temperature_p50": p50,
            "temperature_p90": p90,
            "model_name": "random_forest",
            "model_version": "v1",
            "created_at": now_iso,
        }

    def _produce_point(self, key: str, record: dict):
        if not validate_avro(record, self.pred_schema):
            print(f"[SKIP] Prediction failed Avro validation: {record}")
            return

        if self.dry_run:
            print(f"[DRY-RUN][point] {record}")
            return

        self.producer.produce(
            TOPICS["predictions"],
            key=key.encode("utf-8") if key else None,
            value=json.dumps(record).encode("utf-8"),
        )

    def _produce_quantile(self, key: str, record: dict):
        if not validate_avro(record, self.pred_quant_schema):
            print(f"[SKIP] Quantile prediction failed Avro validation: {record}")
            return

        if self.dry_run:
            print(f"[DRY-RUN][quantiles] {record}")
            return

        self.producer.produce(
            TOPICS["predictions_quantiles"],
            key=key.encode("utf-8") if key else None,
            value=json.dumps(record).encode("utf-8"),
        )

    def _flatten_and_validate(self, payload: dict) -> Optional[pd.Series]:
        if not validate_avro(payload, self.features_schema):
            print("[SKIP] Incoming message failed Avro validation")
            return None

        flat = _flatten_feature_record(payload)
        row = pd.Series(flat)
        return row

    def process_message(self, msg):
        try:
            payload = json.loads(msg.value().decode("utf-8"))
        except Exception as exc:
            print(f"[SKIP] Failed to decode message: {exc}")
            return

        row = self._flatten_and_validate(payload)
        if row is None:
            return

        event_time = payload.get("timestamp") or row.get("timestamp")
        key = msg.key().decode("utf-8") if msg.key() else None

        for horizon, bundle in self.models.items():
            # Build feature vector (fills missing with 0.0)
            X, missing = bundle.build_feature_vector(row)
            if missing:
                print(f"[WARN] Using 0.0 for missing features (horizon {horizon}h): {missing}")

            # Point forecast
            pred = bundle.predict_point(X)
            record = self._to_prediction_record(event_time, horizon, pred)
            self._produce_point(key, record)

            # Quantile forecast (p10/p50/p90) using distribution of tree predictions
            try:
                p10, p50, p90 = bundle.predict_quantiles(X)
                q_record = self._to_quantile_record(event_time, horizon, p10, p50, p90)
                self._produce_quantile(key, q_record)
            except Exception as exc:
                # We tolerate absence of quantile path; point prediction already produced
                print(f"[WARN] Quantile computation failed (horizon {horizon}h): {exc}")

        if not self.dry_run:
            self.producer.flush()

    def run(self):
        print("=" * 70)
        print("🌡️  WEATHER PREDICTION SERVING")
        print("=" * 70)
        print(f"📥 Input : {TOPICS['features']}")
        print(f"📤 Output: {TOPICS['predictions']}")
        print(f"🧠 Models: {list(self.models.keys())}")
        if self.dry_run:
            print("🔎 Mode: DRY RUN (no Kafka produce)")
        print("-" * 70)

        try:
            while True:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    print(f"[WARN] Consumer error: {msg.error()}")
                    continue
                self.process_message(msg)
        except KeyboardInterrupt:
            print("\n[INFO] Stopping prediction consumer.")
        finally:
            self.consumer.close()
            self.producer.flush()


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Kafka streaming prediction consumer")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not produce to Kafka, only print predictions",
    )
    args = parser.parse_args(argv)

    consumer = PredictConsumer(dry_run=args.dry_run)
    consumer.run()


if __name__ == "__main__":
    sys.exit(main())

