# Path: backend/training/build_dataset.py
"""
Builds an offline historical dataset from the `data.features.hourly` topic.

Usage:
    python -m backend.training.build_dataset --horizon 3 --max-messages 5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
from confluent_kafka import Consumer

from backend.config.kafka_config import KAFKA_BOOTSTRAP_SERVERS, TOPICS
from backend.utils.schema_validator import load_avro_schema, validate_avro


FEATURES_SCHEMA = load_avro_schema("backend/schemas/features_weather.avsc")


def _make_consumer(group_id: str | None = None) -> Consumer:
    """
    Build a confluent_kafka consumer using existing bootstrap servers.
    
    If group_id is None, creates a unique group ID based on timestamp
    to allow reading all messages from the beginning.
    """
    if group_id is None:
        from datetime import datetime
        group_id = f"dataset-builder-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    return Consumer(
        {
            "bootstrap.servers": ",".join(KAFKA_BOOTSTRAP_SERVERS),
            "group.id": group_id,
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )


def _flatten_feature_record(record: dict) -> dict:
    """Flatten nested feature payload into a flat row for the dataset."""
    location = record.get("location", {}) or {}
    base = record.get("base_features", {}) or {}
    cyclical = record.get("cyclical_features", {}) or {}
    historical = record.get("historical_features", {}) or {}
    targets = record.get("target_labels", {}) or {}
    
    # Extract rolling stats
    rolling_3h = historical.get("temp_rolling_3h") or {}
    rolling_6h = historical.get("temp_rolling_6h") or {}
    temp_deltas = historical.get("temp_deltas") or {}
    temp_lags = historical.get("temp_lags") or {}
    
    row = {
        "timestamp": record.get("timestamp"),
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "elevation": location.get("elevation"),
        # Base features
        "temperature": base.get("temp_current"),
        "humidity": base.get("humidity_current"),
        "wind_speed": base.get("wind_current"),
        "pressure": base.get("pressure_current"),
        "cloud_cover": base.get("cloud_cover_current"),
        "precipitation": base.get("precipitation_current"),
        # Cyclical features
        "hour_sin": cyclical.get("hour_sin"),
        "hour_cos": cyclical.get("hour_cos"),
        "day_sin": cyclical.get("day_sin"),
        "day_cos": cyclical.get("day_cos"),
        "is_day": cyclical.get("is_day"),
        "is_weekend": cyclical.get("is_weekend"),
        # Historical rolling stats (3h)
        "temp_mean_3h": rolling_3h.get("mean_3h") if isinstance(rolling_3h, dict) else None,
        "temp_std_3h": rolling_3h.get("std_3h") if isinstance(rolling_3h, dict) else None,
        "temp_min_3h": rolling_3h.get("min_3h") if isinstance(rolling_3h, dict) else None,
        "temp_max_3h": rolling_3h.get("max_3h") if isinstance(rolling_3h, dict) else None,
        # Historical rolling stats (6h)
        "temp_mean_6h": rolling_6h.get("mean_6h") if isinstance(rolling_6h, dict) else None,
        "temp_std_6h": rolling_6h.get("std_6h") if isinstance(rolling_6h, dict) else None,
        "temp_min_6h": rolling_6h.get("min_6h") if isinstance(rolling_6h, dict) else None,
        "temp_max_6h": rolling_6h.get("max_6h") if isinstance(rolling_6h, dict) else None,
        # Deltas
        "temp_delta_1h": temp_deltas.get("delta_1h") if isinstance(temp_deltas, dict) else None,
        "temp_delta_3h": temp_deltas.get("delta_3h") if isinstance(temp_deltas, dict) else None,
        "temp_delta_6h": temp_deltas.get("delta_6h") if isinstance(temp_deltas, dict) else None,
        "temp_delta_12h": temp_deltas.get("delta_12h") if isinstance(temp_deltas, dict) else None,
        # Rate of change
        "temp_roc_3h": historical.get("temp_roc_3h"),
        "temp_roc_6h": historical.get("temp_roc_6h"),
        # Lags
        "temp_lag_1h": temp_lags.get("lag_1h") if isinstance(temp_lags, dict) else None,
        "temp_lag_2h": temp_lags.get("lag_2h") if isinstance(temp_lags, dict) else None,
        "temp_lag_3h": temp_lags.get("lag_3h") if isinstance(temp_lags, dict) else None,
        "temp_lag_6h": temp_lags.get("lag_6h") if isinstance(temp_lags, dict) else None,
        "temp_lag_12h": temp_lags.get("lag_12h") if isinstance(temp_lags, dict) else None,
        # Volatility
        "temp_volatility_6h": historical.get("temp_volatility_6h"),
        # Original target labels (from API forecasts)
        "target_temp_1h": targets.get("temp_target_1h"),
        "target_temp_3h": targets.get("temp_target_3h"),
        "target_temp_6h": targets.get("temp_target_6h"),
    }
    
    return row


def build_dataframe(records: Iterable[dict]) -> pd.DataFrame:
    """Convert validated feature records into a sorted pandas DataFrame."""
    rows = [_flatten_feature_record(r) for r in records]
    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def add_prediction_target(df: pd.DataFrame, horizon_hours: int) -> pd.DataFrame:
    """
    Adds a target column: temperature shifted by +horizon_hours.

    Column name example: target_temperature_horizon_3h
    """
    if "temperature" not in df.columns:
        raise ValueError("Column 'temperature' is required to build targets.")

    horizon_rows = int(horizon_hours)
    target_col = f"target_temperature_horizon_{horizon_rows}h"

    df = df.copy()
    df[target_col] = df["temperature"].shift(-horizon_rows)
    return df


def load_existing_dataset(parquet_path: Path) -> pd.DataFrame | None:
    """Load existing dataset if it exists."""
    if not parquet_path.exists():
        return None
    
    try:
        existing_df = pd.read_parquet(parquet_path)
        if "timestamp" in existing_df.columns:
            existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"])
        return existing_df
    except Exception as e:
        print(f"[ATTENTION] Impossible de charger le dataset existant: {e}")
        return None


def get_duplicate_columns() -> list[str]:
    """
    Retourne la liste des colonnes à utiliser pour détecter les doublons complets.
    
    Un doublon est détecté uniquement si TOUTES ces features pertinentes
    pour la prédiction sont identiques (dans une tolérance).
    """
    return [
        "temperature",
        "humidity",
        "wind_speed",
        "pressure",
        "cloud_cover",
        "precipitation",
        # Note: on n'inclut pas les features cycliques (hour_sin, etc.)
        # car elles sont dérivées du timestamp et changent automatiquement
    ]


def merge_datasets(existing_df: pd.DataFrame | None, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge new data with existing dataset, removing duplicates based on ALL relevant features.
    
    Un doublon est détecté uniquement si TOUTES les features météorologiques
    pertinentes sont identiques, pas seulement le timestamp ou la température.
    """
    if existing_df is None or existing_df.empty:
        return new_df.copy()
    
    if new_df.empty:
        return existing_df.copy()
    
    # Combine both dataframes
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    
    # Convert timestamp to datetime if not already
    if "timestamp" in combined.columns:
        combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    
    # Obtenir les colonnes pour détection de doublons
    duplicate_cols = get_duplicate_columns()
    
    # Vérifier quelles colonnes existent dans le dataframe
    available_cols = [col for col in duplicate_cols if col in combined.columns]
    
    if not available_cols:
        # Fallback: utiliser timestamp si aucune feature n'est disponible
        print("[ATTENTION] Aucune feature pertinente trouvee, utilisation du timestamp pour deduplication")
        available_cols = ["timestamp"]
    
    # Arrondir les valeurs numériques pour éviter les faux doublons dus aux erreurs d'arrondi
    # (tolérances: temp 0.01°C, humidity 0.1%, wind 0.01 km/h, pressure 0.1 hPa, etc.)
    rounding_map = {
        "temperature": 2,      # 2 décimales (0.01°C)
        "humidity": 1,         # 1 décimale (0.1%)
        "wind_speed": 2,       # 2 décimales (0.01 km/h)
        "pressure": 1,         # 1 décimale (0.1 hPa)
        "cloud_cover": 1,      # 1 décimale (0.1%)
        "precipitation": 3,    # 3 décimales (0.001 mm)
    }
    
    combined_rounded = combined.copy()
    for col in available_cols:
        if col in rounding_map and col in combined_rounded.columns:
            combined_rounded[col] = combined_rounded[col].round(rounding_map[col])
    
    # Remove duplicates based on ALL relevant features (keep last occurrence)
    # Cela garantit qu'un doublon n'est détecté que si TOUTES les features sont identiques
    combined = combined.drop_duplicates(subset=available_cols, keep="last")
    
    # Sort by timestamp
    if "timestamp" in combined.columns:
        combined = combined.sort_values("timestamp").reset_index(drop=True)
    
    return combined


def save_dataset(df: pd.DataFrame, output_dir: Path | str) -> Tuple[Path, Path]:
    """
    Persist dataset to Parquet and CSV.
    
    Args:
        df: Dataframe to save (should already be merged if needed)
        output_dir: Directory to save files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "weather_features.parquet"
    csv_path = output_dir / "weather_features.csv"

    # Save with proper types and compression
    df.to_parquet(parquet_path, index=False, engine='pyarrow', compression='snappy')
    df.to_csv(csv_path, index=False)
    
    return parquet_path, csv_path


def consume_features(max_messages: int = 5000, idle_loops: int = 5) -> List[dict]:
    """Consume features from Kafka, Avro-validate, and return a list of records."""
    consumer = _make_consumer()
    consumer.subscribe([TOPICS["features"]])

    messages: List[dict] = []
    idle = 0

    try:
        while len(messages) < max_messages and idle < idle_loops:
            msg = consumer.poll(1.0)
            if msg is None:
                idle += 1
                continue

            if msg.error():
                # Log and continue (no logger here to keep module lightweight)
                idle += 1
                continue

            try:
                payload = json.loads(msg.value().decode("utf-8"))
            except Exception:
                idle += 1
                continue

            if not validate_avro(payload, FEATURES_SCHEMA):
                idle += 1
                continue

            messages.append(payload)
            idle = 0  # reset idle when we receive data
    finally:
        consumer.close()

    return messages


def main():
    parser = argparse.ArgumentParser(
        description="Build historical dataset from Kafka features topic.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build dataset with 3h horizon (default)
  python -m backend.training.build_dataset --horizon 3

  # Build dataset with custom horizon and message limit
  python -m backend.training.build_dataset --horizon 6 --max-messages 10000

  # Overwrite existing dataset (no merge)
  python -m backend.training.build_dataset --horizon 3 --overwrite
        """
    )
    parser.add_argument("--horizon", type=int, default=3, 
                       help="Prediction horizon in hours (default: 3).")
    parser.add_argument("--max-messages", type=int, default=5000, 
                       help="Max messages to consume from Kafka (default: 5000).")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/datasets",
        help="Output directory for parquet/csv files (default: data/datasets).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing dataset instead of merging (default: merge/append).",
    )
    parser.add_argument(
        "--idle-loops",
        type=int,
        default=5,
        help="Number of idle loops before stopping consumption (default: 5).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("📊 DATASET BUILDER - Historical Weather Features")
    print("=" * 70)
    print(f"📥 Topic: {TOPICS['features']}")
    print(f"🎯 Horizon: {args.horizon}h")
    print(f"📦 Max messages: {args.max_messages}")
    print(f"💾 Output: {args.output_dir}")
    print(f"🔄 Mode: {'OVERWRITE' if args.overwrite else 'MERGE/APPEND'}")
    print("-" * 70)

    # Consume from Kafka
    print("[INFO] Consommation depuis Kafka...")
    records = consume_features(max_messages=args.max_messages, idle_loops=args.idle_loops)
    if not records:
        raise SystemExit("❌ Aucun message consommé depuis Kafka; impossible de construire le dataset.")

    print(f"[INFO] ✅ {len(records)} messages consommés depuis Kafka")
    
    # Build new dataframe
    print("[INFO] Construction du DataFrame...")
    new_df = build_dataframe(records)
    if new_df.empty:
        raise SystemExit("❌ Aucun enregistrement valide à ajouter au dataset.")

    print(f"[INFO] ✅ DataFrame construit: {len(new_df)} lignes")

    # Load existing dataset if merging
    parquet_path = Path(args.output_dir) / "weather_features.parquet"
    existing_df = None
    if not args.overwrite:
        existing_df = load_existing_dataset(parquet_path)
    
    # Merge datasets if needed
    if existing_df is not None and not existing_df.empty:
        print(f"[INFO] 📂 Dataset existant trouvé: {len(existing_df)} lignes")
        df = merge_datasets(existing_df, new_df)
        added = len(df) - len(existing_df)
        print(f"[INFO] ✅ Après fusion: {len(df)} lignes totales (+{added} nouvelles, doublons supprimés)")
    else:
        df = new_df.copy()
        print(f"[INFO] ✅ Nouveau dataset: {len(df)} lignes")

    # Add prediction target to the complete merged dataset
    print(f"[INFO] Ajout de la target avec horizon {args.horizon}h...")
    df = add_prediction_target(df, args.horizon)
    
    # Count valid targets (non-null)
    target_col = f"target_temperature_horizon_{args.horizon}h"
    valid_targets = df[target_col].notna().sum()
    invalid_targets = len(df) - valid_targets
    
    if valid_targets == 0:
        print(f"⚠️  ATTENTION: Aucune target valide ({invalid_targets}/{len(df)} NaN)")
        print(f"   Raison: Il faut au moins {args.horizon + 1} lignes pour un horizon de {args.horizon}h")
        print(f"   Solution: Augmentez --max-messages ou réduisez --horizon")
    else:
        print(f"[INFO] ✅ Target ajoutée: {valid_targets}/{len(df)} valeurs valides")
        if invalid_targets > 0:
            print(f"   ⚠️  {invalid_targets} valeurs NaN (dernières lignes sans future)")
    
    # Save the merged dataset
    print("[INFO] Sauvegarde du dataset...")
    parquet_path, csv_path = save_dataset(df, args.output_dir)

    print("=" * 70)
    print("✅ DATASET SAUVEGARDÉ")
    print("=" * 70)
    print(f"📊 Total lignes: {len(df)}")
    print(f"📁 Parquet: {parquet_path}")
    print(f"📁 CSV    : {csv_path}")
    print(f"💾 Taille Parquet: {parquet_path.stat().st_size / 1024:.2f} KB")
    print("=" * 70)


if __name__ == "__main__":
    main()

