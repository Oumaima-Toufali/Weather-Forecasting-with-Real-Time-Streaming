# Path: tests/test_build_dataset.py
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from backend.training.build_dataset import (
    add_prediction_target,
    build_dataframe,
    load_existing_dataset,
    merge_datasets,
    save_dataset,
    _flatten_feature_record,
)
from backend.utils.schema_validator import load_avro_schema, validate_avro


def test_add_prediction_target_shift():
    base_time = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
    df = pd.DataFrame(
        {
            "timestamp": [base_time + timedelta(hours=i) for i in range(4)],
            "temperature": [10.0, 11.0, 12.0, 13.0],
        }
    )

    result = add_prediction_target(df, horizon_hours=2)
    target = result["target_temperature_horizon_2h"].tolist()

    assert target[0] == 12.0
    assert target[1] == 13.0
    assert math.isnan(target[2])
    assert math.isnan(target[3])


def test_parquet_export(tmp_path):
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="H"),
            "temperature": [1.0, 2.0, 3.0],
        }
    )
    parquet_path, csv_path = save_dataset(df, tmp_path)
    assert parquet_path.exists()
    assert csv_path.exists()


def test_avro_validation_features_schema():
    schema = load_avro_schema("backend/schemas/features_weather.avsc")
    sample = {
        "location": {"latitude": 1.0, "longitude": 2.0, "elevation": 10.0},
        "timestamp": "2024-01-01T00:00:00Z",
        "base_features": {
            "temp_current": 20.0,
            "humidity_current": 50.0,
            "wind_current": 18.0,
            "pressure_current": 1013.0,
            "cloud_cover_current": 10.0,
            "precipitation_current": 0.0,
        },
        "historical_features": {
            "temp_rolling_3h": {"mean_3h": 20.0, "std_3h": 0.1, "min_3h": 19.9, "max_3h": 20.1},
            "temp_rolling_6h": {"mean_6h": 20.0, "std_6h": 0.2, "min_6h": 19.8, "max_6h": 20.2},
            "temp_deltas": {"delta_1h": 0.1, "delta_3h": 0.2, "delta_6h": 0.3, "delta_12h": None},
            "temp_roc_3h": 0.05,
            "temp_roc_6h": 0.02,
            "temp_lags": {"lag_1h": 19.9, "lag_2h": 19.8, "lag_3h": 19.7, "lag_6h": 19.5, "lag_12h": None},
            "temp_volatility_6h": 0.2,
        },
        "cyclical_features": {
            "hour_sin": 0.0,
            "hour_cos": 1.0,
            "day_sin": 0.0,
            "day_cos": 1.0,
            "is_day": 1,
            "is_weekend": 0,
        },
        "target_labels": {"temp_target_1h": 21.0, "temp_target_3h": 22.0, "temp_target_6h": 23.0},
        "metadata": {
            "feature_engineered_at": "2024-01-01T00:00:02Z",
            "history_size": 10,
            "samples_accumulated": 10,
            "feature_count": 20,
        },
    }

    assert validate_avro(sample, schema)


def test_flatten_feature_record():
    """Test that flattening extracts all features correctly."""
    record = {
        "location": {"latitude": 1.0, "longitude": 2.0, "elevation": 10.0},
        "timestamp": "2024-01-01T00:00:00Z",
        "base_features": {
            "temp_current": 20.0,
            "humidity_current": 50.0,
            "wind_current": 18.0,
            "pressure_current": 1013.0,
            "cloud_cover_current": 10.0,
            "precipitation_current": 0.0,
        },
        "historical_features": {
            "temp_rolling_3h": {"mean_3h": 20.0, "std_3h": 0.1, "min_3h": 19.9, "max_3h": 20.1},
            "temp_rolling_6h": {"mean_6h": 20.0, "std_6h": 0.2, "min_6h": 19.8, "max_6h": 20.2},
            "temp_deltas": {"delta_1h": 0.1, "delta_3h": 0.2, "delta_6h": 0.3, "delta_12h": None},
            "temp_roc_3h": 0.05,
            "temp_roc_6h": 0.02,
            "temp_lags": {"lag_1h": 19.9, "lag_2h": 19.8, "lag_3h": 19.7, "lag_6h": 19.5, "lag_12h": None},
            "temp_volatility_6h": 0.2,
        },
        "cyclical_features": {
            "hour_sin": 0.0,
            "hour_cos": 1.0,
            "day_sin": 0.0,
            "day_cos": 1.0,
            "is_day": 1,
            "is_weekend": 0,
        },
        "target_labels": {"temp_target_1h": 21.0, "temp_target_3h": 22.0, "temp_target_6h": 23.0},
        "metadata": {
            "feature_engineered_at": "2024-01-01T00:00:02Z",
            "history_size": 10,
            "samples_accumulated": 10,
            "feature_count": 20,
        },
    }
    
    flattened = _flatten_feature_record(record)
    
    # Check base features
    assert flattened["temperature"] == 20.0
    assert flattened["humidity"] == 50.0
    assert flattened["wind_speed"] == 18.0
    
    # Check historical features
    assert flattened["temp_mean_3h"] == 20.0
    assert flattened["temp_delta_3h"] == 0.2
    assert flattened["temp_lag_1h"] == 19.9
    
    # Check cyclical features
    assert flattened["hour_sin"] == 0.0
    assert flattened["is_day"] == 1


def test_persistence_load_and_save(tmp_path):
    """Test that dataset can be saved and loaded (persistence)."""
    df1 = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="H"),
            "temperature": [10.0, 11.0, 12.0],
            "humidity": [50, 51, 52],
        }
    )
    
    # Save dataset
    parquet_path, csv_path = save_dataset(df1, tmp_path)
    assert parquet_path.exists()
    assert csv_path.exists()
    
    # Load dataset
    loaded_df = load_existing_dataset(parquet_path)
    assert loaded_df is not None
    assert len(loaded_df) == 3
    assert loaded_df["temperature"].tolist() == [10.0, 11.0, 12.0]


def test_persistence_merge_datasets(tmp_path):
    """Test that datasets are properly merged (simulating stop/restart scenario)."""
    # First dataset
    df1 = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="H"),
            "temperature": [10.0, 11.0, 12.0],
            "humidity": [50, 51, 52],
        }
    )
    
    # Save first dataset
    parquet_path, _ = save_dataset(df1, tmp_path)
    
    # Simulate restart: load existing and add new data
    existing_df = load_existing_dataset(parquet_path)
    assert existing_df is not None
    
    # New data (with some overlap)
    df2 = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01 02:00:00", periods=3, freq="H"),
            "temperature": [12.0, 13.0, 14.0],  # First row overlaps
            "humidity": [52, 53, 54],
        }
    )
    
    # Merge datasets
    merged_df = merge_datasets(existing_df, df2)
    
    # Should have 5 rows (3 from first + 2 new, 1 duplicate removed)
    assert len(merged_df) == 5
    assert merged_df["temperature"].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]
    
    # Save merged dataset
    save_dataset(merged_df, tmp_path)
    
    # Reload and verify persistence
    final_df = load_existing_dataset(parquet_path)
    assert len(final_df) == 5
    assert final_df["temperature"].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0]


def test_build_dataframe_with_features():
    """Test that build_dataframe correctly processes feature records."""
    records = [
        {
            "location": {"latitude": 1.0, "longitude": 2.0, "elevation": 10.0},
            "timestamp": "2024-01-01T00:00:00Z",
            "base_features": {
                "temp_current": 20.0,
                "humidity_current": 50.0,
                "wind_current": 18.0,
                "pressure_current": 1013.0,
                "cloud_cover_current": 10.0,
                "precipitation_current": 0.0,
            },
            "historical_features": {
                "temp_rolling_3h": {"mean_3h": 20.0, "std_3h": 0.1, "min_3h": 19.9, "max_3h": 20.1},
                "temp_rolling_6h": None,
                "temp_deltas": {"delta_1h": 0.1, "delta_3h": 0.2, "delta_6h": 0.3, "delta_12h": None},
                "temp_roc_3h": 0.05,
                "temp_roc_6h": None,
                "temp_lags": {"lag_1h": 19.9, "lag_2h": None, "lag_3h": 19.7, "lag_6h": None, "lag_12h": None},
                "temp_volatility_6h": None,
            },
            "cyclical_features": {
                "hour_sin": 0.0,
                "hour_cos": 1.0,
                "day_sin": 0.0,
                "day_cos": 1.0,
                "is_day": 1,
                "is_weekend": 0,
            },
            "target_labels": {"temp_target_1h": 21.0, "temp_target_3h": 22.0, "temp_target_6h": 23.0},
            "metadata": {
                "feature_engineered_at": "2024-01-01T00:00:02Z",
                "history_size": 10,
                "samples_accumulated": 10,
                "feature_count": 20,
            },
        }
    ]
    
    df = build_dataframe(records)
    
    assert len(df) == 1
    assert "temperature" in df.columns
    assert "temp_mean_3h" in df.columns
    assert "temp_delta_3h" in df.columns
    assert "hour_sin" in df.columns
    assert df["timestamp"].dtype == "datetime64[ns]"

