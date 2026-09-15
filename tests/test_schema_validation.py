# Path: tests/test_schema_validation.py
import json
from pathlib import Path

from backend.utils.schema_validator import load_avro_schema, validate_avro


def test_raw_schema_validation():
    schema = load_avro_schema("backend/schemas/raw_weather.avsc")
    sample = {
        "latitude": 1.0,
        "longitude": 2.0,
        "elevation": 10.0,
        "current": {
            "time": "2024-01-01T00:00:00Z",
            "temperature_2m": 20.0,
            "relative_humidity_2m": 50,
            "precipitation": 0.0,
            "wind_speed_10m": 5.0,
            "wind_direction_10m": 90,
            "cloud_cover": 10,
            "pressure_msl": 1013.0
        },
        "current_units": {
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "precipitation": "mm",
            "wind_speed_10m": "km/h",
            "wind_direction_10m": "°",
            "cloud_cover": "%",
            "pressure_msl": "hPa"
        },
        "hourly": {
            "time": ["2024-01-01T00:00:00Z"],
            "temperature_2m": [20.0],
            "relative_humidity_2m": [50],
            "precipitation_probability": [0],
            "precipitation": [0.0],
            "wind_speed_10m": [5.0],
            "cloud_cover": [10],
            "pressure_msl": [1013.0]
        },
        "metadata": {
            "source": "test",
            "location": "1.0,2.0",
            "timestamp": "2024-01-01T00:00:00Z",
            "fetch_time": "2024-01-01T00:00:00Z"
        }
    }
    assert validate_avro(sample, schema)


def test_cleaned_schema_validation():
    schema = load_avro_schema("backend/schemas/cleaned_weather.avsc")
    sample = {
        "location": {"latitude": 1.0, "longitude": 2.0, "elevation": 10.0},
        "event_time": "2024-01-01T00:00:00Z",
        "current": {
            "time": "2024-01-01T00:00:00Z",
            "temperature_2m": 20.0,
            "relative_humidity_2m": 50,
            "precipitation": 0.0,
            "wind_speed_10m": 18.0,
            "wind_direction_10m": 90,
            "cloud_cover": 10,
            "pressure_msl": 1013.0,
            "units": {
                "temperature": "°C",
                "humidity": "%",
                "precipitation": "mm",
                "wind_speed": "km/h",
                "pressure": "hPa"
            }
        },
        "hourly": {
            "time": ["2024-01-01T00:00:00Z"],
            "temperature_2m": [20.0],
            "relative_humidity_2m": [50],
            "precipitation_probability": [0],
            "precipitation": [0.0],
            "wind_speed_10m": [18.0],
            "cloud_cover": [10],
            "pressure_msl": [1013.0]
        },
        "metadata": {
            "source": "test",
            "location": "1.0,2.0",
            "timestamp": "2024-01-01T00:00:00Z",
            "normalized_at": "2024-01-01T00:00:01Z",
            "data_quality": {
                "completeness": 1.0,
                "has_anomalies": False,
                "validation_errors": []
            }
        }
    }
    assert validate_avro(sample, schema)


def test_features_schema_validation():
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
            "precipitation_current": 0.0
        },
        "historical_features": {
            "temp_rolling_3h": {"mean_3h": 20.0, "std_3h": 0.1, "min_3h": 19.9, "max_3h": 20.1},
            "temp_rolling_6h": {"mean_6h": 20.0, "std_6h": 0.2, "min_6h": 19.8, "max_6h": 20.2},
            "temp_deltas": {"delta_1h": 0.1, "delta_3h": 0.2, "delta_6h": 0.3, "delta_12h": None},
            "temp_roc_3h": 0.05,
            "temp_roc_6h": 0.02,
            "temp_lags": {"lag_1h": 19.9, "lag_2h": 19.8, "lag_3h": 19.7, "lag_6h": 19.5, "lag_12h": None},
            "temp_volatility_6h": 0.2
        },
        "cyclical_features": {
            "hour_sin": 0.0,
            "hour_cos": 1.0,
            "day_sin": 0.0,
            "day_cos": 1.0,
            "is_day": 1,
            "is_weekend": 0
        },
        "target_labels": {"temp_target_1h": 21.0, "temp_target_3h": 22.0, "temp_target_6h": 23.0},
        "metadata": {
            "feature_engineered_at": "2024-01-01T00:00:02Z",
            "history_size": 10,
            "samples_accumulated": 10,
            "feature_count": 20
        }
    }
    assert validate_avro(sample, schema)

