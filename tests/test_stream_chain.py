# Path: tests/test_stream_chain.py
from datetime import datetime, timezone

from backend.processing.normalizer import WeatherNormalizer
from backend.processing.feature_engineering import WeatherFeatureEngineer
from backend.utils.schema_validator import load_avro_schema, validate_avro


def _make_raw_sample():
    now = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc).isoformat()
    return {
        "latitude": 48.8566,
        "longitude": 2.3522,
        "elevation": 35.0,
        "current": {
            "time": now,
            "temperature_2m": 15.2,
            "relative_humidity_2m": 60,
            "precipitation": 0.0,
            "wind_speed_10m": 5.5,
            "wind_direction_10m": 180,
            "cloud_cover": 20,
            "pressure_msl": 1015.0,
        },
        "current_units": {
            "temperature_2m": "°C",
            "relative_humidity_2m": "%",
            "precipitation": "mm",
            "wind_speed_10m": "km/h",
            "wind_direction_10m": "°",
            "cloud_cover": "%",
            "pressure_msl": "hPa",
        },
        "hourly": {
            "time": [now, now, now],
            "temperature_2m": [15.2, 15.6, 16.0],
            "relative_humidity_2m": [60, 58, 55],
            "precipitation_probability": [0, 0, 0],
            "precipitation": [0.0, 0.0, 0.0],
            "wind_speed_10m": [5.5, 6.0, 6.5],
            "cloud_cover": [20, 25, 30],
            "pressure_msl": [1015.0, 1014.5, 1014.0],
        },
        "metadata": {
            "source": "test",
            "location": "48.8566,2.3522",
            "timestamp": now,
            "fetch_time": now,
        },
    }


def test_end_to_end_normalize_and_engineer_features():
    raw_sample = _make_raw_sample()

    cleaned_schema = load_avro_schema("backend/schemas/cleaned_weather.avsc")
    features_schema = load_avro_schema("backend/schemas/features_weather.avsc")

    normalizer = WeatherNormalizer()
    cleaned = normalizer.normalize(raw_sample)

    assert cleaned is not None, "Normalization should return payload"
    assert validate_avro(cleaned, cleaned_schema)

    engineer = WeatherFeatureEngineer(history_size=10)
    features = engineer.engineer_features(cleaned)

    assert features is not None, "Feature engineering should produce payload"
    assert validate_avro(features, features_schema)

