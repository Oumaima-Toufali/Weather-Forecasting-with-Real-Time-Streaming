-- Path: ksql/feature_builder.sql
-- ksqlDB feature builder for weather stream (JSON payloads validated by Avro in Python)
-- Requires ksqlDB server to have JSON processing enabled (no Schema Registry needed)

-- 1) RAW stream (optionnel : pour debug)
CREATE STREAM IF NOT EXISTS RAW_WEATHER (
  latitude DOUBLE,
  longitude DOUBLE,
  elevation DOUBLE,
  current STRUCT<
    time STRING,
    temperature_2m DOUBLE,
    relative_humidity_2m INTEGER,
    precipitation DOUBLE,
    wind_speed_10m DOUBLE,
    wind_direction_10m INTEGER,
    cloud_cover INTEGER,
    pressure_msl DOUBLE
  >,
  current_units STRUCT<
    temperature_2m STRING,
    relative_humidity_2m STRING,
    precipitation STRING,
    wind_speed_10m STRING,
    wind_direction_10m STRING,
    cloud_cover STRING,
    pressure_msl STRING
  >,
  hourly STRUCT<
    time ARRAY<STRING>,
    temperature_2m ARRAY<DOUBLE>,
    relative_humidity_2m ARRAY<INTEGER>,
    precipitation_probability ARRAY<INTEGER>,
    precipitation ARRAY<DOUBLE>,
    wind_speed_10m ARRAY<DOUBLE>,
    cloud_cover ARRAY<INTEGER>,
    pressure_msl ARRAY<DOUBLE>
  >,
  metadata STRUCT<
    source STRING,
    location STRING,
    timestamp STRING,
    fetch_time STRING
  >
) WITH (
  KAFKA_TOPIC='data.raw.stream',
  VALUE_FORMAT='JSON'
);

-- 2) CLEANED stream
CREATE STREAM IF NOT EXISTS CLEANED_WEATHER (
  location STRUCT<latitude DOUBLE, longitude DOUBLE, elevation DOUBLE>,
  event_time STRING,
  current STRUCT<
    time STRING,
    temperature_2m DOUBLE,
    relative_humidity_2m INTEGER,
    precipitation DOUBLE,
    wind_speed_10m DOUBLE,
    wind_direction_10m INTEGER,
    cloud_cover INTEGER,
    pressure_msl DOUBLE,
    units STRUCT<temperature STRING, humidity STRING, precipitation STRING, wind_speed STRING, pressure STRING>
  >,
  hourly STRUCT<
    time ARRAY<STRING>,
    temperature_2m ARRAY<DOUBLE>,
    relative_humidity_2m ARRAY<INTEGER>,
    precipitation_probability ARRAY<INTEGER>,
    precipitation ARRAY<DOUBLE>,
    wind_speed_10m ARRAY<DOUBLE>,
    cloud_cover ARRAY<INTEGER>,
    pressure_msl ARRAY<DOUBLE>
  >,
  metadata STRUCT<
    source STRING,
    location STRING,
    timestamp STRING,
    normalized_at STRING,
    data_quality STRUCT<completeness DOUBLE, has_anomalies BOOLEAN, validation_errors ARRAY<STRING>>
  >
) WITH (
  KAFKA_TOPIC='data.cleaned.stream',
  VALUE_FORMAT='JSON'
);

-- 3) FEATURE stream (output)
CREATE STREAM IF NOT EXISTS FEATURES_HOURLY (
  location STRUCT<latitude DOUBLE, longitude DOUBLE, elevation DOUBLE>,
  timestamp STRING,
  base_features STRUCT<
    temp_current DOUBLE,
    humidity_current DOUBLE,
    wind_current DOUBLE,
    pressure_current DOUBLE,
    cloud_cover_current DOUBLE,
    precipitation_current DOUBLE
  >,
  historical_features STRUCT<
    temp_rolling_3h STRUCT<mean_3h DOUBLE, std_3h DOUBLE, min_3h DOUBLE, max_3h DOUBLE>,
    temp_rolling_6h STRUCT<mean_6h DOUBLE, std_6h DOUBLE, min_6h DOUBLE, max_6h DOUBLE>,
    temp_deltas STRUCT<delta_1h DOUBLE, delta_3h DOUBLE, delta_6h DOUBLE, delta_12h DOUBLE>,
    temp_roc_3h DOUBLE,
    temp_roc_6h DOUBLE,
    temp_lags STRUCT<lag_1h DOUBLE, lag_2h DOUBLE, lag_3h DOUBLE, lag_6h DOUBLE, lag_12h DOUBLE>,
    temp_volatility_6h DOUBLE
  >,
  cyclical_features STRUCT<
    hour_sin DOUBLE,
    hour_cos DOUBLE,
    day_sin DOUBLE,
    day_cos DOUBLE,
    is_day INTEGER,
    is_weekend INTEGER
  >,
  target_labels STRUCT<temp_target_1h DOUBLE, temp_target_3h DOUBLE, temp_target_6h DOUBLE>,
  metadata STRUCT<feature_engineered_at STRING, history_size INTEGER, samples_accumulated INTEGER, feature_count INTEGER>
) WITH (
  KAFKA_TOPIC='data.features.hourly',
  VALUE_FORMAT='JSON'
);

-- 4) Fenêtre tumbling 1h avec agrégations principales
CREATE TABLE CLEANED_1H AS
SELECT
  location->latitude   AS latitude,
  location->longitude  AS longitude,
  WINDOWSTART          AS window_start,
  WINDOWEND            AS window_end,
  AVG(current->temperature_2m) AS temp_mean_1h,
  STDDEV_SAMP(current->temperature_2m) AS temp_std_1h,
  MIN(current->temperature_2m) AS temp_min_1h,
  MAX(current->temperature_2m) AS temp_max_1h,
  AVG(current->relative_humidity_2m) AS humidity_mean_1h,
  AVG(current->wind_speed_10m) AS wind_mean_1h,
  LATEST_BY_OFFSET(current->temperature_2m) AS temp_last,
  EARLIEST_BY_OFFSET(current->temperature_2m) AS temp_first
FROM CLEANED_WEATHER
WINDOW TUMBLING (SIZE 1 HOUR)
GROUP BY location->latitude, location->longitude
EMIT CHANGES;

-- 5) Fenêtre 3h pour deltas
CREATE TABLE CLEANED_3H AS
SELECT
  location->latitude   AS latitude,
  location->longitude  AS longitude,
  WINDOWSTART          AS window_start,
  WINDOWEND            AS window_end,
  AVG(current->temperature_2m) AS temp_mean_3h,
  STDDEV_SAMP(current->temperature_2m) AS temp_std_3h,
  MIN(current->temperature_2m) AS temp_min_3h,
  MAX(current->temperature_2m) AS temp_max_3h,
  LATEST_BY_OFFSET(current->temperature_2m) AS temp_last,
  EARLIEST_BY_OFFSET(current->temperature_2m) AS temp_first
FROM CLEANED_WEATHER
WINDOW TUMBLING (SIZE 3 HOUR)
GROUP BY location->latitude, location->longitude
EMIT CHANGES;

-- 6) Fenêtre 6h pour tendances plus longues
CREATE TABLE CLEANED_6H AS
SELECT
  location->latitude   AS latitude,
  location->longitude  AS longitude,
  WINDOWSTART          AS window_start,
  WINDOWEND            AS window_end,
  AVG(current->temperature_2m) AS temp_mean_6h,
  STDDEV_SAMP(current->temperature_2m) AS temp_std_6h,
  MIN(current->temperature_2m) AS temp_min_6h,
  MAX(current->temperature_2m) AS temp_max_6h,
  LATEST_BY_OFFSET(current->temperature_2m) AS temp_last,
  EARLIEST_BY_OFFSET(current->temperature_2m) AS temp_first
FROM CLEANED_WEATHER
WINDOW TUMBLING (SIZE 6 HOUR)
GROUP BY location->latitude, location->longitude
EMIT CHANGES;

-- 7) Flux final : jointure des fenêtres pour produire les features horaires
CREATE STREAM FEATURES_HOURLY_ENRICHED AS
SELECT
  c1.latitude AS latitude,
  c1.longitude AS longitude,
  c1.window_end AS feature_time,
  STRUCT(
    temp_current := c1.temp_last,
    humidity_current := NULL,
    wind_current := NULL,
    pressure_current := NULL,
    cloud_cover_current := NULL,
    precipitation_current := NULL
  ) AS base_features,
  STRUCT(
    temp_rolling_3h := STRUCT(mean_3h := c3.temp_mean_3h, std_3h := c3.temp_std_3h, min_3h := c3.temp_min_3h, max_3h := c3.temp_max_3h),
    temp_rolling_6h := STRUCT(mean_6h := c6.temp_mean_6h, std_6h := c6.temp_std_6h, min_6h := c6.temp_min_6h, max_6h := c6.temp_max_6h),
    temp_deltas := STRUCT(
      delta_1h := c1.temp_last - c1.temp_first,
      delta_3h := c3.temp_last - c3.temp_first,
      delta_6h := c6.temp_last - c6.temp_first,
      delta_12h := NULL
    ),
    temp_roc_3h := (c3.temp_last - c3.temp_first) / 3.0,
    temp_roc_6h := (c6.temp_last - c6.temp_first) / 6.0,
    temp_lags := STRUCT(
      lag_1h := c1.temp_first,
      lag_2h := NULL,
      lag_3h := c3.temp_first,
      lag_6h := c6.temp_first,
      lag_12h := NULL
    ),
    temp_volatility_6h := c6.temp_std_6h
  ) AS historical_features
FROM CLEANED_1H c1
LEFT JOIN CLEANED_3H c3
  WITHIN 3 HOURS
  ON c1.latitude = c3.latitude AND c1.longitude = c3.longitude
LEFT JOIN CLEANED_6H c6
  WITHIN 6 HOURS
  ON c1.latitude = c6.latitude AND c1.longitude = c6.longitude
EMIT CHANGES;