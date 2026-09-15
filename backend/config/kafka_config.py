# Path: backend/config/kafka_config.py
"""Configuration Kafka pour le projet Weather Forecaster Sur le topic data.raw.weather"""

import json

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS = ['localhost:9092']

# Topics alignés avec la chaîne de traitement
TOPICS = {
    'raw': 'data.raw.stream',
    'cleaned': 'data.cleaned.stream',
    'features': 'data.features.hourly',
    'predictions': 'data.predictions.weather',
    # Probabilistic quantiles (P10/P50/P90) produced by serving
    'predictions_quantiles': 'data.predictions.weather.quantiles'
}

# Producer Configuration
PRODUCER_CONFIG = {
    'bootstrap_servers': KAFKA_BOOTSTRAP_SERVERS,
    'value_serializer': lambda v: json.dumps(v).encode('utf-8'),
    'key_serializer': lambda k: k.encode('utf-8') if k else None,
    'acks': 'all',  # Attendre confirmation de tous les replicas
    'retries': 3,
    'max_in_flight_requests_per_connection': 1  # Garantir l'ordre
}

# Consumer Configuration
CONSUMER_CONFIG = {
    'bootstrap_servers': KAFKA_BOOTSTRAP_SERVERS,
    'value_deserializer': lambda v: json.loads(v.decode('utf-8')),
    'key_deserializer': lambda k: k.decode('utf-8') if k else None,
    'auto_offset_reset': 'earliest',
    'enable_auto_commit': True,
    'group_id': 'weather-processing-group'
}