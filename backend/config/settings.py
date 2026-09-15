# Path: backend/config/settings.py
"""
Configuration centrale du projet Weather Forecaster
"""
import os
from dataclasses import dataclass
from typing import List

@dataclass
class OpenMeteoConfig:
    """Configuration Open-Meteo API"""
    BASE_URL: str = "https://api.open-meteo.com/v1/forecast"
    # Paramètres météo à récupérer
    HOURLY_PARAMS: List[str] = None
    # Localisation par défaut (zone plus volatile : Medellín, Colombie)
    # 6.2442° N, 75.5812° W
    DEFAULT_LATITUDE: float = 6.2442
    DEFAULT_LONGITUDE: float = -75.5812
    # Intervalle de requêtes (minutes)
    POLL_INTERVAL: int = 10
    
    def __post_init__(self):
        if self.HOURLY_PARAMS is None:
            self.HOURLY_PARAMS = [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "wind_speed_10m",
                "cloud_cover",
                "pressure_msl"
            ]

@dataclass
class KafkaConfig:
    """Configuration Kafka (KRaft mode)"""
    BOOTSTRAP_SERVERS: str = "localhost:9092"
    
    # Topics
    TOPIC_RAW: str = "data.raw.weather"
    TOPIC_CLEANED: str = "data.cleaned.weather"
    TOPIC_FEATURES: str = "data.features.weather"
    TOPIC_PREDICTIONS: str = "data.predictions.weather"
    
    # Configuration producer
    PRODUCER_CONFIG: dict = None
    
    # Configuration consumer
    CONSUMER_CONFIG: dict = None
    
    def __post_init__(self):
        if self.PRODUCER_CONFIG is None:
            self.PRODUCER_CONFIG = {
                'bootstrap_servers': self.BOOTSTRAP_SERVERS,
                'value_serializer': lambda v: str(v).encode('utf-8'),
                'acks': 'all',  # Garantie de livraison
                'retries': 3,
                'enable_idempotence': True  # Évite les doublons
            }
        
        if self.CONSUMER_CONFIG is None:
            self.CONSUMER_CONFIG = {
                'bootstrap_servers': self.BOOTSTRAP_SERVERS,
                'group_id': 'weather-processor-group',
                'auto_offset_reset': 'earliest',
                'enable_auto_commit': True
            }

@dataclass
class DataPaths:
    """Chemins des données"""
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    @property
    def RAW_DATA(self) -> str:
        return os.path.join(self.BASE_DIR, "data", "raw")
    
    @property
    def PROCESSED_DATA(self) -> str:
        return os.path.join(self.BASE_DIR, "data", "processed")
    
    @property
    def HISTORICAL_DATA(self) -> str:
        return os.path.join(self.BASE_DIR, "data", "historical")
    
    @property
    def MODELS(self) -> str:
        return os.path.join(self.BASE_DIR, "models", "saved")

@dataclass
class ModelConfig:
    """Configuration des modèles ML"""
    # Horizons de prédiction (heures)
    PREDICTION_HORIZONS: List[int] = None
    
    # Features utilisées
    FEATURE_COLUMNS: List[str] = None
    
    # Target variable
    TARGET_COLUMN: str = "temperature_2m"
    
    # Window sizes pour feature engineering
    WINDOW_SIZES: List[int] = None
    
    def __post_init__(self):
        if self.PREDICTION_HORIZONS is None:
            self.PREDICTION_HORIZONS = [1, 3, 6]  # +1h, +3h, +6h
        
        if self.FEATURE_COLUMNS is None:
            self.FEATURE_COLUMNS = [
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "pressure_msl"
            ]
        
        if self.WINDOW_SIZES is None:
            self.WINDOW_SIZES = [3, 6, 12, 24]  # 3h, 6h, 12h, 24h

# Instances globales
OPEN_METEO = OpenMeteoConfig()
KAFKA = KafkaConfig()
PATHS = DataPaths()
MODEL = ModelConfig()

# Variables d'environnement
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")