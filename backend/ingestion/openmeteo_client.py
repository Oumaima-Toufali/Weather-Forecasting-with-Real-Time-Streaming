# Path: backend/ingestion/openmeteo_client.py
import requests
from datetime import datetime
from typing import Dict, Optional

from backend.config.api_config import WEATHER_VARIABLES, FORECAST_DAYS


class OpenMeteoClient:
    """Client pour récupérer les données météo depuis Open-Meteo API"""
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude
    
    def get_current_weather(self) -> Optional[Dict]:
        """Récupère les données météo actuelles"""
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "current": WEATHER_VARIABLES['current'],
            "hourly": WEATHER_VARIABLES['hourly'],
            "forecast_days": FORECAST_DAYS
        }
        
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            data["metadata"] = {
                "source": "open-meteo",
                "location": f"{self.latitude},{self.longitude}",
                "timestamp": datetime.utcnow().isoformat(),
                "fetch_time": datetime.utcnow().isoformat()
            }
            
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Erreur API Open-Meteo: {e}")
            return None