# Path: backend/processing/normalizer.py
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from backend.utils.schema_validator import load_avro_schema, validate_avro


class WeatherNormalizer:
    """
    Nettoie et standardise les données météo brutes d'Open-Meteo
    """
    
    def __init__(self):
        self.required_fields = [
            'latitude', 'longitude', 'current', 'hourly', 'metadata'
        ]
        self.cleaned_schema = load_avro_schema("backend/schemas/cleaned_weather.avsc")
    
    def validate_structure(self, data: dict) -> bool:
        """Vérifie que les champs essentiels sont présents"""
        for field in self.required_fields:
            if field not in data:
                print(f"⚠️  Champ manquant: {field}")
                return False
        
        # Vérifier current_units
        if 'current_units' not in data:
            print("⚠️  'current_units' manquant")
            return False
        
        return True
    
    def _convert_temperature(self, value, unit: str) -> Optional[float]:
        """Convertit température en °C si nécessaire"""
        val = self._to_float(value)
        if val is None:
            return None
        if unit in ['°C', 'celsius', 'C']:
            return val
        if unit in ['K', 'kelvin']:
            return val - 273.15
        if unit in ['°F', 'fahrenheit', 'F']:
            return (val - 32) * 5 / 9
        return val
    
    def _convert_speed(self, value, unit: str) -> Optional[float]:
        """Convertit vitesse du vent en km/h"""
        val = self._to_float(value)
        if val is None:
            return None
        if unit in ['km/h', 'kmh']:
            return val
        if unit in ['m/s', 'mps']:
            return val * 3.6
        if unit in ['mph']:
            return val * 1.60934
        return val
    
    def _convert_pressure(self, value, unit: str) -> Optional[float]:
        """Convertit pression en hPa"""
        val = self._to_float(value)
        if val is None:
            return None
        if unit in ['hPa', 'hectopascal']:
            return val
        if unit in ['Pa', 'pascal']:
            return val / 100
        if unit in ['kPa']:
            return val * 10
        return val
    
    def clean_current_data(self, current: dict, units: dict) -> Optional[dict]:
        """Nettoie les données actuelles"""
        try:
            temp = self._convert_temperature(current.get('temperature_2m'), units.get('temperature_2m', '°C'))
            wind_speed = self._convert_speed(current.get('wind_speed_10m'), units.get('wind_speed_10m', 'km/h'))
            pressure = self._convert_pressure(current.get('pressure_msl'), units.get('pressure_msl', 'hPa'))
            
            return {
                'time': current.get('time'),
                'temperature_2m': temp,
                'relative_humidity_2m': self._to_int(current.get('relative_humidity_2m')),
                'precipitation': self._to_float(current.get('precipitation')),
                'wind_speed_10m': wind_speed,
                'wind_direction_10m': self._to_int(current.get('wind_direction_10m')),
                'cloud_cover': self._to_int(current.get('cloud_cover')),
                'pressure_msl': pressure,
                'units': {
                    'temperature': '°C',
                    'humidity': '%',
                    'precipitation': units.get('precipitation', 'mm'),
                    'wind_speed': 'km/h',
                    'pressure': 'hPa'
                }
            }
        except Exception as e:
            print(f"❌ Erreur nettoyage current: {e}")
            return None
    
    def clean_hourly_data(self, hourly: dict, units: dict) -> Optional[dict]:
        """Nettoie les prévisions horaires"""
        try:
            temp_unit = units.get('temperature_2m', '°C')
            wind_unit = units.get('wind_speed_10m', 'km/h')
            pressure_unit = units.get('pressure_msl', 'hPa')
            
            return {
                'time': hourly.get('time', []),
                'temperature_2m': [self._convert_temperature(t, temp_unit) for t in hourly.get('temperature_2m', [])],
                'relative_humidity_2m': [self._to_int(h) for h in hourly.get('relative_humidity_2m', [])],
                'precipitation_probability': [self._to_int(p) for p in hourly.get('precipitation_probability', [])],
                'precipitation': [self._to_float(p) for p in hourly.get('precipitation', [])],
                'wind_speed_10m': [self._convert_speed(w, wind_unit) for w in hourly.get('wind_speed_10m', [])],
                'cloud_cover': [self._to_int(c) for c in hourly.get('cloud_cover', [])],
                'pressure_msl': [self._convert_pressure(p, pressure_unit) for p in hourly.get('pressure_msl', [])],
            }
        except Exception as e:
            print(f"❌ Erreur nettoyage hourly: {e}")
            return None
    
    def add_data_quality_metrics(self, data: dict) -> dict:
        """Ajoute des métriques de qualité des données"""
        metrics = {
            'completeness': 0.0,
            'has_anomalies': False,
            'validation_errors': []
        }
        
        # Calcul de complétude (current data)
        current = data.get('current', {})
        expected_fields = ['temperature_2m', 'relative_humidity_2m', 'precipitation', 
                          'wind_speed_10m', 'pressure_msl']
        non_null_fields = sum(1 for f in expected_fields if current.get(f) is not None)
        metrics['completeness'] = non_null_fields / len(expected_fields)
        
        # Détection d'anomalies simples
        temp = current.get('temperature_2m')
        humidity = current.get('relative_humidity_2m')
        
        if temp is not None and (temp < -50 or temp > 60):
            metrics['has_anomalies'] = True
            metrics['validation_errors'].append('temperature_out_of_range')
        
        if humidity is not None and (humidity < 0 or humidity > 100):
            metrics['has_anomalies'] = True
            metrics['validation_errors'].append('humidity_out_of_range')
        
        return metrics
    
    def normalize(self, raw_data: dict) -> Optional[dict]:
        """
        Fonction principale de normalisation
        """
        # Validation de structure
        if not self.validate_structure(raw_data):
            return None
        
        # Nettoyage
        cleaned_current = self.clean_current_data(
            raw_data['current'],
            raw_data['current_units']
        )
        
        cleaned_hourly = self.clean_hourly_data(raw_data['hourly'], raw_data['current_units'])
        
        if not cleaned_current or not cleaned_hourly:
            return None
        
        now_iso = datetime.now(timezone.utc).isoformat()
        
        # Construction données normalisées
        normalized = {
            'location': {
                'latitude': raw_data['latitude'],
                'longitude': raw_data['longitude'],
                'elevation': raw_data.get('elevation', 0.0)
            },
            'event_time': raw_data['metadata'].get('timestamp', cleaned_current.get('time')),
            'current': cleaned_current,
            'hourly': cleaned_hourly,
            'metadata': {
                **raw_data['metadata'],
                'normalized_at': now_iso,
                'data_quality': self.add_data_quality_metrics({
                    'current': cleaned_current,
                    'hourly': cleaned_hourly
                })
            }
        }
        
        # Validation Avro
        if not validate_avro(normalized, self.cleaned_schema):
            print("⚠️  Données normalisées non conformes au schéma Avro")
            return None
        
        return normalized
    
    # Helpers
    @staticmethod
    def _to_float(value) -> Optional[float]:
        """Conversion sécurisée vers float"""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def _to_int(value) -> Optional[int]:
        """Conversion sécurisée vers int"""
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None