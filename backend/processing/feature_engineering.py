# Path: backend/processing/feature_engineering.py
"""
Feature Engineering avec accumulation d'historique
Version CORRIGÉE : calcule sur données historiques réelles
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from collections import deque


class WeatherFeatureEngineer:
    """
    Crée des features temporelles à partir de l'historique réel
    """
    
    def __init__(self, history_size: int = 168):  # 7 jours * 24h
        self.feature_windows = [1, 3, 6, 12, 24]
        
        # Historique en mémoire (deque = efficient pour append/pop)
        self.history_size = history_size
        self.temp_history = deque(maxlen=history_size)
        self.humidity_history = deque(maxlen=history_size)
        self.wind_history = deque(maxlen=history_size)
        self.pressure_history = deque(maxlen=history_size)
        self.timestamps = deque(maxlen=history_size)
        
        self.samples_accumulated = 0
    
    def update_history(self, current_data: dict):
        """
        Ajoute les données actuelles à l'historique
        """
        self.temp_history.append(current_data.get('temperature_2m'))
        self.humidity_history.append(current_data.get('relative_humidity_2m'))
        self.wind_history.append(current_data.get('wind_speed_10m'))
        self.pressure_history.append(current_data.get('pressure_msl'))
        self.timestamps.append(current_data.get('time'))
        
        self.samples_accumulated += 1
    
    def compute_rolling_stats(self, series: deque, window: int) -> dict:
        """Calcule stats sur fenêtre glissante de l'historique"""
        if len(series) < window:
            return {
                f'mean_{window}h': None,
                f'std_{window}h': None,
                f'min_{window}h': None,
                f'max_{window}h': None
            }
        
        # Prendre les N derniers éléments
        recent = list(series)[-window:]
        arr = np.array([x for x in recent if x is not None])
        
        if len(arr) == 0:
            return {
                f'mean_{window}h': None,
                f'std_{window}h': None,
                f'min_{window}h': None,
                f'max_{window}h': None
            }
        
        return {
            f'mean_{window}h': float(np.mean(arr)),
            f'std_{window}h': float(np.std(arr)),
            f'min_{window}h': float(np.min(arr)),
            f'max_{window}h': float(np.max(arr))
        }
    
    def compute_deltas(self, series: deque, hours: List[int]) -> dict:
        """Calcule variations sur historique"""
        deltas = {}
        
        current = series[-1] if len(series) > 0 else None
        
        for h in hours:
            if len(series) >= h + 1 and current is not None:
                past_value = series[-(h + 1)]
                if past_value is not None:
                    deltas[f'delta_{h}h'] = float(current - past_value)
                else:
                    deltas[f'delta_{h}h'] = None
            else:
                deltas[f'delta_{h}h'] = None
        
        return deltas
    
    def compute_rate_of_change(self, series: deque, hours: int) -> Optional[float]:
        """Taux de variation moyen par heure"""
        if len(series) >= hours + 1:
            current = series[-1]
            past = series[-(hours + 1)]
            
            if current is not None and past is not None:
                return float((current - past) / hours)
        
        return None
    
    def create_lag_features(self, series: deque, lags: List[int]) -> dict:
        """Crée features de lag sur historique"""
        lag_features = {}
        
        for lag in lags:
            if len(series) >= lag + 1:
                lag_features[f'lag_{lag}h'] = series[-(lag + 1)]
            else:
                lag_features[f'lag_{lag}h'] = None
        
        return lag_features
    
    def extract_features_from_history(self) -> dict:
        """
        Extrait toutes les features depuis l'historique accumulé
        """
        features = {}
        
        if len(self.temp_history) == 0:
            return features
        
        # 1. ROLLING STATISTICS sur historique
        for window in [3, 6, 12, 24]:
            features[f'temp_rolling_{window}h'] = self.compute_rolling_stats(
                self.temp_history, window
            )
            features[f'humidity_rolling_{window}h'] = self.compute_rolling_stats(
                self.humidity_history, window
            )
            features[f'wind_rolling_{window}h'] = self.compute_rolling_stats(
                self.wind_history, window
            )
        
        # 2. DELTAS (variations)
        features['temp_deltas'] = self.compute_deltas(self.temp_history, [1, 3, 6, 12])
        features['humidity_deltas'] = self.compute_deltas(self.humidity_history, [1, 3, 6])
        features['wind_deltas'] = self.compute_deltas(self.wind_history, [1, 3, 6])
        
        # 3. RATE OF CHANGE
        features['temp_roc_3h'] = self.compute_rate_of_change(self.temp_history, 3)
        features['temp_roc_6h'] = self.compute_rate_of_change(self.temp_history, 6)
        features['wind_roc_3h'] = self.compute_rate_of_change(self.wind_history, 3)
        
        # 4. LAG FEATURES
        features['temp_lags'] = self.create_lag_features(self.temp_history, [1, 2, 3, 6, 12])
        features['humidity_lags'] = self.create_lag_features(self.humidity_history, [1, 3, 6])
        features['wind_lags'] = self.create_lag_features(self.wind_history, [1, 3, 6])
        
        # 5. VOLATILITY (écart-type glissant)
        if len(self.temp_history) >= 6:
            recent_temps = list(self.temp_history)[-6:]
            features['temp_volatility_6h'] = float(np.std([t for t in recent_temps if t is not None]))
        else:
            features['temp_volatility_6h'] = None
        
        return features
    
    def create_target_labels(self, hourly_forecast: dict) -> dict:
        """
        Crée les labels à partir des PRÉVISIONS (c'est OK ici car ce sont les targets)
        """
        temps = hourly_forecast.get('temperature_2m', [])
        
        targets = {}
        
        # Prédire la température dans 1h, 3h, 6h
        if len(temps) >= 2:
            targets['temp_target_1h'] = temps[1]
        
        if len(temps) >= 4:
            targets['temp_target_3h'] = temps[3]
        
        if len(temps) >= 7:
            targets['temp_target_6h'] = temps[6]
        
        return targets
    
    def engineer_features(self, cleaned_data: dict) -> Optional[dict]:
        """
        Fonction principale : génère features depuis historique + prévisions
        
        IMPORTANT:
        - Features calculées sur HISTORIQUE réel
        - Targets pris depuis PRÉVISIONS API (pour training)
        """
        try:
            current = cleaned_data.get('current', {})
            hourly = cleaned_data.get('hourly', {})
            location = cleaned_data.get('location', {})
            metadata = cleaned_data.get('metadata', {})
            
            # 1. Ajouter current à l'historique
            self.update_history(current)
            
            # 2. Features de base (valeurs actuelles)
            base_features = {
                'temp_current': current.get('temperature_2m'),
                'humidity_current': current.get('relative_humidity_2m'),
                'wind_current': current.get('wind_speed_10m'),
                'pressure_current': current.get('pressure_msl'),
                'cloud_cover_current': current.get('cloud_cover'),
                'precipitation_current': current.get('precipitation')
            }
            
            # 3. Features temporelles depuis HISTORIQUE
            historical_features = self.extract_features_from_history()
            
            # 4. Features cycliques (heure du jour)
            current_time = datetime.fromisoformat(current.get('time'))
            hour = current_time.hour
            day_of_week = current_time.weekday()
            
            cyclical_features = {
                'hour_sin': float(np.sin(2 * np.pi * hour / 24)),
                'hour_cos': float(np.cos(2 * np.pi * hour / 24)),
                'day_sin': float(np.sin(2 * np.pi * day_of_week / 7)),
                'day_cos': float(np.cos(2 * np.pi * day_of_week / 7)),
                'is_day': 1 if 6 <= hour <= 20 else 0,
                'is_weekend': 1 if day_of_week >= 5 else 0
            }
            
            # 5. Target labels (prévisions futures)
            target_labels = self.create_target_labels(hourly)
            
            # 6. Assemblage final
            feature_vector = {
                'location': location,
                'timestamp': current.get('time'),
                'base_features': base_features,
                'historical_features': historical_features,
                'cyclical_features': cyclical_features,
                'target_labels': target_labels,
                'metadata': {
                    **metadata,
                    'feature_engineered_at': datetime.utcnow().isoformat(),
                    'history_size': len(self.temp_history),
                    'samples_accumulated': self.samples_accumulated,
                    'feature_count': self._count_all_features(
                        base_features, historical_features, cyclical_features
                    )
                }
            }
            
            return feature_vector
            
        except Exception as e:
            print(f"❌ Erreur feature engineering: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    @staticmethod
    def _count_all_features(base, historical, cyclical) -> int:
        """Compte le nombre total de features non-nulles"""
        count = 0
        
        for feat_dict in [base, historical, cyclical]:
            if isinstance(feat_dict, dict):
                for key, value in feat_dict.items():
                    if isinstance(value, dict):
                        count += len([v for v in value.values() if v is not None])
                    elif value is not None:
                        count += 1
        
        return count
    
    def get_history_summary(self) -> dict:
        """Retourne un résumé de l'historique accumulé"""
        return {
            'samples': len(self.temp_history),
            'max_capacity': self.history_size,
            'temp_range': {
                'min': min(self.temp_history) if self.temp_history else None,
                'max': max(self.temp_history) if self.temp_history else None
            },
            'time_range': {
                'oldest': self.timestamps[0] if self.timestamps else None,
                'newest': self.timestamps[-1] if self.timestamps else None
            }
        }