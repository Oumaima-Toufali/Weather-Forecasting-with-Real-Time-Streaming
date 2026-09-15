# Path: backend/training/historical_data_loader.py
"""
Historical Data Loader - Pipeline d'entraînement offline

Ce script permet d'entraîner des modèles ML sur des données historiques Open-Meteo
SANS utiliser Kafka. L'entraînement est complètement offline.

IMPORTANT:
- Kafka est utilisé uniquement pour la prédiction temps réel
- L'entraînement se fait sur données historiques via l'API Open-Meteo Archive
- Aucune modification du Producer/Consumer Kafka existants

Usage:
    python -m backend.training.historical_data_loader \
        --start-date 2024-01-01 \
        --end-date 2024-01-31 \
        --latitude 33.5731 \
        --longitude -7.5898
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

from backend.config.settings import OPEN_METEO


class HistoricalDataLoader:
    """
    Charge des données historiques depuis Open-Meteo Archive API
    et prépare un dataset pour l'entraînement offline.
    """
    
    ARCHIVE_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    def __init__(self, latitude: float, longitude: float):
        """
        Initialise le loader avec les coordonnées géographiques.
        
        Args:
            latitude: Latitude (Casablanca par défaut: 33.5731)
            longitude: Longitude (Casablanca par défaut: -7.5898)
        """
        self.latitude = latitude
        self.longitude = longitude
    
    def fetch_historical_data(
        self,
        start_date: str,
        end_date: str,
        variables: Optional[List[str]] = None
    ) -> Optional[Dict]:
        """
        Récupère les données historiques depuis l'API Open-Meteo Archive.
        
        Args:
            start_date: Date de début (format: YYYY-MM-DD)
            end_date: Date de fin (format: YYYY-MM-DD)
            variables: Liste des variables à récupérer (défaut: temp, humidity, wind)
        
        Returns:
            Dictionnaire avec les données horaires ou None en cas d'erreur
        """
        if variables is None:
            # Variables pertinentes pour la prédiction de température
            variables = [
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "pressure_msl",  # Pression atmosphérique (importante pour prédiction)
                "cloud_cover",   # Couverture nuageuse
                "precipitation"  # Précipitations
            ]
        
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(variables),
            "timezone": "UTC"
        }
        
        try:
            print(f"[INFO] Recuperation donnees historiques: {start_date} -> {end_date}")
            response = requests.get(self.ARCHIVE_BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if "hourly" not in data:
                print("[WARN] Aucune donnee horaire dans la reponse")
                return None
            
            print(f"[OK] {len(data['hourly']['time'])} points de donnees recuperes")
            return data
            
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Erreur API Open-Meteo Archive: {e}")
            return None
    
    def build_dataframe(self, api_data: Dict) -> pd.DataFrame:
        """
        Construit un pandas DataFrame à partir des données API.
        
        Args:
            api_data: Données brutes de l'API Open-Meteo
        
        Returns:
            DataFrame avec colonnes: timestamp, temperature, humidity, wind_speed, 
            pressure, cloud_cover, precipitation
        """
        hourly = api_data.get("hourly", {})
        
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(hourly.get("time", [])),
            "temperature": hourly.get("temperature_2m", []),
            "humidity": hourly.get("relative_humidity_2m", []),
            "wind_speed": hourly.get("wind_speed_10m", []),
            "pressure": hourly.get("pressure_msl", []),
            "cloud_cover": hourly.get("cloud_cover", []),
            "precipitation": hourly.get("precipitation", [])
        })
        
        # Trier par timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        return df


class DataCleaner:
    """
    Nettoie les données météorologiques (même logique que WeatherNormalizer).
    Supprime les valeurs aberrantes et les valeurs nulles.
    """
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoie le DataFrame en supprimant les valeurs aberrantes et nulles.
        
        Règles de nettoyage (cohérentes avec WeatherNormalizer):
        - Température: supprimer si < -50°C ou > 60°C
        - Humidité: supprimer si < 0% ou > 100%
        - Vent: supprimer si < 0 km/h
        - Pression: supprimer si < 800 hPa ou > 1100 hPa (valeurs aberrantes)
        - Couverture nuageuse: supprimer si < 0% ou > 100%
        - Précipitations: supprimer si < 0 mm
        - Supprimer toutes les lignes avec valeurs nulles
        
        Args:
            df: DataFrame brut
        
        Returns:
            DataFrame nettoyé
        """
        df = df.copy()
        initial_rows = len(df)
        
        # Supprimer les valeurs nulles
        df = df.dropna()
        
        # Supprimer températures aberrantes (< -50 ou > 60)
        df = df[(df["temperature"] >= -50) & (df["temperature"] <= 60)]
        
        # Supprimer humidités aberrantes (< 0 ou > 100)
        if "humidity" in df.columns:
            df = df[(df["humidity"] >= 0) & (df["humidity"] <= 100)]
        
        # Supprimer vents négatifs
        if "wind_speed" in df.columns:
            df = df[df["wind_speed"] >= 0]
        
        # Supprimer pressions aberrantes (< 800 ou > 1100 hPa)
        if "pressure" in df.columns:
            df = df[(df["pressure"] >= 800) & (df["pressure"] <= 1100)]
        
        # Supprimer couverture nuageuse aberrante (< 0 ou > 100)
        if "cloud_cover" in df.columns:
            df = df[(df["cloud_cover"] >= 0) & (df["cloud_cover"] <= 100)]
        
        # Supprimer précipitations négatives
        if "precipitation" in df.columns:
            df = df[df["precipitation"] >= 0]
        
        final_rows = len(df)
        removed = initial_rows - final_rows
        
        if removed > 0:
            print(f"[INFO] Nettoyage: {removed} lignes supprimées ({initial_rows} → {final_rows})")
        
        return df.reset_index(drop=True)


class HistoricalFeatureEngineer:
    """
    Crée des features temporelles à partir des données historiques
    (même logique que WeatherFeatureEngineer mais adapté pour DataFrame).
    """
    
    @staticmethod
    def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ajoute des features temporelles au DataFrame.
        
        Features créées:
        - Lags: 1h, 3h, 6h
        - Rolling mean: 3h, 6h
        - Deltas: 1h, 3h
        - Encodage temporel: heure, sin/cos heure
        
        Args:
            df: DataFrame avec colonnes timestamp, temperature, humidity, wind_speed
        
        Returns:
            DataFrame avec features ajoutées
        """
        df = df.copy()
        
        # 1. LAG FEATURES (valeurs passées de température)
        df["lag_1h"] = df["temperature"].shift(1)
        df["lag_3h"] = df["temperature"].shift(3)
        df["lag_6h"] = df["temperature"].shift(6)
        
        # Lags pour autres variables pertinentes
        if "pressure" in df.columns:
            df["pressure_lag_1h"] = df["pressure"].shift(1)
            df["pressure_lag_3h"] = df["pressure"].shift(3)
        
        if "humidity" in df.columns:
            df["humidity_lag_1h"] = df["humidity"].shift(1)
            df["humidity_lag_3h"] = df["humidity"].shift(3)
        
        # 2. ROLLING MEAN (moyennes glissantes)
        df["rolling_mean_3h"] = df["temperature"].rolling(window=3, min_periods=1).mean()
        df["rolling_mean_6h"] = df["temperature"].rolling(window=6, min_periods=1).mean()
        
        # Rolling mean pour pression (indicateur de changement météo)
        if "pressure" in df.columns:
            df["pressure_rolling_mean_3h"] = df["pressure"].rolling(window=3, min_periods=1).mean()
            df["pressure_rolling_mean_6h"] = df["pressure"].rolling(window=6, min_periods=1).mean()
        
        # 3. DELTAS (variations)
        df["delta_1h"] = df["temperature"].diff(1)
        df["delta_3h"] = df["temperature"].diff(3)
        
        # Deltas pour pression (changement de pression = changement météo)
        if "pressure" in df.columns:
            df["pressure_delta_1h"] = df["pressure"].diff(1)
            df["pressure_delta_3h"] = df["pressure"].diff(3)
        
        # 4. ENCODAGE TEMPOREL (sin/cos pour cyclicité)
        df["hour"] = df["timestamp"].dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        
        # Supprimer colonnes intermédiaires
        df = df.drop(columns=["hour", "day_of_week"])
        
        return df
    
    @staticmethod
    def create_targets(df: pd.DataFrame) -> pd.DataFrame:
        """
        Crée les variables cibles pour l'entraînement.
        
        Targets créées:
        - target_temperature_1h: température dans 1 heure
        - target_temperature_3h: température dans 3 heures
        - target_temperature_6h: température dans 6 heures
        
        Args:
            df: DataFrame avec features
        
        Returns:
            DataFrame avec targets ajoutées
        """
        df = df.copy()
        
        # Shift négatif pour obtenir les valeurs futures
        df["target_temperature_1h"] = df["temperature"].shift(-1)
        df["target_temperature_3h"] = df["temperature"].shift(-3)
        df["target_temperature_6h"] = df["temperature"].shift(-6)
        
        return df


class ModelTrainer:
    """
    Entraîne des modèles ML simples (RandomForest ou XGBoost)
    pour prédire la température à différents horizons.
    """
    
    def __init__(self, model_type: str = "random_forest", n_estimators: int = 100):
        """
        Initialise le trainer.
        
        Args:
            model_type: Type de modèle ("random_forest" ou "xgboost")
            n_estimators: Nombre d'estimateurs (arbres)
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.models = {}
        self.feature_columns = None
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prépare les features pour l'entraînement.
        
        Args:
            df: DataFrame avec toutes les features
        
        Returns:
            Tuple (DataFrame nettoyé, liste des colonnes features)
        """
        # Colonnes à utiliser comme features (toutes les variables pertinentes)
        feature_cols = [
            "temperature",
            "humidity",
            "wind_speed",
            "pressure",      # Pression atmosphérique (très importante)
            "cloud_cover",   # Couverture nuageuse
            "precipitation", # Précipitations
            # Lags température
            "lag_1h",
            "lag_3h",
            "lag_6h",
            # Lags autres variables
            "pressure_lag_1h",
            "pressure_lag_3h",
            "humidity_lag_1h",
            "humidity_lag_3h",
            # Rolling means
            "rolling_mean_3h",
            "rolling_mean_6h",
            "pressure_rolling_mean_3h",
            "pressure_rolling_mean_6h",
            # Deltas
            "delta_1h",
            "delta_3h",
            "pressure_delta_1h",
            "pressure_delta_3h",
            # Encodage temporel
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos"
        ]
        
        # Garder seulement les colonnes qui existent
        available_cols = [col for col in feature_cols if col in df.columns]
        
        # Supprimer les lignes avec NaN dans les features ou targets
        target_cols = ["target_temperature_1h", "target_temperature_3h", "target_temperature_6h"]
        df_clean = df[available_cols + target_cols].dropna()
        
        self.feature_columns = available_cols
        
        return df_clean, available_cols
    
    def train_model(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        horizon: int
    ) -> Tuple[object, Dict[str, float]]:
        """
        Entraîne un modèle pour un horizon donné.
        
        Args:
            X: Features
            y: Target
            horizon: Horizon de prédiction (1, 3, ou 6)
        
        Returns:
            Tuple (modèle entraîné, métriques)
        """
        # Séparer train/test (80/20, séquentiel, pas de shuffle)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Créer le modèle
        if self.model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == "xgboost":
            try:
                import xgboost as xgb
                model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=42,
                    n_jobs=-1
                )
            except ImportError:
                print("[WARN] XGBoost non installe, utilisation de RandomForest")
                model = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                )
        else:
            raise ValueError(f"Type de modèle inconnu: {self.model_type}")
        
        # Entraînement
        print(f"[INFO] Entraînement modèle {horizon}h...")
        model.fit(X_train, y_train)
        
        # Prédictions et métriques
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        metrics = {
            "train_mae": mean_absolute_error(y_train, y_pred_train),
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_pred_train)),
            "train_r2": r2_score(y_train, y_pred_train),
            "test_mae": mean_absolute_error(y_test, y_pred_test),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_pred_test)),
            "test_r2": r2_score(y_test, y_pred_test),
        }
        
        return model, metrics
    
    def train_all_models(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Entraîne les modèles pour les 3 horizons (1h, 3h, 6h).
        
        Args:
            df: DataFrame avec features et targets
        
        Returns:
            Dictionnaire des métriques pour chaque horizon
        """
        # Préparer les features
        df_clean, feature_cols = self.prepare_features(df)
        X = df_clean[feature_cols]
        
        all_metrics = {}
        
        # Entraîner pour chaque horizon
        for horizon in [1, 3, 6]:
            target_col = f"target_temperature_{horizon}h"
            
            if target_col not in df_clean.columns:
                print(f"[WARN] Colonne {target_col} introuvable, sautee")
                continue
            
            y = df_clean[target_col]
            
            # Entraîner
            model, metrics = self.train_model(X, y, horizon)
            self.models[f"{horizon}h"] = model
            all_metrics[f"{horizon}h"] = metrics
            
            # Afficher métriques
            print(f"\n[METRIQUES] Modele {horizon}h:")
            print(f"   Train MAE: {metrics['train_mae']:.3f}°C")
            print(f"   Train RMSE: {metrics['train_rmse']:.3f}°C")
            print(f"   Train R²: {metrics['train_r2']:.3f}")
            print(f"   Test MAE: {metrics['test_mae']:.3f}°C")
            print(f"   Test RMSE: {metrics['test_rmse']:.3f}°C")
            print(f"   Test R²: {metrics['test_r2']:.3f}")
        
        return all_metrics
    
    def save_models(self, output_dir: Path):
        """
        Sauvegarde les modèles entraînés.
        
        Args:
            output_dir: Répertoire de sortie
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for horizon, model in self.models.items():
            filename = output_dir / f"weather_model_{horizon}.pkl"
            joblib.dump(model, filename)
            print(f"[OK] Modele sauvegarde: {filename}")
        
        # Sauvegarder aussi les colonnes features pour la prédiction
        if self.feature_columns:
            metadata = {
                "feature_columns": self.feature_columns,
                "model_type": self.model_type,
                "n_estimators": self.n_estimators
            }
            metadata_path = output_dir / "model_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            print(f"[OK] Metadonnees sauvegardees: {metadata_path}")


def save_dashboard_metrics(
    metrics: Dict[str, Dict[str, float]],
    df_features: pd.DataFrame,
    feature_cols: List[str],
    output_dir: Path,
) -> None:
    """
    Sauvegarde les métriques utiles au dashboard (performance modèle + qualité des données)
    dans un fichier JSON consommé par le dashboard Streamlit.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Qualité des données: complétude des features utilisées par le modèle
    data_quality: Dict[str, Dict[str, float]] = {}
    total_rows = int(len(df_features)) if len(df_features) else 0

    for col in feature_cols:
        if col not in df_features.columns or total_rows == 0:
            continue
        non_null = int(df_features[col].notna().sum())
        completeness_pct = float(non_null * 100.0 / total_rows)
        data_quality[col] = {
            "non_null": non_null,
            "total": total_rows,
            "completeness_pct": round(completeness_pct, 2),
        }

    payload = {
        "model_performance": metrics,
        "data_quality": data_quality,
    }

    metrics_path = output_dir / "dashboard_metrics.json"
    try:
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[OK] Metriques dashboard sauvegardees: {metrics_path}")
    except Exception as exc:  # pragma: no cover - logging only
        print(f"[WARN] Impossible de sauvegarder les metriques dashboard: {exc}")


def main():
    """
    Point d'entrée principal pour le pipeline d'entraînement offline.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline d'entraînement offline sur données historiques Open-Meteo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT:
- Ce script n'utilise PAS Kafka
- L'entraînement est complètement offline sur données historiques
- Kafka est utilisé uniquement pour la prédiction temps réel

Exemples:
  # Entraînement sur janvier 2024 (Casablanca par défaut)
  python -m backend.training.historical_data_loader \\
      --start-date 2024-01-01 \\
      --end-date 2024-01-31

  # Entraînement avec dates et localisation personnalisées
  python -m backend.training.historical_data_loader \\
      --start-date 2024-01-01 \\
      --end-date 2024-03-31 \\
      --latitude 33.5731 \\
      --longitude -7.5898 \\
      --model-type xgboost
        """
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Date de début (format: YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="Date de fin (format: YYYY-MM-DD)"
    )
    parser.add_argument(
        "--latitude",
        type=float,
        default=OPEN_METEO.DEFAULT_LATITUDE,
        help=f"Latitude (défaut: {OPEN_METEO.DEFAULT_LATITUDE})"
    )
    parser.add_argument(
        "--longitude",
        type=float,
        default=OPEN_METEO.DEFAULT_LONGITUDE,
        help=f"Longitude (défaut: {OPEN_METEO.DEFAULT_LONGITUDE})"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="random_forest",
        choices=["random_forest", "xgboost"],
        help="Type de modèle ML (défaut: random_forest)"
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Nombre d'estimateurs (arbres) (défaut: 100)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Répertoire de sortie pour les modèles (défaut: models)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("PIPELINE D'ENTRAINEMENT OFFLINE - Historical Data Loader")
    print("=" * 70)
    print(f"Localisation: ({args.latitude}, {args.longitude})")
    print(f"Periode: {args.start_date} -> {args.end_date}")
    print(f"Modele: {args.model_type} ({args.n_estimators} estimateurs)")
    print(f"Sortie: {args.output_dir}")
    print("-" * 70)
    
    # 1. Charger les données historiques
    loader = HistoricalDataLoader(args.latitude, args.longitude)
    api_data = loader.fetch_historical_data(args.start_date, args.end_date)
    
    if api_data is None:
        raise SystemExit("[ERROR] Impossible de recuperer les donnees historiques")
    
    # 2. Construire le DataFrame
    print("\n[INFO] Construction du DataFrame...")
    df = loader.build_dataframe(api_data)
    print(f"[OK] DataFrame cree: {len(df)} lignes")
    
    # 3. Nettoyer les données
    print("\n[INFO] Nettoyage des donnees...")
    cleaner = DataCleaner()
    df_clean = cleaner.clean_data(df)
    print(f"[OK] Donnees nettoyees: {len(df_clean)} lignes")
    
    # 4. Feature Engineering
    print("\n[INFO] Feature Engineering...")
    engineer = HistoricalFeatureEngineer()
    df_features = engineer.engineer_features(df_clean)
    df_features = engineer.create_targets(df_features)
    print(f"[OK] Features creees: {len(df_features.columns)} colonnes")
    
    # 5. Entraîner les modèles
    print("\n[INFO] Entraînement des modèles...")
    trainer = ModelTrainer(
        model_type=args.model_type,
        n_estimators=args.n_estimators
    )
    metrics = trainer.train_all_models(df_features)
    
    # 6. Sauvegarder les modèles
    print("\n[INFO] Sauvegarde des modèles...")
    trainer.save_models(args.output_dir)

    # 7. Sauvegarder aussi les métriques pour le dashboard
    print("\n[INFO] Sauvegarde des metriques pour le dashboard...")
    feature_cols_for_quality = trainer.feature_columns or []
    save_dashboard_metrics(metrics, df_features, feature_cols_for_quality, Path(args.output_dir))
    
    print("\n" + "=" * 70)
    print("ENTRAINEMENT TERMINE")
    print("=" * 70)
    print(f"Modeles entraines: {len(trainer.models)}")
    print(f"Repertoire: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

