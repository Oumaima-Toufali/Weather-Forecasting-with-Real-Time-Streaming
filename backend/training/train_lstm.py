# Path: backend/training/train_lstm.py
"""
LSTM Training Module - Entraînement offline de modèles LSTM pour prédiction météo

Ce module permet d'entraîner des modèles LSTM (Long Short-Term Memory) pour la
prédiction de température à différents horizons, en complément des modèles tabulaires
(RandomForest/XGBoost).

IMPORTANT:
- Entraînement complètement OFFLINE (pas de Kafka)
- Module indépendant, n'affecte pas le pipeline Kafka existant
- Compatible avec un futur déploiement streaming
- Modèle sauvegardé au format Keras (.h5) pour compatibilité ONNX

Usage:
    python -m backend.training.train_lstm \
        --start-date 2024-01-01 \
        --end-date 2024-03-31 \
        --timesteps 24 \
        --epochs 50 \
        --batch-size 32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    keras = None  # type: ignore
    Sequential = LSTM = Dense = Dropout = EarlyStopping = ModelCheckpoint = None  # type: ignore

from backend.config.settings import OPEN_METEO


class HistoricalDataLoader:
    """
    Charge des données historiques depuis Open-Meteo Archive API.
    Réutilise la logique de historical_data_loader.py pour cohérence.
    """
    
    ARCHIVE_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude
    
    def fetch_historical_data(
        self,
        start_date: str,
        end_date: str,
        variables: Optional[List[str]] = None
    ) -> Optional[Dict]:
        """Récupère les données historiques depuis l'API Open-Meteo Archive."""
        if variables is None:
            variables = [
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
                "pressure_msl",
                "cloud_cover",
                "precipitation"
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
        """Construit un pandas DataFrame à partir des données API."""
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
        
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df


class DataCleaner:
    """Nettoie les données météorologiques (même logique que historical_data_loader)."""
    
    @staticmethod
    def clean_data(df: pd.DataFrame) -> pd.DataFrame:
        """Nettoie le DataFrame en supprimant les valeurs aberrantes et nulles."""
        df = df.copy()
        initial_rows = len(df)
        
        df = df.dropna()
        df = df[(df["temperature"] >= -50) & (df["temperature"] <= 60)]
        
        if "humidity" in df.columns:
            df = df[(df["humidity"] >= 0) & (df["humidity"] <= 100)]
        if "wind_speed" in df.columns:
            df = df[df["wind_speed"] >= 0]
        if "pressure" in df.columns:
            df = df[(df["pressure"] >= 800) & (df["pressure"] <= 1100)]
        if "cloud_cover" in df.columns:
            df = df[(df["cloud_cover"] >= 0) & (df["cloud_cover"] <= 100)]
        if "precipitation" in df.columns:
            df = df[df["precipitation"] >= 0]
        
        final_rows = len(df)
        removed = initial_rows - final_rows
        
        if removed > 0:
            print(f"[INFO] Nettoyage: {removed} lignes supprimees ({initial_rows} -> {final_rows})")
        
        return df.reset_index(drop=True)


class LSTMPreprocessor:
    """
    Prépare les données pour l'entraînement LSTM.
    Crée des séquences temporelles (timesteps) et normalise les données.
    """
    
    def __init__(self, timesteps: int = 24):
        """
        Initialise le preprocessor.
        
        Args:
            timesteps: Nombre d'heures passées à utiliser pour prédire (défaut: 24h)
        """
        self.timesteps = timesteps
        self.feature_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()
        self.feature_columns = None
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prépare les features pour LSTM (sans features temporelles complexes).
        LSTM apprend les patterns temporels directement depuis les séquences.
        
        Args:
            df: DataFrame avec colonnes météo brutes
        
        Returns:
            DataFrame avec features préparées
        """
        df = df.copy()
        
        # Features de base (LSTM apprendra les patterns temporels)
        feature_cols = [
            "temperature",
            "humidity",
            "wind_speed",
            "pressure",
            "cloud_cover",
            "precipitation"
        ]
        
        # Garder seulement les colonnes qui existent
        available_cols = [col for col in feature_cols if col in df.columns]
        self.feature_columns = available_cols
        
        # Encodage temporel simple (pour aider LSTM)
        df["hour"] = df["timestamp"].dt.hour
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        
        # Ajouter les features cycliques
        available_cols.extend(["hour_sin", "hour_cos", "day_sin", "day_cos"])
        self.feature_columns = available_cols
        
        return df
    
    def create_sequences(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        horizon: int = 1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Crée des séquences (single horizon)."""
        X, y = [], []
        for i in range(len(features) - self.timesteps - horizon + 1):
            X.append(features[i : i + self.timesteps])
            y.append(targets[i + self.timesteps + horizon - 1])
        return np.array(X), np.array(y)

    def create_sequences_multi(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        horizons: List[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Crée des séquences multi-horizon (y shape = [n, len(horizons)])."""
        X, Y = [], []
        max_h = max(horizons)
        for i in range(len(features) - self.timesteps - max_h + 1):
            X.append(features[i : i + self.timesteps])
            Y.append([targets[i + self.timesteps + h - 1] for h in horizons])
        return np.array(X), np.array(Y)
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        horizons: List[int] = [1, 3, 6],
        multi_horizon: bool = False,
    ) -> Dict[str, Tuple]:
        """
        Prépare les données pour l'entraînement.
        
        Retour:
            - si multi_horizon=False: {horizon: (X_train, y_train, X_test, y_test)}
            - si multi_horizon=True : {"multi": (X_train, y_train, X_test, y_test, horizons)}
        """
        df_features = self.prepare_features(df)

        feature_data = df_features[self.feature_columns].values
        temperature_data = df["temperature"].values

        feature_data_scaled = self.feature_scaler.fit_transform(feature_data)
        temperature_scaled = self.target_scaler.fit_transform(
            temperature_data.reshape(-1, 1)
        ).flatten()

        if multi_horizon:
            X, Y = self.create_sequences_multi(
                feature_data_scaled, temperature_scaled, horizons
            )
            if len(X) == 0:
                raise SystemExit("[ERROR] Pas assez de donnees pour creer des sequences multi-horizon")
            split_seq = int(len(X) * 0.8)
            X_train, X_test = X[:split_seq], X[split_seq:]
            Y_train, Y_test = Y[:split_seq], Y[split_seq:]
            print(
                f"[INFO] Multi-horizon: {len(X_train)} sequences train, {len(X_test)} sequences test"
            )
            return {"multi": (X_train, Y_train, X_test, Y_test, horizons)}

        # Sinon, un modèle par horizon
        datasets = {}
        for horizon in horizons:
            X, y = self.create_sequences(
                feature_data_scaled,
                temperature_scaled,
                horizon=horizon,
            )
            split_seq = int(len(X) * 0.8)
            X_train, X_test = X[:split_seq], X[split_seq:]
            y_train, y_test = y[:split_seq], y[split_seq:]
            datasets[horizon] = (X_train, y_train, X_test, y_test)
            print(
                f"[INFO] Horizon {horizon}h: {len(X_train)} sequences train, {len(X_test)} sequences test"
            )

        return datasets


class LSTMTrainer:
    """
    Entraîne des modèles LSTM pour la prédiction de température.
    Supporte l'entraînement multi-horizon et multi-output.
    """
    
    def __init__(
        self,
        timesteps: int = 24,
        n_features: int = 10,
        lstm_units: List[int] = [64, 32],
        dropout_rate: float = 0.2,
        multi_horizon: bool = False
    ):
        """
        Initialise le trainer LSTM.
        
        Args:
            timesteps: Nombre de timesteps dans les séquences
            n_features: Nombre de features par timestep
            lstm_units: Liste des unités LSTM par couche (ex: [64, 32])
            dropout_rate: Taux de dropout pour régularisation
            multi_horizon: Si True, entraîne un modèle multi-output (1h, 3h, 6h)
        """
        self.timesteps = timesteps
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.multi_horizon = multi_horizon
        self.models = {}
    
    def build_model(self, output_dim: int = 1) -> keras.Model:
        """
        Construit un modèle LSTM.
        
        Args:
            output_dim: Dimension de la sortie (1 pour single-output, 3 pour multi-horizon)
        
        Returns:
            Modèle Keras compilé
        """
        model = Sequential()

        # Parcours des couches LSTM en contrôlant return_sequences
        for idx, units in enumerate(self.lstm_units):
            is_last = idx == len(self.lstm_units) - 1
            if idx == 0:
                # Première couche avec input_shape
                model.add(
                    LSTM(
                        units=units,
                        return_sequences=not is_last,
                        input_shape=(self.timesteps, self.n_features),
                    )
                )
            else:
                # Couches suivantes
                model.add(
                    LSTM(
                        units=units,
                        return_sequences=not is_last,
                    )
                )
            model.add(Dropout(self.dropout_rate))

        # Couche de sortie
        model.add(Dense(output_dim, activation="linear"))
        
        # Compiler
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["mae"],
        )
        
        return model
    
    def _compute_metrics_single(self, y_true, y_pred, scaler: MinMaxScaler) -> Dict[str, float]:
        """Calcule les métriques en échelle originale (inverse_transform)."""
        y_true_denorm = scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_denorm = scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        return {
            "mae": float(mean_absolute_error(y_true_denorm, y_pred_denorm)),
            "rmse": float(np.sqrt(mean_squared_error(y_true_denorm, y_pred_denorm))),
            "r2": float(r2_score(y_true_denorm, y_pred_denorm)),
        }

    def _compute_metrics_multi(
        self, y_true, y_pred, scaler: MinMaxScaler, horizons: List[int]
    ) -> Dict[str, Dict[str, float]]:
        """Métriques multi-sortie en échelle originale."""
        metrics = {}
        for idx, h in enumerate(horizons):
            true_col = scaler.inverse_transform(y_true[:, idx].reshape(-1, 1)).flatten()
            pred_col = scaler.inverse_transform(y_pred[:, idx].reshape(-1, 1)).flatten()
            metrics[f"{h}h"] = {
                "mae": float(mean_absolute_error(true_col, pred_col)),
                "rmse": float(np.sqrt(mean_squared_error(true_col, pred_col))),
                "r2": float(r2_score(true_col, pred_col)),
            }
        return metrics

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        horizon: int,
        target_scaler: MinMaxScaler,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.1
    ) -> Tuple[keras.Model, Dict[str, float]]:
        """
        Entraîne un modèle LSTM pour un horizon donné.
        
        Args:
            X_train: Séquences d'entraînement
            y_train: Targets d'entraînement
            X_test: Séquences de test
            y_test: Targets de test
            horizon: Horizon de prédiction
            epochs: Nombre d'époques
            batch_size: Taille du batch
            validation_split: Fraction pour validation
        
        Returns:
            Tuple (modèle entraîné, métriques)
        """
        print(f"[INFO] Entrainement modele LSTM {horizon}h...")
        
        # Construire modèle
        model = self.build_model(output_dim=1)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                f'models/lstm_checkpoint_{horizon}h.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-5,
                verbose=1,
            ),
        ]
        
        # Entraînement
        history = model.fit(
            X_train, y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        # Prédictions
        y_pred_train = model.predict(X_train, verbose=0)
        y_pred_test = model.predict(X_test, verbose=0)
        
        # Métriques en échelle originale
        train_metrics = self._compute_metrics_single(y_train, y_pred_train, target_scaler)
        test_metrics = self._compute_metrics_single(y_test, y_pred_test, target_scaler)
        metrics = {
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_r2": train_metrics["r2"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_r2": test_metrics["r2"],
        }
        
        return model, metrics

    def train_model_multi(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        horizons: List[int],
        target_scaler: MinMaxScaler,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.1,
    ) -> Tuple[keras.Model, Dict[str, Dict[str, float]]]:
        """
        Entraîne un modèle LSTM multi-horizon (sortie dimension len(horizons)).
        """
        print("[INFO] Entrainement modele LSTM multi-horizon...")

        model = self.build_model(output_dim=len(horizons))

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
                verbose=1,
            ),
            ModelCheckpoint(
                "models/lstm_checkpoint_multi.h5",
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=5,
                min_lr=1e-5,
                verbose=1,
            ),
        ]

        model.fit(
            X_train,
            y_train,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        y_pred_train = model.predict(X_train, verbose=0)
        y_pred_test = model.predict(X_test, verbose=0)

        train_metrics = self._compute_metrics_multi(y_train, y_pred_train, target_scaler, horizons)
        test_metrics = self._compute_metrics_multi(y_test, y_pred_test, target_scaler, horizons)
        metrics = {}
        for h in train_metrics.keys():
            metrics[h] = {
                "train_mae": train_metrics[h]["mae"],
                "train_rmse": train_metrics[h]["rmse"],
                "train_r2": train_metrics[h]["r2"],
                "test_mae": test_metrics[h]["mae"],
                "test_rmse": test_metrics[h]["rmse"],
                "test_r2": test_metrics[h]["r2"],
            }

        return model, metrics
    
    def train_all_models(
        self,
        datasets: Dict[str, Tuple],
        epochs: int = 50,
        batch_size: int = 32,
        target_scaler: MinMaxScaler | None = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Entraîne les modèles pour tous les horizons.
        
        Args:
            datasets: Dictionnaire {horizon: (X_train, y_train, X_test, y_test)}
            epochs: Nombre d'époques
            batch_size: Taille du batch
        
        Returns:
            Dictionnaire des métriques pour chaque horizon
        """
        all_metrics = {}
        
        # Cas multi-horizon (clé "multi")
        if "multi" in datasets:
            X_train, y_train, X_test, y_test, horizons = datasets["multi"]
            model, metrics = self.train_model_multi(
                X_train, y_train, X_test, y_test,
                horizons=horizons,
                target_scaler=target_scaler if target_scaler else MinMaxScaler(),
                epochs=epochs,
                batch_size=batch_size
            )
            self.models["multi"] = model
            all_metrics.update(metrics)

            print("\n[METRIQUES] Modele LSTM multi-horizon:")
            for h, m in metrics.items():
                print(f"  Horizon {h}:")
                print(f"    Train MAE: {m['train_mae']:.4f}")
                print(f"    Train RMSE: {m['train_rmse']:.4f}")
                print(f"    Train R²: {m['train_r2']:.4f}")
                print(f"    Test MAE: {m['test_mae']:.4f}")
                print(f"    Test RMSE: {m['test_rmse']:.4f}")
                print(f"    Test R²: {m['test_r2']:.4f}")
            return all_metrics

        # Cas un modèle par horizon
        for horizon, (X_train, y_train, X_test, y_test) in datasets.items():
            model, metrics = self.train_model(
                X_train, y_train, X_test, y_test,
                horizon=int(horizon),
                target_scaler=target_scaler if target_scaler else MinMaxScaler(),
                epochs=epochs,
                batch_size=batch_size
            )
            
            self.models[f"{horizon}h"] = model
            all_metrics[f"{horizon}h"] = metrics
            
            print(f"\n[METRIQUES] Modele LSTM {horizon}h:")
            print(f"   Train MAE: {metrics['train_mae']:.4f}")
            print(f"   Train RMSE: {metrics['train_rmse']:.4f}")
            print(f"   Train R²: {metrics['train_r2']:.4f}")
            print(f"   Test MAE: {metrics['test_mae']:.4f}")
            print(f"   Test RMSE: {metrics['test_rmse']:.4f}")
            print(f"   Test R²: {metrics['test_r2']:.4f}")
        
        return all_metrics

    def save_dashboard_metrics(
        self,
        metrics: Dict[str, Dict[str, float]],
        preprocessor: LSTMPreprocessor,
        output_dir: Path,
    ) -> None:
        """
        Sauvegarde les métriques utiles au dashboard (backtest LSTM) dans un JSON.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Qualité des données: complétude des features utilisées par le LSTM
        data_quality: Dict[str, Dict[str, float]] = {}
        # preprocessor.feature_columns a déjà été définie
        # On n'a pas ici le DF complet pour compter NaN, donc on stocke simplement la liste des features utilisées
        # (Optionnel: on pourrait passer un DF pour calculer une vraie complétude. Ici on reste léger.)
        for col in preprocessor.feature_columns or []:
            data_quality[col] = {
                "completeness_pct": 100.0  # par simplification, les séquences LSTM sont construites sans NaN
            }

        payload = {
            "model_performance": metrics,
            "data_quality": data_quality,
        }

        metrics_path = output_dir / "dashboard_metrics.json"
        try:
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"[OK] Metriques dashboard (LSTM) sauvegardees: {metrics_path}")
        except Exception as exc:
            print(f"[WARN] Impossible de sauvegarder les metriques dashboard LSTM: {exc}")
    
    def save_models(self, output_dir: Path, preprocessor: LSTMPreprocessor):
        """
        Sauvegarde les modèles LSTM et les métadonnées.
        
        Args:
            output_dir: Répertoire de sortie
            preprocessor: Preprocessor utilisé (pour sauvegarder scalers)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder modèles Keras
        for horizon, model in self.models.items():
            # si clé "multi" -> nom explicite
            if horizon == "multi":
                filename = output_dir / "lstm_model_multi.h5"
            else:
                filename = output_dir / f"lstm_model_{horizon}.h5"
            model.save(filename)
            print(f"[OK] Modele LSTM sauvegarde: {filename}")
        
        # Sauvegarder métadonnées
        metadata = {
            "model_type": "lstm",
            "timesteps": self.timesteps,
            "n_features": self.n_features,
            "lstm_units": self.lstm_units,
            "dropout_rate": self.dropout_rate,
            "feature_columns": preprocessor.feature_columns,
            "horizons": list(self.models.keys())
        }
        
        metadata_path = output_dir / "lstm_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[OK] Metadonnees sauvegardees: {metadata_path}")
        
        # Note: Pour dénormaliser les prédictions, il faudrait aussi sauvegarder
        # les scalers (feature_scaler et target_scaler) avec joblib ou pickle


def main():
    """Point d'entrée principal pour l'entraînement LSTM."""
    
    if not TF_AVAILABLE:
        raise SystemExit(
            "[ERROR] TensorFlow/Keras non installe. "
            "Installez avec: pip install tensorflow"
        )
    
    parser = argparse.ArgumentParser(
        description="Entrainement offline de modeles LSTM pour prediction meteo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
IMPORTANT:
- Entrainement completement OFFLINE (pas de Kafka)
- Module independant, n'affecte pas le pipeline Kafka existant
- Compatible avec un futur deploiement streaming

Exemples:
  # Entrainement basique (24h timesteps, 50 epochs)
  python -m backend.training.train_lstm \\
      --start-date 2024-01-01 \\
      --end-date 2024-03-31

  # Entrainement avec parametres personnalises
  python -m backend.training.train_lstm \\
      --start-date 2024-01-01 \\
      --end-date 2024-03-31 \\
      --timesteps 48 \\
      --epochs 100 \\
      --batch-size 64 \\
      --lstm-units 128 64
        """
    )
    
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Date de debut (format: YYYY-MM-DD)"
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
        help=f"Latitude (defaut: {OPEN_METEO.DEFAULT_LATITUDE})"
    )
    parser.add_argument(
        "--longitude",
        type=float,
        default=OPEN_METEO.DEFAULT_LONGITUDE,
        help=f"Longitude (defaut: {OPEN_METEO.DEFAULT_LONGITUDE})"
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=24,
        help="Nombre d'heures passees pour predire (defaut: 24)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Nombre d'epoques d'entrainement (defaut: 50)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Taille du batch (defaut: 32)"
    )
    parser.add_argument(
        "--lstm-units",
        type=int,
        nargs="+",
        default=[64, 32],
        help="Unites LSTM par couche (defaut: 64 32)"
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.2,
        help="Taux de dropout pour regularisation (defaut: 0.2)"
    )
    parser.add_argument(
        "--multi-horizon",
        action="store_true",
        help="Entrainer un modele multi-output (1h, 3h, 6h simultanement)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models_lstm",
        help="Repertoire de sortie (defaut: models_lstm)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("LSTM TRAINING - Entrainement offline de modeles LSTM")
    print("=" * 70)
    print(f"Localisation: ({args.latitude}, {args.longitude})")
    print(f"Periode: {args.start_date} -> {args.end_date}")
    print(f"Timesteps: {args.timesteps}h")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"LSTM units: {args.lstm_units}")
    print(f"Sortie: {args.output_dir}")
    print("-" * 70)
    
    # 1. Charger les données
    print("\n[INFO] Chargement des donnees historiques...")
    loader = HistoricalDataLoader(args.latitude, args.longitude)
    api_data = loader.fetch_historical_data(args.start_date, args.end_date)
    
    if api_data is None:
        raise SystemExit("[ERROR] Impossible de recuperer les donnees historiques")
    
    # 2. Construire DataFrame
    print("\n[INFO] Construction du DataFrame...")
    df = loader.build_dataframe(api_data)
    print(f"[OK] DataFrame cree: {len(df)} lignes")
    
    # 3. Nettoyer
    print("\n[INFO] Nettoyage des donnees...")
    cleaner = DataCleaner()
    df_clean = cleaner.clean_data(df)
    print(f"[OK] Donnees nettoyees: {len(df_clean)} lignes")
    
    # 4. Préparer pour LSTM
    print("\n[INFO] Preparation des sequences LSTM...")
    preprocessor = LSTMPreprocessor(timesteps=args.timesteps)
    datasets = preprocessor.prepare_data(
        df_clean,
        horizons=[1, 3, 6],
        multi_horizon=args.multi_horizon,
    )
    
    # 5. Entraîner
    print("\n[INFO] Entrainement des modeles LSTM...")
    n_features = len(preprocessor.feature_columns)
    trainer = LSTMTrainer(
        timesteps=args.timesteps,
        n_features=n_features,
        lstm_units=args.lstm_units,
        dropout_rate=args.dropout_rate,
        multi_horizon=args.multi_horizon
    )
    
    metrics = trainer.train_all_models(
        datasets,
        epochs=args.epochs,
        batch_size=args.batch_size,
        target_scaler=preprocessor.target_scaler,
    )
    
    # 6. Sauvegarder modèles + métriques dashboard
    print("\n[INFO] Sauvegarde des modeles...")
    trainer.save_models(args.output_dir, preprocessor)

    print("[INFO] Sauvegarde des metriques dashboard LSTM...")
    trainer.save_dashboard_metrics(metrics, preprocessor, Path(args.output_dir))
    
    print("\n" + "=" * 70)
    print("ENTRAINEMENT LSTM TERMINE")
    print("=" * 70)
    print(f"Modeles entraines: {len(trainer.models)}")
    print(f"Repertoire: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

