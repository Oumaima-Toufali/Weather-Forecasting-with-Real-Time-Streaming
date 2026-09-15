# Path: backend/processing/feature_consumer.py
"""
Consumer avec Feature Engineering basé sur historique réel
"""

import time
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

from backend.config.kafka_config import CONSUMER_CONFIG, PRODUCER_CONFIG, TOPICS
from backend.processing.feature_engineering import WeatherFeatureEngineer
from backend.utils.schema_validator import load_avro_schema, validate_avro


class FeatureStreamProcessor:
    """
    Consomme data.cleaned.weather, accumule historique,
    crée des features, et produit vers data.features.weather
    """
    
    def __init__(self):
        # Consumer
        consumer_config = CONSUMER_CONFIG.copy()
        consumer_config['group_id'] = 'feature-engineering-group'
        consumer_config['auto_offset_reset'] = 'earliest'  # Lire depuis le début
        
        self.consumer = KafkaConsumer(
            TOPICS['cleaned'],
            **consumer_config
        )
        
        # Producer
        self.producer = KafkaProducer(**PRODUCER_CONFIG)
        
        # Feature engineer (AVEC historique en mémoire)
        self.engineer = WeatherFeatureEngineer(history_size=168)  # 7 jours
        
        self.processed_count = 0
        self.error_count = 0
        self.skipped_duplicates = 0
        
        self.last_features = None  # Pour détecter les duplicatas complets
        self.features_schema = load_avro_schema("backend/schemas/features_weather.avsc")
    
    def is_duplicate(self, current_data: dict) -> bool:
        """
        Détecte si c'est un vrai doublon en comparant TOUTES les features pertinentes.
        
        Un doublon est détecté uniquement si toutes les features météorologiques
        pertinentes pour la prédiction sont identiques (dans une tolérance).
        
        Features comparées:
        - temperature_2m
        - relative_humidity_2m
        - wind_speed_10m
        - pressure_msl
        - cloud_cover
        - precipitation
        """
        if self.last_features is None:
            # Stocker les features actuelles pour la prochaine comparaison
            self.last_features = {
                'temp': current_data.get('temperature_2m'),
                'humidity': current_data.get('relative_humidity_2m'),
                'wind': current_data.get('wind_speed_10m'),
                'pressure': current_data.get('pressure_msl'),
                'cloud': current_data.get('cloud_cover'),
                'precip': current_data.get('precipitation'),
            }
            return False
        
        # Comparer toutes les features pertinentes
        current_features = {
            'temp': current_data.get('temperature_2m'),
            'humidity': current_data.get('relative_humidity_2m'),
            'wind': current_data.get('wind_speed_10m'),
            'pressure': current_data.get('pressure_msl'),
            'cloud': current_data.get('cloud_cover'),
            'precip': current_data.get('precipitation'),
        }
        
        # Tolérances pour comparaison (évite les erreurs d'arrondi)
        TOLERANCES = {
            'temp': 0.01,      # 0.01°C
            'humidity': 0.1,   # 0.1%
            'wind': 0.01,      # 0.01 km/h
            'pressure': 0.1,   # 0.1 hPa
            'cloud': 0.1,      # 0.1%
            'precip': 0.001,   # 0.001 mm
        }
        
        # Vérifier si toutes les features sont identiques (dans la tolérance)
        for key in current_features:
            current_val = current_features[key]
            last_val = self.last_features[key]
            
            # Gérer les valeurs None
            if current_val is None and last_val is None:
                continue
            if current_val is None or last_val is None:
                # Si une valeur est None et l'autre non, ce n'est pas un doublon
                self.last_features = current_features
                return False
            
            # Comparer avec tolérance
            tolerance = TOLERANCES[key]
            if abs(current_val - last_val) > tolerance:
                # Au moins une feature diffère, ce n'est pas un doublon
                self.last_features = current_features
                return False
        
        # Toutes les features sont identiques -> vrai doublon
        return True
    
    def process_message(self, message):
        """Traite un message cleaned et génère des features"""
        try:
            cleaned_data = message.value
            current = cleaned_data.get('current', {})
            
            # Détection doublons complets (compare toutes les features pertinentes)
            if self.is_duplicate(current):
                self.skipped_duplicates += 1
                if self.skipped_duplicates % 10 == 0:
                    print(f"⏭️  Doublons complets skippés: {self.skipped_duplicates}")
                return None
            
            # Feature engineering
            features = self.engineer.engineer_features(cleaned_data)
            
            if features:
                return features
            else:
                print(f"⚠️  Échec feature engineering")
                self.error_count += 1
                return None
                
        except Exception as e:
            print(f"❌ Erreur traitement: {e}")
            self.error_count += 1
            return None
    
    def send_to_features_topic(self, key: str, features: dict) -> bool:
        """Envoie les features vers data.features.weather"""
        try:
            if not validate_avro(features, self.features_schema):
                print("❌ Features invalides vs schéma Avro, message ignoré")
                return False
            
            future = self.producer.send(
                TOPICS['features'],
                key=key,
                value=features
            )
            future.get(timeout=10)
            return True
        except KafkaError as e:
            print(f"❌ Erreur envoi Kafka: {e}")
            return False
    
    def run(self):
        """Boucle principale"""
        print("=" * 70)
        print("🧠 FEATURE ENGINEERING PROCESSOR v2.0")
        print("=" * 70)
        print(f"📥 Input : {TOPICS['cleaned']}")
        print(f"📤 Output: {TOPICS['features']}")
        print(f"💾 Historique: {self.engineer.history_size} samples max")
        print("-" * 70)
        
        try:
            for message in self.consumer:
                # Feature engineering
                features = self.process_message(message)
                
                if features:
                    # Envoi vers topic features
                    success = self.send_to_features_topic(message.key, features)
                    
                    if success:
                        self.processed_count += 1
                        
                        # Affichage détaillé
                        base = features.get('base_features', {})
                        targets = features.get('target_labels', {})
                        meta = features.get('metadata', {})
                        
                        # Résumé historique
                        history = self.engineer.get_history_summary()
                        
                        print(f"✅ Message #{self.processed_count}")
                        print(f"   📊 Historique: {history['samples']}/{self.engineer.history_size} samples")
                        print(f"   🌡️  Temp actuelle: {base.get('temp_current')}°C")
                        
                        # Afficher features historiques si disponibles
                        hist_features = features.get('historical_features', {})
                        temp_rolling_6h = hist_features.get('temp_rolling_6h', {})
                        if temp_rolling_6h and temp_rolling_6h.get('mean_6h'):
                            print(f"   📈 Moyenne 6h: {temp_rolling_6h['mean_6h']:.2f}°C "
                                  f"(min: {temp_rolling_6h.get('min_6h', 'N/A')}, "
                                  f"max: {temp_rolling_6h.get('max_6h', 'N/A')})")
                        
                        # Deltas
                        temp_deltas = hist_features.get('temp_deltas', {})
                        if temp_deltas.get('delta_3h'):
                            print(f"   📉 Delta 3h: {temp_deltas['delta_3h']:+.2f}°C")
                        
                        # Targets
                        print(f"   🎯 Targets: +1h={targets.get('temp_target_1h')}°C, "
                              f"+3h={targets.get('temp_target_3h')}°C, "
                              f"+6h={targets.get('temp_target_6h')}°C")
                        
                        print(f"   ✨ Features: {meta.get('feature_count')} | "
                              f"Erreurs: {self.error_count} | "
                              f"Duplicatas: {self.skipped_duplicates}")
                        print("-" * 70)
                
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n⛔ Processor arrêté")
            self.show_final_summary()
        finally:
            self.cleanup()
    
    def show_final_summary(self):
        """Affiche un résumé final"""
        print("\n" + "=" * 70)
        print("📊 RÉSUMÉ FINAL")
        print("=" * 70)
        print(f"✅ Messages traités: {self.processed_count}")
        print(f"❌ Erreurs: {self.error_count}")
        print(f"⏭️  Duplicatas skippés: {self.skipped_duplicates}")
        
        history = self.engineer.get_history_summary()
        print(f"\n💾 Historique accumulé: {history['samples']} samples")
        print(f"🌡️  Température range: {history['temp_range']['min']}°C - {history['temp_range']['max']}°C")
        print(f"⏰ Période: {history['time_range']['oldest']} → {history['time_range']['newest']}")
        print("=" * 70)
    
    def cleanup(self):
        """Ferme les connexions"""
        print(f"\n🧹 Nettoyage...")
        self.producer.flush()
        self.producer.close()
        self.consumer.close()


def main():
    processor = FeatureStreamProcessor()
    processor.run()


if __name__ == "__main__":
    main()