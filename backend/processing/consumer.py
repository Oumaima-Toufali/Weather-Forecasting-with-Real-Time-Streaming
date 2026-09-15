# Path: backend/processing/consumer.py
import json
import time
from typing import Optional

from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError

from backend.config.kafka_config import CONSUMER_CONFIG, PRODUCER_CONFIG, TOPICS
from backend.processing.normalizer import WeatherNormalizer


class WeatherStreamProcessor:
    """
    Consomme data.raw.weather, normalise les données,
    et produit vers data.cleaned.weather
    """
    
    def __init__(self):
        # Eviter les doublons de paramètres: utiliser uniquement CONSUMER_CONFIG
        self.consumer = KafkaConsumer(
            TOPICS['raw'],
            **CONSUMER_CONFIG
        )
        
        self.producer = KafkaProducer(**PRODUCER_CONFIG)
        self.normalizer = WeatherNormalizer()
        self.processed_count = 0
        
    def process_message(self, message) -> Optional[dict]:
        """Traite un message brut et retourne données nettoyées"""
        try:
            raw_data = message.value
            
            # Normalisation
            cleaned_data = self.normalizer.normalize(raw_data)
            
            if cleaned_data:
                print(f"OK Message normalise: {message.key}")
                return cleaned_data
            else:
                print(f"WARN Echec normalisation: {message.key}")
                return None
                
        except Exception as e:
            print(f"ERROR Traitement: {e}")
            return None
    
    def send_to_cleaned_topic(self, key: str, cleaned_data: dict) -> bool:
        """Envoie les données nettoyées vers data.cleaned.weather"""
        try:
            future = self.producer.send(
                TOPICS['cleaned'],
                key=key,
                value=cleaned_data
            )
            future.get(timeout=10)
            return True
        except KafkaError as e:
            print(f"ERROR Envoi Kafka: {e}")
            return False
    
    def run(self):
        """Boucle principale de traitement"""
        print(f"Consumer demarre... Ecoute sur '{TOPICS['raw']}'")
        print(f"Sortie vers '{TOPICS['cleaned']}'")
        print("-" * 60)
        
        try:
            for message in self.consumer:
                # Traitement
                cleaned_data = self.process_message(message)
                
                if cleaned_data:
                    # Envoi vers topic cleaned
                    key = message.key
                    success = self.send_to_cleaned_topic(key, cleaned_data)
                    
                    if success:
                        self.processed_count += 1
                        print(f"Messages traites: {self.processed_count}")
                        print(f"Temp: {cleaned_data['current']['temperature_2m']} C "
                              f"| Humidite: {cleaned_data['current']['relative_humidity_2m']} %")
                        print("-" * 60)
                
                time.sleep(0.1)  # Petit délai pour éviter surcharge
                
        except KeyboardInterrupt:
            print("\nConsumer arrete par l'utilisateur")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Ferme proprement les connexions"""
        print(f"\nNettoyage... Total traite: {self.processed_count}")
        self.producer.flush()
        self.producer.close()
        self.consumer.close()


def main():
    processor = WeatherStreamProcessor()
    processor.run()


if __name__ == "__main__":
    main()