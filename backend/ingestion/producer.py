# Path: backend/ingestion/producer.py
import os
import time
from datetime import datetime, timezone
from typing import Optional
from collections import deque

from kafka import KafkaProducer
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from backend.config.kafka_config import PRODUCER_CONFIG, TOPICS
from backend.ingestion.openmeteo_client import OpenMeteoClient
from backend.utils.schema_validator import load_avro_schema, validate_avro


class RateLimiter:
    """
    Simple sliding-window limiter to cap API calls per minute.
    Prevents accidental quota overruns on upstream APIs.
    """

    def __init__(self, max_requests_per_minute: int):
        self.max_requests = max_requests_per_minute
        self.events = deque()

    def wait_if_needed(self) -> None:
        now = time.time()
        window_start = now - 60
        # Evict old events
        while self.events and self.events[0] < window_start:
            self.events.popleft()
        if len(self.events) >= self.max_requests:
            sleep_for = self.events[0] + 60 - now
            sleep_for = max(sleep_for, 0)
            print(f"⏳ Rate limit reached ({self.max_requests}/min). Sleeping {sleep_for:.1f}s.")
            time.sleep(sleep_for)
        self.events.append(time.time())


class WeatherProducer:
    def __init__(self):
        self.producer = KafkaProducer(**PRODUCER_CONFIG)
        # Localisation par défaut plus volatile (Medellín), override possible via LATITUDE/LONGITUDE
        # Medellín : 6.2442° N, 75.5812° W
        lat = float(os.getenv("LATITUDE", 6.2442))
        lon = float(os.getenv("LONGITUDE", -75.5812))
        self.client = OpenMeteoClient(latitude=lat, longitude=lon)
        self.poll_interval = int(os.getenv("POLL_INTERVAL_SECONDS", 5))  # 5s par défaut pour limiter la charge
        self.raw_schema = load_avro_schema("backend/schemas/raw_weather.avsc")
        max_req_per_min = int(os.getenv("MAX_REQUESTS_PER_MINUTE", 12))  # 12/min ~ 5s cadence
        self.rate_limiter = RateLimiter(max_req_per_min)
    
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=30),
        retry=retry_if_exception_type(Exception)
    )
    def fetch_payload(self) -> Optional[dict]:
        """Récupère les données météo"""
        self.rate_limiter.wait_if_needed()
        return self.client.get_current_weather()
    
    def run(self):
        print(f"🌤️  Weather Producer démarré (poll every {self.poll_interval}s)")
        topic = TOPICS['raw']
        
        try:
            while True:
                payload = self.fetch_payload()
                
                if payload:
                    timestamp = datetime.now(timezone.utc).isoformat()
                    key = f"weather:{payload['metadata']['location']}:{timestamp}"

                    # Validation Avro avant envoi
                    if not validate_avro(payload, self.raw_schema):
                        print("⚠️  Payload non conforme au schéma Avro, message ignoré")
                        time.sleep(self.poll_interval)
                        continue
                    
                    future = self.producer.send(topic, key=key, value=payload)
                    record_md = future.get(timeout=10)
                    
                    print(f"✅ [{timestamp[:19]}] → partition={record_md.partition} "
                          f"offset={record_md.offset} | "
                          f"Temp={payload['current']['temperature_2m']}°C")
                else:
                    print("⚠️  Aucune donnée récupérée")
                
                time.sleep(self.poll_interval)
                
        except KeyboardInterrupt:
            print("\n⛔ Producer arrêté")
        finally:
            self.producer.close()


if __name__ == "__main__":
    producer = WeatherProducer()
    producer.run()