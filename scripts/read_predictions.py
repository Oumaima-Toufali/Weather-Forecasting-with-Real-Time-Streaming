# Path: scripts/read_predictions.py
"""
Utility script to read predictions from the Kafka topic `data.predictions.weather`.

Usage (from project root, venv activated):

    python scripts/read_predictions.py
"""

from __future__ import annotations

import json

from kafka import KafkaConsumer


def main() -> None:
    topic = "data.predictions.weather"

    consumer = KafkaConsumer(
        topic,
        bootstrap_servers=["localhost:9092"],
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )

    print(f"--- Listening on {topic} ---")
    try:
        for msg in consumer:
            print(msg.value)
    except KeyboardInterrupt:
        print("\nArrêt du consumer.")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()


