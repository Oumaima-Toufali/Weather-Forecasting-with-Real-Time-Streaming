# Path: scripts/setup_topics.py
"""
Création des topics Kafka utilisés par la chaîne de traitement.

Usage:
    python scripts/setup_topics.py --bootstrap localhost:9092 --partitions 3 --replication 1
"""

from argparse import ArgumentParser
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError

DEFAULT_TOPICS = [
    "data.raw.stream",
    "data.cleaned.stream",
    "data.features.hourly",
    "data.predictions.weather",
]


def ensure_topics(bootstrap: str, partitions: int, replication: int):
    admin = KafkaAdminClient(bootstrap_servers=bootstrap, client_id="weather-topic-setup")
    topics = [
        NewTopic(name=t, num_partitions=partitions, replication_factor=replication)
        for t in DEFAULT_TOPICS
    ]
    try:
        admin.create_topics(new_topics=topics, validate_only=False)
        print(f"✅ Topics créés : {', '.join(DEFAULT_TOPICS)}")
    except TopicAlreadyExistsError:
        print("ℹ️  Topics déjà présents, aucune action.")
    finally:
        admin.close()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--bootstrap", default="localhost:9092", help="Adresse bootstrap Kafka")
    parser.add_argument("--partitions", type=int, default=3, help="Nombre de partitions")
    parser.add_argument("--replication", type=int, default=1, help="Facteur de réplication")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ensure_topics(args.bootstrap, args.partitions, args.replication)

