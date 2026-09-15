# Path: backend/training/check_kafka_topic.py
"""
Vérifie combien de messages sont disponibles dans le topic Kafka.
"""

import json
from confluent_kafka import Consumer, TopicPartition
from backend.config.kafka_config import KAFKA_BOOTSTRAP_SERVERS, TOPICS

def check_topic_messages():
    """Vérifie le nombre de messages dans le topic."""
    topic = TOPICS["features"]
    
    # Créer un consumer temporaire
    consumer = Consumer(
        {
            "bootstrap.servers": ",".join(KAFKA_BOOTSTRAP_SERVERS),
            "group.id": f"check-{topic}",
            "auto.offset.reset": "earliest",
        }
    )
    
    try:
        # Obtenir les métadonnées du topic
        metadata = consumer.list_topics(topic, timeout=10)
        
        if topic not in metadata.topics:
            print(f"[ERREUR] Topic '{topic}' n'existe pas")
            return
        
        topic_metadata = metadata.topics[topic]
        partitions = topic_metadata.partitions
        
        print("=" * 60)
        print(f"VERIFICATION DU TOPIC: {topic}")
        print("=" * 60)
        print(f"Partitions: {len(partitions)}")
        print("-" * 60)
        
        total_messages = 0
        
        for partition_id in partitions:
            # Obtenir les offsets low et high
            tp = TopicPartition(topic, partition_id)
            low, high = consumer.get_watermark_offsets(tp, timeout=10)
            messages_in_partition = high - low
            
            print(f"Partition {partition_id}:")
            print(f"  Low offset:  {low}")
            print(f"  High offset: {high}")
            print(f"  Messages:    {messages_in_partition}")
            
            total_messages += messages_in_partition
        
        print("-" * 60)
        print(f"TOTAL MESSAGES: {total_messages}")
        print("=" * 60)
        
        if total_messages == 0:
            print("\n[ATTENTION] Aucun message dans le topic!")
            print("   Les donnees peuvent avoir ete supprimees ou le topic est vide.")
        else:
            print(f"\n[OK] {total_messages} messages disponibles pour recuperation")
            print("   Utilisez: python -m backend.training.recover_dataset")
            
    except Exception as e:
        print(f"[ERREUR] Impossible de verifier le topic: {e}")
    finally:
        consumer.close()

if __name__ == "__main__":
    check_topic_messages()

