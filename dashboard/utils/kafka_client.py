import json
from confluent_kafka import Consumer, TopicPartition
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
import streamlit as st

class KafkaMonitoringClient:
    def __init__(self, bootstrap_servers: str = 'localhost:9092'):
        self.bootstrap_servers = bootstrap_servers
        self.conf = {
            'bootstrap.servers': self.bootstrap_servers,
            'group.id': 'streamlit-monitoring-group',
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False
        }

    def get_consumer(self):
        return Consumer(self.conf)

    def fetch_messages(self, topic: str, max_messages: int = 100, timeout: float = 1.0) -> List[Dict[str, Any]]:
        """Fetch latest messages from a topic."""
        consumer = self.get_consumer()
        messages = []
        try:
            consumer.subscribe([topic])
            
            # Get the high watermark to only read recent messages if needed, 
            # but here we follow the prompt to get a window of data.
            # For simplicity, we'll poll until max_messages or timeout.
            
            count = 0
            start_time = datetime.now()
            while count < max_messages:
                msg = consumer.poll(0.1)
                if msg is None:
                    if (datetime.now() - start_time).total_seconds() > timeout:
                        break
                    continue
                if msg.error():
                    continue
                
                try:
                    payload = json.loads(msg.value().decode('utf-8'))
                    messages.append(payload)
                    count += 1
                except Exception:
                    continue
        finally:
            consumer.close()
        return messages

    def get_kafka_stats(self, topics: List[str]) -> Dict[str, Any]:
        """Get lag and throughput statistics for topics."""
        consumer = self.get_consumer()
        stats = {}
        try:
            for topic in topics:
                metadata = consumer.list_topics(topic, timeout=5)
                if topic not in metadata.topics:
                    stats[topic] = {"error": "Topic not found"}
                    continue
                
                topic_metadata = metadata.topics[topic]
                partitions_info = []
                total_lag = 0
                
                for partition_id in topic_metadata.partitions:
                    tp = TopicPartition(topic, partition_id)
                    low, high = consumer.get_watermark_offsets(tp, timeout=5)
                    
                    # To get actual lag, we'd need the current committed offset for a group
                    # For monitoring, we can show total messages in topic as a proxy or 
                    # use high - low as "available backlog"
                    messages_in_partition = high - low
                    partitions_info.append({
                        "partition": partition_id,
                        "low": low,
                        "high": high,
                        "available": messages_in_partition
                    })
                    total_lag += messages_in_partition
                
                stats[topic] = {
                    "partitions": partitions_info,
                    "total_messages": total_lag,
                    "partition_count": len(topic_metadata.partitions)
                }
        finally:
            consumer.close()
        return stats

@st.cache_resource
def get_monitoring_client():
    return KafkaMonitoringClient()

