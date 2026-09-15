# Path: backend/training/recover_dataset.py
"""
Script de récupération complète du dataset depuis Kafka.

Ce script lit TOUTES les données du topic data.features.hourly depuis le début,
même si elles ont déjà été consommées, en utilisant un nouveau consumer group.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List

import pandas as pd
from confluent_kafka import Consumer

from backend.config.kafka_config import KAFKA_BOOTSTRAP_SERVERS, TOPICS
from backend.utils.schema_validator import load_avro_schema, validate_avro
from backend.training.build_dataset import (
    _flatten_feature_record,
    build_dataframe,
    add_prediction_target,
    save_dataset,
    merge_datasets,
    load_existing_dataset,
)

FEATURES_SCHEMA = load_avro_schema("backend/schemas/features_weather.avsc")


def recover_all_messages_from_kafka(timeout_seconds: int = 30) -> List[dict]:
    """
    Récupère TOUTES les messages depuis le début du topic.
    Utilise un consumer group unique basé sur le timestamp pour éviter les conflits.
    """
    # Créer un group.id unique pour cette récupération
    unique_group_id = f"recovery-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    consumer = Consumer(
        {
            "bootstrap.servers": ",".join(KAFKA_BOOTSTRAP_SERVERS),
            "group.id": unique_group_id,
            "auto.offset.reset": "earliest",  # Lire depuis le début
            "enable.auto.commit": False,  # Ne pas committer pour pouvoir relire
        }
    )
    
    consumer.subscribe([TOPICS["features"]])
    
    messages: List[dict] = []
    no_message_count = 0
    max_no_message = timeout_seconds  # Arrêter après X secondes sans message
    
    print(f"[INFO] Debut de la recuperation avec group.id: {unique_group_id}")
    print(f"[INFO] Lecture depuis le debut du topic: {TOPICS['features']}")
    print(f"[INFO] Timeout: {timeout_seconds} secondes sans message")
    print("-" * 60)
    
    try:
        while no_message_count < max_no_message:
            msg = consumer.poll(1.0)  # Poll toutes les secondes
            
            if msg is None:
                no_message_count += 1
                if no_message_count % 5 == 0:
                    print(f"[INFO] En attente... ({no_message_count}s, {len(messages)} messages recuperes)")
                continue
            
            if msg.error():
                if msg.error().code() == -191:  # PARTITION_EOF
                    print("[INFO] Fin de partition atteinte")
                    no_message_count += 1
                    continue
                else:
                    print(f"[ERREUR] Erreur Kafka: {msg.error()}")
                    no_message_count += 1
                    continue
            
            try:
                payload = json.loads(msg.value().decode("utf-8"))
            except Exception as e:
                print(f"[ATTENTION] Message invalide (JSON): {e}")
                continue
            
            if not validate_avro(payload, FEATURES_SCHEMA):
                print(f"[ATTENTION] Message invalide (Avro schema)")
                continue
            
            messages.append(payload)
            no_message_count = 0  # Reset counter when we get a message
            
            if len(messages) % 10 == 0:
                print(f"[INFO] {len(messages)} messages recuperes...")
                
    except KeyboardInterrupt:
        print("\n[INFO] Interruption par l'utilisateur")
    finally:
        consumer.close()
    
    return messages


def main():
    parser = argparse.ArgumentParser(
        description="Recupere TOUTES les donnees depuis Kafka et reconstruit le dataset complet."
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=3,
        help="Prediction horizon in hours.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Secondes d'attente sans message avant d'arreter.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/datasets",
        help="Output directory for parquet/csv files.",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Fusionner avec le dataset existant (si present).",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("RECUPERATION COMPLETE DU DATASET DEPUIS KAFKA")
    print("=" * 60)
    
    # Récupérer toutes les messages
    records = recover_all_messages_from_kafka(timeout_seconds=args.timeout)
    
    if not records:
        print("[ERREUR] Aucun message recupere depuis Kafka")
        print("[INFO] Verifiez que:")
        print("  1. Kafka est demarre")
        print("  2. Le topic data.features.hourly existe")
        print("  3. Il y a des messages dans le topic")
        return
    
    print(f"\n[OK] {len(records)} messages recuperes depuis Kafka")
    
    # Construire le dataframe
    df = build_dataframe(records)
    if df.empty:
        print("[ERREUR] Aucune donnee valide apres traitement")
        return
    
    print(f"[OK] DataFrame construit: {len(df)} lignes")
    
    # Fusionner avec l'existant si demandé
    if args.merge:
        parquet_path = Path(args.output_dir) / "weather_features.parquet"
        existing_df = load_existing_dataset(parquet_path)
        if existing_df is not None and not existing_df.empty:
            print(f"[INFO] Fusion avec dataset existant ({len(existing_df)} lignes)...")
            df = merge_datasets(existing_df, df)
            print(f"[OK] Apres fusion: {len(df)} lignes")
    
    # Ajouter les targets
    df = add_prediction_target(df, args.horizon)
    
    # Sauvegarder
    parquet_path, csv_path = save_dataset(df, args.output_dir)
    
    print("\n" + "=" * 60)
    print("[OK] DATASET RECUPERE ET SAUVEGARDE")
    print("=" * 60)
    print(f"   Total lignes: {len(df)}")
    print(f"   Parquet: {parquet_path}")
    print(f"   CSV: {csv_path}")
    
    if "timestamp" in df.columns:
        print(f"   Periode: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print("=" * 60)


if __name__ == "__main__":
    main()

