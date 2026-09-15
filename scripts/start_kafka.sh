#!/bin/bash
# Script d'installation Kafka en mode KRaft (sans ZooKeeper)

set -e

echo "📦 Installation de Kafka en mode KRaft..."

# Variables
KAFKA_VERSION="3.6.1"
SCALA_VERSION="2.13"
KAFKA_DIR="kafka_${SCALA_VERSION}-${KAFKA_VERSION}"
KAFKA_TAR="${KAFKA_DIR}.tgz"
DOWNLOAD_URL="https://downloads.apache.org/kafka/${KAFKA_VERSION}/${KAFKA_TAR}"

# Téléchargement
if [ ! -d "$KAFKA_DIR" ]; then
    echo "⬇️  Téléchargement de Kafka ${KAFKA_VERSION}..."
    wget -q $DOWNLOAD_URL
    
    echo "📂 Extraction..."
    tar -xzf $KAFKA_TAR
    rm $KAFKA_TAR
    
    echo "✅ Kafka installé dans ${KAFKA_DIR}"
else
    echo "✅ Kafka déjà installé"
fi

# Configuration KRaft
cd $KAFKA_DIR

echo "🔧 Configuration KRaft..."

# Génération UUID cluster
CLUSTER_ID=$(bin/kafka-storage.sh random-uuid)
echo "🆔 Cluster ID: $CLUSTER_ID"

# Format du stockage
bin/kafka-storage.sh format -t $CLUSTER_ID -c config/kraft/server.properties

echo "✅ Kafka configuré en mode KRaft!"
echo ""
echo "🚀 Pour démarrer Kafka, exécute :"
echo "   cd $KAFKA_DIR"
echo "   bin/kafka-server-start.sh config/kraft/server.properties"
echo ""
echo "📋 Pour créer les topics, exécute ensuite :"
echo "   python scripts/setup_topics.py"