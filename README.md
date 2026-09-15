# 🌡️ Weather Forecaster

Système de prédiction météorologique en temps réel utilisant l'apprentissage automatique et le streaming de données avec Apache Kafka.

## 📋 Description

Weather Forecaster est une plateforme complète de prédiction météorologique qui ingère des données en temps réel via l'API Open-Meteo, les traite dans un pipeline Kafka, et génère des prédictions de température à court terme (1h, 3h, 6h) à l'aide de modèles de machine learning.

## ✨ Fonctionnalités

- 🔄 **Ingestion en temps réel** : Collecte automatique des données météorologiques via Open-Meteo API
- 📊 **Pipeline de streaming** : Traitement des données avec Apache Kafka (KRaft mode)
- 🤖 **Modèles ML multiples** : Support pour RandomForest, XGBoost et LSTM
- 📈 **Feature Engineering** : Génération automatique de features temporelles (lags, rolling means, deltas)
- 🎯 **Prédictions probabilistes** : Point forecasts et quantiles (P10/P50/P90)
- 📱 **Dashboard interactif** : Visualisation en temps réel avec Streamlit
- ✅ **Validation de schémas** : Utilisation d'Avro pour la validation des données

## 🏗️ Architecture

```
API Open-Meteo
    ↓
Producer (Ingestion)
    ↓
Topic: data.raw.stream
    ↓
Normalizer (Nettoyage)
    ↓
Topic: data.cleaned.stream
    ↓
Feature Engineering
    ↓
Topic: data.features.hourly
    ↓
ML Serving (RandomForest/LSTM)
    ↓
Topics: data.predictions.weather & data.predictions.weather.quantiles
    ↓
Dashboard Streamlit
```

## 🛠️ Technologies

- **Backend** : Python 3.10+
- **Streaming** : Apache Kafka (KRaft mode)
- **ML** : scikit-learn, XGBoost, TensorFlow (LSTM)
- **Dashboard** : Streamlit, Plotly
- **Data** : Pandas, PyArrow (Parquet)
- **Serialization** : Avro (FastAvro)

## 📦 Installation

### Prérequis

- Python 3.10 ou supérieur
- Java 17+ (pour Kafka)
- Apache Kafka (mode KRaft)

### Configuration

1. **Cloner le repository**
```bash
git clone https://github.com/votre-username/weather-forecaster.git
cd weather-forecaster
```

2. **Créer un environnement virtuel**
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

3. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

4. **Configurer Kafka**

Téléchargez et installez Apache Kafka, puis configurez le mode KRaft. Voir la section [Configuration Kafka](#configuration-kafka) pour plus de détails.

## 🚀 Démarrage rapide

### 1. Démarrer Kafka

```bash
# Windows (PowerShell)
.\scripts\start_kafka.ps1

# Linux/Mac
./scripts/start_kafka.sh
```

### 2. Créer les topics Kafka

```bash
# Windows
.\scripts\create_topics.ps1

# Linux/Mac
./scripts/create_topics.sh
```

### 3. Lancer l'ingestion des données

```bash
python -m backend.ingestion.producer
```

### 4. Lancer le traitement des données

```bash
# Terminal 1: Normalisation
python -m backend.processing.consumer

# Terminal 2: Feature Engineering
python -m backend.processing.feature_consumer
```

### 5. Entraîner un modèle (optionnel)

```bash
# Entraînement sur données historiques
python -m backend.training.historical_data_loader \
    --start-date 2024-01-01 \
    --end-date 2024-03-31
```

### 6. Lancer le serving ML

```bash
python -m backend.serving.predict_consumer
```

### 7. Lancer le dashboard

```bash
streamlit run dashboard/streamlit_app.py
```

## 📚 Utilisation

### Entraînement de modèles

#### RandomForest / XGBoost

```bash
python -m backend.training.historical_data_loader \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    --model-type random_forest \
    --n-estimators 100
```

#### LSTM (Deep Learning)

```bash
python -m backend.training.train_lstm \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    --timesteps 24 \
    --epochs 50
```

### Génération de dataset

```bash
python -m backend.training.build_dataset \
    --horizon 3 \
    --max-messages 10000
```

### Export ONNX (optionnel)

```bash
python scripts/export_rf_to_onnx.py
```

## 📁 Structure du projet

```
weather-forecaster/
├── backend/
│   ├── config/          # Configuration (Kafka, API, etc.)
│   ├── ingestion/       # Producer Kafka
│   ├── processing/      # Consumers (normalization, features)
│   ├── training/        # Entraînement des modèles
│   ├── serving/         # ML serving
│   ├── models/          # Modèles ML
│   └── schemas/         # Schémas Avro
├── dashboard/           # Application Streamlit
├── data/               # Données (raw, processed, datasets)
├── models/             # Modèles entraînés
├── scripts/            # Scripts utilitaires
└── tests/              # Tests unitaires
```

## 🔧 Configuration Kafka

Pour configurer Kafka en mode KRaft, modifiez `config/kraft/server.properties` :

```properties
process.roles=broker,controller
node.id=1
controller.quorum.voters=1@localhost:9093
listeners=PLAINTEXT://:9092,CONTROLLER://:9093
log.dirs=/path/to/kafka/kraft-combined-logs
```

## 📊 Modèles disponibles

- **RandomForest** : Modèle par défaut, rapide et interprétable
- **XGBoost** : Performance améliorée pour datasets volumineux
- **LSTM** : Deep learning pour patterns temporels complexes

Voir `docs/LSTM_VS_RANDOMFOREST.md` pour un guide de choix.

## 🧪 Tests

```bash
pytest tests/
```

## 📝 Documentation

- `docs/LSTM_VS_RANDOMFOREST.md` : Guide de choix entre modèles
- `docs/DUPLICATE_DETECTION_LOGIC.md` : Logique de détection des doublons

## 🤝 Contribution

Les contributions sont les bienvenues ! N'hésitez pas à ouvrir une issue ou soumettre une pull request.

## 📄 Licence

Ce projet est sous licence MIT.

## 👤 Auteur

Eng. Toufali Oumaima


