# Logique de Détection des Doublons

## 📋 Vue d'ensemble

Le système détecte et supprime les **vrais doublons complets** en comparant **TOUTES les features météorologiques pertinentes** pour la prédiction, et non pas uniquement la température ou le timestamp.

## 🎯 Principe

**Un doublon n'est détecté que si TOUTES les features suivantes sont identiques :**
- `temperature` (température)
- `humidity` (humidité)
- `wind_speed` (vitesse du vent)
- `pressure` (pression)
- `cloud_cover` (couverture nuageuse)
- `precipitation` (précipitation)

## 🔍 Implémentation

### 1. Dans le Feature Consumer (`backend/processing/feature_consumer.py`)

**Méthode `is_duplicate()`** :
- Compare **toutes les 6 features météorologiques** entre le message actuel et le précédent
- Utilise des **tolérances** pour éviter les faux doublons dus aux erreurs d'arrondi :
  - Température : ±0.01°C
  - Humidité : ±0.1%
  - Vent : ±0.01 km/h
  - Pression : ±0.1 hPa
  - Nuages : ±0.1%
  - Précipitation : ±0.001 mm

**Logique** :
```python
Si TOUTES les features sont identiques (dans la tolérance) → DOUBLON
Si AU MOINS UNE feature diffère → PAS UN DOUBLON (données valides)
```

### 2. Dans le Dataset Builder (`backend/training/build_dataset.py`)

**Fonction `merge_datasets()`** :
- Utilise `pandas.drop_duplicates()` avec **toutes les features pertinentes** comme clé
- Arrondit les valeurs numériques avant comparaison pour éviter les faux doublons
- Conserve la dernière occurrence en cas de doublon

**Colonnes utilisées** :
```python
["temperature", "humidity", "wind_speed", "pressure", "cloud_cover", "precipitation"]
```

## ✅ Exemples

### Cas 1 : Vrai doublon (supprimé)
```
Message 1: temp=20.0°C, humidity=60%, wind=10 km/h, pressure=1013 hPa, cloud=50%, precip=0 mm
Message 2: temp=20.0°C, humidity=60%, wind=10 km/h, pressure=1013 hPa, cloud=50%, precip=0 mm
→ DOUBLON détecté et supprimé
```

### Cas 2 : Même température mais autres features différentes (conservé)
```
Message 1: temp=20.0°C, humidity=60%, wind=10 km/h, pressure=1013 hPa, cloud=50%, precip=0 mm
Message 2: temp=20.0°C, humidity=65%, wind=12 km/h, pressure=1013 hPa, cloud=50%, precip=0 mm
→ PAS UN DOUBLON (humidity et wind diffèrent) → CONSERVÉ
```

### Cas 3 : Timestamp différent mais features identiques (doublon)
```
Message 1 (10:00): temp=20.0°C, humidity=60%, wind=10 km/h, pressure=1013 hPa, cloud=50%, precip=0 mm
Message 2 (10:05): temp=20.0°C, humidity=60%, wind=10 km/h, pressure=1013 hPa, cloud=50%, precip=0 mm
→ DOUBLON détecté (même si timestamp différent)
```

## 🛡️ Protection contre les faux doublons

1. **Tolérances numériques** : Évite de considérer comme doublons des valeurs légèrement différentes dues aux arrondis
2. **Comparaison complète** : Une seule feature différente suffit pour considérer que ce n'est pas un doublon
3. **Gestion des valeurs None** : Si une feature est None et l'autre non, ce n'est pas un doublon

## 📊 Impact sur le pipeline

- **Feature Consumer** : Évite de publier des messages dupliqués vers `data.features.hourly`
- **Dataset Builder** : Évite d'ajouter des lignes dupliquées au dataset final
- **Qualité des données** : Seuls les vrais doublons complets sont supprimés, préservant la diversité des données

## 🔧 Configuration

Les features utilisées pour la détection sont définies dans :
- `backend/processing/feature_consumer.py` : méthode `is_duplicate()`
- `backend/training/build_dataset.py` : fonction `get_duplicate_columns()`

Pour modifier les features comparées, éditez ces deux fonctions.

