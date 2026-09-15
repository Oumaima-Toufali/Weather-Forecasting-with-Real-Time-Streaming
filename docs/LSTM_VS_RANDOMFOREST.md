# LSTM vs RandomForest - Guide de choix

Ce document explique quand utiliser les modèles LSTM vs RandomForest/XGBoost pour la prédiction météorologique dans le projet Weather Forecaster.

## Vue d'ensemble

Le projet Weather Forecaster propose deux approches complémentaires pour la prédiction de température :

1. **Modèles tabulaires** (RandomForest/XGBoost) : `historical_data_loader.py`
2. **Modèles LSTM** (Deep Learning) : `train_lstm.py`

## Comparaison des approches

### RandomForest / XGBoost

**Avantages :**
- ✅ **Rapidité d'entraînement** : Quelques secondes à quelques minutes
- ✅ **Interprétabilité** : Features importantes facilement identifiables
- ✅ **Robustesse** : Moins sensible aux hyperparamètres
- ✅ **Faible consommation mémoire** : Modèles légers (~1-10 MB)
- ✅ **Pas de GPU requis** : Fonctionne sur CPU
- ✅ **Déploiement simple** : Compatible avec scikit-learn ecosystem

**Inconvénients :**
- ❌ **Features manuelles** : Nécessite de créer manuellement les lags, rolling means, etc.
- ❌ **Patterns complexes** : Moins efficace pour capturer des dépendances temporelles très longues
- ❌ **Non-séquentiel** : Traite chaque échantillon indépendamment

**Quand l'utiliser :**
- Développement rapide et prototypage
- Besoin d'interprétabilité (comprendre quelles features sont importantes)
- Ressources limitées (CPU uniquement, mémoire limitée)
- Déploiement sur edge devices
- Petits datasets (< 10k échantillons)

### LSTM (Long Short-Term Memory)

**Avantages :**
- ✅ **Apprentissage automatique** : Apprend les patterns temporels directement depuis les séquences
- ✅ **Dépendances longues** : Peut capturer des patterns sur plusieurs jours/semaines
- ✅ **Séquentiel** : Prend en compte l'ordre temporel naturel
- ✅ **Multi-horizon** : Peut prédire plusieurs horizons simultanément
- ✅ **Performance potentielle** : Peut surpasser les modèles tabulaires sur datasets volumineux

**Inconvénients :**
- ❌ **Entraînement long** : Plusieurs minutes à heures selon la taille des données
- ❌ **GPU recommandé** : Entraînement beaucoup plus rapide sur GPU
- ❌ **Hyperparamètres sensibles** : Nécessite tuning (timesteps, units, dropout, etc.)
- ❌ **Modèles lourds** : Fichiers plus volumineux (~10-100 MB)
- ❌ **Black box** : Moins interprétable
- ❌ **Dépendances** : Nécessite TensorFlow/Keras

**Quand l'utiliser :**
- Datasets volumineux (> 10k échantillons)
- Patterns temporels complexes à capturer
- Besoin de prédictions multi-horizon simultanées
- GPU disponible pour l'entraînement
- Performance maximale recherchée

## Recommandations par cas d'usage

### Cas d'usage 1 : Prototypage rapide
**→ Utiliser RandomForest**
- Entraînement en quelques secondes
- Résultats rapidement exploitables
- Facile à déboguer et interpréter

### Cas d'usage 2 : Production avec contraintes
**→ Utiliser RandomForest/XGBoost**
- Déploiement simple (pas de TensorFlow)
- Faible latence de prédiction
- Compatible avec ONNX pour optimisation

### Cas d'usage 3 : Datasets volumineux (> 1 mois)
**→ Utiliser LSTM**
- Meilleure capacité à apprendre des patterns complexes
- Performance potentiellement supérieure

### Cas d'usage 4 : Prédictions multi-horizon simultanées
**→ Utiliser LSTM avec multi-horizon**
- Un seul modèle pour prédire 1h, 3h, 6h
- Cohérence entre les horizons

### Cas d'usage 5 : Edge deployment / IoT
**→ Utiliser RandomForest**
- Modèles légers
- Pas de dépendance TensorFlow
- Prédiction rapide sur CPU

## Performance attendue

### Sur dataset de 3 mois (2184 échantillons)

**RandomForest :**
- 1h : R² ≈ 0.92, MAE ≈ 0.66°C
- 3h : R² ≈ 0.69, MAE ≈ 1.30°C
- 6h : R² ≈ 0.51, MAE ≈ 1.83°C

**LSTM (avec tuning) :**
- 1h : R² ≈ 0.93-0.95, MAE ≈ 0.50-0.60°C
- 3h : R² ≈ 0.75-0.85, MAE ≈ 1.00-1.20°C
- 6h : R² ≈ 0.60-0.70, MAE ≈ 1.50-1.70°C

*Note : Les performances LSTM dépendent fortement du tuning des hyperparamètres.*

## Workflow recommandé

1. **Commencer par RandomForest** pour avoir une baseline rapide
2. **Évaluer les performances** sur le dataset de test
3. **Si insuffisant**, essayer LSTM avec tuning
4. **Comparer** les deux approches sur métriques métier
5. **Choisir** selon contraintes de déploiement

## Commandes d'entraînement

### RandomForest (baseline rapide)
```bash
python -m backend.training.historical_data_loader \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    --model-type random_forest
```

### LSTM (performance maximale)
```bash
python -m backend.training.train_lstm \
    --start-date 2024-01-01 \
    --end-date 2024-03-31 \
    --timesteps 24 \
    --epochs 50 \
    --batch-size 32
```

## Conclusion

Les deux approches sont complémentaires :
- **RandomForest** : Choix par défaut pour rapidité et simplicité
- **LSTM** : Choix pour performance maximale sur datasets volumineux

Le projet supporte les deux, permettant de choisir selon les contraintes du déploiement.

