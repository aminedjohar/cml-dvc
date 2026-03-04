# churn-cml-dvc — MLOps Pipeline de bout en bout

> Atelier 4 — MLOps | DVC + DagsHub + GitHub Actions + CML  
> Pr. Soufiane HAMIDA | Janvier 2026

---

## 🎯 Objectif

Ce projet met en place un pipeline MLOps complet pour un modèle de prédiction de **churn client** (désabonnement), incluant :

- Le versioning des données et modèles avec **DVC**
- Le stockage des artefacts sur **DagsHub** (alternative gratuite à AWS S3)
- Un pipeline **CI/CD GitHub Actions** qui entraîne, évalue et pousse automatiquement les artefacts
- La publication automatique d'un **rapport de métriques** (via CML) à chaque push

---

## 🏗️ Architecture du projet

```
churn-cml-dvc/
├── .dvc/
│   └── config              # Remote DagsHub (sans secrets)
├── .github/
│   └── workflows/
│       └── ci.yml          # Pipeline CI/CD GitHub Actions
├── data/                   # Données gérées par DVC (non versionnées Git)
├── models/                 # Modèle + métriques gérés par DVC
├── data.dvc                # Métadonnées DVC pour data/
├── models.dvc              # Métadonnées DVC pour models/
├── script.py               # Script d'entraînement ML
├── requirements.txt        # Dépendances Python
└── README.md
```

---

## 🔧 Stack technique

| Outil | Rôle | Remplace |
|-------|------|----------|
| **DVC 3.63.0** | Versioning des données et modèles | — |
| **DagsHub** | Stockage des artefacts (remote DVC) | AWS S3 |
| **GitHub Actions** | Pipeline CI/CD automatisé | — |
| **CML** | Publication automatique du rapport | — |
| **scikit-learn** | Modèle RandomForest | — |
| **imbalanced-learn** | Rééquilibrage SMOTE | — |

---

## 🤖 Modèle ML

- **Problème** : Classification binaire — prédire si un client va churner
- **Algorithme** : Random Forest (100 estimateurs)
- **Rééquilibrage** : SMOTE (Synthetic Minority Oversampling Technique)
- **Dataset** : 1000 clients synthétiques avec 6 features

**Features utilisées :**
- `age`, `tenure`, `monthly_charges`, `num_products`, `has_internet`, `num_complaints`

**Résultats obtenus :**
- Accuracy : **0.66**
- F1-score : **0.66**

---

## ⚙️ Pipeline CI/CD

À chaque `git push`, le pipeline GitHub Actions exécute automatiquement :

1. ✅ **Checkout** du repo
2. ✅ **Installation** de Python 3.11 + dépendances
3. ✅ **Configuration** du remote DVC (DagsHub)
4. ✅ **DVC pull** — récupère la dernière version des données/modèles depuis DagsHub
5. ✅ **Entraînement** du modèle (`python script.py`)
6. ✅ **DVC add + push** — pousse les nouveaux artefacts vers DagsHub
7. ✅ **Rapport CML** — publie métriques + matrice de confusion en commentaire GitHub

---

## 🚀 Reproduire en local

### 1. Cloner le repo
```bash
git clone https://github.com/aminedjohar/cml-dvc.git
cd cml-dvc
```

### 2. Créer l'environnement Python
```bash
python3.11 -m venv venv
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
pip install "dvc[http]==3.63.0"
pip install "pathspec==0.11.2"
```

### 3. Configurer DVC (DagsHub)
```bash
dvc remote modify myremote --local auth basic
dvc remote modify myremote --local user <DAGSHUB_USERNAME>
dvc remote modify myremote --local password <DAGSHUB_TOKEN>
```

### 4. Récupérer les données et entraîner
```bash
dvc pull
python script.py
```

---

## 🔐 Secrets GitHub requis

| Secret | Description |
|--------|-------------|
| `DAGSHUB_USERNAME` | Nom d'utilisateur DagsHub |
| `DAGSHUB_TOKEN` | Token d'accès DagsHub |

---

## 📊 Rapport automatique CML

À chaque push, CML publie automatiquement un commentaire sur le commit GitHub contenant :
- Les métriques du modèle (Accuracy, F1-score, classification report)
- La matrice de confusion

---

## 🔄 Flux MLOps complet

```
Code change (git push)
        ↓
GitHub Actions déclenché
        ↓
Install deps + Configure DVC
        ↓
dvc pull (récupère données depuis DagsHub)
        ↓
python script.py (entraînement)
        ↓
dvc push (artefacts → DagsHub)
        ↓
CML report (métriques → commentaire GitHub)
```

Ce cycle garantit : versioning du code (Git) + versioning des données/modèles (DVC + DagsHub) + reproductibilité (CI) + traçabilité (CML).