import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from imblearn.over_sampling import SMOTE
import joblib

# ── Reproductibilité ──────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Dossiers de sortie ────────────────────────────────────────
os.makedirs("data",   exist_ok=True)
os.makedirs("models", exist_ok=True)

# ── 1. Génération d'un dataset churn synthétique ─────────────
print(">>> Génération du dataset...")
n = 1000
df = pd.DataFrame({
    "age":             np.random.randint(18, 70, n),
    "tenure":          np.random.randint(0,  72, n),
    "monthly_charges": np.round(np.random.uniform(20, 120, n), 2),
    "num_products":    np.random.randint(1,  5,  n),
    "has_internet":    np.random.randint(0,  2,  n),
    "num_complaints":  np.random.randint(0,  10, n),
})
# Churn : probabilité plus haute si beaucoup de plaintes + faible ancienneté
prob = (df["num_complaints"] / 10) * 0.6 + (1 - df["tenure"] / 72) * 0.4
df["churn"] = (np.random.rand(n) < prob).astype(int)

df.to_csv("data/churn_data.csv", index=False)
print(f"   Dataset: {df.shape}  |  churn rate: {df['churn'].mean():.1%}")

# ── 2. Préparation ────────────────────────────────────────────
X = df.drop("churn", axis=1)
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Rééquilibrage avec SMOTE
smote = SMOTE(random_state=SEED)
X_res, y_res = smote.fit_resample(X_train, y_train)
print(f"   Après SMOTE : {dict(zip(*np.unique(y_res, return_counts=True)))}")

# ── 3. Entraînement ───────────────────────────────────────────
print(">>> Entraînement du modèle...")
clf = RandomForestClassifier(n_estimators=100, random_state=SEED)
clf.fit(X_res, y_res)

# ── 4. Évaluation ─────────────────────────────────────────────
y_pred = clf.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
f1     = f1_score(y_test, y_pred, average="weighted")

print(f"   Accuracy : {acc:.4f}")
print(f"   F1-score : {f1:.4f}")

# ── 5. Sauvegarde des métriques ───────────────────────────────
metrics_path = "models/metrics.txt"
with open(metrics_path, "w") as f:
    f.write(f"Accuracy  : {acc:.4f}\n")
    f.write(f"F1-score  : {f1:.4f}\n\n")
    f.write(classification_report(y_test, y_pred))
print(f"   Métriques sauvegardées → {metrics_path}")

# ── 6. Matrice de confusion ───────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"], ax=ax)
ax.set_xlabel("Prédit")
ax.set_ylabel("Réel")
ax.set_title("Matrice de confusion")
plt.tight_layout()
fig.savefig("models/conf_matrix.png", dpi=100)
plt.close()
print("   Matrice de confusion sauvegardée → models/conf_matrix.png")

# ── 7. Sauvegarde du modèle ───────────────────────────────────
model_path = "models/model.joblib"
joblib.dump(clf, model_path)
print(f"   Modèle sauvegardé → {model_path}")

print(">>> Terminé ✓")