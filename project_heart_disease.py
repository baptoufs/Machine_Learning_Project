# project_heart_disease_UCI.py
"""
Projet complet Machine Learning — UCI Heart Disease (Cleveland)
Dataset officiel :
https://archive.ics.uci.edu/ml/datasets/Heart+Disease

Fichier requis : heart_uci.csv
Il s’agit du fichier UCI 'processed.cleveland.data' renommé.

Pipeline :
1. Chargement + nettoyage UCI (remplacement des '?')
2. EDA (stats + graphiques)
3. Prétraitement (imputation + scaling)
4. Modélisation (LogReg, RandomForest, KNN)
5. Hyperparam tuning
6. Évaluations (F1, Precision, Recall, ROC, AUC)
7. Export résultats
"""

# ----------------------------
# IMPORTS
# ----------------------------
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

import joblib

# ----------------------------
# CONFIG
# ----------------------------
DATA_PATH = "heart_uci.csv" 
OUTPUT_DIR = "results_uci"
TEST_SIZE = 0.2
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# 1. CHARGEMENT DU DATASET UCI
# ----------------------------
column_names = [
    'age','sex','cp','trestbps','chol','fbs','restecg','thalach',
    'exang','oldpeak','slope','ca','thal','target'
]

df = pd.read_csv(
    DATA_PATH,
    header=None,
    names=column_names,
    na_values='?'        # les "?" sont remplacés par NaN
)

print("\n--- Aperçu du dataset UCI ---")
print(df.head())
print("\nTaille :", df.shape)


# ----------------------------
# 2. EDA
# ----------------------------
print("\n--- Valeurs manquantes ---")
print(df.isna().sum())

# Statistiques
df.describe().to_csv(os.path.join(OUTPUT_DIR, "summary_stats.csv"))

# Plot 1 : distribution cible
plt.figure(figsize=(6,4))
sns.countplot(x='target', data=df)
plt.title("Distribution de la cible (UCI Heart Disease)")
plt.savefig(os.path.join(OUTPUT_DIR, "target_distribution.png"))
plt.close()

# Heatmap corrélation (numériques)
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Corrélations — UCI Heart Disease")
plt.savefig(os.path.join(OUTPUT_DIR, "heatmap_corr.png"))
plt.close()


# ----------------------------
# 3. PRÉTRAITEMENT
# ----------------------------

# Conversion de la cible en binaire :
# UCI : 0 = absence, 1–4 = présence
df["target"] = (df["target"] > 0).astype(int)

X = df.drop(columns=["target"])
y = df["target"]

# Séparer train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=SEED
)

# Colonnes numériques (toutes ici)
num_cols = X.columns

# Imputation
imputer = SimpleImputer(strategy="median")
X_train_imp = imputer.fit_transform(X_train)
X_test_imp = imputer.transform(X_test)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imp)
X_test_scaled = scaler.transform(X_test_imp)


# ----------------------------
# 4. MODÈLES
# ----------------------------
models = {
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "RandomForest": RandomForestClassifier(random_state=SEED),
    "KNN": KNeighborsClassifier()
}

param_grids = {
    "LogisticRegression": {
        'C': [0.01, 0.1, 1, 10]
    },
    "RandomForest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10]
    },
    "KNN": {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform','distance']
    }
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

results = []


# ----------------------------
# 5. ENTRAÎNEMENT + TUNING + ÉVALUATION
# ----------------------------
for name, model in models.items():

    print(f"\n### Entraînement : {name} ###")

    grid = GridSearchCV(
        model,
        param_grids[name],
        scoring="f1",
        cv=cv,
        n_jobs=-1
    )
    grid.fit(X_train_scaled, y_train)

    best = grid.best_estimator_
    print("Meilleurs hyperparamètres :", grid.best_params_)

    y_pred = best.predict(X_test_scaled)

    # Probabilités
    if hasattr(best, "predict_proba"):
        y_proba = best.predict_proba(X_test_scaled)[:,1]
    else:
        y_proba = None

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True)

    # AUC
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    results.append({
        "model": name,
        "f1": report['1']["f1-score"],
        "precision": report['1']["precision"],
        "recall": report['1']["recall"],
        "auc": auc,
        "best_params": grid.best_params_
    })

    # Sauvegarde du modèle
    joblib.dump(best, os.path.join(OUTPUT_DIR, f"{name}.joblib"))

    # Rapport texte
    with open(os.path.join(OUTPUT_DIR, f"{name}_report.txt"), "w") as f:
        f.write(classification_report(y_test, y_pred))

    # Matrice de confusion
    plt.figure(figsize=(5,4))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix — {name}")
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_confusion.png"))
    plt.close()

    # ROC
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6,5))
        plt.plot(fpr, tpr)
        plt.plot([0,1], [0,1], linestyle="--")
        plt.title(f"ROC Curve — {name} (AUC={auc:.3f})")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_ROC.png"))
        plt.close()


# ----------------------------
# 6. EXPORT RÉSUMÉ FINAL
# ----------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "summary_results.csv"), index=False)

print("\n--- ANALYSE TERMINÉE ---")
print("Les résultats se trouvent dans :", OUTPUT_DIR)
print(results_df)
