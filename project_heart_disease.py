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