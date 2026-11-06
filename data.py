
"""
Breast Cancer Detection and Classification Using Supervised Machine Learning
Dataset: Wisconsin Diagnostic Breast Cancer (WDBC) - UCI ML Repository
         (569 records, 30 numerical features, diagnosis: benign / malignant)

Setup (run once in the VS Code terminal):
    pip install numpy pandas scikit-learn matplotlib seaborn

Run:
    python breast_cancer_vscode.py
Charts are saved into an 'outputs' folder next to this script.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 1. DATA COLLECTION
# ------------------------------------------------------------------
# The WDBC dataset ships with scikit-learn, so no download is needed.
# (To use a CSV from UCI/Kaggle instead, replace this block with
#  df = pd.read_csv("data.csv") and set the target column accordingly.)
raw = load_breast_cancer(as_frame=True)
df = raw.frame.copy()
# In scikit-learn: 0 = malignant, 1 = benign. Flip so 1 = malignant.
df["diagnosis"] = 1 - df["target"]
df = df.drop(columns="target")

print("=" * 60)
print("1. DATA COLLECTION")
print("=" * 60)
print("Shape:", df.shape)
print(df.head(), "\n")

# ------------------------------------------------------------------
# 2. PREPROCESSING AND CLEANING
# ------------------------------------------------------------------
print("=" * 60)
print("2. DATA PREPROCESSING")
print("=" * 60)
print("Missing values :", int(df.isnull().sum().sum()))
print("Duplicate rows :", int(df.duplicated().sum()))
print("Data types     :", df.dtypes.value_counts().to_dict())
df = df.drop_duplicates().dropna()

counts = df["diagnosis"].value_counts().rename({0: "Benign", 1: "Malignant"})
print("\nClass distribution:\n", counts, "\n")

plt.figure(figsize=(5, 4))
sns.barplot(x=counts.index, y=counts.values, palette=["#4C9F70", "#D1495B"])
plt.title("Class Distribution")
plt.ylabel("Number of tumours")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "class_distribution.png"), dpi=150)
plt.close()

X = df.drop(columns="diagnosis")
y = df["diagnosis"]

# ------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT (80% / 20%, stratified)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
print(f"Training records: {len(X_train)} | Testing records: {len(X_test)}\n")

# ------------------------------------------------------------------
# 4. MODEL TRAINING
# Logistic Regression is scaled (standardised); tree models do not need it.
# ------------------------------------------------------------------
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# ------------------------------------------------------------------
# 5. EVALUATION
# ------------------------------------------------------------------
print("=" * 60)
print("3. MODEL EVALUATION (malignant = positive class)")
print("=" * 60)

results, matrices = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    cm = confusion_matrix(y_test, pred)
    matrices[name] = cm

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1-score": f1_score(y_test, pred),
        "CV Mean": cv_scores.mean(),
        "CV Std": cv_scores.std(),
    })

    print(f"\n--- {name} ---")
    print(classification_report(y_test, pred, target_names=["Benign", "Malignant"]))
    print("Confusion matrix [[TN FP] [FN TP]]:\n", cm)
    print(f"5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

results_df = pd.DataFrame(results).set_index("Model").round(4)
print("\n" + "=" * 60)
print("SUMMARY TABLE")
print("=" * 60)
print(results_df)
results_df.to_csv(os.path.join(OUT_DIR, "model_results.csv"))

# ------------------------------------------------------------------
# 6. CHARTS
# ------------------------------------------------------------------
# Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (name, cm) in zip(axes, matrices.items()):
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "confusion_matrices.png"), dpi=150)
plt.close()

# Model comparison
results_df[["Accuracy", "Precision", "Recall", "F1-score"]].plot(
    kind="bar", figsize=(9, 5), ylim=(0.8, 1.02), rot=0)
plt.title("Model Comparison")
plt.ylabel("Score")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "model_comparison.png"), dpi=150)
plt.close()

# Feature importance (Random Forest)
rf = models["Random Forest"]
importance = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 most important features (Random Forest):")
print(importance.head(10).round(4))

plt.figure(figsize=(8, 5))
importance.head(10).sort_values().plot(kind="barh", color="#3E6FB0")
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_importance.png"), dpi=150)
plt.close()

# ------------------------------------------------------------------
# 7. FINAL MODEL SELECTION
# Chosen by highest malignant recall (fewest missed cancers), then CV accuracy.
# ------------------------------------------------------------------
best = results_df.sort_values(["Recall", "CV Mean"], ascending=False).index[0]
print("\n" + "=" * 60)
print(f"FINAL MODEL SELECTED: {best}")
print("=" * 60)
print(results_df.loc[best])
print(f"\nCharts and results saved in: {OUT_DIR}")
print("Note: academic decision-support tool only - not for real diagnosis.")