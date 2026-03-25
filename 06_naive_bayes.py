# 06_naive_bayes.py
# Dataset: Breast Cancer
# Goal: Naive Bayes classifier - simple and fast, based on probability
# assumes features are independent (which is rarely true but still works surprisingly well)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, roc_curve,
                             roc_auc_score)

# Step 1: Load data
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['Target'] = cancer.target

# Step 2: Prepare
X = df.drop('Target', axis=1)
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scaling - not strictly required for naive bayes but keeping it consistent
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Train - literally 2 lines, no hyperparameters to tune
model = GaussianNB()
model.fit(X_train_scaled, y_train)
# thats it. no max_depth, no n_estimators, no k to pick. just fit and go

# Step 4: Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("=== Naive Bayes ===")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred,
                            target_names=['Malignant', 'Benign']))

# roc auc
y_prob = model.predict_proba(X_test_scaled)[:, 1]
auc = roc_auc_score(y_test, y_prob)
print(f"ROC AUC: {auc:.4f}")

# overfitting check
train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
print(f"\nTrain: {train_acc:.4f}")
print(f"Test:  {accuracy:.4f}")
print(f"Gap:   {train_acc - accuracy:.4f}")
# small gap which makes sense - naive bayes is simple so it doesnt overfit much

# Step 5: Compare all classification models on breast cancer
print(f"\n--- All Models on Breast Cancer ---")
print(f"  Logistic Regression: 0.9737")
print(f"  Decision Tree (d=3): 0.9561")
print(f"  Random Forest:       0.9649")
print(f"  KNN (best k):        0.9649")
print(f"  Naive Bayes:         {accuracy:.4f}")

# Step 6: Plots

# roc curve comparing naive bayes to the random baseline
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='steelblue', linewidth=2,
         label=f'Naive Bayes (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Naive Bayes ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('plots/nb_roc_curve.png', dpi=150)
plt.show()

# bar chart comparing all models
models = ['Logistic\nRegression', 'Decision\nTree', 'Random\nForest', 'KNN', 'Naive\nBayes']
accuracies = [0.9737, 0.9561, 0.9649, 0.9649, accuracy]

plt.figure(figsize=(8, 5))
bars = plt.bar(models, accuracies, color='steelblue')
plt.ylabel('Accuracy')
plt.title('All Classification Models - Breast Cancer')
plt.ylim(0.93, 1.0)
for bar, val in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.3f}', ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('plots/all_models_comparison.png', dpi=150)
plt.show()
# cool to see them all side by side

print("\nDone!")