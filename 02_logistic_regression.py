# 02_logistic_regression.py
# predicting breast cancer - malignant or benign
# first time doing classification instead of regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay,
                             roc_curve, roc_auc_score)
from sklearn.preprocessing import StandardScaler

# load data - 569 patients, 30 features
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['Target'] = cancer.target  # 0 = malignant, 1 = benign
df.to_csv('data/breast_cancer.csv', index=False)

# explore
print("=== First 5 Rows ===")
print(df.head())
print(f"\nShape: {df.shape}")

# checking if classes are balanced
print(f"\nBenign (1):    {(df['Target'] == 1).sum()}")
print(f"Malignant (0): {(df['Target'] == 0).sum()}")

# prepare
X = df.drop('Target', axis=1)
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# scaling - logistic regression needs features on same scale
# fit on train only, transform both to avoid leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# train
model = LogisticRegression(random_state=42, max_iter=10000)
model.fit(X_train_scaled, y_train)

# predict and evaluate
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n=== Test Results ===")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n=== Confusion Matrix ===")
print(cm)

tn, fp, fn, tp = cm.ravel()
print(f"\nTrue Negatives:  {tn}")
print(f"False Positives: {fp}")
print(f"False Negatives: {fn}")
print(f"True Positives:  {tp}")

# precision recall f1
print(f"\n=== Classification Report ===")
print(classification_report(y_test, y_pred,
                            target_names=['Malignant', 'Benign']))

# roc auc - how well does the model separate the two classes
y_prob = model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
print(f"ROC AUC: {auc:.4f}")

# overfitting check
train_pred = model.predict(X_train_scaled)
train_acc = accuracy_score(y_train, train_pred)
print(f"\nTrain Accuracy: {train_acc:.4f}")
print(f"Test Accuracy:  {accuracy:.4f}")
print(f"Gap:            {train_acc - accuracy:.4f}")

# --- plots ---

# confusion matrix plot
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=['Malignant', 'Benign']).plot(ax=ax)
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png', dpi=150)
plt.show()

# roc curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='steelblue', linewidth=2,
         label=f'Logistic Regression (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
plt.savefig('plots/roc_curve.png', dpi=150)
plt.show()

# top 10 important features
importance = pd.Series(np.abs(model.coef_[0]), index=X.columns)
top10 = importance.sort_values(ascending=True).tail(10)
plt.figure(figsize=(8, 5))
top10.plot(kind='barh', color='steelblue')
plt.xlabel('Absolute Coefficient')
plt.title('Top 10 Features')
plt.tight_layout()
plt.savefig('plots/log_reg_feature_importance.png', dpi=150)
plt.show()

print("\nDone!")