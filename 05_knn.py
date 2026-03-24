# 05_knn.py
# Dataset: Breast Cancer
# Goal: KNN classifier - predicts based on closest neighbors
# need to scale features since KNN uses distance

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report)

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

# scaling is important for KNN - it uses distance between points
# if one feature is 0-1 and another is 0-1000 the big one dominates
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Train with k=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("=== KNN (k=5) ===")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred,
                            target_names=['Malignant', 'Benign']))

# Step 4: Finding the best K
# trying k from 1 to 25 to see which works best
k_values = range(1, 26)
train_scores = []
test_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    train_scores.append(accuracy_score(y_train, knn.predict(X_train_scaled)))
    test_scores.append(accuracy_score(y_test, knn.predict(X_test_scaled)))

best_k = k_values[np.argmax(test_scores)]
best_acc = max(test_scores)
print(f"\nBest K: {best_k} (accuracy: {best_acc:.4f})")

# Step 5: What happens without scaling?
knn_noscale = KNeighborsClassifier(n_neighbors=5)
knn_noscale.fit(X_train, y_train)
noscale_acc = accuracy_score(y_test, knn_noscale.predict(X_test))

print(f"\n--- Scaling Matters ---")
print(f"  With scaling:    {accuracy:.4f}")
print(f"  Without scaling: {noscale_acc:.4f}")
# proves why scaling is important for distance-based algorithms

# Step 6: Plots

# k vs accuracy
plt.figure(figsize=(8, 5))
plt.plot(k_values, train_scores, 'b-o', markersize=4, label='Train')
plt.plot(k_values, test_scores, 'r-o', markersize=4, label='Test')
plt.xlabel('K (number of neighbors)')
plt.ylabel('Accuracy')
plt.title('KNN: Finding the Best K')
plt.legend()
plt.tight_layout()
plt.savefig('plots/knn_k_vs_accuracy.png', dpi=150)
plt.show()
# k=1 overfits (memorizes), larger k is smoother

# scaled vs unscaled comparison
plt.figure(figsize=(6, 4))
bars = plt.bar(['With Scaling', 'Without Scaling'],
               [accuracy, noscale_acc],
               color=['steelblue', 'coral'])
plt.ylabel('Accuracy')
plt.title('KNN: Does Scaling Matter?')
plt.ylim(0.9, 1.0)
for bar, val in zip(bars, [accuracy, noscale_acc]):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
             f'{val:.3f}', ha='center')
plt.tight_layout()
plt.savefig('plots/knn_scaling_comparison.png', dpi=150)
plt.show()

# comparing all classification models so far
print(f"\n--- All Models on Breast Cancer ---")
print(f"  Logistic Regression: 0.9737")
print(f"  Decision Tree (d=3): 0.9561")
print(f"  Random Forest:       0.9649")
print(f"  KNN (k={best_k}):          {best_acc:.4f}")

print("\nDone!")