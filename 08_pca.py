# 08_pca.py
# Dataset: Breast Cancer (30 features)
# Goal: PCA - reduce dimensions while keeping most of the information
# also seeing if a model works just as well with fewer features

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Step 1: Load data
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['Target'] = cancer.target

print("=== Breast Cancer Dataset ===")
print(f"Shape: {df.shape}")
print(f"Features: {len(cancer.feature_names)}")
# 30 features is a lot, can we reduce without losing much?

# Step 2: Scale first - PCA needs scaled data
X = df.drop('Target', axis=1)
y = df['Target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: PCA with all components to see how much each one explains
pca_full = PCA()
pca_full.fit(X_scaled)

# how much variance does each component explain
explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

print(f"\n=== Variance Explained ===")
for i in range(5):
    print(f"  PC{i + 1}: {explained[i]:.4f} ({cumulative[i] * 100:.1f}% cumulative)")

# how many components to keep 95% of the information?
n_95 = np.argmax(cumulative >= 0.95) + 1
print(f"\nNeed {n_95} components to keep 95% of variance (out of 30)")

# Step 4: Reduce to 2 components for visualization
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

print(f"\n2 components explain {pca_2d.explained_variance_ratio_.sum() * 100:.1f}% of variance")

# Step 5: Does a model still work with fewer features?
# comparing logistic regression with all 30 features vs PCA-reduced

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# all 30 features
lr_full = LogisticRegression(random_state=42, max_iter=10000)
lr_full.fit(X_train, y_train)
acc_full = accuracy_score(y_test, lr_full.predict(X_test))

# with PCA - trying different numbers of components
components_to_try = [2, 5, 10, 15, 20, 30]
pca_accuracies = []

for n in components_to_try:
    pca = PCA(n_components=n)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    lr = LogisticRegression(random_state=42, max_iter=10000)
    lr.fit(X_train_pca, y_train)
    pca_accuracies.append(accuracy_score(y_test, lr.predict(X_test_pca)))

print(f"\n=== Accuracy with Different Components ===")
print(f"  All 30 features: {acc_full:.4f}")
for n, acc in zip(components_to_try, pca_accuracies):
    print(f"  PCA ({n:2d} components): {acc:.4f}")
# interesting that even 10 components gets close to using all 30

# Step 6: Plots

# variance explained by each component
plt.figure(figsize=(8, 5))
plt.bar(range(1, 11), explained[:10], color='steelblue', alpha=0.7, label='Individual')
plt.plot(range(1, 11), cumulative[:10], 'r-o', label='Cumulative')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.title('How Much Does Each Component Explain?')
plt.legend()
plt.tight_layout()
plt.savefig('plots/pca_variance_explained.png', dpi=150)
plt.show()

# 2d scatter colored by actual class
plt.figure(figsize=(8, 6))
colors = ['coral', 'steelblue']
labels = ['Malignant', 'Benign']
for i in range(2):
    mask = y == i
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                c=colors[i], label=labels[i], alpha=0.5)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0] * 100:.1f}%)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1] * 100:.1f}%)')
plt.title('Breast Cancer Data in 2D (PCA)')
plt.legend()
plt.tight_layout()
plt.savefig('plots/pca_2d_scatter.png', dpi=150)
plt.show()
# can see the two groups separate pretty well even in just 2 dimensions

# accuracy vs number of components
plt.figure(figsize=(8, 5))
plt.plot(components_to_try, pca_accuracies, 'b-o', label='PCA')
plt.axhline(y=acc_full, color='red', linestyle='--', label=f'All features ({acc_full:.3f})')
plt.xlabel('Number of Components')
plt.ylabel('Accuracy')
plt.title('PCA: How Many Components Do We Need?')
plt.legend()
plt.tight_layout()
plt.savefig('plots/pca_components_vs_accuracy.png', dpi=150)
plt.show()

print("\nDone!")