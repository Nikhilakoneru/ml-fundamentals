# 04_random_forest.py
# Dataset: Breast Cancer + California Housing
# Goal: Random forest on both - should be better than a single decision tree

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, r2_score,
                             mean_squared_error)

# --- PART 1: Classification on Breast Cancer ---

# Step 1: Load data
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['Target'] = cancer.target

# Step 2: Prepare - no scaling needed same as decision tree
X = df.drop('Target', axis=1)
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Train with 100 trees
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("=== Random Forest Classification ===")
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")

# oob score - each tree has ~37% data it never saw, use that as validation
rf_oob = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
rf_oob.fit(X_train, y_train)
print(f"OOB Score: {rf_oob.oob_score_:.4f}")

print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred,
                            target_names=['Malignant', 'Benign']))

# overfitting check
train_acc = accuracy_score(y_train, rf.predict(X_train))
print(f"Train: {train_acc:.4f}")
print(f"Test:  {accuracy:.4f}")
print(f"Gap:   {train_acc - accuracy:.4f}")

# Step 4: does adding more trees help?
tree_counts = [10, 25, 50, 100, 200, 300, 500]
scores = []

for n in tree_counts:
    forest = RandomForestClassifier(n_estimators=n, random_state=42)
    forest.fit(X_train, y_train)
    scores.append(accuracy_score(y_test, forest.predict(X_test)))

# Step 5: Plots

plt.figure(figsize=(8, 5))
plt.plot(tree_counts, scores, 'b-o')
plt.xlabel('Number of Trees')
plt.ylabel('Test Accuracy')
plt.title('How Many Trees Do We Need?')
plt.tight_layout()
plt.savefig('plots/rf_n_trees_vs_accuracy.png', dpi=150)
plt.show()
# looks like after ~100 trees it flattens out

importance = pd.Series(rf.feature_importances_, index=X.columns)
top10 = importance.sort_values(ascending=True).tail(10)
plt.figure(figsize=(8, 5))
top10.plot(kind='barh', color='steelblue')
plt.xlabel('Importance')
plt.title('Random Forest - Top 10 Features')
plt.tight_layout()
plt.savefig('plots/rf_feature_importance.png', dpi=150)
plt.show()

# --- PART 2: Regression on California Housing ---

print("\n--- Regression: California Housing ---")

housing = fetch_california_housing()
df_h = pd.DataFrame(housing.data, columns=housing.feature_names)
df_h['Price'] = housing.target

X_h = df_h.drop('Price', axis=1)
y_h = df_h['Price']

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_h, y_h, test_size=0.2, random_state=42
)

rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_h, y_train_h)

y_pred_h = rf_reg.predict(X_test_h)
test_r2 = r2_score(y_test_h, y_pred_h)
rmse = np.sqrt(mean_squared_error(y_test_h, y_pred_h))

print(f"\nTest R2: {test_r2:.4f}")
print(f"RMSE:    {rmse:.4f} (${rmse * 100000:,.0f})")

# how does it stack up against everything so far
print(f"\n--- All Models on Calif Housing ---")
print(f"  Linear Regression:  0.5758")
print(f"  Decision Tree (d=5): 0.6220")
print(f"  Random Forest:       {test_r2:.4f}")
# nice improvement

plt.figure(figsize=(8, 6))
plt.scatter(y_test_h, y_pred_h, alpha=0.3, color='steelblue', s=10)
plt.plot([0, 5], [0, 5], color='red', linestyle='--', label='Perfect')
plt.xlabel('Actual Price ($100,000s)')
plt.ylabel('Predicted Price ($100,000s)')
plt.title('Random Forest: Actual vs Predicted')
plt.legend()
plt.tight_layout()
plt.savefig('plots/rf_regression_actual_vs_pred.png', dpi=150)
plt.show()

print("\nDone!")