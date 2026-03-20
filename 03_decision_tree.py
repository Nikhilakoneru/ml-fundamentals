# 03_decision_tree.py
# Dataset: Breast Cancer (classification) + California Housing (regression)
# Goal: Trying decision trees on both types of problems

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, r2_score,
                             mean_squared_error)

# --- PART 1: Classification on Breast Cancer ---

# Step 1: Load data
cancer = load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['Target'] = cancer.target  # 0 = malignant, 1 = benign


# Step 2: Prepare
X = df.drop('Target', axis=1)
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# no scaling needed for trees unlike logistic regression

# Step 3: Train
# trying a full tree first to see if it overfits
tree_full = DecisionTreeClassifier(random_state=42)
tree_full.fit(X_train, y_train)

print("Full Tree (no limits):")
print(f"  Train: {accuracy_score(y_train, tree_full.predict(X_train)):.4f}")
print(f"  Test:  {accuracy_score(y_test, tree_full.predict(X_test)):.4f}")
print(f"  Depth: {tree_full.get_depth()}")
# yep it overfits, train is perfect but test drops

# now limiting depth
tree_pruned = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_pruned.fit(X_train, y_train)

print(f"\nPruned Tree (max_depth=3):")
print(f"  Train: {accuracy_score(y_train, tree_pruned.predict(X_train)):.4f}")
print(f"  Test:  {accuracy_score(y_test, tree_pruned.predict(X_test)):.4f}")

# Step 4: Evaluate
y_pred = tree_pruned.predict(X_test)

print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred,
                            target_names=['Malignant', 'Benign']))

# Step 5: Testing different depths
depths = range(1, 16)
train_scores = []
test_scores = []

for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, random_state=42)
    tree.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, tree.predict(X_train)))
    test_scores.append(accuracy_score(y_test, tree.predict(X_test)))

# Step 6: Plots

plt.figure(figsize=(8, 5))
plt.plot(depths, train_scores, 'b-o', label='Train')
plt.plot(depths, test_scores, 'r-o', label='Test')
plt.xlabel('Max Depth')
plt.ylabel('Accuracy')
plt.title('Depth vs Accuracy')
plt.legend()
plt.tight_layout()
plt.savefig('plots/dt_depth_vs_accuracy.png', dpi=150)
plt.show()

# you can actually see the tree which is pretty cool
plt.figure(figsize=(20, 8))
plot_tree(tree_pruned,
          feature_names=cancer.feature_names,
          class_names=['Malignant', 'Benign'],
          filled=True, rounded=True, fontsize=8)
plt.title('Decision Tree (max_depth=3)')
plt.tight_layout()
plt.savefig('plots/dt_tree_visualization.png', dpi=150)
plt.show()

importance = pd.Series(tree_pruned.feature_importances_, index=X.columns)
top10 = importance.sort_values(ascending=True).tail(10)
plt.figure(figsize=(8, 5))
top10.plot(kind='barh', color='steelblue')
plt.xlabel('Importance')
plt.title('Top 10 Features')
plt.tight_layout()
plt.savefig('plots/dt_feature_importance.png', dpi=150)
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

# same thing - full tree vs pruned
tree_reg_full = DecisionTreeRegressor(random_state=42)
tree_reg_full.fit(X_train_h, y_train_h)

tree_reg_pruned = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_reg_pruned.fit(X_train_h, y_train_h)

print(f"\nFull Tree:")
print(f"  Train R2: {r2_score(y_train_h, tree_reg_full.predict(X_train_h)):.4f}")
print(f"  Test R2:  {r2_score(y_test_h, tree_reg_full.predict(X_test_h)):.4f}")

y_pred_h = tree_reg_pruned.predict(X_test_h)
test_r2 = r2_score(y_test_h, y_pred_h)
rmse = np.sqrt(mean_squared_error(y_test_h, y_pred_h))

print(f"\nPruned Tree (depth=5):")
print(f"  Test R2: {test_r2:.4f}")
print(f"  RMSE:    {rmse:.4f} (${rmse * 100000:,.0f})")

# comparing to script 01
print(f"\nLinear Regression R2 was 0.5758")
print(f"Decision Tree R2 is     {test_r2:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test_h, y_pred_h, alpha=0.3, color='steelblue', s=10)
plt.plot([0, 5], [0, 5], color='red', linestyle='--', label='Perfect')
plt.xlabel('Actual Price ($100,000s)')
plt.ylabel('Predicted Price ($100,000s)')
plt.title('Decision Tree: Actual vs Predicted')
plt.legend()
plt.tight_layout()
plt.savefig('plots/dt_regression_actual_vs_pred.png', dpi=150)
plt.show()

print("\nDone!")