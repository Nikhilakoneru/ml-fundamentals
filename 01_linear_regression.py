# 01_linear_regression.py
# Dataset: California Housing (built into sklearn)
# Goal: Predict house prices using Linear Regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Step 1: Load the dataset
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target
df.to_csv('data/california_housing.csv', index=False)

# Step 2: Explore the data
print("=== First 5 Rows ===")
print(df.head())
print(f"\nShape: {df.shape}")
print(f"\n=== Summary Stats ===")
print(df.describe())

# Step 3: Prepare the data
X = df.drop('Price', axis=1)
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples:  {X_test.shape[0]}")

# Step 4: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

print(f"\n=== What the Model Learned ===")
for feature, weight in zip(X.columns, model.coef_):
    print(f"  {feature:12s} -> {weight:>10.4f}")
print(f"  {'Intercept':12s} -> {model.intercept_:>10.4f}")

# Step 5: Predict and evaluate
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n=== Test Results ===")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}   (in $100,000s)")
print(f"MAE:  {mae:.4f}   (in $100,000s)")
print(f"R2:   {r2:.4f}")
print(f"\nIn real dollars:")
print(f"  Average prediction error: ${rmse * 100000:,.0f}")

# Step 6: Visualize

# Plot 1: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='steelblue', s=10)
plt.plot([0, 5], [0, 5], color='red', linestyle='--', label='Perfect predictions')
plt.xlabel('Actual Price ($100,000s)')
plt.ylabel('Predicted Price ($100,000s)')
plt.title('Linear Regression: Actual vs Predicted')
plt.legend()
plt.tight_layout()
plt.savefig('plots/actual_vs_predicted.png', dpi=150)
plt.show()

# Plot 2: Residuals
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, alpha=0.3, color='steelblue', s=10)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Predicted Price')
plt.ylabel('Residual')
plt.title('Residual Plot')
plt.tight_layout()
plt.savefig('plots/residuals.png', dpi=150)
plt.show()

# Plot 3: Feature Importance
importance = pd.Series(np.abs(model.coef_), index=X.columns)
importance = importance.sort_values(ascending=True)
plt.figure(figsize=(8, 5))
importance.plot(kind='barh', color='steelblue')
plt.xlabel('Absolute Weight')
plt.title('Feature Importance (Linear Regression)')
plt.tight_layout()
plt.savefig('plots/feature_importance.png', dpi=150)
plt.show()

print("\nPlots saved in plots/ folder.")
print("Done!")