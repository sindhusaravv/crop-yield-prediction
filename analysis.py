import pandas as pd

# Load the dataset
df = pd.read_csv("Custom_Crops_yield_Historical_Dataset.csv")

# Define target variable
y = df["Yield_kg_per_ha"]

# Define input features
X = df[
    [
        "Area_ha",
        "pH",
        "Total_N_kg",
        "Total_P_kg",
        "Total_K_kg",
        "Temperature_C",
        "Humidity_%",
        "Rainfall_mm",
        "Wind_Speed_m_s",
        "Solar_Radiation_MJ_m2_day"
    ]
]

print("===== INPUT FEATURES (X) =====")
print(X.head())

print("\n===== TARGET (y) =====")
print(y.head())

print("\n===== X SHAPE =====")
print(X.shape)

print("\n===== y SHAPE =====")
print(y.shape)
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n===== TRAINING DATA =====")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\n===== TESTING DATA =====")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
from sklearn.linear_model import LinearRegression

# Create the Linear Regression model
linear_model = LinearRegression()

# Train the model
linear_model.fit(X_train, y_train)

# Make predictions
y_pred = linear_model.predict(X_test)

print("\n===== LINEAR REGRESSION =====")
print("First 10 actual values:")
print(y_test.head(10).values)

print("\nFirst 10 predicted values:")
print(y_pred[:10])
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n===== MODEL EVALUATION =====")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R² Score:", r2)
from sklearn.preprocessing import PolynomialFeatures

# Create polynomial features of degree 2
poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Train Linear Regression on polynomial features
poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Make predictions
y_poly_pred = poly_model.predict(X_test_poly)

print("\n===== POLYNOMIAL REGRESSION (DEGREE 2) =====")
print("First 10 predictions:")
print(y_poly_pred[:10])
# Evaluate Polynomial Regression
poly_mae = mean_absolute_error(y_test, y_poly_pred)
poly_mse = mean_squared_error(y_test, y_poly_pred)
poly_rmse = np.sqrt(poly_mse)
poly_r2 = r2_score(y_test, y_poly_pred)

print("\n===== POLYNOMIAL REGRESSION EVALUATION =====")
print("MAE:", poly_mae)
print("MSE:", poly_mse)
print("RMSE:", poly_rmse)
print("R² Score:", poly_r2)
from sklearn.linear_model import Ridge

# Create Ridge Regression model
ridge_model = Ridge(alpha=1.0)

# Train the model
ridge_model.fit(X_train, y_train)

# Make predictions
y_ridge_pred = ridge_model.predict(X_test)

print("\n===== RIDGE REGRESSION =====")
print("First 10 predictions:")
print(y_ridge_pred[:10])
# Evaluate Ridge Regression
ridge_mae = mean_absolute_error(y_test, y_ridge_pred)
ridge_mse = mean_squared_error(y_test, y_ridge_pred)
ridge_rmse = np.sqrt(ridge_mse)
ridge_r2 = r2_score(y_test, y_ridge_pred)

print("\n===== RIDGE REGRESSION EVALUATION =====")
print("MAE:", ridge_mae)
print("MSE:", ridge_mse)
print("RMSE:", ridge_rmse)
print("R² Score:", ridge_r2)
from sklearn.preprocessing import StandardScaler

# Standardize the input features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n===== FEATURE SCALING =====")
print("Scaled training data shape:", X_train_scaled.shape)
print("Scaled testing data shape:", X_test_scaled.shape)
# Ridge Regression with scaled features
scaled_ridge_model = Ridge(alpha=1.0)

scaled_ridge_model.fit(X_train_scaled, y_train)

# Make predictions
y_scaled_ridge_pred = scaled_ridge_model.predict(X_test_scaled)

print("\n===== SCALED RIDGE REGRESSION =====")
print("First 10 predictions:")
print(y_scaled_ridge_pred[:10])
# Evaluate Scaled Ridge Regression
scaled_ridge_mae = mean_absolute_error(y_test, y_scaled_ridge_pred)
scaled_ridge_mse = mean_squared_error(y_test, y_scaled_ridge_pred)
scaled_ridge_rmse = np.sqrt(scaled_ridge_mse)
scaled_ridge_r2 = r2_score(y_test, y_scaled_ridge_pred)

print("\n===== SCALED RIDGE EVALUATION =====")
print("MAE:", scaled_ridge_mae)
print("MSE:", scaled_ridge_mse)
print("RMSE:", scaled_ridge_rmse)
print("R² Score:", scaled_ridge_r2)
from sklearn.linear_model import BayesianRidge

# Create Bayesian Ridge Regression model
bayesian_model = BayesianRidge()

# Train the model using scaled features
bayesian_model.fit(X_train_scaled, y_train)

# Make predictions
y_bayesian_pred = bayesian_model.predict(X_test_scaled)

print("\n===== BAYESIAN LINEAR REGRESSION =====")
print("First 10 predictions:")
print(y_bayesian_pred[:10])
# Evaluate Bayesian Linear Regression
bayesian_mae = mean_absolute_error(y_test, y_bayesian_pred)
bayesian_mse = mean_squared_error(y_test, y_bayesian_pred)
bayesian_rmse = np.sqrt(bayesian_mse)
bayesian_r2 = r2_score(y_test, y_bayesian_pred)

print("\n===== BAYESIAN LINEAR REGRESSION EVALUATION =====")
print("MAE:", bayesian_mae)
print("MSE:", bayesian_mse)
print("RMSE:", bayesian_rmse)
print("R² Score:", bayesian_r2)
# Compare all models
results = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Polynomial Regression",
        "Ridge Regression",
        "Scaled Ridge Regression",
        "Bayesian Linear Regression"
    ],
    "MAE": [
        mae,
        poly_mae,
        ridge_mae,
        scaled_ridge_mae,
        bayesian_mae
    ],
    "RMSE": [
        rmse,
        poly_rmse,
        ridge_rmse,
        scaled_ridge_rmse,
        bayesian_rmse
    ],
    "R2 Score": [
        r2,
        poly_r2,
        ridge_r2,
        scaled_ridge_r2,
        bayesian_r2
    ]
})

print("\n===== MODEL COMPARISON =====")
print(results)
# ===== CLIMATE STRESS SIMULATION =====

# Take one sample from the test dataset
sample = X_test.iloc[[0]].copy()

# Predict normal yield
normal_prediction = scaled_ridge_model.predict(
    scaler.transform(sample)
)[0]

# Create climate stress scenario
stress_sample = sample.copy()

# Increase temperature by 2°C
stress_sample["Temperature_C"] += 2

# Decrease rainfall by 20%
stress_sample["Rainfall_mm"] *= 0.80

# Predict yield under climate stress
stress_prediction = scaled_ridge_model.predict(
    scaler.transform(stress_sample)
)[0]

# Calculate change
yield_change = stress_prediction - normal_prediction
percentage_change = (yield_change / normal_prediction) * 100

print("\n===== CLIMATE STRESS SIMULATION =====")
print("Normal predicted yield:", normal_prediction, "kg/ha")
print("Climate stress predicted yield:", stress_prediction, "kg/ha")
print("Change in yield:", yield_change, "kg/ha")
print("Percentage change:", percentage_change, "%")
# ===== ACTUAL VS PREDICTED YIELD =====

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_scaled_ridge_pred, alpha=0.3)

plt.xlabel("Actual Yield (kg/ha)")
plt.ylabel("Predicted Yield (kg/ha)")
plt.title("Actual vs Predicted Crop Yield")

# Perfect prediction reference line
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()
# ===== CORRELATION WITH CROP YIELD =====

correlations = df.select_dtypes(include="number").corr()["Yield_kg_per_ha"]
correlations = correlations.sort_values(ascending=False)

plt.figure(figsize=(10, 6))

correlations.drop("Yield_kg_per_ha").plot(kind="bar")

plt.xlabel("Features")
plt.ylabel("Correlation with Yield")
plt.title("Feature Correlation with Crop Yield")
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.savefig("feature_correlation.png", dpi=300)
plt.show()
# ===== RAINFALL VS CROP YIELD =====

plt.figure(figsize=(8, 6))

plt.scatter(
    df["Rainfall_mm"],
    df["Yield_kg_per_ha"],
    alpha=0.2
)

plt.xlabel("Rainfall (mm)")
plt.ylabel("Crop Yield (kg/ha)")
plt.title("Rainfall vs Crop Yield")

plt.tight_layout()
plt.savefig("rainfall_vs_yield.png", dpi=300)
plt.show()
# ===== QUANTILE ANALYSIS =====

yield_quantiles = df["Yield_kg_per_ha"].quantile(
    [0.25, 0.50, 0.75]
)

print("\n===== YIELD QUANTILES =====")
print("25th Percentile:", yield_quantiles[0.25])
print("50th Percentile (Median):", yield_quantiles[0.50])
print("75th Percentile:", yield_quantiles[0.75])