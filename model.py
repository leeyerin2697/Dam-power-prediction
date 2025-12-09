import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error
import numpy as np

# ===== 1. Load Data =====
df = pd.read_csv("한국수자원공사_수문현황정보_일별.csv", encoding='utf-8')

# ===== 2. Feature and Target Columns =====
features = [
    'water_level', 
    'storage_volume', 
    'inflow_rate', 
    'total_discharge', 
    'rainfall', 
    'cumulative_rainfall', 
    'storage_ratio'
]
target = 'power_discharge'

# Rename columns (Korean → English)
df = df.rename(columns={
    '저수위': 'water_level',
    '저수량': 'storage_volume',
    '유입량': 'inflow_rate',
    '총방류량': 'total_discharge',
    '강수량': 'rainfall',
    '금년누가강우량': 'cumulative_rainfall',
    '저수율': 'storage_ratio',
    '발전방류량': 'power_discharge',
    '댐명': 'dam_name'
})

# ===== 3. Select Columns and Remove Missing / Invalid Values =====
data = df[features + [target, 'dam_name']].dropna()

data = data[
    (data['water_level'] >= 0) &
    (data['storage_volume'] >= 0) &
    (data['inflow_rate'] >= 0) &
    (data['total_discharge'] >= 0) &
    (data['rainfall'] >= 0) &
    (data['cumulative_rainfall'] >= 0) &
    (data['storage_ratio'] >= 0)
]

print("Original dataset size:", len(data))

# ===== 4. Sampling Large Dataset =====
if len(data) > 50000:
    data = data.sample(n=50000, random_state=42)
    print("Sampled dataset size:", len(data))

# ===== 5. Define X and y =====
X = data[features].astype('float32')
y = data[target].astype('float32')

# ===== 6. Train/Test Split =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== 7-0. Linear Regression (Baseline Model) =====
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
pred_linear = linear_model.predict(X_test)

# ===== 7. Polynomial Regression (Nonlinear Model) =====
poly_model = make_pipeline(
    StandardScaler(),
    PolynomialFeatures(degree=2),
    LinearRegression()
)

poly_model.fit(X_train, y_train)
pred_poly = poly_model.predict(X_test)

# ===== 8. Random Forest Model =====
rf_model = RandomForestRegressor(
    n_estimators=30,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
pred_rf = rf_model.predict(X_test)

# ===== 9. Model Performance Comparison =====
print("\n====== Model Performance Comparison ======")

print("Linear Regression MSE:", mean_squared_error(y_test, pred_linear))
print("Polynomial Regression MSE:", mean_squared_error(y_test, pred_poly))
print("Random Forest MSE:", mean_squared_error(y_test, pred_rf))

print("Linear Regression R2:", r2_score(y_test, pred_linear))
print("Polynomial Regression R2:", r2_score(y_test, pred_poly))
print("Random Forest R2:", r2_score(y_test, pred_rf))

print("Linear Regression MAE:", mean_absolute_error(y_test, pred_linear))
print("Polynomial Regression MAE:", mean_absolute_error(y_test, pred_poly))
print("Random Forest MAE:", mean_absolute_error(y_test, pred_rf))

print("Linear Regression RMSE:", np.sqrt(mean_squared_error(y_test, pred_linear)))
print("Polynomial Regression RMSE:", np.sqrt(mean_squared_error(y_test, pred_poly)))
print("Random Forest RMSE:", np.sqrt(mean_squared_error(y_test, pred_rf)))

# ===== 10. Visualization =====
# RMSE Bar Chart
models = ['Linear Regression', 'Polynomial Regression', 'Random Forest']
Rmse_values = [
    np.sqrt(mean_squared_error(y_test, pred_linear)),
    np.sqrt(mean_squared_error(y_test, pred_poly)),
    np.sqrt(mean_squared_error(y_test, pred_rf))
]

plt.figure()
plt.bar(models, Rmse_values)
plt.title("Model Comparison (RMSE)")
plt.xlabel("Model")
plt.ylabel("RMSE")
plt.show()

# MSE Bar Chart
models = ['Linear Regression', 'Polynomial Regression', 'Random Forest']
mse_values = [
    mean_squared_error(y_test, pred_linear),
    mean_squared_error(y_test, pred_poly),
    mean_squared_error(y_test, pred_rf)
]

plt.figure()
plt.bar(models, mse_values)
plt.title("Model Comparison (MSE)")
plt.xlabel("Model")
plt.ylabel("MSE")
plt.show()

# R2 Bar Chart
r2_values = [
    r2_score(y_test, pred_linear),
    r2_score(y_test, pred_poly),
    r2_score(y_test, pred_rf)
]

plt.figure()
plt.bar(models, r2_values)
plt.title("Model Comparison (R² Score)")
plt.xlabel("Model")
plt.ylabel("R²")
plt.show()


# MAE Bar Chart
models = ['Linear Regression', 'Polynomial Regression', 'Random Forest']
mae_values = [
    mean_absolute_error(y_test, pred_linear),
    mean_absolute_error(y_test, pred_poly),
    mean_absolute_error(y_test, pred_rf)
]

plt.figure()
plt.bar(models, mse_values)
plt.title("Model Comparison (MAE)")
plt.xlabel("Model")
plt.ylabel("MAE")
plt.show()

# Actual vs Predicted Plots
plt.figure()
plt.scatter(y_test, pred_linear)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])
plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.show()

plt.figure()
plt.scatter(y_test, pred_poly)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])
plt.title("Polynomial Regression: Actual vs Predicted")
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.show()

plt.figure()
plt.scatter(y_test, pred_rf)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()])
plt.title("Random Forest: Actual vs Predicted")
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.show()


# ===== 11. Feature Importance (Random Forest) =====
importances = rf_model.feature_importances_

feature_importance = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n=== Feature Importance (Random Forest) ===")
print(feature_importance)
plt.figure()
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel("Feature Importance")
plt.ylabel("Variables")
plt.title("Feature Importance - Random Forest")
plt.gca().invert_yaxis()
plt.show()


print("====== Random Forest Hyperparameter Experiments ======")

n_estimators_list = [10, 50, 100]
max_depth_list = [5, 10, 20]

for n in n_estimators_list:
    for d in max_depth_list:
        test_rf = RandomForestRegressor(
            n_estimators=n,
            max_depth=d,
            random_state=42,
            n_jobs=-1
        )
        test_rf.fit(X_train, y_train)
        pred = test_rf.predict(X_test)

        mse = mean_squared_error(y_test, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, pred)

        print(f"n_estimators={n}, max_depth={d} -> RMSE: {rmse:.2f}, R2: {r2:.3f}")

