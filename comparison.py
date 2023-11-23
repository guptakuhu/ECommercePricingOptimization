import time
import pandas as pd
import PythonWithPyro.py
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset (assuming you've already preprocessed it)
data = pd.read_csv('/kaggle/input/retail-price-optimization/retail_price.csv')

# Split data into features (X) and target (y)
X = data.drop(columns=["unit_price"])
y = data["unit_price"]

# Initialize models
pyro_model = PyroPricingModel()  # Replace with your actual Pyro model
linear_model = LinearRegression()

# Measure inference time for Pyro model
start_time = time.time()
pyro_model.fit(X, y)
pyro_inference_time = time.time() - start_time

# Measure inference time for Linear Regression model
start_time = time.time()
linear_model.fit(X, y)
linear_inference_time = time.time() - start_time

# Predict prices using both models
pyro_predicted_prices = pyro_model.predict(X)
linear_predicted_prices = linear_model.predict(X)

# Evaluate accuracy (MSE)
pyro_mse = mean_squared_error(y, pyro_predicted_prices)
linear_mse = mean_squared_error(y, linear_predicted_prices)

print(f"Pyro MSE: {pyro_mse:.2f}")
print(f"Linear Regression MSE: {linear_mse:.2f}")

# Compare efficiency
print(f"Pyro Inference Time: {pyro_inference_time:.2f} seconds")
print(f"Linear Regression Inference Time: {linear_inference_time:.2f} seconds")

# Interpretability: Explain feature importance for Linear Regression model
coefficients = linear_model.coef_
feature_importance = dict(zip(X.columns, coefficients))
print("Feature Importance (Linear Regression):")
for feature, importance in feature_importance.items():
    print(f"{feature}: {importance:.2f}")
