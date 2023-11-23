import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('/kaggle/input/retail-price-optimization/retail_price.csv')

# Preprocessing
categorical_features = ["product_category_name", "month_year", "s"]  # Relevant categorical features
data = pd.get_dummies(data, columns=categorical_features, drop_first=True)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=["unit_price"]), data["unit_price"], test_size=0.2)

# Fit a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict prices on test data
y_pred = model.predict(X_test)

# Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error on historical data: {mse:.2f}")

# Get pricing coefficients
coefficients = model.coef_
intercept = model.intercept_

print("Pricing Coefficients:")
for feature, coef in zip(X_train.columns, coefficients):
    print(f"{feature}: {coef:.2f}")
print(f"Intercept: {intercept:.2f}")
