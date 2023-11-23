import pandas as pd
import torch
import pyro
import pyro.distributions as dist
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv('/kaggle/input/retail-price-optimization/retail_price.csv')

# Preprocessing
scaler = MinMaxScaler()
numeric_features = ["qty", "total_price", "freight_price", "product_name_lenght", "product_description_lenght", "product_photos_qty", "product_weight_g", "product_score", "customers", "weekday", "weekend", "holiday", "lag_price"]
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=["unit_price"]), data["unit_price"], test_size=0.2)

class PyroPricingModel:
    # Define the probabilistic model using Pyro
    def pricing_model(data):
        price_mean = pyro.param("price_mean", torch.tensor(100.0))
        price_std = pyro.param("price_std", torch.tensor(10.0), constraint=dist.constraints.positive)

        with pyro.plate("data", len(data)):
            price = pyro.sample("price", dist.Normal(price_mean, price_std), obs=data["unit_price"])

        return price

    # Train the model
    def train_model(data):
        pyro.clear_param_store()
        optimizer = torch.optim.Adam({"lr": 0.01})
        svi = pyro.infer.SVI(pricing_model, guide=None, optim=optimizer, loss=pyro.infer.Trace_ELBO())

        for _ in range(1000):
            loss = svi.step(data)

    # Train the model
    train_model(X_train)

    # Predict prices on test data
    def predict_prices(data):
        samples = pricing_model(data)
        predicted_prices = samples.mean(dim=0)
        return predicted_prices

    # Get predicted prices
    predicted_prices = predict_prices(X_test)

    # Evaluate model performance
    mse = mean_squared_error(y_test, predicted_prices)
    print(f"Mean Squared Error on historical data: {mse:.2f}")

