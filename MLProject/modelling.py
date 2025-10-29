import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import os
import argparse

# --- Ambil argumen dari MLflow Project ---
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="housing_preprocessed.csv")
args = parser.parse_args()

# --- Load dataset ---
data_path = os.path.join(os.path.dirname(__file__), args.data_path)
df = pd.read_csv(data_path)

# Misal kolom terakhir adalah target
X = df.drop("target", axis=1)
y = df["target"]

# --- Split data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Evaluate ---
score = model.score(X_test, y_test)
print(f"Model R^2 Score: {score}")

# --- Log ke MLflow ---
with mlflow.start_run():
    mlflow.log_param("data_path", args.data_path)
    mlflow.log_metric("r2_score", score)
    mlflow.sklearn.log_model(model, "model")
