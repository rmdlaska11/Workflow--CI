import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Set experiment name
mlflow.set_experiment("Level Basic")

# Aktifkan autolog MLflow
mlflow.sklearn.autolog()

# Load data
data = pd.read_csv("train_data.csv")

# Konversi kolom integer ke float64 untuk menghindari masalah missing value
X = data.drop(columns="Weather Type")
X = X.astype({col: 'float64' for col in X.select_dtypes('int').columns})
y = data["Weather Type"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Jalankan MLflow run
with mlflow.start_run():
    # Parameter model
    n_estimators = 505
    max_depth = 37

    # Train model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
    model.fit(X_train, y_train)

    # Prediksi untuk infer signature
    predictions = model.predict(X_test)
    signature = infer_signature(X_test, predictions)

    # Log model dengan signature dan input_example
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_test.iloc[:5]
    )

    # Log metrik akurasi
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    print(f"Model logged to MLflow with accuracy: {accuracy:.4f}")