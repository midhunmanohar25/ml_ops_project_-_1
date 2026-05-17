
# # Fetch from environment or default to local
# # MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
# # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# # MODEL_NAME = os.getenv("MODEL_NAME", "Car_Price_Pipeline")
# # MODEL_VERSION = os.getenv("MODEL_VERSION", "1")

# # model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
# # pipeline = mlflow.sklearn.load_model(model_uri)
# MODEL_VERSION = "1.0.0.0"
# model_path = "pipeline.joblib"
# pipeline = joblib.load(model_path)

# # Access Gradient Boosting model from pipeline
# model = pipeline.named_steps['model']


import joblib
import pandas as pd
import numpy as np
import os
import mlflow.sklearn

# Connect to your remote DagsHub MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = os.getenv("MODEL_NAME", "Car_Price_Pipeline")
# You can pass "latest", a number like "2", or an alias like "champion"
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest") 

# Construct the model registry URI
model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"

# This will download the model directly from MLflow into memory when FastAPI starts!
pipeline = mlflow.sklearn.load_model(model_uri)
model = pipeline.named_steps['model']



def predict_output(user_input: dict):

    # Convert the single input dictionary into a single-row DataFrame
    df_input = pd.DataFrame([user_input])
    
    # 1. Prediction (on log scale)
    pred_log = pipeline.predict(df_input)[0]
    
    # Reverse Log transform to get the true predicted price
    predicted_price = float(np.expm1(pred_log))

    # 2. Collect predictions from all trees
    tree_predictions = []
    
    # Transform the input data once outside the loop for better performance
    X_transformed = pipeline.named_steps['preprocessor'].transform(df_input)
    
    for estimator in model.estimators_.flatten():
        pred_tree = estimator.predict(X_transformed)[0]
        tree_predictions.append(pred_tree)
    
    tree_predictions = np.array(tree_predictions)
    
    # Standard deviation on the log scale
    std_dev = np.std(tree_predictions)
    
    # 3. Calculate 95% confidence intervals ON THE LOG SCALE first
    lower_bound_log = pred_log - 1.96 * std_dev
    upper_bound_log = pred_log + 1.96 * std_dev
    
    # Convert confidence bounds back from log scale
    lower_bound = max(0.0, float(np.expm1(lower_bound_log)))
    upper_bound = float(np.expm1(upper_bound_log))
    
    # Convert std dev back just for the heuristic confidence score
    std_dev_price = float(np.expm1(std_dev))
    confidence_score = max(
        0.0,
        min(100.0, 100.0 - (std_dev_price / predicted_price) * 100.0)
    )
    
    # ---------------- Response ----------------
    # Use .iloc[0] to extract single scalar values out of the pandas DataFrame
    return {
        "predicted_price": round(predicted_price, 2),

        "confidence_interval": {
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2)
        },

        "model_confidence_percent": round(confidence_score, 2),

        "prediction_summary": {
            "car_age": int(df_input["car_age"].iloc[0]),
            "is_luxury": bool(df_input["is_luxury"].iloc[0]),
            "kms_per_year": round(float(df_input["kms_per_year"].iloc[0]), 2)
        }
    }