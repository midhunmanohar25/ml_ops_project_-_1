import mlflow
import mlflow.sklearn
import os
import pandas as pd
import numpy as np



# Fetch from environment or default to local
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_NAME = os.getenv("MODEL_NAME", "Car_Price_Pipeline")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1")

model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
pipeline = mlflow.sklearn.load_model(model_uri)

# Access Gradient Boosting model from pipeline
model = pipeline.named_steps['model']


def predict_output(user_input: dict):

    user_input = pd.DataFrame([user_input])
    
    # Prediction
    pred_log = pipeline.predict(user_input)[0]
    
    # Reverse Log transform
    predicted_price = float(np.exmp1(pred_log))

    
    # Collect predictions from all trees
    tree_predictions = []
    
    for estimator in model.estimators_.flatten():
        pred_tree = estimator.predict(
            pipeline.named_steps['preprocessor'].transform(user_input)
        )[0]
        
        tree_predictions.append(pred_tree)
    
    tree_predictions = np.array(tree_predictions)
    
    # Standard deviation
    std_dev = np.std(tree_predictions)
    
    # Convert std dev back from log scale
    std_dev_price = float(np.expm1(std_dev))
    
    # 95% confidence interal
    lower_bound = max(0, predicted_price - 1.96 * std_dev_price)
    upper_bound = predicted_price + 1.96 * std_dev_price
    
    # Confidence score (custom heuristic)
    confidence_score = max(
        0,
        min(100, 100 - (std_dev_price / predicted_price) * 100)
    )
    
    # ---------------- Response ----------------
    return {
        "predicted_price": round(predicted_price, 2),

        "confidence_interval": {
            "lower_bound": round(lower_bound, 2),
            "upper_bound": round(upper_bound, 2)
        },

        "model_confidence_percent": round(confidence_score, 2),

        "prediction_summary": {
            "car_age": user_input.car_age,
            "is_luxury": bool(user_input.is_luxury),
            "kms_per_year": round(user_input.kms_per_year, 2)
        }
    }