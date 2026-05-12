from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
from pydantic import BaseModel
import uvicorn

# 1. Initialize FastAPI
app = FastAPI(title="Used Car Price API")

# 2. Define the Input Schema using Pydantic
class CarInput(BaseModel):
    name: str
    company: str
    year: int
    kms_driven: float
    fuel_type: str

# 3. Load the pre-trained Pipeline
# Using a Pipeline is better because it handles preprocessing automatically
model_path = "pipeline.joblib"
pipeline = joblib.load(model_path)


@app.get("/")
def home():
    return {"status": "API is online", "message": "Send a POST request to /predict"}

@app.get("/welcome")
def welcome():
    return {"message": "Hi welcome to the Mlops project app created by Midhun Manohar"}

@app.post("/predict")
def predict(data: CarInput):
    # --- Feature Engineering (Must match your training logic) ---
    luxury_brands = ['bmw', 'audi', 'mercedes', 'jaguar', 'mini', 'land rover']
    is_luxury = 1 if data.company.lower() in luxury_brands else 0
    current_year = 2020
    car_age = current_year - data.year
    kms_per_year = data.kms_driven / (car_age + 1)

    # 4. Create DataFrame for Pipeline
    # Pipelines expect a DataFrame with matching column names
    input_df = pd.DataFrame([{
        'name': data.name,
        'company': data.company,
        'year': data.year,
        'kms_driven': data.kms_driven,
        'fuel_type': data.fuel_type,
        'car_age': car_age,
        'is_luxury': is_luxury,
        'kms_per_year': kms_per_year
    }])

    # 5. Make Prediction
    pred_log_score = pipeline.predict(input_df)
    
    # Reverse log transform if necessary (np.expm1)
    # If you didn't log-transform during training, just use pred_log_score[0]
    final_price = np.expm1(pred_log_score[0])

    return {
        "input_received": data.dict(),
        "predicted_price": round(float(final_price), 2)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # 127.0.0.1