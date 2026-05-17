from fastapi import FastAPI
from starlette.responses import JSONResponse
from schema.user_input import UserInput
from schema.prediction_response import PredictionResponse
from model.predict import predict_output, model, MODEL_VERSION
import uvicorn

app = FastAPI()

# human readable
@app.get('/')
def home():
    return {"status": "API is online", 'message': 'Used Car Price Prediction API'}

@app.get('/health')
def health_check():
    return {
        'status': 'OK',
        'version' : MODEL_VERSION,
        'model_loaded' : model is not None
    }
    
@app.post('/predict', response_model=PredictionResponse)
def predict(data: UserInput):
    
    user_input = {
        'name': data.name,
        'company': data.company,
        'year': data.year,
        'kms_driven': data.kms_driven,
        'fuel_type': data.fuel_type,
        'car_age': data.car_age,
        'is_luxury': data.is_luxury,
        'kms_per_year': data.kms_per_year
    }
    
    try:
        
        prediction = predict_output(user_input)
        
        return JSONResponse(status_code=200, content={'response': prediction})
    
    except Exception as e:

        return JSONResponse(status_code=500, content=str(e))
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    # 127.0.0.1