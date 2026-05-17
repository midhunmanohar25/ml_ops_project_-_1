from pydantic import BaseModel, Field

class ConfidenceInterval(BaseModel):
    lower_bound: float = Field(
        ...,
        description="Lower bound of the predicted price range",
        example=450000.0
    )

    upper_bound: float = Field(
        ...,
        description="Upper bound of the predicted price range",
        example=550000.0
    )


class PredictionSummary(BaseModel):
    car_age: int = Field(
        ...,
        description="Age of the car in years",
        example=5
    )

    is_luxury: bool = Field(
        ...,
        description="Whether the car belongs to a luxury brand",
        example=True
    )

    kms_per_year: float = Field(
        ...,
        description="Average kilometers driven per year",
        example=12000.5
    )



# ----------------------The Main class-------------------------------


class PredictionResponse(BaseModel):
    predicted_price: float = Field(
        ...,
        description="Predicted resale price of the car",
        example=525000.75
    )

    confidence_interval: ConfidenceInterval

    model_confidence_percent: float = Field(
        ...,
        description="Estimated confidence score of the model prediction",
        example=91.4
    )

    prediction_summary: PredictionSummary
    