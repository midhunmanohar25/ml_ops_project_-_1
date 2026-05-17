from pydantic import BaseModel, Field, computed_field, field_validator
from typing import Literal, Annotated


class UserInput(BaseModel):
    
    name: Annotated[str, Field(..., description='Name of the Car', examples=['Maruti Suzuki Alto'])]
    company: Annotated[str, Field(..., description='Name of the Company that the Car belongs to', examples=['Maruti'])]
    year: Annotated[int, Field(..., description='Car Purchased Year')]
    kms_driven: Annotated[float, Field(...,gt=0, description='Kilometer Driven in Float')]
    fuel_type: Annotated[Literal['Diesel', 'Petrol'], Field(..., description='Fuel Type of the Car')]
    
    @computed_field
    @property
    def is_luxury(self) -> int:
        luxury_brands = ['bmw', 'audi', 'mercedes', 'jaguar', 'mini', 'land rover']
        if self.company.lower() in luxury_brands:
            return 1
        else:
            return 0
    
    @computed_field
    @property
    def car_age(self) -> int:
        current_year = 2020
        return current_year - self.year
    
    @computed_field
    @property
    def kms_per_year(self) -> float:
        return self.kms_driven / (self.car_age + 1)