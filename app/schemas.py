# Schemas for sepsis prediction API requests and responses
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# Define a model for vital observations
class VitalObservation(BaseModel):
    timestamp: Optional[str] = Field(default=None, description="ISO timestamp string")
    heart_rate: Optional[float] = Field(default=None)
    temperature: Optional[float] = Field(default=None)
    oxygen_saturation: Optional[float] = Field(default=None)
    respiratory_rate: Optional[float] = Field(default=None)
    blood_pressure: Optional[float] = Field(default=None)

# Define a model for lab observations
class LabObservation(BaseModel):
    timestamp: Optional[str] = Field(default=None, description="ISO timestamp string")
    white_cell_count: Optional[float] = Field(default=None)
    crp: Optional[float] = Field(default=None)
    lactate: Optional[float] = Field(default=None)
    creatinine: Optional[float] = Field(default=None)
    platelet_count: Optional[float] = Field(default=None)

# Define a model for sepsis prediction requests
class SepsisPredictionRequest(BaseModel):
    patient_id: int = Field(..., example=101, description='Unique identifier for the patient')
    age: int = Field(..., example=50)
    gender: str = Field(..., example='Male')
    comorbidity_count: int = Field(..., example=2)
    vitals: List[VitalObservation] = Field(default=[])
    labs: List[LabObservation] = Field(default=[])  

# Define a model for sepsis prediction responses
class SepsisPredictionResponse(BaseModel):
    patient_id: int 
    sepsis_risk_score: float
    risk_category: str
    prediction_window: str
    key_risk_factors: List[str]

# Define a model for health check responses
class HealthCheckResponse(BaseModel):
    status: str
    service: str
    model_loaded: bool
    version: str
    