from pydantic import BaseModel
from typing import List, Optional

class Prediction(BaseModel):
    label: str
    confidence: float
    probabilities: List[float]

class InferenceResponse(BaseModel):
    model: str
    frames_analyzed: int
    sampled_fps: float
    duration_seconds: float
    prediction: Prediction
    trajectory: List[dict]
    notes: List[str]
    pspi_available: bool
    pspi_message: str
