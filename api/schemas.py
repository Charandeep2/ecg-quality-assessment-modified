"""
Pydantic Schemas for the Adaptive ECG Quality Assessment FastAPI backend.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "2.0.0-adaptive"
    device: str
    model_loaded: bool
    dataset_records_available: int


class PredictionRequest(BaseModel):
    record_id: Optional[str] = Field(None, description="PhysioNet record ID (e.g. '1002867')")
    signal: Optional[List[List[float]]] = Field(
        None, description="Optional raw signal array of shape (5000, 12)"
    )
    adaptive_preprocessing: bool = Field(
        True, description="Apply 0.5 Hz Butterworth filter if baseline wander > 0.08"
    )
    adaptive_cwt: bool = Field(
        True, description="Expand CWT ceiling to 50 Hz if high-frequency noise > 0.05"
    )
    adaptive_rank: bool = Field(
        True, description="Use 95% cumulative singular-value energy rank instead of fixed (50, 20, 6)"
    )


class LeadInspectionItem(BaseModel):
    lead: str
    minimum: float
    maximum: float
    std: float
    unique_values: int
    constant_lead: bool
    contains_dropout_marker: bool


class PredictionResponse(BaseModel):
    record_id: Optional[str] = None
    prediction: str = Field(..., description="'Acceptable' or 'Unacceptable'")
    prediction_class: int = Field(..., description="0 for Acceptable, 1 for Unacceptable")
    probability_unacceptable: float
    probability_acceptable: float
    filter_applied: bool
    max_frequency_hz: float
    tucker_rank: List[int]
    reconstruction_error: float
    compression_ratio: float
    parameters_stored: int
    characteristics: Dict[str, float]
    flat_leads: List[str]
    dropout_leads: List[str]
    core_slice: Optional[List[List[float]]] = None
    xai_factors: Optional[List[Dict[str, Any]]] = None
    condition_cluster: Optional[Dict[str, Any]] = None


class InspectionResponse(BaseModel):
    record_id: Optional[str] = None
    flat_leads: List[str]
    dropout_leads: List[str]
    characteristics: Dict[str, float]
    leads: List[LeadInspectionItem]


class DecompositionResponse(BaseModel):
    record_id: Optional[str] = None
    rank: List[int]
    core_shape: List[int]
    compression_ratio: float
    reconstruction_error: float
    parameters_stored: int
    original_shape: List[int]
    core_slice: Optional[List[List[float]]] = None


class RecordSummary(BaseModel):
    record_id: str
    label: Optional[str] = None
    group_size: Optional[int] = 1

