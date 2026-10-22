"""Pydantic request/response schemas for FastAPI inference service."""
from pydantic import BaseModel, Field
from typing import List


class PredictRequest(BaseModel):
    """Request schema for /predict endpoint."""

    occupancy: List[float] = Field(
        ...,
        description="Binary occupancy vector [52], exactly K ones",
        min_items=52,
        max_items=52
    )
    K: int = Field(
        ...,
        description="Decap budget (total number of ones in occupancy)",
        ge=1,
        le=52
    )
    frequency_mhz: float = Field(
        ...,
        description="Frequency in MHz (1-600)",
        ge=1.0,
        le=600.0
    )

    class Config:
        json_schema_extra = {
            "example": {
                "occupancy": [0.0] * 26 + [1.0] * 26,
                "K": 26,
                "frequency_mhz": 200.0
            }
        }


class PredictResponse(BaseModel):
    """Response schema for /predict endpoint."""

    spectrum: List[float] = Field(
        description="Impedance spectrum [231], 1-600 MHz log-scale (log Ω)",
        min_items=231,
        max_items=231
    )
    heatmap: List[List[float]] = Field(
        description="Physical Ω heatmap [64, 64]"
    )
    metadata: dict = Field(
        default_factory=dict,
        description="Metadata: occupancy_sum, K, frequency_mhz"
    )
