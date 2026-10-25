"""
FastAPI Route: Dataset Navigation & Record Retrieval.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
import pandas as pd

import core
from api.schemas import RecordSummary

router = APIRouter(prefix="/records", tags=["Records"])


@router.get("", response_model=List[RecordSummary])
def list_records(
    limit: int = Query(50, ge=1, le=1000, description="Max records to return"),
    label: Optional[str] = Query(None, description="Filter by 'Acceptable' or 'Unacceptable'"),
):
    """List available PhysioNet 2011 Set A records with their ground truth labels and cluster groups."""
    data, groups_df, chars_df, _ = core.load_project_metadata()

    if data.empty:
        raise HTTPException(status_code=500, detail="Dataset metadata could not be loaded.")

    df = data.copy()
    if label and isinstance(label, str):
        df = df[df["label"].str.lower() == label.lower()]

    records = []
    max_count = int(limit) if isinstance(limit, (int, float)) else 50
    for _, row in df.head(max_count).iterrows():
        records.append(RecordSummary(
            record_id=str(row["record_id"]),
            label=str(row.get("label", "Unknown")),
            group_size=int(row.get("group_size", 1)),
        ))

    return records


@router.get("/{record_id}")
def get_record_data(record_id: str, downsample: int = Query(5, ge=1, le=50)):
    """
    Get raw ECG waveform data and header metadata for a given record.
    Optionally downsamples time series for quick web charting.
    """
    try:
        signal, fields = core.load_dataset_ecg(record_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Record {record_id} error: {exc}")

    sampled = signal[::downsample, :]
    times = [round(i / core.FS, 4) for i in range(0, signal.shape[0], downsample)]

    leads_data = {}
    for idx, lead_name in enumerate(core.LEAD_NAMES[: signal.shape[1]]):
        leads_data[lead_name] = [round(float(v), 4) for v in sampled[:, idx]]

    return {
        "record_id": record_id,
        "sampling_frequency": fields.get("fs", core.FS),
        "total_samples": signal.shape[0],
        "duration_seconds": round(signal.shape[0] / core.FS, 2),
        "time": times,
        "leads": leads_data,
    }

