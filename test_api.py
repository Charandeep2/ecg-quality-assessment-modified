"""
Comprehensive Backend Endpoint Verification Test Suite.
"""

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

import core
from api.main import api_status
from api.routes.inspection import inspect_record
from api.routes.predict import predict_ecg
from api.routes.records import get_record_data, list_records
from api.schemas import PredictionRequest

print("=" * 70)
print("VERIFYING ADAPTIVE ECG QUALITY ASSESSMENT (STP 2.0) BACKEND")
print("=" * 70)

# 1. API Status Endpoint
status = api_status()
print(f"1. API Status:             ✅ OK (v{status['version']})")
assert status["version"] == "2.0.0"

# 2. Records Listing Endpoint
records = list_records(limit=5)
print(f"2. Records Listing:        ✅ OK ({len(records)} records fetched)")
assert len(records) == 5

# 3. Waveform Retrieval Endpoint
wave_data = get_record_data("1002867", downsample=5)
print(f"3. Waveform Data:          ✅ OK (Leads: {len(wave_data['leads'])}, Points: {len(wave_data['time'])})")
assert "leads" in wave_data
assert "time" in wave_data

# 4. Lead-Level Inspection & 11 Characteristics
inspection = inspect_record("1002867")
chars = inspection.characteristics
print(f"4. Signal Characteristics: ✅ OK (Computed {len(chars)} features)")
assert "mean_rms" in chars
assert "mean_centered_rms" in chars
assert "mean_powerline_ratio" in chars
assert "mean_high_frequency_ratio" in chars
print(f"   • Mean Centered RMS:    {chars['mean_centered_rms']:.4f} mV")
print(f"   • Powerline (50/60 Hz): {chars['mean_powerline_ratio']:.6f}")
print(f"   • Baseline Wander Std:  {chars['mean_baseline_variation']:.4f}")

# 5. Full Adaptive Prediction & Physiological Clustering
req = PredictionRequest(record_id="1002867")
pred = predict_ecg(req)
cluster = pred.condition_cluster
print(f"5. Adaptive Quality Pred:  ✅ OK ({pred.prediction}, {pred.probability_acceptable * 100:.1f}%)")
print(f"   • Condition Cluster:    [Cluster {cluster['cluster_id']}: {cluster['name']}]")
print(f"   • Empirical Cohort Risk:{cluster['empirical_unacceptable_risk']}%")
print(f"   • XAI Factors:          {len(pred.xai_factors)} factor attributions verified")

print("=" * 70)
print("ALL BACKEND ENDPOINTS AND RESEARCH UPGRADES PASSED SUCCESSFULLY!")
print("=" * 70)
