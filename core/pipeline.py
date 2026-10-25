"""
Unified End-to-End ECG Quality Assessment Pipeline.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from core.classifier import load_quality_classifier, predict_ecg_quality
from core.config import (
    DEFAULT_MAX_FREQUENCY,
    FIXED_RANK,
    MAX_RANK,
    MODEL_FILE,
)
from core.cwt import create_cwt_tensor, select_cwt_frequency
from core.data_loader import find_record, load_dataset_ecg
from core.clustering import predict_signal_condition
from core.preprocessing import (
    calculate_characteristics,
    high_pass_filter,
    inspect_leads,
    should_apply_highpass,
)
from core.tucker import (
    calculate_reconstruction_error,
    calculate_tucker,
    compression_ratio,
    estimate_energy_rank,
    representation_size,
)


class ECGPipeline:
    """
    End-to-end pipeline for 12-lead ECG quality assessment using
    Continuous Wavelet Transforms (CWT) and Higher-Order Tucker Decomposition.
    """

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = Path(model_path) if model_path else MODEL_FILE
        self.classifier_artifact = load_quality_classifier(self.model_path)

    def process_record(
        self,
        record_id: str,
        adaptive_preprocessing: bool = False,
        adaptive_cwt: bool = False,
        adaptive_rank: bool = False,
    ) -> Dict[str, Any]:
        """
        Process a recording given its PhysioNet record ID.
        """
        signal, fields = load_dataset_ecg(record_id)
        result = self.process_signal(
            signal,
            adaptive_preprocessing=adaptive_preprocessing,
            adaptive_cwt=adaptive_cwt,
            adaptive_rank=adaptive_rank,
        )
        result["record_id"] = record_id
        result["fields"] = fields
        return result

    def process_signal(
        self,
        signal: np.ndarray,
        adaptive_preprocessing: bool = False,
        adaptive_cwt: bool = False,
        adaptive_rank: bool = False,
    ) -> Dict[str, Any]:
        """
        Execute full end-to-end processing pipeline on a 12-lead ECG signal:
        1. Lead inspection and statistical characterization
        2. Preprocessing (raw or adaptive 0.5 Hz highpass)
        3. CWT time-frequency tensor generation (40 Hz or adaptive 50 Hz)
        4. Tucker decomposition (fixed rank (50, 20, 6) or 95% energy adaptive rank)
        5. Compression metrics and reconstruction error
        6. Quality classification (Acceptable vs Unacceptable)

        Returns:
            Dict containing signal inspection, features, decomposition, error, and predictions.
        """
        # Step 1: Inspection, Characteristics, and Physiological Condition Clustering
        lead_inspection_df = inspect_leads(signal)
        characteristics = calculate_characteristics(signal)
        condition_cluster = predict_signal_condition(characteristics)

        # Step 2: Adaptive Preprocessing
        filter_applied = False
        if adaptive_preprocessing and should_apply_highpass(characteristics):
            working_signal = high_pass_filter(signal)
            filter_applied = True
        else:
            working_signal = signal.copy()

        # Step 3: CWT Frequency Selection
        if adaptive_cwt:
            max_freq = select_cwt_frequency(characteristics)
        else:
            max_freq = DEFAULT_MAX_FREQUENCY

        tensor, frequencies = create_cwt_tensor(
            working_signal, maximum_frequency=max_freq
        )

        # Step 4: Tucker Rank Selection & Decomposition
        if adaptive_rank:
            rank = estimate_energy_rank(tensor, max_rank=MAX_RANK)
        else:
            rank = FIXED_RANK

        core, factors = calculate_tucker(tensor, rank=rank)

        # Step 5: Compression & Reconstruction Metrics
        recon_error = calculate_reconstruction_error(tensor, core, factors)
        param_count = representation_size(tensor.shape, rank)
        comp_ratio = compression_ratio(tensor.shape, rank)

        # Step 6: Quality Prediction (Fixed pipeline inference)
        prediction_result = predict_ecg_quality(signal, model_path=self.model_path)

        return {
            "prediction": prediction_result["label"],
            "prediction_class": prediction_result["prediction"],
            "probability_unacceptable": prediction_result["probability_unacceptable"],
            "condition_cluster": condition_cluster,
            "characteristics": characteristics,
            "lead_inspection": lead_inspection_df,
            "filter_applied": filter_applied,
            "max_frequency": max_freq,
            "tensor": tensor,
            "frequencies": frequencies,
            "rank": rank,
            "core": core,
            "factors": factors,
            "reconstruction_error": recon_error,
            "representation_size": param_count,
            "compression_ratio": comp_ratio,
            "features": prediction_result["features"],
        }

