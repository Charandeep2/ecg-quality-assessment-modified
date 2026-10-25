"""
Core modular package for Adaptive ECG Quality Assessment (Scalogram-Tensor Processing).
"""

from core.classifier import (
    calculate_metrics,
    load_quality_classifier,
    predict_ecg_quality,
)
from core.config import (
    CLASS_NAMES,
    DATA_DIR,
    DEFAULT_MAX_FREQUENCY,
    ENERGY_THRESHOLD,
    FIXED_FEATURE_DIM,
    FIXED_RANK,
    FS,
    HIGH_MAX_FREQUENCY,
    LABEL_MAP,
    LEAD_NAMES,
    MAX_RANK,
    MODEL_FILE,
    MODEL_INFO_FILE,
    PROJECT_ROOT,
    RESULTS_DIR,
    GROUP_FILE,
    CHAR_FILE,
    LEAD_FILE,
    RANK_FILE,
    COMPLETE_RANK_FILE,
    WAVELET,
)
from core.cwt import (
    compute_lead_scalogram,
    create_cwt_tensor,
    select_cwt_frequency,
)
from core.data_loader import (
    detect_duplicate_records,
    find_record,
    load_dataset_ecg,
    load_project_metadata,
    load_rank_results,
)
from core.features import (
    extract_adaptive_core_features,
    extract_fixed_classifier_features,
    pad_core_tensor,
)
from core.pipeline import ECGPipeline
from core.clustering import (
    CLUSTERING_FEATURES,
    CLUSTER_PROFILES,
    get_cluster_model,
    predict_signal_condition,
)
from core.preprocessing import (
    calculate_characteristics,
    clean_ecg_adaptive,
    high_pass_filter,
    inspect_leads,
    should_apply_highpass,
)
from core.tucker import (
    calculate_reconstruction_error,
    calculate_tucker,
    compression_ratio,
    estimate_energy_rank,
    get_device,
    parse_saved_rank,
    representation_size,
)
from core.visualization import (
    plot_3d_tensor,
    plot_core_slice,
    plot_ecg,
    plot_scalogram,
)

__all__ = [
    # Config
    "FS",
    "LEAD_NAMES",
    "WAVELET",
    "FIXED_RANK",
    "MAX_RANK",
    "ENERGY_THRESHOLD",
    "FIXED_FEATURE_DIM",
    "DEFAULT_MAX_FREQUENCY",
    "HIGH_MAX_FREQUENCY",
    "LABEL_MAP",
    "CLASS_NAMES",
    "MODEL_FILE",
    "MODEL_INFO_FILE",
    "PROJECT_ROOT",
    "DATA_DIR",
    "RESULTS_DIR",
    "GROUP_FILE",
    "CHAR_FILE",
    "LEAD_FILE",
    "RANK_FILE",
    "COMPLETE_RANK_FILE",
    # Data Loader
    "find_record",
    "load_dataset_ecg",
    "load_project_metadata",
    "load_rank_results",
    "detect_duplicate_records",
    # Preprocessing
    "calculate_characteristics",
    "inspect_leads",
    "high_pass_filter",
    "should_apply_highpass",
    "clean_ecg_adaptive",
    # CWT
    "create_cwt_tensor",
    "compute_lead_scalogram",
    "select_cwt_frequency",
    # Tucker
    "get_device",
    "calculate_tucker",
    "estimate_energy_rank",
    "calculate_reconstruction_error",
    "representation_size",
    "compression_ratio",
    "parse_saved_rank",
    # Features
    "extract_fixed_classifier_features",
    "pad_core_tensor",
    "extract_adaptive_core_features",
    # Classifier
    "load_quality_classifier",
    "predict_ecg_quality",
    "calculate_metrics",
    # Visualization
    "plot_ecg",
    "plot_scalogram",
    "plot_3d_tensor",
    "plot_core_slice",
    # Pipeline
    "ECGPipeline",
    # Clustering
    "CLUSTERING_FEATURES",
    "CLUSTER_PROFILES",
    "get_cluster_model",
    "predict_signal_condition",
]

