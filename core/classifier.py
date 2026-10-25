"""
Model inference and classification performance evaluation.
"""

import os
import warnings
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from core.config import CLASS_NAMES, FIXED_FEATURE_DIM, MODEL_FILE
from core.features import extract_fixed_classifier_features


def load_quality_classifier(
    model_path: Optional[Union[str, Path]] = None
) -> Optional[Dict[str, Any]]:
    """
    Load the trained Random Forest quality classifier artifact from disk.

    The returned artifact dict contains:
    - 'model': Trained RandomForestClassifier instance
    - 'n_features': Expected feature dimension (6000)
    - 'class_mapping': Dict {0: 'Acceptable', 1: 'Unacceptable'}
    - 'feature_method': Description string of feature pipeline
    """
    path = Path(model_path) if model_path else MODEL_FILE
    if not path.exists():
        return None
    return joblib.load(path)


def predict_ecg_quality(
    signal: np.ndarray,
    model_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Execute full inference pipeline for a single 12-lead ECG recording:
    1. Load model artifact
    2. Extract fixed 6,000-dimensional CWT + Tucker core features
    3. Run Random Forest prediction and probability estimation

    Returns:
        dict:
            - 'label': 'Acceptable' or 'Unacceptable'
            - 'prediction': 0 (Acceptable) or 1 (Unacceptable)
            - 'probability_unacceptable': Posterior probability of unacceptable class
            - 'features': Extracted (1, 6000) feature vector
            - 'tensor': Intermediate (5000, 100, 12) CWT tensor
            - 'core': Intermediate (50, 20, 6) Tucker core tensor
    """
    artifact = load_quality_classifier(model_path)
    if artifact is None:
        target_path = model_path or MODEL_FILE
        raise FileNotFoundError(f"Trained model artifact not found at: {target_path}")

    features, tensor, core = extract_fixed_classifier_features(signal)

    expected_dim = artifact.get("n_features", FIXED_FEATURE_DIM)
    if features.shape[1] != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch: extracted {features.shape[1]}, model expects {expected_dim}."
        )

    model = artifact["model"]
    prediction = int(model.predict(features)[0])

    probabilities = model.predict_proba(features)[0]
    class_positions = {int(label): idx for idx, label in enumerate(model.classes_)}
    prob_unacceptable = float(probabilities[class_positions[1]])

    class_mapping = artifact.get("class_mapping", CLASS_NAMES)
    label_str = class_mapping.get(prediction, CLASS_NAMES.get(prediction, str(prediction)))

    return {
        "label": label_str,
        "prediction": prediction,
        "probability_unacceptable": prob_unacceptable,
        "features": features,
        "tensor": tensor,
        "core": core,
    }


def calculate_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
) -> Dict[str, Union[float, int]]:
    """
    Calculate comprehensive binary classification metrics for ECG quality assessment.
    Positive class is Unacceptable (1); Negative class is Acceptable (0).

    Metrics:
    - Sensitivity: True Positive Rate (detection rate of unacceptable records)
    - Specificity: True Negative Rate (correct retention of acceptable records)
    - Precision: Positive Predictive Value
    - Accuracy: Overall correct classifications
    - F1-Score: Harmonic mean of precision and sensitivity
    - Confusion Matrix counts: TN, FP, FN, TP
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, zero_division=0))
    sens = float(recall_score(y_t, y_p, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_t, y_p, zero_division=0))

    return {
        "accuracy": acc,
        "precision": prec,
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total": len(y_t),
    }

