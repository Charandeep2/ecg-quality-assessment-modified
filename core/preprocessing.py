"""
Signal quality inspection, statistical characterization, and adaptive preprocessing.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from scipy.signal import butter, sosfiltfilt

from core.config import FS, LEAD_NAMES


def calculate_characteristics(signal: np.ndarray, fs: int = FS) -> Dict[str, float]:
    """
    Compute the full suite of 11 candidate statistical and spectral signal characteristics
    (as established in Adaptive Scalogram-Tensor Processing research, NIT Goa):

    1. Magnitude / Energy:
       - mean_rms, max_rms: Raw root-mean-square amplitude across leads
       - mean_centered_rms, max_centered_rms: DC-offset-subtracted dynamic RMS
       - mean_abs_amplitude: Mean absolute electrical potential
    2. Variation / Baseline:
       - mean_std, max_std: Standard deviation per lead
       - mean_baseline_variation, max_baseline_variation: Standard deviation of 200ms trend
    3. Frequency / Noise:
       - mean_high_frequency_ratio, max_high_frequency_ratio: Spectral energy >= 20 Hz
    4. Powerline Interference:
       - mean_powerline_ratio, max_powerline_ratio: Mains electrical grid hum (50/60 Hz)
    """
    if signal.ndim != 2:
        raise ValueError("Signal must have shape (samples, leads).")

    # 1. Magnitude & Centered Measures
    rms_values = np.sqrt(np.mean(signal ** 2, axis=0))
    centered_signal = signal - np.mean(signal, axis=0)
    centered_rms_values = np.sqrt(np.mean(centered_signal ** 2, axis=0))
    std_values = np.std(signal, axis=0)
    abs_values = np.mean(np.abs(signal), axis=0)

    # 2. Baseline Wander via 200ms Moving-Average
    window = max(3, int(fs * 0.2))
    kernel = np.ones(window) / window
    baseline_values = []
    for lead in range(signal.shape[1]):
        trend = np.convolve(signal[:, lead], kernel, mode="same")
        baseline_values.append(float(np.std(trend)))

    # 3. Spectral Frequency & Powerline Ratios (Periodogram PSD)
    hf_ratios = []
    pl_ratios = []
    for lead in range(signal.shape[1]):
        lead_std = std_values[lead]
        if lead_std <= 1e-8:
            hf_ratios.append(0.0)
            pl_ratios.append(0.0)
            continue

        freqs, pxx = scipy_signal.periodogram(signal[:, lead], fs=fs)
        total_p = float(np.sum(pxx))
        if total_p <= 1e-12:
            hf_ratios.append(0.0)
            pl_ratios.append(0.0)
            continue

        # High-frequency power ratio (muscle tremor / EMG >= 20 Hz)
        hf_p = float(np.sum(pxx[freqs >= 20.0]))
        hf_ratios.append(hf_p / total_p)

        # Powerline power ratio (50 Hz and 60 Hz mains grid hum)
        pl_p = float(np.sum(pxx[(freqs >= 49.0) & (freqs <= 51.0)])) + \
               float(np.sum(pxx[(freqs >= 59.0) & (freqs <= 61.0)]))
        pl_ratios.append(pl_p / total_p)

    return {
        "mean_rms": float(np.mean(rms_values)),
        "max_rms": float(np.max(rms_values)),
        "mean_centered_rms": float(np.mean(centered_rms_values)),
        "max_centered_rms": float(np.max(centered_rms_values)),
        "mean_std": float(np.mean(std_values)),
        "max_std": float(np.max(std_values)),
        "mean_abs_amplitude": float(np.mean(abs_values)),
        "mean_baseline_variation": float(np.mean(baseline_values)),
        "max_baseline_variation": float(np.max(baseline_values)),
        "mean_high_frequency_ratio": float(np.mean(hf_ratios)),
        "max_high_frequency_ratio": float(np.max(hf_ratios)),
        "mean_powerline_ratio": float(np.mean(pl_ratios)),
        "max_powerline_ratio": float(np.max(pl_ratios)),
    }


def inspect_leads(
    signal: np.ndarray, lead_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Inspect each ECG lead individually to detect physical acquisition artifacts:
    - Constant / flat leads (std <= 1e-8)
    - Clipped or missing sensor dropout markers (-80 mV artifact in CinC 2011)
    - Amplitude bounds (min, max) and count of unique digital levels
    """
    if lead_names is None:
        lead_names = LEAD_NAMES[: signal.shape[1]]

    rows = []
    for index in range(signal.shape[1]):
        values = signal[:, index]
        std_val = float(np.std(values))
        unique_count = int(np.unique(values).size)

        rows.append({
            "Lead": lead_names[index] if index < len(lead_names) else f"Lead_{index+1}",
            "Minimum": float(np.min(values)),
            "Maximum": float(np.max(values)),
            "Standard deviation": std_val,
            "Unique values": unique_count,
            "Constant lead": bool(std_val <= 1e-8),
            "Contains -80 marker": bool(
                np.any(np.isclose(values, -80.0, atol=1e-6))
            ),
        })

    return pd.DataFrame(rows)


def high_pass_filter(
    signal: np.ndarray, cutoff: float = 0.5, order: int = 4, fs: int = FS
) -> np.ndarray:
    """
    Apply a zero-phase 4th-order Butterworth high-pass filter (0.5 Hz default cutoff)
    using Second-Order Sections (SOS) to suppress baseline wander.
    """
    sos = butter(order, cutoff, btype="highpass", fs=fs, output="sos")
    filtered = np.column_stack([
        sosfiltfilt(sos, signal[:, lead])
        for lead in range(signal.shape[1])
    ])
    return filtered.astype(np.float32)


def should_apply_highpass(
    characteristics: Dict[str, float],
    baseline_threshold: float = 0.08,
) -> bool:
    """
    Adaptive preprocessing rule (Week 6 & Week 8):
    Determines whether a high-pass filter should be applied based on the
    computed baseline variation.
    """
    return characteristics.get("mean_baseline_variation", 0.0) >= baseline_threshold


def clean_ecg_adaptive(
    signal: np.ndarray,
    fs: int = FS,
    baseline_threshold: float = 0.08,
) -> Tuple[np.ndarray, bool, Dict[str, float]]:
    """
    Execute adaptive preprocessing:
    1. Compute signal characteristics
    2. Decide whether high-pass filtering is needed
    3. Return (filtered_or_raw_signal, filter_was_applied, characteristics)
    """
    chars = calculate_characteristics(signal, fs=fs)
    apply_hp = should_apply_highpass(chars, baseline_threshold=baseline_threshold)

    processed_signal = high_pass_filter(signal, fs=fs) if apply_hp else signal.copy()
    return processed_signal, apply_hp, chars

