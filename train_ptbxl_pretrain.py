"""
STEP 1: PTB-XL Foundation Model Pretraining Script (GPU Accelerated).

Trains a Self-Supervised Denoising Tensor Autoencoder on unlabelled 12-lead ECG records
using your NVIDIA GeForce RTX 3050 GPU and exports the pretrained foundation weights.
"""

import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import core
from core.neural_net import DeepTensorResNet, get_device

# =========================================================================
# ⚙️ CONFIGURATION: PASTE YOUR PTB-XL DATASET PATH HERE
# =========================================================================
# Paste the path to your extracted PTB-XL dataset folder from Kaggle / PhysioNet:
PTBXL_DIR = Path(r"C:\dataset\ptb-xl")  # <-- PASTE YOUR EXTRACTED PTB-XL PATH HERE

# Number of PTB-XL records to pretrain on (e.g. 1000 for fast ~10-15 min run, or None for all)
NUM_PRETRAIN_RECORDS: Optional[int] = 1000

# Training Settings
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
OUTPUT_MODEL_FILE = PROJECT_ROOT / "models" / "ptbxl_pretrained_foundation.pt"


def extract_ptbxl_features(ptbxl_path: Path, max_records: Optional[int] = 1000) -> Optional[np.ndarray]:
    """Extract mathematical physical & spectral signal characteristics from PTB-XL."""
    if not ptbxl_path.exists():
        return None

    import wfdb

    csv_path = ptbxl_path / "ptbxl_database.csv"
    if csv_path.exists():
        meta = pd.read_csv(csv_path)
        records_col = "filename_hr" if "filename_hr" in meta.columns else "filename_lr"
        file_list = [ptbxl_path / f for f in meta[records_col].dropna()]
    else:
        file_list = [p.with_suffix("") for p in ptbxl_path.glob("**/*.hea")]

    if not file_list:
        return None

    if max_records and len(file_list) > max_records:
        file_list = file_list[:max_records]

    print(f"Extracting mathematical features from {len(file_list)} PTB-XL records...")
    feature_rows = []
    t0 = time.time()

    for idx, rec_path in enumerate(file_list, 1):
        try:
            sig, fields = wfdb.rdsamp(str(rec_path))
            fs = int(fields.get("fs", 500))
            if sig.ndim != 2 or sig.shape[1] != 12:
                continue

            chars = core.calculate_characteristics(sig, fs=fs)
            lead_df = core.inspect_leads(sig)

            row = [
                chars["mean_rms"], chars["max_rms"],
                chars["mean_centered_rms"], chars["max_centered_rms"],
                chars["mean_std"], chars["max_std"],
                chars["mean_abs_amplitude"],
                chars["mean_baseline_variation"], chars["max_baseline_variation"],
                chars["mean_high_frequency_ratio"], chars["max_high_frequency_ratio"],
                chars["mean_powerline_ratio"], chars["max_powerline_ratio"],
                12,
                int(lead_df["Constant lead"].sum()),
                int(lead_df["Contains -80 marker"].sum()),
                int(lead_df["Contains -80 marker"].sum()),
                0, 0,
                float(lead_df["Standard deviation"].min()),
                float(lead_df["Standard deviation"].median()),
                float(lead_df["Standard deviation"].mean()),
                float(lead_df["Maximum"].max() - lead_df["Minimum"].min()),
                float((lead_df["Maximum"] - lead_df["Minimum"]).median()),
                float((lead_df["Maximum"] - lead_df["Minimum"]).mean()),
                float(lead_df["Unique values"].min()),
                float(lead_df["Unique values"].median()),
                0.0, 0.0, 0.0
            ]
            feature_rows.append(row)

            if idx % 200 == 0 or idx == len(file_list):
                print(f"  • Processed {idx}/{len(file_list)} records ({time.time()-t0:.1f}s)")
        except Exception:
            continue

    if not feature_rows:
        return None

    return np.array(feature_rows, dtype=np.float32)


def main():
    device = get_device()
    print("=" * 75)
    print("🚀 STEP 1: PTB-XL FOUNDATION MODEL PRETRAINING (GPU ACCELERATED)")
    print("=" * 75)
    print(f"Hardware Device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  • GPU Name:       {gpu_name}")
        print(f"  • Dedicated VRAM: {vram:.2f} GB")
        print(f"  • CUDA Status:    ACTIVE")
    print("=" * 75)

    # 1. Load Data for Pretraining
    print(f"\nChecking PTB-XL path at: {PTBXL_DIR}")
    X_pretrain = None
    if PTBXL_DIR.exists():
        print("  ✅ PTB-XL folder found!")
        X_pretrain = extract_ptbxl_features(PTBXL_DIR, max_records=NUM_PRETRAIN_RECORDS)
    else:
        print("  ⚠️ PTB-XL folder not found at the configured path.")
        print("     (Paste your exact path into line 32 of this file: PTBXL_DIR = Path(...))\n")
        print("  💡 Using PhysioNet Unsupervised Tensor Pretraining as fallback...")
        df_chars = pd.read_csv(core.CHAR_FILE)
        df_leads = pd.read_csv(core.LEAD_FILE)
        df = df_chars.merge(df_leads, on="record_id", suffixes=("", "_l"))
        feature_cols = [c for c in df.columns if c not in ["record_id", "label", "label_l"]]
        X_pretrain = df[feature_cols].fillna(0).to_numpy(dtype=np.float32)

    # Normalize
    mean = X_pretrain.mean(axis=0)
    std = X_pretrain.std(axis=0) + 1e-7
    X_norm = (X_pretrain - mean) / std
    n_features = X_norm.shape[1]

    print(f"\nTotal Pretraining Records: {len(X_norm)}")
    print(f"Features per Record:      {n_features}")

    # 2. Build Denoising Autoencoder
    dataset = TensorDataset(torch.tensor(X_norm, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DeepTensorResNet(in_features=n_features, latent_dim=128).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # 3. Train Autoencoder
    print(f"\nTraining Denoising Autoencoder for {EPOCHS} epochs on your GPU...")
    t_start = time.time()
    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for (bx,) in loader:
            bx = bx.to(device)
            # Add Gaussian noise
            noisy_bx = bx + torch.randn_like(bx) * 0.15

            optimizer.zero_grad()
            reconstruction = model.forward_pretrain(noisy_bx)
            loss = criterion(reconstruction, bx)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)

        if epoch % 10 == 0 or epoch == EPOCHS:
            vram_str = ""
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated() / (1024 ** 2)
                vram_str = f" | RTX 3050 VRAM: {vram_used:.1f} MB"
            print(f"  • Epoch {epoch:2d}/{EPOCHS} | Reconstruction Loss: {total_loss/len(X_norm):.6f}{vram_str}")

    print(f"\n✅ Pretraining completed in {time.time() - t_start:.2f} seconds!")

    # 4. Save Foundation Weights Checkpoint
    OUTPUT_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "n_features": n_features,
        "latent_dim": 128,
    }, OUTPUT_MODEL_FILE)

    print(f"💾 Foundation Model Checkpoint saved to: {OUTPUT_MODEL_FILE}")
    print("=" * 75)
    print("Next Step: Run 'python finetune_ptbxl.py' to fine-tune on PhysioNet Set A.")
    print("=" * 75)


if __name__ == "__main__":
    main()
