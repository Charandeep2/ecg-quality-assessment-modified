"""
STEP 4: Production Model Export & Integration Verification Script.

Bundles the fine-tuned deep neural network weights with feature normalization
parameters and exports a production-grade PyTorch artifact ready for API and
dashboard deployment.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import core
from core.neural_net import DeepTensorResNet, get_device

FINETUNED_INPUT_FILE = PROJECT_ROOT / "models" / "ptbxl_finetuned_model.pt"
PRODUCTION_OUTPUT_FILE = PROJECT_ROOT / "models" / "ptbxl_production_model.pt"


def main():
    device = get_device()
    print("=" * 75)
    print("📦 STEP 4: PRODUCTION MODEL EXPORT & INTEGRATION VERIFICATION")
    print("=" * 75)

    if not FINETUNED_INPUT_FILE.exists():
        print(f"⚠️ Fine-tuned checkpoint not found at: {FINETUNED_INPUT_FILE}")
        print("   Please run Step 2 first: 'python finetune_ptbxl.py'")
        sys.exit(1)

    print(f"Loading checkpoint: {FINETUNED_INPUT_FILE}")
    checkpoint = torch.load(FINETUNED_INPUT_FILE, map_location=device, weights_only=False)

    state_dict = checkpoint["state_dict"]
    mean = checkpoint["mean"]
    std = checkpoint["std"]
    feature_names = checkpoint.get("feature_names", [])
    threshold = checkpoint.get("threshold", 0.42)
    val_acc = checkpoint.get("val_accuracy", 0.0)
    val_sens = checkpoint.get("val_sensitivity", 0.0)

    # Instantiate production model
    n_features = len(mean)
    model = DeepTensorResNet(in_features=n_features, latent_dim=128).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"  • Input Features:    {n_features}")
    print(f"  • Decision Cutoff:   {threshold:.2f}")
    print(f"  • Model Parameters:  {sum(p.numel() for p in model.parameters()):,}")

    # 1. Verification on Test Records
    print("\nVerifying inferences on real patient ECG recordings...")

    # Load 2 real test cases
    df_chars = pd.read_csv(core.CHAR_FILE)
    df_leads = pd.read_csv(core.LEAD_FILE)
    df = df_chars.merge(df_leads, on="record_id", suffixes=("", "_l"))

    for test_id, expected in [("1002867", "Acceptable"), ("1002603", "Unacceptable")]:
        row = df[df["record_id"].astype(str) == str(test_id)]
        if not row.empty:
            feature_cols = [c for c in df.columns if c not in ["record_id", "label", "label_l"]]
            x_raw = row[feature_cols].fillna(0).to_numpy(dtype=np.float32)[0]
            x_norm = (x_raw - mean) / std

            x_tensor = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model.forward_classify(x_tensor)
                prob = float(torch.softmax(logits, dim=1)[0, 1].cpu().item())
                pred = "Unacceptable" if prob >= threshold else "Acceptable"

            status = "✅ PASS" if pred == expected else "⚠️ MISMATCH"
            print(f"  • Record {test_id} (Expected: {expected:<12}) -> Predicted: {pred:<12} (Risk: {prob*100:5.1f}%) | {status}")

    # 2. Package and Export Final Production Artifact
    production_payload = {
        "model_architecture": "DeepTensorResNet",
        "state_dict": state_dict,
        "mean": mean,
        "std": std,
        "feature_names": feature_names,
        "optimal_threshold": threshold,
        "validation_accuracy": val_acc,
        "validation_sensitivity": val_sens,
        "class_mapping": {0: "Acceptable", 1: "Unacceptable"},
    }

    PRODUCTION_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(production_payload, PRODUCTION_OUTPUT_FILE)

    print("\n" + "=" * 75)
    print(f"🎉 PRODUCTION ARTIFACT EXPORTED SUCCESSFULLY!")
    print(f"File Location: {PRODUCTION_OUTPUT_FILE}")
    print("=" * 75)


if __name__ == "__main__":
    main()
