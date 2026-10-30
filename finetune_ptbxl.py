"""
STEP 2: Supervised Fine-Tuning Script (GPU Accelerated).

Loads the pretrained foundation model checkpoint from Step 1 and fine-tunes
it on the PhysioNet Challenge 2011 Set A dataset (998 records) using Focal Loss
on your NVIDIA GeForce RTX 3050 GPU.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import core
from core.neural_net import DeepTensorResNet, FocalLoss, get_device

PRETRAINED_MODEL_FILE = PROJECT_ROOT / "models" / "ptbxl_pretrained_foundation.pt"
OUTPUT_FINETUNED_FILE = PROJECT_ROOT / "models" / "ptbxl_finetuned_model.pt"

BATCH_SIZE = 64
FINETUNE_EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
DECISION_THRESHOLD = 0.42


def load_target_data():
    """Load PhysioNet 2011 Set A labeled records."""
    df_groups = pd.read_csv(core.GROUP_FILE)
    df_chars = pd.read_csv(core.CHAR_FILE)
    df_leads = pd.read_csv(core.LEAD_FILE)

    df = df_groups.merge(df_chars, on="record_id", suffixes=("", "_c")).merge(
        df_leads, on="record_id", suffixes=("", "_l")
    )

    feature_cols = [
        c for c in df.columns
        if c not in ["record_id", "label", "label_c", "label_l", "signal_hash", "group_size"]
    ]

    X = df[feature_cols].fillna(0).to_numpy(dtype=np.float32)
    y = (df["label"] == "Unacceptable").astype(int).to_numpy()

    return X, y, feature_cols


def main():
    device = get_device()
    print("=" * 75)
    print("🎯 STEP 2: SUPERVISED FINE-TUNING ON PHYSIO-NET SET A (GPU ACCELERATED)")
    print("=" * 75)
    print(f"Hardware Device: {device}")
    if torch.cuda.is_available():
        print(f"  • GPU Device: {torch.cuda.get_device_name(0)}")
    print("=" * 75)

    # 1. Load Pretrained Foundation Weights
    if not PRETRAINED_MODEL_FILE.exists():
        print(f"⚠️ Pretrained foundation checkpoint not found at: {PRETRAINED_MODEL_FILE}")
        print("   Please run Step 1 first: 'python train_ptbxl_pretrain.py'")
        sys.exit(1)

    print(f"Loading pretrained foundation checkpoint: {PRETRAINED_MODEL_FILE}")
    checkpoint = torch.load(PRETRAINED_MODEL_FILE, map_location=device, weights_only=False)

    # 2. Load PhysioNet Target Dataset
    X_raw, y_raw, feature_names = load_target_data()
    n_features = X_raw.shape[1]
    print(f"\nLoaded PhysioNet Target Dataset:")
    print(f"  • Total Records:          {len(y_raw)}")
    print(f"  • Acceptable Records (0): {(y_raw == 0).sum()}")
    print(f"  • Unacceptable Records (1): {(y_raw == 1).sum()}")
    print(f"  • Features Count:         {n_features}")

    # Standardize using checkpoint statistics or target
    mean = checkpoint.get("mean", X_raw.mean(axis=0))
    std = checkpoint.get("std", X_raw.std(axis=0) + 1e-7)
    X_norm = (X_raw - mean) / std

    # Train / Validation Split (80% train, 20% holdout validation)
    X_train, X_val, y_train, y_val = train_test_split(
        X_norm, y_raw, test_size=0.20, random_state=42, stratify=y_raw
    )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    # 3. Initialize Model with Pretrained Weights
    model = DeepTensorResNet(in_features=n_features, latent_dim=128).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    print("  ✅ Pretrained encoder weights successfully loaded into model!")

    # 4. Supervised Fine-Tuning with Focal Loss
    criterion = FocalLoss(alpha=torch.tensor([1.0, 3.43]), gamma=2.0, device=device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=FINETUNE_EPOCHS)

    print(f"\nFine-tuning for {FINETUNE_EPOCHS} epochs with Focal Loss on your RTX 3050 GPU...")
    t_start = time.time()
    model.train()
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model.forward_classify(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)
        scheduler.step()

        if epoch % 10 == 0 or epoch == FINETUNE_EPOCHS:
            vram_str = ""
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated() / (1024 ** 2)
                vram_str = f" | VRAM: {vram_used:.1f} MB"
            print(f"  • Epoch {epoch:2d}/{FINETUNE_EPOCHS} | Focal Loss: {total_loss/len(X_train):.6f}{vram_str}")

    print(f"\n✅ Fine-tuning completed in {time.time() - t_start:.2f} seconds!")

    # 5. Validation Performance
    model.eval()
    with torch.no_grad():
        logits = model.forward_classify(torch.tensor(X_val, dtype=torch.float32).to(device))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = (probs >= DECISION_THRESHOLD).astype(int)

    val_acc = accuracy_score(y_val, preds) * 100
    val_sens = recall_score(y_val, preds) * 100
    val_spec = (confusion_matrix(y_val, preds)[0, 0] / (y_val == 0).sum()) * 100
    val_f1 = f1_score(y_val, preds) * 100
    cm = confusion_matrix(y_val, preds)

    print("\n" + "=" * 75)
    print("📈 VALIDATION PERFORMANCE (20% HOLDOUT):")
    print("=" * 75)
    print(f"  • Accuracy:    {val_acc:.2f}%")
    print(f"  • Sensitivity: {val_sens:.2f}% (Unacceptable detection)")
    print(f"  • Specificity: {val_spec:.2f}% (Acceptable retention)")
    print(f"  • F1-Score:    {val_f1:.2f}%")
    print(f"  • Confusion:   TN={cm[0,0]}, FP={cm[0,1]}, FN={cm[1,0]}, TP={cm[1,1]}")
    print("=" * 75)

    # 6. Save Fine-Tuned Checkpoint
    OUTPUT_FINETUNED_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "mean": mean,
        "std": std,
        "feature_names": feature_names,
        "threshold": DECISION_THRESHOLD,
        "val_accuracy": val_acc,
        "val_sensitivity": val_sens,
    }, OUTPUT_FINETUNED_FILE)

    print(f"\n💾 Fine-Tuned Model Checkpoint saved to: {OUTPUT_FINETUNED_FILE}")
    print("=" * 75)
    print("Next Step: Run 'python evaluate_ptbxl_cv.py' to run 5-Fold Stratified Cross-Validation.")
    print("=" * 75)


if __name__ == "__main__":
    main()
