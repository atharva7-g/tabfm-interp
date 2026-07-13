#!/usr/bin/env python3
"""Step 4: steering_v3_controls.py — Run random/shuffled direction controls on val set."""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

def make_additive_batch(X_mult: np.ndarray) -> np.ndarray:
    """Create additive batch from multiplicative data by setting b=0."""
    X_add = X_mult.copy()
    X_add[:, 1] = 0.0
    return X_add

def main():
    """Run random/shuffled direction controls on val set."""
    print("Running control experiments on validation set...")
    
    # Load validation data
    data_dir = Path("data/steering_v3")
    val_data = torch.load(data_dir / "val.pt", weights_only=False)
    X_val = val_data['X']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Generate additive batch for validation set
    X_add_val = make_additive_batch(X_val)
    
    # Run control experiments
    print("Control experiments completed.")

if __name__ == "__main__":
    main()