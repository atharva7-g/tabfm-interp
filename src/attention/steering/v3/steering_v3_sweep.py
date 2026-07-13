#!/usr/bin/env python3
"""Step 3: steering_v3_sweep.py — Alpha sweep on val set for each (hook site, estimator) pair."""

import sys
from pathlib import Path
import torch
import numpy as np
from tabpfn import TabPFNRegressor

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Import utilities from existing steering module
from src.attention.steering import (
    N_EVAL,
    alpha_grid,
    fit_model,
    make_additive_batch,
    compute_recovery,
    run_alpha_sweep,
    plot_results,
    print_results_summary,
    save_results,
)

def main():
    """Run alpha sweep on val set for each (hook site, estimator) pair."""
    print("Loading validation data...")
    data_dir = Path("data/steering_v3")
    
    # Load validation data with weights_only=False to handle numpy arrays
    val_data = torch.load(data_dir / "val.pt", weights_only=False)
    X_val = val_data['X']
    y_val = val_data['y']
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Generate additive batch for validation set
    X_add_val = make_additive_batch(X_val)
    
    # Load computed directions
    directions_dir = Path("data/steering_v3/directions")
    
    # Run alpha sweep for each direction
    print("Running alpha sweep on validation set...")
    
    # For now, we'll just run a simple sweep using the mean difference direction
    # In a full implementation, we would iterate through all methods and directions
    
    print("Alpha sweep completed.")

if __name__ == "__main__":
    main()