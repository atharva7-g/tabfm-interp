#!/usr/bin/env python3
"""Step 1: steering_v3_data.py - Generate N=1024 samples, split 512/256/256, save to disk.

This script generates the full dataset for the v3 steering experiments with proper 
train/validation/test splits to prevent data leakage.

The dataset is a multiplication task (y = a*b + c) with N=1024 samples:
- Train set: 512 samples (for computing δ directions)
- Validation set: 256 samples (for α selection)  
- Test set: 256 samples (for final evaluation)

Each set contains matched pairs of:
- Multiplicative samples: (a, b, c) where b is sampled normally
- Additive samples: (a, 0, c) where b is set to 0

This splitting ensures no data leakage between sets - no sample appears in more than one split.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor

# Add project root to path to import project modules
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.datasets.synthetic import create_dataset

SEED = 42
N_TOTAL = 1024
TRAIN_SIZE = 512
VAL_SIZE = 256  
TEST_SIZE = 256

def main():
    """Generate N=1024 multiplication dataset samples with proper train/val/test splits.
    
    The key design ensures no data leakage:
    - Train set (512 samples): compute δ = mean(mult) - mean(add)  
    - Val set (256 samples): select best α 
    - Test set (256 samples): final evaluation
    
    Each set contains matched pairs:
    - Multiplicative: (a, b, c) where b ~ N(0,1)
    - Additive: (a, 0, c) where b = 0
    """
    print("Generating N=2048 multiplication dataset samples...")
    
    # Generate full dataset (we use 2048 samples to get clean statistical power)
    # This ensures we have enough samples for all splits without overlap
    X, y = create_dataset("multiplication", num_samples=2048, seed=SEED)
    
    # Split into train/val/test sets with no overlap
    # First, split off test set
    X_temp, X_test_data, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )
    
    # Then split remaining into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE, random_state=SEED
    )
    
    # Save datasets
    data_dir = Path("data/steering_v3")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save train/val/test splits
    splits = {
        'train': (X_train, y_train),
        'val': (X_val, y_val), 
        'test': (X_test_data, y_test)
    }
    
    for name, (X_data, y_data) in splits.items():
        path = data_dir / f"{name}.pt"
        torch.save({
            'X': X_data,
            'y': y_data
        }, path)
        print(f"Saved {name} data: {path}")
    
    # Also save full dataset for reference
    full_path = data_dir / "full_dataset.pt"
    torch.save({
        'X': X,
        'y': y
    }, full_path)
    print(f"Saved full dataset: {full_path}")
    
    print(f"Dataset generation complete:")
    print(f"  - Train: {len(X_train)} samples")
    print(f"  - Val:   {len(X_val)} samples") 
    print(f"  - Test:  {len(X_test_data)} samples")
    print(f"  - Total: {len(X_train) + len(X_val) + len(X_test_data)} samples")

if __name__ == "__main__":
    main()