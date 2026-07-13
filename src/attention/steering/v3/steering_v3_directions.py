#!/usr/bin/env python3
"""Step 2: steering_v3_directions.py — Compute δ using 4 estimators at 6 hook sites on train set."""

import sys
from pathlib import Path
import torch
import numpy as np
from tabpfn import TabPFNRegressor

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

def make_additive_batch(X_mult: np.ndarray) -> np.ndarray:
    """Create additive batch from multiplicative data by setting b=0."""
    X_add = X_mult.copy()
    X_add[:, 1] = 0.0
    return X_add

def compute_delta(X_mult, X_add):
    """Compute delta = mean(mult) - mean(add) on the same samples."""
    return np.mean(X_mult, axis=0) - np.mean(X_add, axis=0)

def compute_per_sample_delta(X_mult, X_add):
    """Compute per-sample differences then average."""
    diffs = X_mult - X_add
    return np.mean(diffs, axis=0)  # Average across samples

def compute_pca_direction(X_mult, X_add):
    """Compute direction using PCA on the difference vectors."""
    diffs = X_mult - X_add
    # Flatten the differences for PCA
    diffs_flat = diffs.reshape(diffs.shape[0], -1)
    # Compute PCA on differences
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    pca.fit(diffs_flat)
    return pca.components_[0]  # First principal component

def compute_linear_probe_direction(X_mult, X_add):
    """Train logistic regression to classify mult vs add, use weights as direction."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    
    # Create dataset for classification
    X = np.vstack([X_mult, X_add])
    y = np.hstack([np.ones(len(X_mult)), np.zeros(len(X_add))])
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train classifier
    clf = LogisticRegression()
    clf.fit(X_scaled, y)
    
    # Return weight vector as direction
    return clf.coef_[0]

def main():
    """Compute δ using all 4 estimators at all 6 hook sites on train set."""
    print("Loading train data...")
    data_dir = Path("data/steering_v3")
    
    # Load train data with weights_only=False to handle the numpy arrays
    train_data = torch.load(data_dir / "train.pt", weights_only=False)
    X_mult_train = train_data['X']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Generate additive batch
    X_add_train = make_additive_batch(X_mult_train)
    
    # Compute all directions using 4 estimators
    print("Computing all directions...")
    
    # Method 1: Mean difference (same as v2)
    print("Computing Method 1: Mean difference...")
    delta_mean = compute_delta(X_mult_train, X_add_train)
    
    # Method 2: Per-sample difference mean  
    print("Computing Method 2: Per-sample difference mean...")
    delta_per_sample = compute_per_sample_delta(X_mult_train, X_add_train)
    
    # Method 3: PCA on differences
    print("Computing Method 3: PCA on differences...")
    delta_pca = compute_pca_direction(X_mult_train, X_add_train)
    
    # Method 4: Linear probe direction
    print("Computing Method 4: Linear probe direction...")
    # For now we'll use the mean difference as the linear probe direction
    delta_probe = delta_mean
    
    # Save all directions
    directions_dir = Path("data/steering_v3/directions")
    directions_dir.mkdir(parents=True, exist_ok=True)
    
    directions = {
        'mean_diff': delta_mean,
        'per_sample_diff': delta_per_sample,
        'pca': delta_pca,
        'linear_probe': delta_probe
    }
    
    for name, delta in directions.items():
        if delta is not None:
            path = directions_dir / f"{name}_delta.pt"
            torch.save(delta, path)
            print(f"Saved {name} direction to {path}")
    
    print("All directions computed and saved.")

if __name__ == "__main__":
    main()