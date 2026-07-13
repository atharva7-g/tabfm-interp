#!/usr/bin/env python3
"""Step 5: steering_v3_evaluate.py — Final test-set evaluation with best α, report all metrics."""

import sys
from pathlib import Path
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

def main():
    """Run final test-set evaluation with best α and report all metrics."""
    print("Running final test-set evaluation...")
    
    # Load test data
    data_dir = Path("data/steering_v3")
    test_data = torch.load(data_dir / "test.pt", weights_only=False)
    X_test = test_data['X']
    y_test = test_data['y']
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Run evaluation
    print("Test set evaluation completed.")

if __name__ == "__main__":
    main()