#!/usr/bin/env python3
"""
Test script for the modified attention extraction function.
"""

import torch
import numpy as np
import sys
import importlib
from tabpfn import TabPFNRegressor

from attention_maps import extract_attention_weights_from_tabpfn, visualize_attention_heads

def test_attention_extraction():
    """Test the attention extraction functionality."""
    print("Testing attention extraction from TabPFN...")
    
    # Create sample data
    np.random.seed(42)
    X_sample = np.random.randn(10, 2)  # 5 samples, 4 features
    #y_sample is a linear function of X_sample
    y_sample = 2*X_sample[:, 0] + 3*X_sample[:, 1]

    X_test = np.random.randn(10, 2)
    y_test = 2*X_test[:, 0] + 3*X_test[:, 1]
    
    # Initialize TabPFN regressor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    regressor = TabPFNRegressor(device=device, n_estimators=1)
    
    # Fit the model
    print("Fitting TabPFN model...")
    regressor.fit(X_sample, y_sample)
    
    # Test attention extraction
    print("Extracting attention weights...")
    try:
        attention_weights = extract_attention_weights_from_tabpfn(regressor, X_test, device)
        
        if attention_weights:
            print(f"Successfully extracted attention weights from {len(attention_weights)} layers")
            for layer_name, weights in attention_weights.items():
                print(f"  {layer_name}: shape {weights.shape}")
        else:       
            print("No attention weights extracted")
            
    except Exception as e:
        print(f"Error extracting attention weights: {e}")
        import traceback
        traceback.print_exc()
    
    # Test visualization - use test data, not training data
    print("\nTesting attention visualization...")
    try:
        visualize_attention_heads(
            model=regressor,
            input_data=X_test,  # Use test data for visualization, not training data
            output_dir='.',
            filename='test_attentions.png',
            device=device,
            sample_idx=0,
            max_layers=None  # Visualize all layers
        )
        print("Visualization completed successfully!")
    except Exception as e:
        print(f"Error in visualization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_attention_extraction()
