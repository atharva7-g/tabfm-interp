#try to patch the activations from one run to another run and trying to influence the answer of the second run
#keep runs small, ten samples each so can examine by eye
import torch
import numpy as np

import torch
import numpy as np
import random
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
import torch.nn as nn
from typing import List, Tuple, Dict

def set_seed(seed):
    """Set random seeds for reproducibility across all sources of randomness"""
    random.seed(seed)  # Python's random module
    np.random.seed(seed)  # NumPy
    torch.manual_seed(seed)  # PyTorch CPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # PyTorch GPU
        torch.cuda.manual_seed_all(seed)  # All GPU devices
    # Set deterministic behavior for PyTorch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dummy_dataset(num_samples=10, num_features=2):
    X = np.random.randn(num_samples, num_features)
    y = np.random.randn(num_samples)
    return X, y

def train_tabpfn_regressor(X, y):
    regressor = TabPFNRegressor(n_estimators=1,device='cuda' if torch.cuda.is_available() else 'cpu')
    regressor.fit(X, y)
    return regressor

def test_attention_activations(regressor, model, X_data: np.ndarray, device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """Extract activations from all transformer layers"""
    
    # Track call counts for each hook
    call_counts = {}
    
    def attention_hook(name):
        call_counts[name] = 0
        def hook(model, input, output):
            if hasattr(model, 'compute_qkv') and hasattr(model, '_compute'):
                call_counts[name] += 1
                hidden_states = input[0] if isinstance(input, tuple) else input
                print(f"{name} (call #{call_counts[name]}): {hidden_states.shape}")
        return hook
    
    hook_handles = []
    for i, layer in enumerate(model.transformer_encoder.layers):
        if hasattr(layer, 'self_attn_between_features') and hasattr(layer, 'self_attn_between_items'):
            handle1 = layer.self_attn_between_features.register_forward_hook(attention_hook(f'layer_{i}_features'))
            handle2 = layer.self_attn_between_items.register_forward_hook(attention_hook(f'layer_{i}_items'))
            hook_handles.append(handle1)
            hook_handles.append(handle2)
    
    with torch.no_grad():
        _ = regressor.predict(X_data)
    
    for handle in hook_handles:
        handle.remove()
    
    # Print summary
    print("\n=== Call Summary ===")
    for name in sorted(call_counts.keys()):
        print(f"{name}: {call_counts[name]} calls")
    
    return None


def main():
    set_seed(42)
    X_data1, y_data1 = create_dummy_dataset(num_samples=10, num_features=2)
    X_data2, y_data2 = create_dummy_dataset(num_samples=10, num_features=2)
    regressor = train_tabpfn_regressor(X_data1, y_data1)
    test_attention_activations(regressor, regressor.model_, X_data2)

if __name__ == "__main__":
    main()


