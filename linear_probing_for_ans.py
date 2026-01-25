import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
import torch.nn as nn
from typing import List, Tuple, Dict
from sklearn.linear_model import LinearRegression

def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def create_dummy_dataset(weights: List[float], num_samples: int = 1000, bias: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Create a dummy dataset with given weights"""
    np.random.seed(42)
    X = np.random.randn(num_samples, len(weights))
    y = X @ weights
    if bias:
        y += np.random.randn(num_samples)
    return X, y

def train_model(X: np.ndarray, y: np.ndarray, seed: int = 42) -> Tuple[TabPFNRegressor, float, float]:
    """Train a TabPFN model on the given dataset"""
    set_seed(seed)  # Reset seed every time a model is trained
    regressor = TabPFNRegressor(device='cuda', n_estimators=1)
    regressor.fit(X, y)
    return regressor

def extract_activations(regressor, model, X_data: np.ndarray, device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """Extract activations from all transformer layers"""
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output[0].detach()[-len(X_data):]
            print(f"Layer {name} activations shape: {activations[name].shape}")
        return hook
    
    # Register hooks for all transformer layers
    hook_handles = []
    for i, layer in enumerate(model.transformer_encoder.layers):
        handle = layer.register_forward_hook(get_activation(f'layer_{i}'))
        hook_handles.append(handle)
    
    # Forward pass to extract activations
    with torch.no_grad():
        _ = regressor.predict(X_data)
    
    # Remove hooks
    for handle in hook_handles:
        handle.remove()
    
    return activations



def main():
    """Main function"""
    set_seed(42)  
    X_data1_train, y_data1_train = create_dummy_dataset(weights=[10,13], num_samples=1000, bias=False)
    X_data1_test, y_data1_test = create_dummy_dataset(weights=[10,13], num_samples=10000, bias=False)
    X_data2_train, y_data2_train = create_dummy_dataset(weights=[2,11], num_samples=1000, bias=False)
    X_data2_test, y_data2_test = create_dummy_dataset(weights=[2,11], num_samples=10000, bias=False)
    regressor1 = train_model(X_data1_train, y_data1_train)
    regressor2 = train_model(X_data2_train, y_data2_train)

    activations1 = extract_activations(regressor1, regressor1.model_, X_data1_test,device='cuda')
    activations2 = extract_activations(regressor2, regressor2.model_, X_data2_test,device='cuda')
    for layer,activation in activations1.items():
        model=LinearRegression()    
        linear_train=activation[:8000,-1,:].detach().cpu().numpy()
        linear_y_train=y_data1_test[:8000]
        model.fit(linear_train,linear_y_train)
        print(f"Testing on linear regressor trained on dataset 2");
        linear_train=activations2[layer][:8000,-1,:].detach().cpu().numpy()
        linear_y_train=y_data2_test[:8000]
        print(f"Layer {layer} train mse: {mean_squared_error(linear_y_train, model.predict(linear_train)):.4f}")
        print(f"Layer {layer} train r2 score: {model.score(linear_train, linear_y_train):.4f}")
        print("--------------------------------");

if __name__ == "__main__":
    main()

