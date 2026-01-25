import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
import torch.nn as nn
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt

def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# Replace linear weight-based dataset with a dataset for y = a/b + c
def create_ab_dataset(num_samples: int = 1000, noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Create dataset where X has columns [a, b, c] and y = a/b + c + noise.
    b is sampled away from zero to avoid division issues."""
    a = np.random.randn(num_samples)
    # sample b away from zero (uniform between 0.5 and 2.0) with random sign
    b = np.random.uniform(0.5, 2.0, size=num_samples) * np.random.choice([-1.0, 1.0], size=num_samples)
    c = np.random.randn(num_samples)
    X = np.stack([a, b, c], axis=1)
    y = a * b + c
    if noise_std > 0:
        y = y + np.random.randn(num_samples) * noise_std
    return X, y

# class LinearProbe(nn.Module):
#     """1-layer neural network probe for activation analysis"""
#     def __init__(self, input_dim: int, output_dim: int = 1, hidden_dim: int = 256):
#         super(LinearProbe, self).__init__()
#         self.network = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, output_dim)
#         )

#     def forward(self, x):
#         return self.network(x)

class LinearProbe(nn.Module):
    # linear probe with no hidden layer and no activation
    def __init__(self, input_dim: int, output_dim: int = 1):
            super(LinearProbe, self).__init__()
            self.linear = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.linear(x)

def extract_activations(regressor, model, X_data: np.ndarray, device: str = 'cuda') -> Dict[str, torch.Tensor]:
    """Extract activations from all transformer layers"""
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output[0].detach()[-len(X_data):]
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

def train_linear_probe(activation_train: torch.Tensor, target_train: torch.Tensor,
                       activation_eval: torch.Tensor, target_eval: torch.Tensor,
                       device: str = 'cuda', epochs: int = 100, lr: float = 0.001) -> Tuple[LinearProbe, float, float, float]:
    """Train a linear probe given explicit train/eval activation splits.
    Returns: probe, train_loss, eval_loss, eval_r2"""
    # Flatten activations
    activation_size = activation_train.shape[1] * activation_train.shape[2]
    activation_train_flat = activation_train.view(-1, activation_size)
    activation_eval_flat = activation_eval.view(-1, activation_size)
    
    # Move to device and cast
    activation_train_flat = activation_train_flat.to(device).float()
    activation_eval_flat = activation_eval_flat.to(device).float()
    target_train = target_train.to(device).float()
    target_eval = target_eval.to(device).float()
    
    # Initialize probe
    probe = LinearProbe(input_dim=activation_size, output_dim=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(probe.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = probe(activation_train_flat).squeeze()
        loss = criterion(outputs, target_train)
        loss.backward()
        optimizer.step()
    
    # Compute final training loss and evaluate on provided eval split
    with torch.no_grad():
        train_outputs = probe(activation_train_flat).squeeze()
        train_loss = criterion(train_outputs, target_train)
        eval_outputs = probe(activation_eval_flat).squeeze()
        eval_loss = criterion(eval_outputs, target_eval)
        r2 = r2_score(target_eval.cpu().numpy(), eval_outputs.cpu().numpy())
    
    return probe, train_loss.item(), eval_loss.item(), r2

def main():
    # Set random seed
    set_seed(42)
    
    # Configuration
    samples_per_dataset = 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    TARGET_LAYER = 'layer_10'  # kept for reference but we will sweep all layers
    
    print(f"Using device: {device}")
    print(f"Will sweep all layers and train probes to predict a/b from activations")
    
    # Create a single dataset
    X, y = create_ab_dataset(samples_per_dataset, noise_std=0.0)
    mean_a_over_b = np.mean(X[:,0] * X[:,1])
    print(f"Generated single dataset (samples={samples_per_dataset}), mean(a/b)={mean_a_over_b:.4f}")
    
    # Split dataset into train/test for TabPFN
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    
    # Train TabPFN on training split
    regressor = TabPFNRegressor(device=device, n_estimators=1)
    regressor.fit(X_train, y_train)
    
    # Evaluate TabPFN performance on test split
    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"TabPFN MSE on held-out split: {mse:.4f}, R²: {r2:.4f}")
    
    # Extract activations only from the held-out TabPFN test split
    activations_test = extract_activations(regressor, regressor.model_, X_test, device)
    
    # Prepare probe data: per-sample ground-truth a/b on X_test
    a_over_b_targets = X_test[:, 0] * X_test[:, 1]  # numpy array
    
    # Create a single index split on the test set to be reused across layers
    num_samples = next(iter(activations_test.values())).shape[0]
    indices = np.arange(num_samples)
    idx_train, idx_eval = train_test_split(indices, test_size=0.5, random_state=42)
    
    # Sweep all layers
    layer_keys = sorted(
        activations_test.keys(),
        key=lambda k: int(k.split('_')[1]) if ('_' in k and k.split('_')[-1].isdigit()) else k
    )
    
    results = []
    for layer_name in layer_keys:
        layer_acts = activations_test[layer_name].cpu()  # (N, seq_len, hidden)
        # select same indices across layers
        probe_train_acts = layer_acts[idx_train]
        probe_eval_acts = layer_acts[idx_eval]
        probe_train_targets = torch.tensor(a_over_b_targets[idx_train], dtype=torch.float32)
        probe_eval_targets = torch.tensor(a_over_b_targets[idx_eval], dtype=torch.float32)
    
        # Train probe for this layer
        probe, train_loss, eval_loss, eval_r2 = train_linear_probe(
            probe_train_acts, probe_train_targets,
            probe_eval_acts, probe_eval_targets,
            device, epochs=100, lr=0.001
        )
    
        # Compute predicted mean on eval split for reporting
        activation_size = probe_eval_acts.shape[1] * probe_eval_acts.shape[2]
        eval_flat = probe_eval_acts.view(-1, activation_size).to(device).float()
        with torch.no_grad():
            preds = probe(eval_flat).squeeze().cpu().numpy()
        pred_mean = preds.mean() if preds.size > 0 else float('nan')
        true_mean = probe_eval_targets.numpy().mean()
    
        results.append({
            'layer': layer_name,
            'train_loss': train_loss,
            'eval_loss': eval_loss,
            'eval_r2': eval_r2,
            'true_mean': true_mean,
            'pred_mean': pred_mean
        })
    
        print(f"{layer_name}: TrainLoss={train_loss:.4f}, EvalLoss={eval_loss:.4f}, EvalR2={eval_r2:.4f}, TrueMean={true_mean:.4f}, PredMean={pred_mean:.4f}")
    
    # Summary: print per-layer and best layer by eval R2
    print("\n" + "="*50)
    print("LAYER SWEEP SUMMARY")
    print("="*50)
    best = max(results, key=lambda r: r['eval_r2'])
    for r in results:
        print(f"{r['layer']}: EvalR2={r['eval_r2']:.4f}, EvalLoss={r['eval_loss']:.4f}, TrueMean={r['true_mean']:.4f}, PredMean={r['pred_mean']:.4f}")
    print("-"*50)
    print(f"Best layer by Eval R2: {best['layer']} (EvalR2={best['eval_r2']:.4f})")
    
    # Final summary
    print("\nFINAL SUMMARY")
    print(f"TabPFN test MSE: {mse:.4f}, R2: {r2:.4f}")
    print(f"Dataset mean(a/b): {mean_a_over_b:.4f}")
    
    # Create and save graph of R2 results across layers
    layer_numbers = [int(r['layer'].split('_')[1]) for r in results]
    r2_values = [r['eval_r2'] for r in results]
    
    # Sort by layer number to ensure correct order
    sorted_data = sorted(zip(layer_numbers, r2_values))
    layer_numbers_sorted, r2_values_sorted = zip(*sorted_data)
    
    plt.figure(figsize=(10, 6))
    plt.plot(layer_numbers_sorted, r2_values_sorted, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel('Layer')
    plt.ylabel('R²')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('r2_intermediate_value_probe_across_layers.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nGraph saved to r2_across_layers.png")

if __name__ == "__main__":
    main()
