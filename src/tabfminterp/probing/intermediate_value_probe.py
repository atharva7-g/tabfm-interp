import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
import torch.nn as nn
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt

def set_seed(seed):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def create_mul_plus_c_dataset(num_samples: int = 1000, noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Dataset with columns [a, b, c] and target y = a * b + c."""
    a = np.random.randn(num_samples)
    b = np.random.randn(num_samples)
    c = np.random.randn(num_samples)
    X = np.stack([a, b, c], axis=1)
    y = a * b + c
    if noise_std > 0:
        y = y + np.random.randn(num_samples) * noise_std
    return X, y


def create_div_plus_c_dataset(num_samples: int = 1000, noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Dataset with columns [a, b, c] and target y = a / b + c. b is kept away from zero."""
    a = np.random.randn(num_samples)
    b = np.random.uniform(0.5, 2.0, size=num_samples) * np.random.choice([-1.0, 1.0], size=num_samples)
    c = np.random.randn(num_samples)
    X = np.stack([a, b, c], axis=1)
    y = a / b + c
    if noise_std > 0:
        y = y + np.random.randn(num_samples) * noise_std
    return X, y


def create_mul_plus_cd_dataset(num_samples: int = 1000, noise_std: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Dataset with columns [a, b, c, d] and target y = a * b + c * d."""
    a = np.random.randn(num_samples)
    b = np.random.randn(num_samples)
    c = np.random.randn(num_samples)
    d = np.random.randn(num_samples)
    X = np.stack([a, b, c, d], axis=1)
    y = a * b + c * d
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

class ProbeMLP(nn.Module):
    """Configurable probe MLP with optional hidden layers."""

    def __init__(self, input_dim: int, output_dim: int = 1, hidden_layers: Optional[List[int]] = None):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers or []:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

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
                       device: str = 'cuda', epochs: int = 100, lr: float = 0.001,
                       hidden_layers: Optional[List[int]] = None) -> Tuple[ProbeMLP, float, float, float]:
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
    probe = ProbeMLP(input_dim=activation_size, output_dim=1, hidden_layers=hidden_layers).to(device)
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
    set_seed(42)
    samples_per_dataset = 1000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print("Will sweep all layers for multiple hidden relations.")
    
    experiment_configs = [
        {
            'name': 'Recover a*b from y = a*b + c',
            'dataset_fn': create_mul_plus_c_dataset,
            'target_fn': lambda X: X[:, 0] * X[:, 1],
            'plot_label': 'a*b (a*b + c)'
        },
        {
            'name': 'Recover a/b from y = a/b + c',
            'dataset_fn': create_div_plus_c_dataset,
            'target_fn': lambda X: X[:, 0] / X[:, 1],
            'plot_label': 'a/b (a/b + c)'
        },
        # {
        #     'name': 'Recover a*b from y = a*b + c*d',
        #     'dataset_fn': create_mul_plus_cd_dataset,
        #     'target_fn': lambda X: X[:, 0] * X[:, 1],
        #     'plot_label': 'a*b (a*b + c*d)'
        # }
    ]
    
    all_plot_data = []
    experiment_summaries = []
    
    for config in experiment_configs:
        print("\n" + "=" * 60)
        print(f"Experiment: {config['name']}")
        print("=" * 60)
        
        X, y = config['dataset_fn'](samples_per_dataset, noise_std=0.0)
        hidden_stat = config['target_fn'](X).mean()
        print(f"Generated dataset (samples={samples_per_dataset}), hidden term mean={hidden_stat:.4f}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        regressor = TabPFNRegressor(device=device, n_estimators=1)
        regressor.fit(X_train, y_train)
        
        y_pred = regressor.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"TabPFN MSE on held-out split: {mse:.4f}, R²: {r2:.4f}")
        
        activations_test = extract_activations(regressor, regressor.model_, X_test, device)
        target_values = config['target_fn'](X_test)
        
        num_samples = next(iter(activations_test.values())).shape[0]
        indices = np.arange(num_samples)
        idx_train, idx_eval = train_test_split(indices, test_size=0.5, random_state=42)
        
        layer_keys = sorted(
            activations_test.keys(),
            key=lambda k: int(k.split('_')[1]) if ('_' in k and k.split('_')[-1].isdigit()) else k
        )
        
        results = []
        for layer_name in layer_keys:
            layer_acts = activations_test[layer_name].cpu()
            probe_train_acts = layer_acts[idx_train]
            probe_eval_acts = layer_acts[idx_eval]
            probe_train_targets = torch.tensor(target_values[idx_train], dtype=torch.float32)
            probe_eval_targets = torch.tensor(target_values[idx_eval], dtype=torch.float32)
            
            probe, train_loss, eval_loss, eval_r2 = train_linear_probe(
                probe_train_acts, probe_train_targets,
                probe_eval_acts, probe_eval_targets,
                device, epochs=100, lr=0.001
            )
            
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
        
        best = max(results, key=lambda r: r['eval_r2'])
        experiment_summaries.append({
            'name': config['name'],
            'mse': mse,
            'r2': r2,
            'best_layer': best['layer'],
            'best_r2': best['eval_r2']
        })
        
        layer_numbers = [int(r['layer'].split('_')[1]) for r in results]
        r2_values = [r['eval_r2'] for r in results]
        sorted_data = sorted(zip(layer_numbers, r2_values))
        layer_numbers_sorted, r2_values_sorted = zip(*sorted_data)
        all_plot_data.append({
            'label': config['plot_label'],
            'layers': list(layer_numbers_sorted),
            'r2': list(r2_values_sorted)
        })
        
        print("-" * 60)
        print(f"Best layer by Eval R2 for {config['name']}: {best['layer']} (EvalR2={best['eval_r2']:.4f})")
    
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    for summary in experiment_summaries:
        print(f"{summary['name']}: TabPFN MSE={summary['mse']:.4f}, R²={summary['r2']:.4f}, "
              f"BestLayer={summary['best_layer']} (EvalR2={summary['best_r2']:.4f})")
    
    plt.figure(figsize=(10, 6))
    for plot_item in all_plot_data:
        plt.plot(
            plot_item['layers'],
            plot_item['r2'],
            marker='o',
            linewidth=2,
            markersize=8,
            label=plot_item['label']
        )
    plt.xlabel('Layer')
    plt.ylabel('R²')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path = 'r2_intermediate_value_probe_across_layers.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nGraph saved to {output_path}")

    run_probe_complexity_experiment(device, samples_per_dataset, layer_name='layer_8')


def run_probe_complexity_experiment(device: str, samples_per_dataset: int, layer_name: str = 'layer_8'):
    """Run probe complexity sweeps for multiple relations on a fixed layer and plot R² vs hidden depth."""
    print("\n" + "=" * 60)
    print(f"Probe complexity sweep for {layer_name}")
    print("=" * 60)

    relation_configs = [
        {
            'name': 'Recover a*b from y = a*b + c',
            'dataset_fn': create_mul_plus_c_dataset,
            'target_fn': lambda X: X[:, 0] * X[:, 1],
            'plot_label': 'a*b (a*b + c)'
        },
        {
            'name': 'Recover a/b from y = a/b + c',
            'dataset_fn': create_div_plus_c_dataset,
            'target_fn': lambda X: X[:, 0] / X[:, 1],
            'plot_label': 'a/b (a/b + c)'
        }
    ]

    hidden_layer_configs = [
        [],
        [128],
        [128, 128],
        [128, 128, 128],
        [128, 128, 128, 128]
    ]

    plot_data = []
    for relation in relation_configs:
        print("\n" + "-" * 60)
        print(f"Relation: {relation['name']}")
        print("-" * 60)

        X, y = relation['dataset_fn'](samples_per_dataset, noise_std=0.0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
        hidden_targets = relation['target_fn'](X_test)

        regressor = TabPFNRegressor(device=device, n_estimators=1)
        regressor.fit(X_train, y_train)

        activations_test = extract_activations(regressor, regressor.model_, X_test, device)
        if layer_name not in activations_test:
            print(f"Layer {layer_name} not found. Available layers: {list(activations_test.keys())}")
            continue

        layer_acts = activations_test[layer_name].cpu()
        indices = np.arange(layer_acts.shape[0])
        idx_train, idx_eval = train_test_split(indices, test_size=0.5, random_state=42)

        probe_train_acts = layer_acts[idx_train]
        probe_eval_acts = layer_acts[idx_eval]
        probe_train_targets = torch.tensor(hidden_targets[idx_train], dtype=torch.float32)
        probe_eval_targets = torch.tensor(hidden_targets[idx_eval], dtype=torch.float32)

        complexity_results = []
        for config in hidden_layer_configs:
            _, _, _, eval_r2 = train_linear_probe(
                probe_train_acts,
                probe_train_targets,
                probe_eval_acts,
                probe_eval_targets,
                device=device,
                epochs=100,
                lr=0.001,
                hidden_layers=config
            )
            complexity_results.append(eval_r2)
            config_label = f"{len(config)} hidden layers" if config else "0 hidden layers"
            print(f"{config_label}: EvalR2={eval_r2:.4f}")

        plot_data.append({
            'label': relation['plot_label'],
            'r2': complexity_results
        })

    if not plot_data:
        print("No probe complexity data collected; graph will not be generated.")
        return

    x_values = [len(cfg) for cfg in hidden_layer_configs]
    plt.figure(figsize=(8, 5))
    for data in plot_data:
        plt.plot(
            x_values,
            data['r2'],
            marker='o',
            linewidth=2,
            label=data['label']
        )
    plt.xlabel('Number of hidden layers in probe')
    plt.ylabel('R² on held-out split')
    plt.title(f'Probe complexity vs R² (Layer {layer_name.split("_")[-1]})')
    plt.grid(True, alpha=0.3)
    plt.xticks(x_values)
    plt.legend()
    plt.tight_layout()
    output_path = f'r2_probe_complexity_{layer_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Probe complexity graph saved to {output_path}")

if __name__ == "__main__":
    main()
