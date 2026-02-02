from numpy import signedinteger
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNRegressor
import torch.nn as nn
from typing import List, Tuple, Dict


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def create_simple_dataset(num_samples: int = 4000, noise_std: float = 0.0, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Create simple dataset with a + b expression.
    X = [a, b], y = a + b
    Returns: X, y
    """
    rng = np.random.RandomState(seed)
    a = rng.randn(num_samples)
    b = rng.randn(num_samples)
    y = a+b
    X = np.stack([a, b], axis=1).astype(np.float32)

    if noise_std > 0:
        y = y + rng.randn(num_samples) * noise_std

    return X, y.astype(np.float32)


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 1):
        super(LinearProbe, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


@torch.no_grad()
def get_layer_names(model) -> List[str]:
    return [f"layer_{i}" for i, _ in enumerate(model.transformer_encoder.layers)]


@torch.no_grad()
def run_and_cache_activations(
    regressor: TabPFNRegressor,
    model,
    X_np: np.ndarray,
    target_layers: List[str],
) -> Tuple[Dict[str, torch.Tensor], np.ndarray]:
    """Capture activations (keeping sample dimension intact) while returning the corresponding predictions."""
    activations: Dict[str, torch.Tensor] = {k: None for k in target_layers}

    def make_hook(name: str):
        def hook(module, inputs, output):
            test_activations = output
            print(f'cache {name}')
            print(test_activations.shape)
            print('--------------------------------')
            activations[name] = test_activations.detach()
        return hook

    handles = []
    for i, layer in enumerate(model.transformer_encoder.layers):
        name = f"layer_{i}"
        if name in target_layers:
            handles.append(layer.register_forward_hook(make_hook(name)))

    y_pred = regressor.predict(X_np)

    for h in handles:
        h.remove()

    return activations, y_pred


def apply_activation_patching(regressor: TabPFNRegressor, model, X_np: np.ndarray, patch_dict: Dict[str, torch.Tensor]) -> np.ndarray:
    """Patch activations from patch_dict into the forward pass for X_np at the last position.
    X_np is always one sample (shape: 1 x num_features).
    patch_dict contains activations of shape (1, 4, 192) that will replace the last sample
    in the forward pass activations of shape (a, 4, 192).
    """
    def make_patch_hook(name: str):
        def hook(module, inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            # patch_dict[name] is shape (1, 4, 192), replace the last sample in out
            patched_activation = patch_dict[name].to(out.device, dtype=out.dtype)
            # Replace the last index along dimension 0: (a, 4, 192) -> replace [-1:] with patched
            # out_patched = out.clone()
            #print("out_patched.shape", out.shape)
            #print("patched_activation.shape", patched_activation.shape)
            # out_patched[-1:] = patched_activation
            out_patched = out.clone()
            out_patched[:,3000,:-1,:] = patched_activation[:,:,:-1,:]
            return out_patched
        return hook

    handles = []
    for name in patch_dict.keys():
        idx = int(name.split('_')[1])
        handles.append(model.transformer_encoder.layers[idx].register_forward_hook(make_patch_hook(name)))

    y_pred = regressor.predict(X_np)

    for h in handles:
        h.remove()

    return y_pred

def run_experiment():
    set_seed(100)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1) Train TabPFN on simple a + b dataset
    X, y = create_simple_dataset(num_samples=6000, noise_std=0.0, seed=0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    regressor = TabPFNRegressor(device=device, n_estimators=1)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"TabPFN MSE on held-out split: {mse:.4f}, R²: {r2:.4f}")

    model = regressor.model_
    layer_names = get_layer_names(model)

    # 2) Select anchor sample (destination for patching) and source samples
    # Anchor: one specific input we'll patch into
    # Sources: several different inputs we'll extract activations from
    anchor_idx = 0
    num_sources = 5
    source_indices = np.arange(1, num_sources + 1)  # Take next few samples as sources
    
    X_anchor = X_test[anchor_idx:anchor_idx+1].astype(np.float32)  # Shape: (1, 2)
    y_anchor_true = y_test[anchor_idx]
    
    X_sources = X_test[source_indices].astype(np.float32)  # Shape: (num_sources, 2)
    y_sources_true = y_test[source_indices]

    # Combine anchor and sources so we can run a single forward pass
    X_eval = np.concatenate([X_anchor, X_sources], axis=0)

    # 3) Cache activations for anchor and all source samples
    print("\n" + "="*50)
    print("CACHING ACTIVATIONS")
    print("="*50)
    acts_batch, y_eval = run_and_cache_activations(regressor, model, X_eval, layer_names)

    y_anchor_base = float(y_eval[0])
    print(f"\nAnchor sample: a={X_anchor[0, 0]:.4f}, b={X_anchor[0, 1]:.4f}")
    print(f"  True y: {y_anchor_true:.4f}, Predicted: {y_anchor_base:.4f}")
    
    print(f"\nSource samples:")
    for i, idx in enumerate(source_indices):
        print(f"  Source {i+1}: a={X_sources[i, 0]:.4f}, b={X_sources[i, 1]:.4f}, True y: {y_sources_true[i]:.4f}")

    # 4) For each source sample, patch its activations into anchor at each layer
    print("\n" + "="*50)
    print("PATCHING SWEEPS (source → anchor, per layer)")
    print("="*50)
    
    all_results = []
    
    for source_idx, source_num in enumerate(source_indices):
        print(f"\n--- Patching Source {source_num} into Anchor ---")
        source_label = f"Source_{source_num}"

        y_source_true = y_sources_true[source_idx]
        
        # Extract activations for this specific source (need to slice from batch)
        acts_source_single = {}
        for layer_name in layer_names:
            all_activations_at_layer = acts_batch[layer_name]
            batch_source_idx = source_idx + 1  # offset by anchor
            acts_source_single[layer_name] = all_activations_at_layer[:,3000+batch_source_idx:3000+batch_source_idx+1, ...]

        results_source = []
        for name in layer_names:
            # Patch source activations into anchor at this layer
            y_anchor_patched = apply_activation_patching(
                regressor, model, X_eval, 
                {name: acts_source_single[name]}
            )
            y_anchor_patched_val = y_anchor_patched[0]
            
            # Measure change: how much did prediction move toward source's true value?
            pred_change = y_anchor_patched_val - y_anchor_base
            target_change = y_source_true - y_anchor_base
            alignment = pred_change * target_change  # Positive if moving in right direction
            
            error_anchor = np.abs(y_anchor_patched_val - y_anchor_true)
            error_base = np.abs(y_anchor_base - y_anchor_true)
            error_delta = error_anchor - error_base
            
            results_source.append({
                'layer': name,
                'source': source_label,
                'y_anchor_base': float(y_anchor_base),
                'y_anchor_patched': float(y_anchor_patched_val),
                'y_source_true': float(y_source_true),
                'pred_change': float(pred_change),
                'target_change': float(target_change),
                'alignment': float(alignment),
                'error_delta': float(error_delta),
            })
            
            print(f"{name}: base={y_anchor_base:.4f} → patched={y_anchor_patched_val:.4f} "
                  f"(source={y_source_true:.4f}), Δ={pred_change:+.5f}, align={alignment:+.5f}")
        
        all_results.extend(results_source)
    
    # 5) Summary across all sources and layers
    print("\n" + "="*50)
    print("SUMMARY ACROSS ALL SOURCES")
    print("="*50)
    print("Layer       Avg |Δ|    Max |Δ|    Avg Align   Sources with Align > 0")
    for name in layer_names:
        layer_results = [r for r in all_results if r['layer'] == name]
        avg_abs_delta = np.mean([abs(r['pred_change']) for r in layer_results])
        max_abs_delta = np.max([abs(r['pred_change']) for r in layer_results])
        avg_align = np.mean([r['alignment'] for r in layer_results])
        pos_align_count = sum(1 for r in layer_results if r['alignment'] > 0)
        print(f"{name:>8}  {avg_abs_delta:>10.5f}  {max_abs_delta:>10.5f}  {avg_align:>11.5f}  {pos_align_count:>21}")

    # 6) Find most influential layers
    layer_stats = {}
    for name in layer_names:
        layer_results = [r for r in all_results if r['layer'] == name]
        layer_stats[name] = {
            'avg_abs_delta': np.mean([abs(r['pred_change']) for r in layer_results]),
            'avg_alignment': np.mean([r['alignment'] for r in layer_results]),
        }
    
    most_influential = max(layer_stats.items(), key=lambda x: x[1]['avg_abs_delta'])
    best_aligned = max(layer_stats.items(), key=lambda x: x[1]['avg_alignment'])
    
    print("\n" + "-"*50)
    print(f"Most influential layer (highest avg |Δ|): {most_influential[0]}, avg |Δ|={most_influential[1]['avg_abs_delta']:.5f}")
    print(f"Best aligned layer (highest avg alignment): {best_aligned[0]}, avg align={best_aligned[1]['avg_alignment']:.5f}")

    # 7) Final summary
    print("\n" + "="*50)
    print("FINAL SUMMARY")
    print("="*50)
    print(f"Held-out MSE: {mse:.4f}, R²: {r2:.4f}")
    print(f"Anchor sample: a={X_anchor[0, 0]:.4f}, b={X_anchor[0, 1]:.4f}, y_true={y_anchor_true:.4f}, y_pred={y_anchor_base:.4f}")
    print("Interpretation guide:")
    print("- Positive alignment: patching source activations moves anchor prediction toward source's true value")
    print("- Negative alignment: patching moves prediction away from source's true value")
    print("- High |Δ|: large change in prediction when patching at this layer")
    print("- Layers with high |Δ| and positive alignment are most informative about feature transfer")


if __name__ == "__main__":
    run_experiment()
