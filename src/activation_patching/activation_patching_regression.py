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


def create_joint_dataset(
    num_samples: int = 4000, noise_std: float = 0.0, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create dataset with op_flag enabling both behaviors on matched inputs.
    X = [a, b, c, op_flag], op_flag in {0.0(mul), 1.0(div)}
    y = a*b + c if op_flag==0 else a/b + c
    Returns: X, y, y_mul, y_div for convenience
    """
    rng = np.random.RandomState(seed)
    a = rng.randn(num_samples)
    b = rng.uniform(0.5, 2.0, size=num_samples) * rng.choice(
        [-1.0, 1.0], size=num_samples
    )
    c = rng.randn(num_samples)
    op_flag = rng.randint(0, 2, size=num_samples).astype(float)

    y_mul = a * b + c
    y_div = a / b + c
    y = np.where(op_flag == 0.0, y_mul, y_div)
    X = np.stack([a, b, c, op_flag], axis=1).astype(np.float32)

    if noise_std > 0:
        y = y + rng.randn(num_samples) * noise_std

    return X, y.astype(np.float32), y_mul.astype(np.float32), y_div.astype(np.float32)


def build_paired_batch(
    a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Given arrays of a,b,c of same length K, build paired inputs for mul and div.
    Returns: X_div[K,4], y_div[K], X_mul[K,4], y_mul[K]
    """
    x_div = np.stack([a, b, c, np.ones_like(a, dtype=np.float32)], axis=1).astype(
        np.float32
    )
    x_mul = np.stack([a, b, c, np.zeros_like(a, dtype=np.float32)], axis=1).astype(
        np.float32
    )
    y_div = (a / b + c).astype(np.float32)
    y_mul = (a * b + c).astype(np.float32)
    return x_div, y_div, x_mul, y_mul


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
    regressor: TabPFNRegressor, model, X_np: np.ndarray, target_layers: List[str]
) -> Dict[str, torch.Tensor]:
    """Capture activations and keep original 4D shape, slicing last len(X_np) along the samples axis (dim 1)."""
    activations: Dict[str, torch.Tensor] = {k: None for k in target_layers}

    expected = X_np.shape[0]

    def make_hook(name: str):
        def hook(module, inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            # Expect out shape like (batch=1, samples=N_total, seq_len, hidden). Slice last expected along dim 1
            if out.dim() >= 2 and out.shape[1] >= expected:
                activations[name] = out.detach()[:, -expected:, ...]
            else:
                activations[name] = out.detach()

        return hook

    handles = []
    for i, layer in enumerate(model.transformer_encoder.layers):
        name = f"layer_{i}"
        if name in target_layers:
            handles.append(layer.register_forward_hook(make_hook(name)))

    _ = regressor.predict(X_np)

    for h in handles:
        h.remove()

    return activations


def apply_activation_patching(
    regressor: TabPFNRegressor,
    model,
    X_np: np.ndarray,
    patch_dict: Dict[str, torch.Tensor],
) -> np.ndarray:
    """Patch by slicing cached tensors along samples axis (dim 1) on every internal forward call.
    No pointer; each call receives the last `need` samples so repeated calls work.
    """
    expected = X_np.shape[0]

    # Normalize cached to have samples on dim 1 and at least expected entries
    cached_norm: Dict[str, torch.Tensor] = {}
    for name, cached in patch_dict.items():
        if cached.dim() >= 2 and cached.shape[1] >= expected:
            cached_norm[name] = cached[:, -expected:, ...]
        else:
            cached_norm[name] = cached

    def make_patch_hook(name: str):
        def hook(module, inputs, output):
            out = output[0] if isinstance(output, (tuple, list)) else output
            tensor = cached_norm[name]
            if out.dim() >= 2 and tensor.dim() >= 2:
                need = out.shape[1]
                have = tensor.shape[1]
                if have < need:
                    # Fallback: tile to satisfy need
                    reps = (need + have - 1) // have
                    tiled = tensor.repeat(1, reps, *([1] * (tensor.dim() - 2)))
                    use = tiled[:, -need:, ...]
                else:
                    use = tensor[:, -need:, ...]
                return use.to(out.device, dtype=out.dtype)
            return tensor.to(out.device, dtype=out.dtype)

        return hook

    handles = []
    for name in patch_dict.keys():
        idx = int(name.split("_")[1])
        handles.append(
            model.transformer_encoder.layers[idx].register_forward_hook(
                make_patch_hook(name)
            )
        )

    y_pred = regressor.predict(X_np)

    for h in handles:
        h.remove()

    return y_pred


def train_linear_probe_for_feature(
    layer_acts: Dict[str, torch.Tensor],
    feature_values: np.ndarray,
    device: str = "cuda",
    epochs: int = 200,
    lr: float = 1e-3,
) -> Dict[str, Dict[str, float]]:
    """Train a single-layer linear probe per layer to predict a scalar feature.
    Supports activations captured as 4D (1, N, T, H) by reshaping to (N, T, H) locally.
    Returns dict: layer -> {train_loss, eval_loss, eval_r2}
    """
    results: Dict[str, Dict[str, float]] = {}

    # Determine N from the first layer's activations
    first_acts = next(iter(layer_acts.values()))
    if first_acts.dim() == 4 and first_acts.shape[0] == 1:
        n = first_acts.shape[1]
    else:
        n = first_acts.shape[0]

    indices = np.arange(n)
    idx_train, idx_eval = train_test_split(indices, test_size=0.5, random_state=42)

    y = torch.tensor(feature_values, dtype=torch.float32)
    y_train = y[idx_train]
    y_eval = y[idx_eval]

    device_t = device if torch.cuda.is_available() else "cpu"

    for layer_name, acts in sorted(
        layer_acts.items(), key=lambda kv: int(kv[0].split("_")[1])
    ):
        # Normalize acts to (N, seq_len, hidden) locally for probing
        if acts.dim() == 4 and acts.shape[0] == 1:
            acts_norm = acts[0]  # (N, T, H)
        else:
            acts_norm = acts

        act_train = acts_norm[idx_train]
        act_eval = acts_norm[idx_eval]
        flat_dim = act_train.shape[1] * act_train.shape[2]
        X_train = act_train.view(-1, flat_dim).to(device_t).float()
        X_eval = act_eval.view(-1, flat_dim).to(device_t).float()
        y_train_d = y_train.to(device_t)
        y_eval_d = y_eval.to(device_t)

        probe = LinearProbe(input_dim=flat_dim, output_dim=1).to(device_t)
        criterion = nn.MSELoss()
        optim = torch.optim.SGD(probe.parameters(), lr=lr)

        for _ in range(epochs):
            optim.zero_grad()
            pred = probe(X_train).squeeze()
            loss = criterion(pred, y_train_d)
            loss.backward()
            optim.step()

        with torch.no_grad():
            train_pred = probe(X_train).squeeze()
            eval_pred = probe(X_eval).squeeze()
            train_loss = criterion(train_pred, y_train_d).item()
            eval_loss = criterion(eval_pred, y_eval_d).item()
            r2 = r2_score(y_eval.cpu().numpy(), eval_pred.cpu().numpy())

        results[layer_name] = {
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "eval_r2": r2,
        }

    return results


def run_experiment():
    set_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Train TabPFN on joint dataset
    X, y, y_mul_all, y_div_all = create_joint_dataset(
        num_samples=6000, noise_std=0.0, seed=0
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=42
    )

    regressor = TabPFNRegressor(device=device, n_estimators=1)
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"TabPFN MSE on held-out split: {mse:.4f}, R²: {r2:.4f}")

    model = regressor.model_
    layer_names = get_layer_names(model)

    # 2) Build paired batch from a random subset of the test set: use (a,b,c) from X_test, generate both op modes
    K = X_test.shape[0]
    sel_idx = np.arange(X_test.shape[0])
    a = X_test[sel_idx, 0].astype(np.float32)
    b = X_test[sel_idx, 1].astype(np.float32)
    c = X_test[sel_idx, 2].astype(np.float32)
    X_div, y_div, X_mul, y_mul = build_paired_batch(a, b, c)

    # Base predictions
    y_div_base = regressor.predict(X_div)
    y_mul_base = regressor.predict(X_mul)

    # 3) Cache activations for both conditions for all layers
    acts_div = run_and_cache_activations(regressor, model, X_div, layer_names)
    acts_mul = run_and_cache_activations(regressor, model, X_mul, layer_names)

    # 4) Dual probes: op_flag decodability and a/b decodability
    # For op_flag decoding on paired batch, the true flag vector for X_div is 1, for X_mul is 0.
    # We'll probe on X_div activations (flag=1), and separately on X_mul activations (flag=0), then average R² as a sanity check.
    op_flag_div = np.ones(K, dtype=np.float32)
    op_flag_mul = np.zeros(K, dtype=np.float32)

    probe_op_div = train_linear_probe_for_feature(
        acts_div, op_flag_div, device=device, epochs=150, lr=1e-3
    )
    probe_op_mul = train_linear_probe_for_feature(
        acts_mul, op_flag_mul, device=device, epochs=150, lr=1e-3
    )

    # a_over_b decodability (independent of op flag)
    a_over_b = (a / b).astype(np.float32)
    probe_div_feature = train_linear_probe_for_feature(
        acts_div, a_over_b, device=device, epochs=150, lr=1e-3
    )
    probe_mul_feature = train_linear_probe_for_feature(
        acts_mul, a_over_b, device=device, epochs=150, lr=1e-3
    )

    print("\n" + "=" * 50)
    print("DUAL PROBES")
    print("=" * 50)
    print("Layer  EvalR2(op@div)  EvalR2(op@mul)  EvalR2(a/b@div)  EvalR2(a/b@mul)")
    for name in layer_names:
        r_op_d = probe_op_div[name]["eval_r2"]
        r_op_m = probe_op_mul[name]["eval_r2"]
        r_ab_d = probe_div_feature[name]["eval_r2"]
        r_ab_m = probe_mul_feature[name]["eval_r2"]
        print(
            f"{name:>8}  {r_op_d:>12.4f}  {r_op_m:>12.4f}  {r_ab_d:>14.4f}  {r_ab_m:>14.4f}"
        )

    # 5) Denoising and Noising sweeps per layer
    print("\n" + "=" * 50)
    print("PATCHING SWEEPS (per layer)")
    print("=" * 50)

    results = []
    for name in layer_names:
        # Denoising: patch div activations into mul run at this layer, measure movement toward y_div
        y_mul_patched = apply_activation_patching(
            regressor, model, X_mul, {name: acts_div[name]}
        )
        delta_to_div = np.mean(
            np.abs(y_mul_base - y_div) - np.abs(y_mul_patched - y_div)
        )

        # Noising: patch mul activations into div run at this layer, measure movement away from y_div
        y_div_patched = apply_activation_patching(
            regressor, model, X_div, {name: acts_mul[name]}
        )
        delta_from_div = np.mean(
            np.abs(y_div_base - y_div) - np.abs(y_div_patched - y_div)
        )  # expect negative if harm

        results.append(
            {
                "layer": name,
                "denoise_delta_to_div": float(delta_to_div),
                "noise_delta_from_div": float(delta_from_div),
                "op_probe_div_r2": float(probe_op_div[name]["eval_r2"]),
                "op_probe_mul_r2": float(probe_op_mul[name]["eval_r2"]),
                "ab_probe_div_r2": float(probe_div_feature[name]["eval_r2"]),
                "ab_probe_mul_r2": float(probe_mul_feature[name]["eval_r2"]),
            }
        )

        print(
            f"{name}: DenoiseΔ→div={delta_to_div:+.5f}  NoiseΔ(from div)={delta_from_div:+.5f}  "
            f"opR2(div/mul)={probe_op_div[name]['eval_r2']:.3f}/{probe_op_mul[name]['eval_r2']:.3f}  "
            f"a/bR2(div/mul)={probe_div_feature[name]['eval_r2']:.3f}/{probe_mul_feature[name]['eval_r2']:.3f}"
        )

    # 6) Report best layers by denoising improvement and by noising harm
    best_denoise = max(results, key=lambda r: r["denoise_delta_to_div"])
    worst_noise = min(
        results, key=lambda r: r["noise_delta_from_div"]
    )  # most negative (largest harm)

    print("\n" + "-" * 50)
    print(
        f"Best denoising (div→mul patched): {best_denoise['layer']}  Δ={best_denoise['denoise_delta_to_div']:+.5f}"
    )
    print(
        f"Strongest noising harm (mul→div patched): {worst_noise['layer']}  Δ={worst_noise['noise_delta_from_div']:+.5f}"
    )

    # 7) Final summary
    print("\nFINAL SUMMARY")
    print(f"Held-out MSE: {mse:.4f}, R²: {r2:.4f}")
    print("Interpretation guide:")
    print(
        "- Denoising positive Δ indicates sufficiency at that layer to restore div behavior when patching into mul run."
    )
    print(
        "- Noising negative Δ indicates necessity at that layer for correct div behavior."
    )
    print(
        "- Prefer layers with high |Δ| where op-flag probe R² is low/moderate and a/b probe R² is high."
    )


if __name__ == "__main__":
    run_experiment()
